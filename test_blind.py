"""Tests for Blind."""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch, tmp_path):
    """Each test gets a fresh database."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("BLIND_DB", str(db_path))
    # Patch the module-level DB_PATH
    import blind.config
    import blind.db

    monkeypatch.setattr(blind.config, "DB_PATH", db_path)
    monkeypatch.setattr(blind.db, "DB_PATH", db_path)


@pytest.fixture
def db():
    from blind.db import get_db

    return get_db()


class TestDatabase:
    def test_creates_tables(self, db):
        tables = {t["name"] for t in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert {"comparisons", "responses", "elo_ratings", "elo_history"} <= tables

    def test_idempotent(self, db):
        from blind.db import get_db

        db2 = get_db()
        assert db2 is not None


class TestElo:
    def test_expected_score_equal(self):
        from blind.elo import expected_score

        assert expected_score(1500, 1500) == pytest.approx(0.5)

    def test_expected_score_higher_wins(self):
        from blind.elo import expected_score

        assert expected_score(1700, 1500) > 0.5

    def test_winner_gains(self, db):
        from blind.elo import update_elo

        update_elo(db, "a", "b", "code")
        ra = db.execute("SELECT rating FROM elo_ratings WHERE model_id='a'").fetchone()["rating"]
        rb = db.execute("SELECT rating FROM elo_ratings WHERE model_id='b'").fetchone()["rating"]
        assert ra > 1500 > rb

    def test_draw_equalizes(self, db):
        from blind.elo import update_elo

        update_elo(db, "a", "b", "code")  # a leads
        ra_before = db.execute("SELECT rating FROM elo_ratings WHERE model_id='a'").fetchone()["rating"]
        update_elo(db, "a", "b", "code", draw=True)
        ra_after = db.execute("SELECT rating FROM elo_ratings WHERE model_id='a'").fetchone()["rating"]
        assert ra_after < ra_before  # leader loses points in draw

    def test_separate_categories(self, db):
        from blind.elo import update_elo

        update_elo(db, "a", "b", "code")
        update_elo(db, "b", "a", "writing")
        ra_code = db.execute("SELECT rating FROM elo_ratings WHERE model_id='a' AND category='code'").fetchone()["rating"]
        ra_write = db.execute("SELECT rating FROM elo_ratings WHERE model_id='a' AND category='writing'").fetchone()["rating"]
        assert ra_code > 1500 > ra_write

    def test_records_history(self, db):
        from blind.elo import update_elo

        update_elo(db, "a", "b", "code")
        history = db.execute("SELECT * FROM elo_history").fetchall()
        assert len(history) == 2  # one for each model

    def test_confidence_low_n(self):
        from blind.elo import confidence_interval

        assert confidence_interval(2, 1, 0) is None

    def test_confidence_sufficient(self):
        from blind.elo import confidence_interval

        ci = confidence_interval(10, 8, 2)
        assert 0 < ci < 400


class TestAutoCategorize:
    def test_code(self):
        from blind.analysis import auto_categorize

        assert auto_categorize("write a Python function") == "code"

    def test_writing(self):
        from blind.analysis import auto_categorize

        assert auto_categorize("write me a poem") == "writing"

    def test_math(self):
        from blind.analysis import auto_categorize

        assert auto_categorize("solve this equation") == "math"

    def test_fallback(self):
        from blind.analysis import auto_categorize

        assert auto_categorize("hello there") == "general"


class TestCost:
    def test_known_model(self):
        from blind.providers import estimate_cost

        assert estimate_cost("gpt-4o", 1000, 500) == (1000 * 2.50 + 500 * 10.00) / 1_000_000

    def test_unknown_model(self):
        from blind.providers import estimate_cost

        assert estimate_cost("unknown", 1000, 500) == 0.0


class TestPositionBias:
    def test_no_data(self, db):
        from blind.analysis import check_position_bias

        assert check_position_bias(db) is None

    def test_detects_bias(self, db):
        from blind.analysis import check_position_bias

        for _ in range(12):
            db.execute("INSERT INTO comparisons (id, prompt, status, winner_position) VALUES (?, 'x', 'voted', 'A')", (os.urandom(6).hex(),))
        db.commit()
        assert "A" in check_position_bias(db)

    def test_no_bias_when_balanced(self, db):
        from blind.analysis import check_position_bias

        for i in range(12):
            db.execute("INSERT INTO comparisons (id, prompt, status, winner_position) VALUES (?, 'x', 'voted', ?)", (os.urandom(6).hex(), "AB"[i % 2]))
        db.commit()
        assert check_position_bias(db) is None


class TestStreak:
    def test_detects(self, db):
        from blind.analysis import check_streak

        for i in range(5):
            db.execute("INSERT INTO comparisons (id, prompt, status, winner_model, created_at) VALUES (?, 'x', 'voted', 'a', datetime('now', ?))", (os.urandom(6).hex(), f"-{5 - i} minutes"))
        db.commit()
        assert "a" in check_streak(db, "a")

    def test_no_streak(self, db):
        from blind.analysis import check_streak

        assert check_streak(db, "a") is None


class TestDemo:
    def test_responses_differ(self):
        from blind.providers import demo_response

        a, _, _, _ = demo_response("model-alpha", "test")
        b, _, _, _ = demo_response("model-beta", "test")
        assert a != b


class TestConfig:
    def test_default(self, tmp_path, monkeypatch):
        import blind.config

        monkeypatch.setattr(blind.config, "CONFIG_PATH", tmp_path / "nope.json")
        assert len(blind.config.load_config()["models"]) >= 2

    def test_custom(self, tmp_path, monkeypatch):
        import blind.config

        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"models": [{"id": "x"}]}))
        monkeypatch.setattr(blind.config, "CONFIG_PATH", p)
        assert blind.config.load_config()["models"][0]["id"] == "x"


class TestCompareFlow:
    def test_demo_vote(self, db, monkeypatch):
        from blind.commands import cmd_compare

        monkeypatch.setattr("builtins.input", lambda _: "A")
        args = MagicMock(prompt=["test"], category="code", models=2, demo=True, rank=False)
        cmd_compare(args)
        assert db.execute("SELECT COUNT(*) as n FROM comparisons WHERE status='voted'").fetchone()["n"] == 1

    def test_demo_tie(self, db, monkeypatch):
        from blind.commands import cmd_compare

        monkeypatch.setattr("builtins.input", lambda _: "tie")
        args = MagicMock(prompt=["test"], category=None, models=2, demo=True, rank=False)
        cmd_compare(args)
        assert db.execute("SELECT COUNT(*) as n FROM comparisons WHERE status='tie'").fetchone()["n"] == 1

    def test_demo_skip(self, db, monkeypatch):
        from blind.commands import cmd_compare

        monkeypatch.setattr("builtins.input", lambda _: "skip")
        args = MagicMock(prompt=["test"], category=None, models=2, demo=True, rank=False)
        cmd_compare(args)
        assert db.execute("SELECT COUNT(*) as n FROM comparisons WHERE status='skipped'").fetchone()["n"] == 1

    def test_demo_rank(self, db, monkeypatch):
        from blind.commands import cmd_compare

        monkeypatch.setattr("builtins.input", lambda _: "BAC")
        args = MagicMock(prompt=["test"], category="general", models=3, demo=True, rank=True)
        cmd_compare(args)
        assert db.execute("SELECT COUNT(*) as n FROM comparisons WHERE status='ranked'").fetchone()["n"] == 1


class TestExport:
    def test_creates_csv(self, db, tmp_path):
        from blind.commands import cmd_export

        db.execute("INSERT INTO comparisons (id, prompt, category, status, winner_model) VALUES ('x', 'test', 'code', 'voted', 'a')")
        db.execute("INSERT INTO responses (comparison_id, model_id, label, content, latency_ms) VALUES ('x', 'a', 'A', 'resp', 100)")
        db.commit()
        out = tmp_path / "out.csv"
        cmd_export(MagicMock(output=str(out)))
        assert "a" in out.read_text()
