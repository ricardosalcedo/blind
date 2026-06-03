"""Tests for Blind — personal blind LLM comparison tool."""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

os.environ["BLIND_DB"] = ":memory:"
os.environ["BLIND_CONFIG"] = "/tmp/blind_test_config.json"

import blind


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch, tmp_path):
    """Each test gets a fresh SQLite database."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("BLIND_DB", str(db_path))
    blind.DB_PATH = db_path
    yield


@pytest.fixture
def db():
    return blind.get_db()


# ──────────────────────────────────────────────────────────────────────────────
# Database
# ──────────────────────────────────────────────────────────────────────────────

class TestDatabase:
    def test_get_db_creates_tables(self, db):
        tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        names = {t["name"] for t in tables}
        assert "comparisons" in names
        assert "responses" in names
        assert "elo_ratings" in names

    def test_get_db_idempotent(self, db):
        db2 = blind.get_db()
        assert db2 is not None


# ──────────────────────────────────────────────────────────────────────────────
# Elo
# ──────────────────────────────────────────────────────────────────────────────

class TestElo:
    def test_expected_score_equal(self):
        assert blind.expected_score(1500, 1500) == pytest.approx(0.5)

    def test_expected_score_higher_rated(self):
        score = blind.expected_score(1700, 1500)
        assert score > 0.5
        assert score < 1.0

    def test_expected_score_lower_rated(self):
        score = blind.expected_score(1300, 1500)
        assert score < 0.5
        assert score > 0.0

    def test_update_elo_winner_gains(self, db):
        blind.update_elo(db, "model-a", "model-b", "code")
        ra = db.execute("SELECT rating FROM elo_ratings WHERE model_id='model-a' AND category='code'").fetchone()
        rb = db.execute("SELECT rating FROM elo_ratings WHERE model_id='model-b' AND category='code'").fetchone()
        assert ra["rating"] > 1500
        assert rb["rating"] < 1500

    def test_update_elo_draw(self, db):
        # First give unequal ratings
        blind.update_elo(db, "model-a", "model-b", "code")
        ra_before = db.execute("SELECT rating FROM elo_ratings WHERE model_id='model-a' AND category='code'").fetchone()["rating"]
        # Now draw
        blind.update_elo(db, "model-a", "model-b", "code", draw=True)
        ra_after = db.execute("SELECT rating FROM elo_ratings WHERE model_id='model-a' AND category='code'").fetchone()["rating"]
        # Winner from before should lose rating in a draw (they were favored)
        assert ra_after < ra_before

    def test_update_elo_tracks_wins_losses(self, db):
        blind.update_elo(db, "model-a", "model-b", "code")
        blind.update_elo(db, "model-a", "model-b", "code")
        row = db.execute("SELECT wins, losses FROM elo_ratings WHERE model_id='model-a' AND category='code'").fetchone()
        assert row["wins"] == 2
        assert row["losses"] == 0

    def test_update_elo_separate_categories(self, db):
        blind.update_elo(db, "model-a", "model-b", "code")
        blind.update_elo(db, "model-b", "model-a", "writing")
        ra_code = db.execute("SELECT rating FROM elo_ratings WHERE model_id='model-a' AND category='code'").fetchone()["rating"]
        ra_write = db.execute("SELECT rating FROM elo_ratings WHERE model_id='model-a' AND category='writing'").fetchone()["rating"]
        assert ra_code > 1500
        assert ra_write < 1500

    def test_elo_confidence_low_n(self):
        assert blind.elo_confidence(2, 1, 0) is None

    def test_elo_confidence_sufficient_n(self):
        ci = blind.elo_confidence(10, 8, 2)
        assert ci is not None
        assert ci > 0
        assert ci < 400  # Should be reasonable


# ──────────────────────────────────────────────────────────────────────────────
# Auto-categorize
# ──────────────────────────────────────────────────────────────────────────────

class TestAutoCategorize:
    def test_code_detection(self):
        assert blind.auto_categorize("write a Python function to sort") == "code"
        assert blind.auto_categorize("debug this error in my class") == "code"

    def test_writing_detection(self):
        assert blind.auto_categorize("write me a poem about rain") == "writing"
        assert blind.auto_categorize("draft an email to my boss") == "writing"

    def test_math_detection(self):
        assert blind.auto_categorize("solve this equation for x") == "math"
        assert blind.auto_categorize("calculate the probability") == "math"

    def test_research_detection(self):
        assert blind.auto_categorize("explain quantum computing") == "research"
        assert blind.auto_categorize("what is the difference between TCP and UDP") == "research"

    def test_creative_detection(self):
        assert blind.auto_categorize("brainstorm ideas for a startup") == "creative"

    def test_general_fallback(self):
        assert blind.auto_categorize("hello there") == "general"


# ──────────────────────────────────────────────────────────────────────────────
# Cost estimation
# ──────────────────────────────────────────────────────────────────────────────

class TestCost:
    def test_known_model(self):
        cost = blind.estimate_cost("gpt-4o", 1000, 500)
        assert cost > 0
        assert cost == (1000 * 2.50 + 500 * 10.00) / 1_000_000

    def test_unknown_model(self):
        assert blind.estimate_cost("unknown-model", 1000, 500) == 0.0

    def test_zero_tokens(self):
        assert blind.estimate_cost("gpt-4o", 0, 0) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Position bias
# ──────────────────────────────────────────────────────────────────────────────

class TestPositionBias:
    def test_no_data(self, db):
        assert blind.check_position_bias(db) is None

    def test_insufficient_data(self, db):
        for _ in range(5):
            db.execute("INSERT INTO comparisons (id, prompt, status, winner_position) VALUES (?, 'x', 'voted', 'A')",
                       (os.urandom(6).hex(),))
        db.commit()
        assert blind.check_position_bias(db) is None  # <10 comparisons

    def test_detects_bias(self, db):
        for i in range(12):
            db.execute("INSERT INTO comparisons (id, prompt, status, winner_position) VALUES (?, 'x', 'voted', 'A')",
                       (os.urandom(6).hex(),))
        db.commit()
        result = blind.check_position_bias(db)
        assert result is not None
        assert "A" in result

    def test_no_bias_when_balanced(self, db):
        for i in range(10):
            pos = "A" if i % 2 == 0 else "B"
            db.execute("INSERT INTO comparisons (id, prompt, status, winner_position) VALUES (?, 'x', 'voted', ?)",
                       (os.urandom(6).hex(), pos))
        db.commit()
        assert blind.check_position_bias(db) is None


# ──────────────────────────────────────────────────────────────────────────────
# Streak detection
# ──────────────────────────────────────────────────────────────────────────────

class TestStreak:
    def test_no_streak(self, db):
        assert blind.check_streak(db, "model-a") is None

    def test_detects_streak(self, db):
        for i in range(5):
            db.execute("INSERT INTO comparisons (id, prompt, status, winner_model, created_at) VALUES (?, 'x', 'voted', 'model-a', datetime('now', ?))",
                       (os.urandom(6).hex(), f"-{5-i} minutes"))
        db.commit()
        result = blind.check_streak(db, "model-a")
        assert result is not None
        assert "model-a" in result

    def test_no_streak_when_mixed(self, db):
        models = ["model-a", "model-a", "model-b", "model-a", "model-a"]
        for i, m in enumerate(models):
            db.execute("INSERT INTO comparisons (id, prompt, status, winner_model, created_at) VALUES (?, 'x', 'voted', ?, datetime('now', ?))",
                       (os.urandom(6).hex(), m, f"-{5-i} minutes"))
        db.commit()
        assert blind.check_streak(db, "model-a") is None


# ──────────────────────────────────────────────────────────────────────────────
# Demo mode
# ──────────────────────────────────────────────────────────────────────────────

class TestDemo:
    def test_demo_response_returns_content(self):
        content, lat, in_t, out_t = blind.demo_response("model-alpha", "test prompt")
        assert content and len(content) > 0
        assert lat > 0
        assert in_t > 0
        assert out_t > 0

    def test_demo_models_differ(self):
        a, _, _, _ = blind.demo_response("model-alpha", "test")
        b, _, _, _ = blind.demo_response("model-beta", "test")
        c, _, _, _ = blind.demo_response("model-gamma", "test")
        assert a != b != c

    def test_all_demo_models_have_responses(self):
        for m in blind.DEMO_MODELS:
            content, _, _, _ = blind.demo_response(m["id"], "any prompt")
            assert content is not None


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_load_default_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(blind, "CONFIG_PATH", tmp_path / "nonexistent.json")
        config = blind.load_config()
        assert "models" in config
        assert len(config["models"]) >= 2

    def test_load_custom_config(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({"models": [{"id": "test"}], "default_models_per_round": 3}))
        monkeypatch.setattr(blind, "CONFIG_PATH", cfg_path)
        config = blind.load_config()
        assert config["models"][0]["id"] == "test"


# ──────────────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────────────

class TestExport:
    def test_export_creates_csv(self, db, tmp_path, monkeypatch, capsys):
        # Insert test data
        db.execute("INSERT INTO comparisons (id, prompt, category, status, winner_model) VALUES ('abc', 'test', 'code', 'voted', 'model-a')")
        db.execute("INSERT INTO responses (comparison_id, model_id, label, content, latency_ms) VALUES ('abc', 'model-a', 'A', 'response', 100)")
        db.commit()

        out = tmp_path / "out.csv"
        args = MagicMock(output=str(out))
        blind.cmd_export(args)

        assert out.exists()
        content = out.read_text()
        assert "model-a" in content
        assert "code" in content


# ──────────────────────────────────────────────────────────────────────────────
# Integration: full compare flow
# ──────────────────────────────────────────────────────────────────────────────

class TestCompareFlow:
    def test_demo_compare_stores_data(self, db, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "A")
        args = MagicMock(prompt=["test prompt"], category="code", models=2, demo=True, rank=False)
        blind.cmd_compare(args)

        comp = db.execute("SELECT * FROM comparisons WHERE status='voted'").fetchone()
        assert comp is not None
        assert comp["category"] == "code"

        resps = db.execute("SELECT * FROM responses WHERE comparison_id=?", (comp["id"],)).fetchall()
        assert len(resps) == 2

    def test_demo_compare_tie(self, db, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "tie")
        args = MagicMock(prompt=["test"], category=None, models=2, demo=True, rank=False)
        blind.cmd_compare(args)

        comp = db.execute("SELECT * FROM comparisons WHERE status='tie'").fetchone()
        assert comp is not None

    def test_demo_compare_skip(self, db, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "skip")
        args = MagicMock(prompt=["test"], category=None, models=2, demo=True, rank=False)
        blind.cmd_compare(args)

        comp = db.execute("SELECT * FROM comparisons WHERE status='skipped'").fetchone()
        assert comp is not None

    def test_demo_rank_mode(self, db, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "BAC")
        args = MagicMock(prompt=["test"], category="general", models=3, demo=True, rank=True)
        blind.cmd_compare(args)

        comp = db.execute("SELECT * FROM comparisons WHERE status='ranked'").fetchone()
        assert comp is not None
