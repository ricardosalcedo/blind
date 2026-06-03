#!/usr/bin/env python3
"""Blind — Personal blind LLM comparison. Build your own model rankings."""

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

import httpx

DB_PATH = Path(os.environ.get("BLIND_DB", Path.home() / ".blind" / "blind.db"))
CONFIG_PATH = Path(os.environ.get("BLIND_CONFIG", Path.home() / ".blind" / "config.json"))

DEFAULT_CONFIG = {
    "models": [
        {
            "id": "claude-sonnet",
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "api_key_env": "ANTHROPIC_API_KEY",
        },
        {"id": "gpt-4o", "provider": "openai", "model": "gpt-4o", "api_key_env": "OPENAI_API_KEY"},
        {"id": "gemini-pro", "provider": "google", "model": "gemini-2.5-pro", "api_key_env": "GOOGLE_API_KEY"},
    ],
    "default_models_per_round": 2,
    "elo_k_factor": 32,
    "timeout_seconds": 60,
    "max_retries": 2,
}

MODEL_COSTS = {  # (input, output) per 1M tokens
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
}

DEMO_MODELS = [
    {"id": "model-alpha", "provider": "demo", "model": "alpha", "api_key_env": "_"},
    {"id": "model-beta", "provider": "demo", "model": "beta", "api_key_env": "_"},
    {"id": "model-gamma", "provider": "demo", "model": "gamma", "api_key_env": "_"},
]

CATEGORY_KEYWORDS = {
    "code": [
        "code",
        "function",
        "implement",
        "debug",
        "error",
        "class",
        "api",
        "algorithm",
        "refactor",
        "test",
        "bug",
        "python",
        "javascript",
    ],
    "writing": ["write", "essay", "poem", "story", "email", "blog", "haiku", "letter", "article", "rewrite", "draft"],
    "math": ["calculate", "solve", "equation", "proof", "integral", "probability", "formula", "sum", "derivative"],
    "research": ["explain", "compare", "difference", "history", "why does", "how does", "what is", "analyze"],
    "creative": ["imagine", "brainstorm", "idea", "invent", "design", "create", "suggest", "name"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Database
# ──────────────────────────────────────────────────────────────────────────────


def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS comparisons (
            id TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            created_at TEXT DEFAULT (datetime('now')),
            winner_model TEXT,
            winner_position TEXT,
            vote_type TEXT DEFAULT 'pick',
            status TEXT DEFAULT 'pending'
        );
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comparison_id TEXT NOT NULL REFERENCES comparisons(id),
            model_id TEXT NOT NULL,
            label TEXT NOT NULL,
            content TEXT NOT NULL,
            latency_ms INTEGER,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0,
            rank INTEGER
        );
        CREATE TABLE IF NOT EXISTS elo_ratings (
            model_id TEXT NOT NULL,
            category TEXT NOT NULL,
            rating REAL DEFAULT 1500,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            draws INTEGER DEFAULT 0,
            PRIMARY KEY (model_id, category)
        );
        CREATE TABLE IF NOT EXISTS elo_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            category TEXT NOT NULL,
            rating REAL NOT NULL,
            recorded_at TEXT DEFAULT (datetime('now'))
        );
    """)
    for col, tbl in [("winner_position", "comparisons"), ("vote_type", "comparisons"), ("rank", "responses")]:
        try:
            db.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} TEXT")
        except Exception:
            pass
    return db


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────


def auto_categorize(prompt):
    prompt_lower = prompt.lower()
    scores = {cat: sum(1 for k in kw if k in prompt_lower) for cat, kw in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def estimate_cost(model_name, in_tok, out_tok):
    if model_name in MODEL_COSTS:
        i, o = MODEL_COSTS[model_name]
        return (in_tok * i + out_tok * o) / 1_000_000
    return 0.0


def check_position_bias(db):
    rows = db.execute(
        "SELECT winner_position, COUNT(*) as n FROM comparisons WHERE status='voted' AND winner_position IS NOT NULL GROUP BY winner_position"
    ).fetchall()
    total = sum(r["n"] for r in rows)
    if total < 10:
        return None
    for r in rows:
        if r["n"] / total > 0.75:
            return f"⚠️  Position bias: you pick '{r['winner_position']}' {r['n'] / total:.0%} of the time."
    return None


def check_streak(db, model_id, n=5):
    """Check if a model has won the last N comparisons."""
    rows = db.execute(
        "SELECT winner_model FROM comparisons WHERE status='voted' ORDER BY created_at DESC LIMIT ?", (n,)
    ).fetchall()
    if len(rows) >= n and all(r["winner_model"] == model_id for r in rows):
        return f"🔥 {model_id} has won {n} in a row!"
    return None


def length_bias_warning(results):
    """Warn if response lengths differ by 3x+."""
    lengths = [len(r["content"].split()) for r in results]
    if max(lengths) >= 3 * min(lengths) and min(lengths) > 10:
        short_idx = lengths.index(min(lengths))
        long_idx = lengths.index(max(lengths))
        return f"  ℹ️  Length disparity: {labels[long_idx]} is {max(lengths) // min(lengths)}x longer than {labels[short_idx]}. Longer ≠ better."
    return None


# ──────────────────────────────────────────────────────────────────────────────
# LLM Providers
# ──────────────────────────────────────────────────────────────────────────────


async def call_model_async(client, model_cfg, prompt, timeout=60, retries=2):
    provider = model_cfg["provider"]
    model = model_cfg["model"]
    api_key = os.environ.get(model_cfg.get("api_key_env", ""), "")
    if not api_key and provider not in ("bedrock",):
        return None, 0, 0, 0

    for attempt in range(retries + 1):
        start = time.time()
        try:
            if provider == "openai":
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048},
                    timeout=timeout,
                )
                data = resp.json()
                if "error" in data:
                    raise Exception(data["error"]["message"])
                u = data.get("usage", {})
                return (
                    data["choices"][0]["message"]["content"],
                    int((time.time() - start) * 1000),
                    u.get("prompt_tokens", 0),
                    u.get("completion_tokens", 0),
                )

            elif provider == "anthropic":
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                    json={"model": model, "max_tokens": 2048, "messages": [{"role": "user", "content": prompt}]},
                    timeout=timeout,
                )
                data = resp.json()
                if "error" in data:
                    raise Exception(data["error"]["message"])
                u = data.get("usage", {})
                return (
                    data["content"][0]["text"],
                    int((time.time() - start) * 1000),
                    u.get("input_tokens", 0),
                    u.get("output_tokens", 0),
                )

            elif provider == "google":
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=timeout,
                )
                data = resp.json()
                if "error" in data:
                    raise Exception(data["error"]["message"])
                u = data.get("usageMetadata", {})
                return (
                    data["candidates"][0]["content"]["parts"][0]["text"],
                    int((time.time() - start) * 1000),
                    u.get("promptTokenCount", 0),
                    u.get("candidatesTokenCount", 0),
                )

            elif provider == "openai-compatible":
                base_url = model_cfg.get("base_url", "http://localhost:11434/v1")
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048},
                    timeout=timeout,
                )
                data = resp.json()
                u = data.get("usage", {})
                return (
                    data["choices"][0]["message"]["content"],
                    int((time.time() - start) * 1000),
                    u.get("prompt_tokens", 0),
                    u.get("completion_tokens", 0),
                )

            elif provider == "bedrock":
                # AWS Bedrock via boto3 (uses default credential chain)
                import boto3

                region = model_cfg.get("region", os.environ.get("AWS_REGION", "us-west-2"))
                bedrock = boto3.client("bedrock-runtime", region_name=region)
                body = json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 2048,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                )
                br_resp = bedrock.invoke_model(modelId=model, body=body, contentType="application/json")
                data = json.loads(br_resp["body"].read())
                u = data.get("usage", {})
                return (
                    data["content"][0]["text"],
                    int((time.time() - start) * 1000),
                    u.get("input_tokens", 0),
                    u.get("output_tokens", 0),
                )

        except (httpx.TimeoutException, httpx.ConnectError):
            if attempt < retries:
                await asyncio.sleep(1 * (attempt + 1))
                continue
            print(f"  ⚠ {model_cfg['id']} timed out", file=sys.stderr)
            return None, 0, 0, 0
        except Exception as e:
            if attempt < retries:
                await asyncio.sleep(1)
                continue
            print(f"  ⚠ {model_cfg['id']}: {e}", file=sys.stderr)
            return None, 0, 0, 0
    return None, 0, 0, 0


async def call_models_parallel(models, prompt, config):
    timeout = config.get("timeout_seconds", 60)
    retries = config.get("max_retries", 2)
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*[call_model_async(client, m, prompt, timeout, retries) for m in models])


def demo_response(model_id, prompt):
    latency = random.randint(400, 1200)
    responses = {
        "model-alpha": f"Here's a direct answer:\n\n{prompt.split()[-1].title()} involves three key principles:\n1. Simplicity in design\n2. Composability of components\n3. Clear separation of concerns\n\nLess complexity leads to more maintainable systems.",
        "model-beta": "Let me break this down with an example.\n\nThink of it like LEGO — each piece has a purpose, but you combine them freely.\n\n- Start with the basics\n- Layer complexity gradually\n- Test each addition independently\n\n```\nresult = compose(step1, step2, step3)\n```\n\nGood abstractions compound over time. The key is to identify the right level of abstraction for your problem domain and stick with it consistently.",
        "model-gamma": "Oh, this is a fun one! 🎯\n\nMost people overthink this. The secret: find patterns, make them repeatable.\n\nThink of it as a conversation between current-you and future-you. What would future-you want to know?\n\nKeep it simple, keep it human, keep iterating.",
    }
    return responses.get(model_id, f"Response: {prompt}"), latency, 50, random.randint(80, 200)


# ──────────────────────────────────────────────────────────────────────────────
# Elo
# ──────────────────────────────────────────────────────────────────────────────


def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def update_elo(db, winner_id, loser_id, category, k=32, draw=False):
    for mid in (winner_id, loser_id):
        db.execute("INSERT OR IGNORE INTO elo_ratings (model_id, category) VALUES (?, ?)", (mid, category))
    ra = db.execute("SELECT rating FROM elo_ratings WHERE model_id=? AND category=?", (winner_id, category)).fetchone()[
        "rating"
    ]
    rb = db.execute("SELECT rating FROM elo_ratings WHERE model_id=? AND category=?", (loser_id, category)).fetchone()[
        "rating"
    ]
    ea, eb = expected_score(ra, rb), expected_score(rb, ra)
    if draw:
        db.execute(
            "UPDATE elo_ratings SET rating=?, draws=draws+1 WHERE model_id=? AND category=?",
            (ra + k * (0.5 - ea), winner_id, category),
        )
        db.execute(
            "UPDATE elo_ratings SET rating=?, draws=draws+1 WHERE model_id=? AND category=?",
            (rb + k * (0.5 - eb), loser_id, category),
        )
    else:
        db.execute(
            "UPDATE elo_ratings SET rating=?, wins=wins+1 WHERE model_id=? AND category=?",
            (ra + k * (1 - ea), winner_id, category),
        )
        db.execute(
            "UPDATE elo_ratings SET rating=?, losses=losses+1 WHERE model_id=? AND category=?",
            (rb + k * (0 - eb), loser_id, category),
        )
    db.commit()
    # Record history for charting
    for mid in (winner_id, loser_id):
        r = db.execute("SELECT rating FROM elo_ratings WHERE model_id=? AND category=?", (mid, category)).fetchone()
        db.execute(
            "INSERT INTO elo_history (model_id, category, rating) VALUES (?, ?, ?)", (mid, category, r["rating"])
        )
    db.commit()


def elo_confidence(wins, losses, draws):
    """95% confidence interval width (approximation)."""
    n = wins + losses + draws
    if n < 5:
        return None  # Not enough data
    # Approximate CI: ±400/sqrt(n)
    return 400 / math.sqrt(n)


# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────

labels = "ABCDEFGH"


def cmd_compare(args):
    config = load_config()
    db = get_db()
    demo = args.demo

    if args.prompt:
        prompt = " ".join(args.prompt)
    else:
        print("Enter your prompt (Ctrl+D to finish):")
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("No prompt provided.")
        return

    category = args.category or auto_categorize(prompt)

    if demo:
        available = DEMO_MODELS
    else:
        available = [
            m for m in config["models"] if m.get("provider") == "bedrock" or os.environ.get(m.get("api_key_env", ""))
        ]
        if len(available) < 2:
            print("Need at least 2 models with API keys set.")
            print("Keys needed:", [m["api_key_env"] for m in config["models"]])
            print("\nUse --demo to try with mock models")
            return

    n = min(args.models or config["default_models_per_round"], len(available))
    selected = random.sample(available, n)
    comp_id = hashlib.sha256(f"{prompt}{time.time()}".encode()).hexdigest()[:12]

    print(f"\n⏳ Sending to {n} models{'  [demo]' if demo else ' in parallel'}...")
    if demo:
        raw = [demo_response(m["id"], prompt) for m in selected]
    else:
        raw = asyncio.run(call_models_parallel(selected, prompt, config))

    results = []
    for i, (content, lat, in_t, out_t) in enumerate(raw):
        if content:
            cost = estimate_cost(selected[i]["model"], in_t, out_t) if not demo else 0
            results.append(
                {
                    "model": selected[i],
                    "content": content,
                    "latency": lat,
                    "in_tokens": in_t,
                    "out_tokens": out_t,
                    "cost": cost,
                }
            )

    if len(results) < 2:
        print("Not enough models responded.")
        return

    random.shuffle(results)

    # Store
    db.execute("INSERT INTO comparisons (id, prompt, category) VALUES (?, ?, ?)", (comp_id, prompt, category))
    for i, r in enumerate(results):
        db.execute(
            "INSERT INTO responses (comparison_id, model_id, label, content, latency_ms, input_tokens, output_tokens, cost_usd) VALUES (?,?,?,?,?,?,?,?)",
            (
                comp_id,
                r["model"]["id"],
                labels[i],
                r["content"],
                r["latency"],
                r["in_tokens"],
                r["out_tokens"],
                r["cost"],
            ),
        )
    db.commit()

    # Display
    print(f"\n{'=' * 60}")
    print(f"  Comparison: {comp_id} | Category: {category}")
    print(f"{'=' * 60}")
    for i, r in enumerate(results):
        wc = len(r["content"].split())
        print(f"\n{'─' * 60}")
        print(f"  Response {labels[i]}  ({r['latency']}ms · {wc} words)")
        print(f"{'─' * 60}")
        print(r["content"])

    # Length bias warning
    lengths = [len(r["content"].split()) for r in results]
    if max(lengths) >= 3 * min(lengths) and min(lengths) > 10:
        short_i = lengths.index(min(lengths))
        long_i = lengths.index(max(lengths))
        print(
            f"\n  ℹ️  {labels[long_i]} is {max(lengths) // min(lengths)}x longer than {labels[short_i]}. Longer ≠ better."
        )

    # Vote
    choices = "/".join(labels[: len(results)])
    print(f"\n{'=' * 60}")

    if args.rank and len(results) > 2:
        print("  Rank all responses best→worst (e.g. 'BAC'):")
    else:
        print(f"  Which is better? [{choices}] or [tie] or [skip]")

    bias = check_position_bias(db)
    if bias:
        print(f"  {bias}")

    vote = input("  > ").strip().upper()

    # Handle ranking mode
    if args.rank and len(vote) == len(results) and all(c in labels[: len(results)] for c in vote):
        db.execute("UPDATE comparisons SET status='ranked', vote_type='rank' WHERE id=?", (comp_id,))
        for pos, letter in enumerate(vote):
            labels.index(letter)
            db.execute("UPDATE responses SET rank=? WHERE comparison_id=? AND label=?", (pos + 1, comp_id, letter))
        # Update Elo: each pair where higher-ranked beats lower-ranked
        for i in range(len(vote)):
            for j in range(i + 1, len(vote)):
                wi = labels.index(vote[i])
                li = labels.index(vote[j])
                update_elo(db, results[wi]["model"]["id"], results[li]["model"]["id"], category)
        db.commit()
        print(f"\n  ✓ Ranked: {' > '.join(vote)}")
        _reveal(results)
        return

    if vote == "SKIP":
        db.execute("UPDATE comparisons SET status='skipped' WHERE id=?", (comp_id,))
        db.commit()
        print("  Skipped.")
        return

    if vote == "TIE":
        db.execute("UPDATE comparisons SET status='tie' WHERE id=?", (comp_id,))
        db.commit()
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                update_elo(db, results[i]["model"]["id"], results[j]["model"]["id"], category, draw=True)
        print("  Tie recorded.")
        _reveal(results)
        return

    if vote in labels[: len(results)]:
        wi = labels.index(vote)
        winner = results[wi]
        db.execute(
            "UPDATE comparisons SET status='voted', winner_model=?, winner_position=?, vote_type='pick' WHERE id=?",
            (winner["model"]["id"], vote, comp_id),
        )
        db.commit()
        for i, r in enumerate(results):
            if i != wi:
                update_elo(db, winner["model"]["id"], r["model"]["id"], category)
        print(f"\n  ✓ You picked {vote}.")
        _reveal(results)
        # Streak check
        for r in results:
            s = check_streak(db, r["model"]["id"])
            if s:
                print(f"  {s}")
    else:
        print("  Invalid choice.")


def _reveal(results):
    print("\n  🔍 Reveal:")
    for i, r in enumerate(results):
        cost = f"  ${r['cost']:.4f}" if r["cost"] > 0 else ""
        print(f"    {labels[i]} = {r['model']['id']}{cost}")


def cmd_stats(args):
    db = get_db()
    cat = args.category

    if cat:
        rows = db.execute("SELECT * FROM elo_ratings WHERE category=? ORDER BY rating DESC", (cat,)).fetchall()
        title = cat
    else:
        rows = db.execute(
            "SELECT model_id, 'all' as category, AVG(rating) as rating, SUM(wins) as wins, SUM(losses) as losses, SUM(draws) as draws FROM elo_ratings GROUP BY model_id ORDER BY rating DESC"
        ).fetchall()
        title = "All categories (averaged)"

    if not rows:
        print("  No data yet. Run `blind compare` to start.")
        return

    print(f"\n📊 Rankings — {title}")
    print(f"  {'Model':<20} {'Elo':>7} {'±CI':>5} {'W':>4} {'L':>4} {'D':>4} {'Win%':>6}")
    print(f"  {'─' * 55}")
    for r in rows:
        total = r["wins"] + r["losses"] + r["draws"]
        win_pct = f"{r['wins'] / total * 100:.0f}%" if total > 0 else "—"
        ci = elo_confidence(r["wins"], r["losses"], r["draws"])
        ci_str = f"±{ci:.0f}" if ci else "  —"
        print(
            f"  {r['model_id']:<20} {r['rating']:>7.0f} {ci_str:>5} {r['wins']:>4} {r['losses']:>4} {r['draws']:>4} {win_pct:>6}"
        )

    total = db.execute("SELECT COUNT(*) as n FROM comparisons WHERE status IN ('voted','tie','ranked')").fetchone()
    print(f"\n  Total comparisons: {total['n']}")

    cost_row = db.execute("SELECT SUM(cost_usd) as total FROM responses").fetchone()
    if cost_row["total"] and cost_row["total"] > 0:
        print(f"  Total cost: ${cost_row['total']:.4f}")

    bias = check_position_bias(db)
    if bias:
        print(f"\n  {bias}")


def cmd_head2head(args):
    """Show pairwise win rates between models."""
    db = get_db()
    models = [r["model_id"] for r in db.execute("SELECT DISTINCT model_id FROM elo_ratings").fetchall()]
    if len(models) < 2:
        print("Need at least 2 models with data.")
        return

    print("\n🥊 Head-to-Head Win Rates")
    print(f"  {'':>15}", end="")
    for m in models:
        print(f" {m[:8]:>8}", end="")
    print()

    for m1 in models:
        print(f"  {m1:>15}", end="")
        for m2 in models:
            if m1 == m2:
                print(f" {'—':>8}", end="")
            else:
                wins = db.execute(
                    """
                    SELECT COUNT(*) as n FROM comparisons c
                    JOIN responses r1 ON r1.comparison_id=c.id AND r1.model_id=?
                    JOIN responses r2 ON r2.comparison_id=c.id AND r2.model_id=?
                    WHERE c.winner_model=? AND c.status='voted'
                """,
                    (m1, m2, m1),
                ).fetchone()["n"]
                total = db.execute(
                    """
                    SELECT COUNT(*) as n FROM comparisons c
                    JOIN responses r1 ON r1.comparison_id=c.id AND r1.model_id=?
                    JOIN responses r2 ON r2.comparison_id=c.id AND r2.model_id=?
                    WHERE c.status IN ('voted','tie')
                """,
                    (m1, m2),
                ).fetchone()["n"]
                rate = f"{wins}/{total}" if total > 0 else "0/0"
                print(f" {rate:>8}", end="")
        print()


def cmd_insights(args):
    """Analyze patterns in your voting behavior."""
    db = get_db()
    total = db.execute("SELECT COUNT(*) as n FROM comparisons WHERE status IN ('voted','tie','ranked')").fetchone()["n"]
    if total < 5:
        print("Need at least 5 comparisons for insights.")
        return

    print(f"\n💡 Insights ({total} comparisons)\n")

    # 1. Category dominance
    cat_winners = db.execute("""
        SELECT category, winner_model, COUNT(*) as n FROM comparisons
        WHERE status='voted' GROUP BY category, winner_model ORDER BY category, n DESC
    """).fetchall()
    cat_map = {}
    for r in cat_winners:
        cat_map.setdefault(r["category"], []).append((r["winner_model"], r["n"]))
    for cat, winners in cat_map.items():
        if winners and len(winners) > 1:
            top = winners[0]
            total_cat = sum(w[1] for w in winners)
            if top[1] / total_cat > 0.6:
                print(f"  📌 {top[0]} dominates in '{cat}' ({top[1]}/{total_cat} = {top[1] / total_cat:.0%})")

    # 2. Length preference
    shorter_wins = 0
    longer_wins = 0
    rows = db.execute("""
        SELECT c.id, c.winner_model FROM comparisons c WHERE c.status='voted'
    """).fetchall()
    for comp in rows:
        resps = db.execute(
            "SELECT model_id, LENGTH(content) as len FROM responses WHERE comparison_id=?", (comp["id"],)
        ).fetchall()
        if len(resps) >= 2:
            winner_len = next((r["len"] for r in resps if r["model_id"] == comp["winner_model"]), 0)
            avg_len = sum(r["len"] for r in resps) / len(resps)
            if winner_len < avg_len * 0.8:
                shorter_wins += 1
            elif winner_len > avg_len * 1.2:
                longer_wins += 1

    if shorter_wins + longer_wins > 5:
        if shorter_wins > longer_wins * 1.5:
            print(f"  📏 You tend to prefer shorter responses ({shorter_wins} vs {longer_wins} longer wins)")
        elif longer_wins > shorter_wins * 1.5:
            print(
                f"  📏 You tend to prefer longer/more detailed responses ({longer_wins} vs {shorter_wins} shorter wins)"
            )

    # 3. Latency preference
    faster_wins = 0
    rows2 = db.execute("SELECT c.id, c.winner_model FROM comparisons c WHERE c.status='voted'").fetchall()
    for comp in rows2:
        resps = db.execute("SELECT model_id, latency_ms FROM responses WHERE comparison_id=?", (comp["id"],)).fetchall()
        if len(resps) >= 2:
            winner_lat = next((r["latency_ms"] for r in resps if r["model_id"] == comp["winner_model"]), 0)
            if winner_lat == min(r["latency_ms"] for r in resps):
                faster_wins += 1
    if total > 10 and faster_wins / total > 0.65:
        print(f"  ⚡ The faster model wins {faster_wins / total:.0%} of the time — you might value speed")

    # 4. Position bias
    bias = check_position_bias(db)
    if bias:
        print(f"  {bias}")

    # 5. Confidence
    low_conf = db.execute("""
        SELECT model_id, category, wins+losses+draws as n FROM elo_ratings WHERE wins+losses+draws < 5
    """).fetchall()
    if low_conf:
        print(f"\n  ⚠️  Low confidence ratings ({len(low_conf)} model-category pairs with <5 comparisons)")


def cmd_replay(args):
    """Re-run a past comparison with current models."""
    db = get_db()
    comp = db.execute(
        "SELECT * FROM comparisons WHERE id LIKE ? ORDER BY created_at DESC LIMIT 1", (f"{args.id}%",)
    ).fetchone()
    if not comp:
        print(f"Comparison '{args.id}' not found.")
        return

    print(f"Replaying: {comp['prompt'][:80]}")
    # Inject the prompt back into compare
    sys.argv = ["blind", "compare"] + (["--demo"] if args.demo else []) + ["-c", comp["category"], comp["prompt"]]
    main()


def cmd_history(args):
    db = get_db()
    rows = db.execute(
        "SELECT id, prompt, category, winner_model, status, created_at FROM comparisons ORDER BY created_at DESC LIMIT ?",
        (args.limit,),
    ).fetchall()
    if not rows:
        print("No history yet.")
        return

    print(f"\n📜 Last {len(rows)} comparisons:")
    for r in rows:
        p = r["prompt"][:50] + "…" if len(r["prompt"]) > 50 else r["prompt"]
        w = r["winner_model"] or r["status"]
        print(f"  {r['id'][:8]} [{r['created_at'][:10]}] {r['category']:<10} {w:<16} {p}")


def cmd_categories(args):
    db = get_db()
    rows = db.execute(
        "SELECT category, COUNT(*) as n FROM comparisons WHERE status IN ('voted','tie','ranked') GROUP BY category ORDER BY n DESC"
    ).fetchall()
    if not rows:
        print("No categories yet.")
        return
    print("\n📁 Categories:")
    for r in rows:
        print(f"  {r['category']:<20} {r['n']} comparisons")


def cmd_chart(args):
    """ASCII Elo trend chart."""
    db = get_db()
    cat_filter = args.category

    if cat_filter:
        rows = db.execute(
            "SELECT model_id, rating, recorded_at FROM elo_history WHERE category=? ORDER BY recorded_at",
            (cat_filter,),
        ).fetchall()
    else:
        rows = db.execute("SELECT model_id, rating, recorded_at FROM elo_history ORDER BY recorded_at").fetchall()

    if not rows:
        print("No history yet. Run some comparisons first.")
        return

    # Group by model
    series = {}
    for r in rows:
        series.setdefault(r["model_id"], []).append(r["rating"])

    # Chart dimensions
    width = min(60, max(len(pts) for pts in series.values()))
    height = 15

    # Get global min/max
    all_ratings = [r for pts in series.values() for r in pts]
    lo = min(all_ratings) - 10
    hi = max(all_ratings) + 10
    span = hi - lo if hi > lo else 1

    # Symbols per model
    symbols = "●○◆◇■□▲△"
    model_ids = sorted(series.keys())

    print(f"\n📈 Elo Trend{f' — {cat_filter}' if cat_filter else ''}")
    print(f"  {hi:.0f} ┐")

    # Build grid
    grid = [[" "] * width for _ in range(height)]
    for mi, mid in enumerate(model_ids):
        pts = series[mid]
        # Resample to width
        if len(pts) > width:
            step = len(pts) / width
            sampled = [pts[int(i * step)] for i in range(width)]
        else:
            sampled = pts

        sym = symbols[mi % len(symbols)]
        for x, rating in enumerate(sampled):
            y = int((rating - lo) / span * (height - 1))
            y = max(0, min(height - 1, y))
            grid[height - 1 - y][x] = sym

    # Print grid
    for row_idx, row in enumerate(grid):
        if row_idx == height // 2:
            mid_val = (hi + lo) / 2
            print(f"  {mid_val:.0f} ┤{''.join(row)}")
        else:
            print(f"       │{''.join(row)}")

    print(f"  {lo:.0f} ┘{'─' * width}")
    print(f"       {'oldest':<{width - 6}}{'latest':>6}")

    # Legend
    print("\n  Legend:")
    for mi, mid in enumerate(model_ids):
        sym = symbols[mi % len(symbols)]
        current = series[mid][-1]
        print(f"    {sym} {mid:<20} (current: {current:.0f})")


def cmd_export(args):
    db = get_db()
    out = args.output or "blind_export.csv"
    rows = db.execute("""
        SELECT c.created_at, c.category, c.prompt, c.winner_model, c.status,
               r.model_id, r.label, r.latency_ms, r.input_tokens, r.output_tokens, r.cost_usd, r.rank
        FROM comparisons c JOIN responses r ON r.comparison_id=c.id ORDER BY c.created_at DESC
    """).fetchall()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "date",
                "category",
                "prompt",
                "winner",
                "status",
                "model",
                "label",
                "latency_ms",
                "in_tokens",
                "out_tokens",
                "cost_usd",
                "rank",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["created_at"],
                    r["category"],
                    r["prompt"][:100],
                    r["winner_model"],
                    r["status"],
                    r["model_id"],
                    r["label"],
                    r["latency_ms"],
                    r["input_tokens"],
                    r["output_tokens"],
                    r["cost_usd"],
                    r["rank"],
                ]
            )
    print(f"✓ Exported {len(rows)} rows to {out}")


def cmd_import_csv(args):
    """Import comparisons from CSV (from another Blind instance)."""
    db = get_db()
    count = 0
    with open(args.file) as f:
        for row in csv.DictReader(f):
            if row.get("winner") and row.get("model") and row.get("category"):
                # Reconstruct Elo updates from win/loss data
                update_elo(db, row["winner"], row["model"], row["category"])
                count += 1
    print(f"✓ Imported {count} records from {args.file}")


def cmd_reset(args):
    if not args.confirm:
        print("Delete all data? Use --confirm")
        return
    db = get_db()
    db.executescript("DELETE FROM responses; DELETE FROM comparisons; DELETE FROM elo_ratings;")
    print("✓ All data reset.")


def cmd_init(args):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists() and not args.force:
        print(f"Config exists at {CONFIG_PATH}. Use --force to overwrite.")
        return
    with open(CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print(f"✓ Config created at {CONFIG_PATH}")


def cmd_config(args):
    print(json.dumps(load_config(), indent=2))


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return DEFAULT_CONFIG


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        prog="blind", description="Personal blind LLM comparison — build your own model rankings"
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("compare", aliases=["c"], help="Run a blind comparison")
    p.add_argument("prompt", nargs="*")
    p.add_argument("-c", "--category", help="Category (auto-detected if omitted)")
    p.add_argument("-n", "--models", type=int, help="Models per round")
    p.add_argument("--demo", action="store_true", help="Mock models, no API keys")
    p.add_argument("--rank", action="store_true", help="Rank all responses (not just pick winner)")

    p = sub.add_parser("stats", aliases=["s"], help="Elo rankings")
    p.add_argument("-c", "--category")

    p = sub.add_parser("head2head", aliases=["h2h"], help="Pairwise win rates")

    p = sub.add_parser("insights", aliases=["i"], help="Voting pattern analysis")

    p = sub.add_parser("replay", aliases=["r"], help="Re-run a past comparison")
    p.add_argument("id", help="Comparison ID (prefix match)")
    p.add_argument("--demo", action="store_true")

    p = sub.add_parser("history", aliases=["h"], help="Past comparisons")
    p.add_argument("-n", "--limit", type=int, default=20)

    sub.add_parser("categories", aliases=["cat"], help="List categories")

    p = sub.add_parser("chart", help="ASCII Elo trend chart")
    p.add_argument("-c", "--category")

    p = sub.add_parser("export", help="Export to CSV")
    p.add_argument("-o", "--output")

    p = sub.add_parser("import", help="Import from CSV")
    p.add_argument("file")

    p = sub.add_parser("reset", help="Delete all data")
    p.add_argument("--confirm", action="store_true")

    p = sub.add_parser("init", help="Create config")
    p.add_argument("--force", action="store_true")

    sub.add_parser("config", help="Show config")

    args = parser.parse_args()
    cmds = {
        "compare": cmd_compare,
        "c": cmd_compare,
        "stats": cmd_stats,
        "s": cmd_stats,
        "head2head": cmd_head2head,
        "h2h": cmd_head2head,
        "insights": cmd_insights,
        "i": cmd_insights,
        "replay": cmd_replay,
        "r": cmd_replay,
        "history": cmd_history,
        "h": cmd_history,
        "categories": cmd_categories,
        "cat": cmd_categories,
        "chart": cmd_chart,
        "export": cmd_export,
        "import": cmd_import_csv,
        "reset": cmd_reset,
        "init": cmd_init,
        "config": cmd_config,
    }
    if args.command in cmds:
        cmds[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
