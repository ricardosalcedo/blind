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
from datetime import datetime
from pathlib import Path

import httpx

DB_PATH = Path(os.environ.get("BLIND_DB", Path.home() / ".blind" / "blind.db"))
CONFIG_PATH = Path(os.environ.get("BLIND_CONFIG", Path.home() / ".blind" / "config.json"))

DEFAULT_CONFIG = {
    "models": [
        {"id": "claude-sonnet", "provider": "anthropic", "model": "claude-sonnet-4-20250514", "api_key_env": "ANTHROPIC_API_KEY"},
        {"id": "gpt-4o", "provider": "openai", "model": "gpt-4o", "api_key_env": "OPENAI_API_KEY"},
        {"id": "gemini-pro", "provider": "google", "model": "gemini-2.5-pro", "api_key_env": "GOOGLE_API_KEY"},
    ],
    "default_models_per_round": 2,
    "elo_k_factor": 32,
    "timeout_seconds": 60,
    "max_retries": 2,
}

# Cost per 1M tokens (input, output) — approximate
MODEL_COSTS = {
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
    "code": ["code", "function", "implement", "debug", "error", "class", "api", "algorithm", "refactor", "test", "bug"],
    "writing": ["write", "essay", "poem", "story", "email", "blog", "haiku", "letter", "article", "rewrite"],
    "math": ["calculate", "solve", "equation", "proof", "integral", "probability", "formula"],
    "research": ["explain", "compare", "difference", "history", "why", "how does", "what is"],
    "creative": ["imagine", "brainstorm", "idea", "invent", "design", "create", "suggest"],
}


# --- Database ---

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
            cost_usd REAL DEFAULT 0
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
    """)
    # Migration: add columns if missing
    try:
        db.execute("ALTER TABLE comparisons ADD COLUMN winner_position TEXT")
    except:
        pass
    try:
        db.execute("ALTER TABLE responses ADD COLUMN input_tokens INTEGER DEFAULT 0")
    except:
        pass
    try:
        db.execute("ALTER TABLE responses ADD COLUMN output_tokens INTEGER DEFAULT 0")
    except:
        pass
    try:
        db.execute("ALTER TABLE responses ADD COLUMN cost_usd REAL DEFAULT 0")
    except:
        pass
    return db


# --- Auto-categorize ---

def auto_categorize(prompt):
    """Guess category from prompt keywords."""
    prompt_lower = prompt.lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for k in keywords if k in prompt_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# --- Cost calculation ---

def estimate_cost(model_name, input_tokens, output_tokens):
    """Estimate cost in USD."""
    if model_name in MODEL_COSTS:
        in_rate, out_rate = MODEL_COSTS[model_name]
        return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
    return 0.0


# --- LLM Providers ---

async def call_model_async(client, model_cfg, prompt, timeout=60, retries=2):
    """Call a model with retries. Returns (content, latency_ms, input_tokens, output_tokens)."""
    provider = model_cfg["provider"]
    model = model_cfg["model"]
    api_key = os.environ.get(model_cfg["api_key_env"], "")

    if not api_key:
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
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return content, int((time.time() - start) * 1000), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

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
                content = data["content"][0]["text"]
                usage = data.get("usage", {})
                return content, int((time.time() - start) * 1000), usage.get("input_tokens", 0), usage.get("output_tokens", 0)

            elif provider == "google":
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=timeout,
                )
                data = resp.json()
                if "error" in data:
                    raise Exception(data["error"]["message"])
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                usage = data.get("usageMetadata", {})
                return content, int((time.time() - start) * 1000), usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)

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
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return content, int((time.time() - start) * 1000), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            if attempt < retries:
                await asyncio.sleep(1 * (attempt + 1))
                continue
            print(f"  ⚠ {model_cfg['id']} timed out after {retries + 1} attempts", file=sys.stderr)
            return None, 0, 0, 0
        except Exception as e:
            if attempt < retries:
                await asyncio.sleep(1)
                continue
            print(f"  ⚠ {model_cfg['id']} failed: {e}", file=sys.stderr)
            return None, 0, 0, 0

    return None, 0, 0, 0


async def call_models_parallel(models, prompt, config):
    """Call all models in parallel."""
    timeout = config.get("timeout_seconds", 60)
    retries = config.get("max_retries", 2)
    async with httpx.AsyncClient() as client:
        tasks = [call_model_async(client, m, prompt, timeout, retries) for m in models]
        return await asyncio.gather(*tasks)


# --- Demo mode ---

def demo_response(model_id, prompt):
    """Generate a synthetic response that varies by model."""
    styles = {"model-alpha": (400, 800), "model-beta": (600, 1200), "model-gamma": (500, 1000)}
    lo, hi = styles.get(model_id, (500, 900))
    latency = random.randint(lo, hi)
    responses = {
        "model-alpha": f"Here's a direct answer:\n\n{prompt.split()[-1].title()} involves three key principles:\n1. Simplicity in design\n2. Composability of components\n3. Clear separation of concerns\n\nThe most important takeaway: less complexity leads to more maintainable systems.",
        "model-beta": f"Great question! Let me break this down with an example.\n\nThink of it like building with LEGO blocks — each piece has a specific shape and purpose, but you can combine them in countless ways.\n\nConsider this scenario:\n- Start with the basics\n- Layer on complexity gradually\n- Test each addition independently\n\n```\nresult = compose(step1, step2, step3)\n```\n\nThe key insight: good abstractions compound over time.",
        "model-gamma": f"Oh, this is a fun one! 🎯\n\nHere's the thing — most people overthink this. The secret is finding patterns and making them repeatable.\n\nThink of it as a conversation between your current self and your future self. What would future-you want to know? Start there.\n\nBottom line: keep it simple, keep it human, keep iterating.",
    }
    return responses.get(model_id, f"Response about: {prompt}"), latency, 50, random.randint(80, 200)


# --- Elo ---

def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def update_elo(db, winner_id, loser_id, category, k=32, draw=False):
    for mid in (winner_id, loser_id):
        db.execute("INSERT OR IGNORE INTO elo_ratings (model_id, category) VALUES (?, ?)", (mid, category))

    row_w = db.execute("SELECT rating FROM elo_ratings WHERE model_id=? AND category=?", (winner_id, category)).fetchone()
    row_l = db.execute("SELECT rating FROM elo_ratings WHERE model_id=? AND category=?", (loser_id, category)).fetchone()
    ra, rb = row_w["rating"], row_l["rating"]
    ea, eb = expected_score(ra, rb), expected_score(rb, ra)

    if draw:
        new_ra = ra + k * (0.5 - ea)
        new_rb = rb + k * (0.5 - eb)
        db.execute("UPDATE elo_ratings SET rating=?, draws=draws+1 WHERE model_id=? AND category=?", (new_ra, winner_id, category))
        db.execute("UPDATE elo_ratings SET rating=?, draws=draws+1 WHERE model_id=? AND category=?", (new_rb, loser_id, category))
    else:
        new_ra = ra + k * (1 - ea)
        new_rb = rb + k * (0 - eb)
        db.execute("UPDATE elo_ratings SET rating=?, wins=wins+1 WHERE model_id=? AND category=?", (new_ra, winner_id, category))
        db.execute("UPDATE elo_ratings SET rating=?, losses=losses+1 WHERE model_id=? AND category=?", (new_rb, loser_id, category))
    db.commit()


# --- Position bias detection ---

def check_position_bias(db):
    """Check if user always picks the same position (A/B)."""
    rows = db.execute(
        "SELECT winner_position, COUNT(*) as n FROM comparisons WHERE status='voted' AND winner_position IS NOT NULL GROUP BY winner_position"
    ).fetchall()
    if not rows:
        return None
    total = sum(r["n"] for r in rows)
    if total < 10:
        return None
    for r in rows:
        ratio = r["n"] / total
        if ratio > 0.75:
            return f"⚠️  Position bias detected: you pick '{r['winner_position']}' {ratio:.0%} of the time. Try reading in reverse order."
    return None


# --- Commands ---

def cmd_compare(args):
    """Run a blind comparison."""
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

    # Auto-categorize if not specified
    category = args.category or auto_categorize(prompt)

    if demo:
        available = DEMO_MODELS
    else:
        available = [m for m in config["models"] if os.environ.get(m["api_key_env"])]
        if len(available) < 2:
            print("Need at least 2 models with API keys configured.")
            print("Set environment variables:", [m["api_key_env"] for m in config["models"]])
            print("\nTip: use --demo to try with mock models")
            return

    n = min(args.models or config["default_models_per_round"], len(available))
    selected = random.sample(available, n)

    comp_id = hashlib.sha256(f"{prompt}{time.time()}".encode()).hexdigest()[:12]

    # Call models (parallel for real, sequential for demo)
    print(f"\n⏳ Sending to {n} models{'  [demo]' if demo else ' in parallel'}...")

    if demo:
        raw_results = [(demo_response(m["id"], prompt)) for m in selected]
    else:
        raw_results = asyncio.run(call_models_parallel(selected, prompt, config))

    results = []
    for i, (content, latency, in_tok, out_tok) in enumerate(raw_results):
        if content:
            cost = estimate_cost(selected[i]["model"], in_tok, out_tok) if not demo else 0
            results.append({
                "model": selected[i], "content": content,
                "latency": latency, "in_tokens": in_tok, "out_tokens": out_tok, "cost": cost,
            })

    if len(results) < 2:
        print("Not enough models responded.")
        return

    # Shuffle and label
    random.shuffle(results)
    labels = "ABCDEFGH"

    # Store
    db.execute("INSERT INTO comparisons (id, prompt, category) VALUES (?, ?, ?)", (comp_id, prompt, category))
    for i, r in enumerate(results):
        db.execute(
            "INSERT INTO responses (comparison_id, model_id, label, content, latency_ms, input_tokens, output_tokens, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (comp_id, r["model"]["id"], labels[i], r["content"], r["latency"], r["in_tokens"], r["out_tokens"], r["cost"]),
        )
    db.commit()

    # Display
    print(f"\n{'='*60}")
    print(f"  Comparison: {comp_id} | Category: {category}")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        word_count = len(r["content"].split())
        print(f"\n{'─'*60}")
        print(f"  Response {labels[i]}  ({r['latency']}ms, {word_count} words)")
        print(f"{'─'*60}")
        print(r["content"])

    # Vote
    choices = "/".join(labels[:len(results)])
    print(f"\n{'='*60}")
    print(f"  Which is better? [{choices}] or [tie] or [skip]")

    # Position bias warning
    bias = check_position_bias(db)
    if bias:
        print(f"  {bias}")

    vote = input("  > ").strip().upper()

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
        print("  Recorded as tie.")
        _reveal(results, labels)
        return

    if vote in labels[:len(results)]:
        winner_idx = labels.index(vote)
        winner = results[winner_idx]
        db.execute("UPDATE comparisons SET status='voted', winner_model=?, winner_position=? WHERE id=?",
                   (winner["model"]["id"], vote, comp_id))
        db.commit()
        for i, r in enumerate(results):
            if i != winner_idx:
                update_elo(db, winner["model"]["id"], r["model"]["id"], category)
        print(f"\n  ✓ You picked {vote}.")
        _reveal(results, labels)
    else:
        print("  Invalid choice.")


def _reveal(results, labels):
    print("\n  🔍 Reveal:")
    for i, r in enumerate(results):
        cost_str = f"  ${r['cost']:.4f}" if r["cost"] > 0 else ""
        print(f"    {labels[i]} = {r['model']['id']}{cost_str}")


def cmd_stats(args):
    """Show rankings."""
    db = get_db()
    category = args.category

    if category:
        rows = db.execute("SELECT * FROM elo_ratings WHERE category=? ORDER BY rating DESC", (category,)).fetchall()
        print(f"\n📊 Rankings — {category}")
    else:
        rows = db.execute(
            "SELECT model_id, 'all' as category, AVG(rating) as rating, SUM(wins) as wins, SUM(losses) as losses, SUM(draws) as draws FROM elo_ratings GROUP BY model_id ORDER BY rating DESC"
        ).fetchall()
        print("\n📊 Rankings — All categories (averaged)")

    if not rows:
        print("  No data yet. Run `blind compare` to start.")
        return

    print(f"  {'Model':<20} {'Elo':>7} {'W':>4} {'L':>4} {'D':>4} {'Win%':>6}")
    print(f"  {'─'*50}")
    for r in rows:
        total = r["wins"] + r["losses"] + r["draws"]
        win_pct = f"{r['wins']/total*100:.0f}%" if total > 0 else "—"
        print(f"  {r['model_id']:<20} {r['rating']:>7.0f} {r['wins']:>4} {r['losses']:>4} {r['draws']:>4} {win_pct:>6}")

    total = db.execute("SELECT COUNT(*) as n FROM comparisons WHERE status IN ('voted','tie')").fetchone()
    print(f"\n  Total comparisons: {total['n']}")

    # Cost summary
    cost_row = db.execute("SELECT SUM(cost_usd) as total FROM responses").fetchone()
    if cost_row["total"] and cost_row["total"] > 0:
        print(f"  Total cost: ${cost_row['total']:.4f}")

    # Position bias check
    bias = check_position_bias(db)
    if bias:
        print(f"\n  {bias}")


def cmd_history(args):
    """Show comparison history."""
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
        prompt_short = r["prompt"][:50] + "…" if len(r["prompt"]) > 50 else r["prompt"]
        winner = r["winner_model"] or r["status"]
        print(f"  [{r['created_at'][:10]}] {r['category']:<10} {winner:<16} {prompt_short}")


def cmd_categories(args):
    """List categories."""
    db = get_db()
    rows = db.execute(
        "SELECT category, COUNT(*) as n FROM comparisons WHERE status IN ('voted','tie') GROUP BY category ORDER BY n DESC"
    ).fetchall()

    if not rows:
        print("No categories yet.")
        return

    print("\n📁 Categories:")
    for r in rows:
        print(f"  {r['category']:<20} {r['n']} comparisons")


def cmd_export(args):
    """Export data to CSV."""
    db = get_db()
    out = args.output or "blind_export.csv"

    rows = db.execute("""
        SELECT c.created_at, c.category, c.prompt, c.winner_model, c.status,
               r.model_id, r.label, r.latency_ms, r.input_tokens, r.output_tokens, r.cost_usd
        FROM comparisons c JOIN responses r ON r.comparison_id = c.id
        ORDER BY c.created_at DESC
    """).fetchall()

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "category", "prompt", "winner", "status", "model", "label", "latency_ms", "in_tokens", "out_tokens", "cost_usd"])
        for r in rows:
            w.writerow([r["created_at"], r["category"], r["prompt"][:100], r["winner_model"], r["status"],
                        r["model_id"], r["label"], r["latency_ms"], r["input_tokens"], r["output_tokens"], r["cost_usd"]])

    print(f"✓ Exported {len(rows)} rows to {out}")


def cmd_reset(args):
    """Reset all data."""
    if not args.confirm:
        print("This will delete all comparisons and ratings. Use --confirm to proceed.")
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
    print("  Edit it to add/remove models, then set API key env vars.")


def cmd_config(args):
    config = load_config()
    print(json.dumps(config, indent=2))


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return DEFAULT_CONFIG


# --- Main ---

def main():
    parser = argparse.ArgumentParser(prog="blind", description="Personal blind LLM comparison — build your own model rankings")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("compare", aliases=["c"], help="Run a blind comparison")
    p.add_argument("prompt", nargs="*", help="Prompt (or pipe via stdin)")
    p.add_argument("-c", "--category", help="Task category (auto-detected if omitted)")
    p.add_argument("-n", "--models", type=int, help="Number of models to compare")
    p.add_argument("--demo", action="store_true", help="Use mock models (no API keys needed)")

    p = sub.add_parser("stats", aliases=["s"], help="Show Elo rankings")
    p.add_argument("-c", "--category", help="Filter by category")

    p = sub.add_parser("history", aliases=["h"], help="Show comparison history")
    p.add_argument("-n", "--limit", type=int, default=20)

    sub.add_parser("categories", aliases=["cat"], help="List categories")

    p = sub.add_parser("export", help="Export data to CSV")
    p.add_argument("-o", "--output", help="Output file (default: blind_export.csv)")

    p = sub.add_parser("reset", help="Reset all data")
    p.add_argument("--confirm", action="store_true")

    p = sub.add_parser("init", help="Initialize config")
    p.add_argument("--force", action="store_true")

    sub.add_parser("config", help="Show config")

    args = parser.parse_args()
    commands = {
        "compare": cmd_compare, "c": cmd_compare,
        "stats": cmd_stats, "s": cmd_stats,
        "history": cmd_history, "h": cmd_history,
        "categories": cmd_categories, "cat": cmd_categories,
        "export": cmd_export,
        "reset": cmd_reset,
        "init": cmd_init,
        "config": cmd_config,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
