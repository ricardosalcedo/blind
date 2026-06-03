#!/usr/bin/env python3
"""Blind — Personal blind LLM comparison. Build your own model rankings."""

import argparse
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
            status TEXT DEFAULT 'pending'
        );
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comparison_id TEXT NOT NULL REFERENCES comparisons(id),
            model_id TEXT NOT NULL,
            label TEXT NOT NULL,
            content TEXT NOT NULL,
            latency_ms INTEGER,
            tokens_used INTEGER
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
    return db


# --- LLM Providers ---

def call_model(model_cfg, prompt):
    """Call a model and return (content, latency_ms, tokens)."""
    provider = model_cfg["provider"]
    model = model_cfg["model"]
    api_key = os.environ.get(model_cfg["api_key_env"], "")

    if not api_key:
        return None, 0, 0

    start = time.time()
    try:
        if provider == "openai":
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048},
                timeout=60,
            )
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)

        elif provider == "anthropic":
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                json={"model": model, "max_tokens": 2048, "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )
            data = resp.json()
            content = data["content"][0]["text"]
            tokens = data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0)

        elif provider == "google":
            resp = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=60,
            )
            data = resp.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)

        elif provider == "openai-compatible":
            base_url = model_cfg.get("base_url", "http://localhost:11434/v1")
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            resp = httpx.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048},
                timeout=60,
            )
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)

        else:
            return None, 0, 0

        latency = int((time.time() - start) * 1000)
        return content, latency, tokens

    except Exception as e:
        print(f"  ⚠ {model_cfg['id']} failed: {e}", file=sys.stderr)
        return None, 0, 0


# --- Elo ---

def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def update_elo(db, winner_id, loser_id, category, k=32, draw=False):
    """Update Elo ratings after a comparison."""
    for mid in (winner_id, loser_id):
        db.execute(
            "INSERT OR IGNORE INTO elo_ratings (model_id, category) VALUES (?, ?)",
            (mid, category),
        )

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


# --- Commands ---

DEMO_MODELS = [
    {"id": "model-alpha", "provider": "demo", "model": "alpha", "api_key_env": "_"},
    {"id": "model-beta", "provider": "demo", "model": "beta", "api_key_env": "_"},
    {"id": "model-gamma", "provider": "demo", "model": "gamma", "api_key_env": "_"},
]


def demo_response(model_id, prompt):
    """Generate a synthetic response that varies by model."""
    styles = {
        "model-alpha": ("concise and technical", 400, 800),
        "model-beta": ("detailed with examples", 600, 1200),
        "model-gamma": ("creative and conversational", 500, 1000),
    }
    style, lo, hi = styles.get(model_id, ("neutral", 500, 900))
    latency = random.randint(lo, hi)
    # Use hash to make responses deterministic per prompt+model but different across models
    seed = hashlib.md5(f"{model_id}:{prompt}".encode()).hexdigest()
    responses = {
        "model-alpha": f"Here's a direct answer:\n\n{prompt.split()[-1].title()} can be understood through three key principles:\n1. Simplicity in design\n2. Composability of components\n3. Clear separation of concerns\n\nThe most important takeaway is that less complexity leads to more maintainable systems.",
        "model-beta": f"Great question! Let me break this down with an example.\n\nThink of {prompt.split()[0].lower()} like building with LEGO blocks. Each piece has a specific shape and purpose, but you can combine them in countless ways.\n\nFor instance, consider this scenario:\n- Start with the basics\n- Layer on complexity gradually\n- Test each addition independently\n\nHere's a concrete example:\n```\nresult = compose(step1, step2, step3)\n```\n\nThe key insight is that good abstractions compound over time.",
        "model-gamma": f"Oh, this is a fun one! 🎯\n\nSo here's the thing — most people overthink {prompt.split()[-1].lower()}. The secret is that it's really just about finding patterns and making them repeatable.\n\nI like to think of it as a conversation between your current self and your future self. What would future-you want to know? Start there.\n\nBottom line: keep it simple, keep it human, keep iterating.",
    }
    return responses.get(model_id, f"Response about: {prompt}"), latency, random.randint(100, 500)


def cmd_compare(args):
    """Run a blind comparison."""
    config = load_config()
    db = get_db()
    demo = args.demo

    # Get prompt
    if args.prompt:
        prompt = " ".join(args.prompt)
    else:
        print("Enter your prompt (Ctrl+D to finish):")
        prompt = sys.stdin.read().strip()

    if not prompt:
        print("No prompt provided.")
        return

    category = args.category or "general"

    if demo:
        available = DEMO_MODELS
    else:
        # Select models
        available = [m for m in config["models"] if os.environ.get(m["api_key_env"])]
        if len(available) < 2:
            print("Need at least 2 models with API keys configured.")
            print("Set environment variables:", [m["api_key_env"] for m in config["models"]])
            print("\nTip: use --demo to try with mock models")
            return

    n = min(args.models or config["default_models_per_round"], len(available))
    selected = random.sample(available, n)

    # Generate comparison ID
    comp_id = hashlib.sha256(f"{prompt}{time.time()}".encode()).hexdigest()[:12]

    # Call models
    print(f"\n⏳ Sending to {n} models{'  [demo mode]' if demo else ''}...")
    results = []
    for model_cfg in selected:
        if demo:
            content, latency, tokens = demo_response(model_cfg["id"], prompt)
        else:
            content, latency, tokens = call_model(model_cfg, prompt)
        if content:
            results.append({"model": model_cfg, "content": content, "latency": latency, "tokens": tokens})

    if len(results) < 2:
        print("Not enough models responded successfully.")
        return

    # Shuffle and assign labels
    random.shuffle(results)
    labels = "ABCDEFGH"

    # Store comparison
    db.execute("INSERT INTO comparisons (id, prompt, category) VALUES (?, ?, ?)", (comp_id, prompt, category))
    for i, r in enumerate(results):
        db.execute(
            "INSERT INTO responses (comparison_id, model_id, label, content, latency_ms, tokens_used) VALUES (?, ?, ?, ?, ?, ?)",
            (comp_id, r["model"]["id"], labels[i], r["content"], r["latency"], r["tokens"]),
        )
    db.commit()

    # Display responses
    print(f"\n{'='*60}")
    print(f"Comparison: {comp_id} | Category: {category}")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"\n{'─'*60}")
        print(f"  Response {labels[i]}  ({r['latency']}ms)")
        print(f"{'─'*60}")
        print(r["content"])

    # Vote
    print(f"\n{'='*60}")
    print(f"Which is better? [{'/'.join(labels[:len(results)])}] or [tie] or [skip]")
    vote = input("> ").strip().upper()

    if vote == "SKIP":
        db.execute("UPDATE comparisons SET status='skipped' WHERE id=?", (comp_id,))
        db.commit()
        print("Skipped.")
        return

    if vote == "TIE":
        db.execute("UPDATE comparisons SET status='tie' WHERE id=?", (comp_id,))
        db.commit()
        # Update Elo as draw for all pairs
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                update_elo(db, results[i]["model"]["id"], results[j]["model"]["id"], category, draw=True)
        print("Recorded as tie.")
        _reveal(results, labels)
        return

    if vote in labels[:len(results)]:
        winner_idx = labels.index(vote)
        winner = results[winner_idx]
        db.execute("UPDATE comparisons SET status='voted', winner_model=? WHERE id=?", (winner["model"]["id"], comp_id))
        db.commit()
        # Update Elo: winner beats all others
        for i, r in enumerate(results):
            if i != winner_idx:
                update_elo(db, winner["model"]["id"], r["model"]["id"], category)
        print(f"\n✓ You picked {vote}.")
        _reveal(results, labels)
    else:
        print("Invalid choice.")


def _reveal(results, labels):
    """Reveal which model was which."""
    print("\n🔍 Reveal:")
    for i, r in enumerate(results):
        print(f"  {labels[i]} = {r['model']['id']}")


def cmd_stats(args):
    """Show current rankings."""
    db = get_db()
    category = args.category

    if category:
        rows = db.execute(
            "SELECT * FROM elo_ratings WHERE category=? ORDER BY rating DESC", (category,)
        ).fetchall()
        print(f"\n📊 Rankings — {category}")
    else:
        rows = db.execute(
            "SELECT model_id, 'all' as category, AVG(rating) as rating, SUM(wins) as wins, SUM(losses) as losses, SUM(draws) as draws FROM elo_ratings GROUP BY model_id ORDER BY rating DESC"
        ).fetchall()
        print("\n📊 Rankings — All categories")

    if not rows:
        print("  No data yet. Run `blind compare` to start.")
        return

    print(f"  {'Model':<20} {'Elo':>7} {'W':>4} {'L':>4} {'D':>4}")
    print(f"  {'─'*43}")
    for r in rows:
        print(f"  {r['model_id']:<20} {r['rating']:>7.0f} {r['wins']:>4} {r['losses']:>4} {r['draws']:>4}")

    total = db.execute("SELECT COUNT(*) as n FROM comparisons WHERE status='voted' OR status='tie'").fetchone()
    print(f"\n  Total comparisons: {total['n']}")


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
        prompt_short = r["prompt"][:60] + "..." if len(r["prompt"]) > 60 else r["prompt"]
        winner = r["winner_model"] or r["status"]
        print(f"  [{r['created_at'][:10]}] {r['category']:<10} {winner:<15} {prompt_short}")


def cmd_categories(args):
    """List categories with comparison counts."""
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


def cmd_init(args):
    """Initialize config."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists() and not args.force:
        print(f"Config exists at {CONFIG_PATH}. Use --force to overwrite.")
        return

    with open(CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)

    print(f"✓ Config created at {CONFIG_PATH}")
    print("  Edit it to add/remove models, then set API key env vars.")


def cmd_config(args):
    """Show current config."""
    config = load_config()
    print(json.dumps(config, indent=2))


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return DEFAULT_CONFIG


# --- Main ---

def main():
    parser = argparse.ArgumentParser(prog="blind", description="Personal blind LLM comparison tool")
    sub = parser.add_subparsers(dest="command")

    # compare
    p = sub.add_parser("compare", aliases=["c"], help="Run a blind comparison")
    p.add_argument("prompt", nargs="*", help="Prompt text (or pipe via stdin)")
    p.add_argument("-c", "--category", help="Task category (e.g. code, writing, research)")
    p.add_argument("-n", "--models", type=int, help="Number of models to compare")
    p.add_argument("--demo", action="store_true", help="Use mock models (no API keys needed)")

    # stats
    p = sub.add_parser("stats", aliases=["s"], help="Show rankings")
    p.add_argument("-c", "--category", help="Filter by category")

    # history
    p = sub.add_parser("history", aliases=["h"], help="Show comparison history")
    p.add_argument("-n", "--limit", type=int, default=20)

    # categories
    sub.add_parser("categories", aliases=["cat"], help="List categories")

    # init
    p = sub.add_parser("init", help="Initialize config")
    p.add_argument("--force", action="store_true")

    # config
    sub.add_parser("config", help="Show config")

    args = parser.parse_args()

    commands = {
        "compare": cmd_compare, "c": cmd_compare,
        "stats": cmd_stats, "s": cmd_stats,
        "history": cmd_history, "h": cmd_history,
        "categories": cmd_categories, "cat": cmd_categories,
        "init": cmd_init,
        "config": cmd_config,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
