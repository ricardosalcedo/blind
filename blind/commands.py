"""CLI command implementations."""

import asyncio
import csv
import hashlib
import json
import random
import sys
import time

from .analysis import auto_categorize, check_position_bias, check_streak, get_available_models
from .chart import render_chart
from .config import CONFIG_PATH, DEFAULT_CONFIG, DEMO_MODELS, load_config
from .db import get_db
from .elo import confidence_interval, update_elo
from .providers import call_models_parallel, demo_response, estimate_cost

LABELS = "ABCDEFGH"


def cmd_compare(args):
    """Run a blind comparison between models."""
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
    available = DEMO_MODELS if demo else get_available_models(config)

    if len(available) < 2:
        print("Need at least 2 models with API keys set.")
        print("Keys needed:", [m["api_key_env"] for m in config["models"]])
        print("\nUse --demo to try with mock models")
        return

    n = min(args.models or config["default_models_per_round"], len(available))
    selected = random.sample(available, n)
    comp_id = hashlib.sha256(f"{prompt}{time.time()}".encode()).hexdigest()[:12]

    # Call models
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

    # Shuffle to blind
    random.shuffle(results)

    # Store
    db.execute("INSERT INTO comparisons (id, prompt, category) VALUES (?, ?, ?)", (comp_id, prompt, category))
    for i, r in enumerate(results):
        db.execute(
            "INSERT INTO responses (comparison_id, model_id, label, content, latency_ms, input_tokens, output_tokens, cost_usd) VALUES (?,?,?,?,?,?,?,?)",
            (
                comp_id,
                r["model"]["id"],
                LABELS[i],
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
        print(f"  Response {LABELS[i]}  ({r['latency']}ms · {wc} words)")
        print(f"{'─' * 60}")
        print(r["content"])

    # Length bias warning
    lengths = [len(r["content"].split()) for r in results]
    if max(lengths) >= 3 * min(lengths) and min(lengths) > 10:
        short_i, long_i = lengths.index(min(lengths)), lengths.index(max(lengths))
        print(
            f"\n  ℹ️  {LABELS[long_i]} is {max(lengths) // min(lengths)}x longer than {LABELS[short_i]}. Longer ≠ better."
        )

    # Vote prompt
    choices = "/".join(LABELS[: len(results)])
    print(f"\n{'=' * 60}")
    if args.rank and len(results) > 2:
        print("  Rank all responses best→worst (e.g. 'BAC'):")
    else:
        print(f"  Which is better? [{choices}] or [tie] or [skip]")

    bias = check_position_bias(db)
    if bias:
        print(f"  {bias}")

    vote = input("  > ").strip().upper()
    _process_vote(db, vote, results, comp_id, category, args.rank)


def _process_vote(db, vote, results, comp_id, category, rank_mode):
    """Process the user's vote."""
    # Rank mode
    if rank_mode and len(vote) == len(results) and all(c in LABELS[: len(results)] for c in vote):
        db.execute("UPDATE comparisons SET status='ranked', vote_type='rank' WHERE id=?", (comp_id,))
        for pos, letter in enumerate(vote):
            db.execute("UPDATE responses SET rank=? WHERE comparison_id=? AND label=?", (pos + 1, comp_id, letter))
        for i in range(len(vote)):
            for j in range(i + 1, len(vote)):
                wi, li = LABELS.index(vote[i]), LABELS.index(vote[j])
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

    if vote in LABELS[: len(results)]:
        wi = LABELS.index(vote)
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
        for r in results:
            s = check_streak(db, r["model"]["id"])
            if s:
                print(f"  {s}")
    else:
        print("  Invalid choice.")


def _reveal(results):
    """Show which model was which."""
    print("\n  🔍 Reveal:")
    for i, r in enumerate(results):
        cost = f"  ${r['cost']:.4f}" if r["cost"] > 0 else ""
        print(f"    {LABELS[i]} = {r['model']['id']}{cost}")


def cmd_stats(args):
    """Show Elo rankings."""
    db = get_db()
    cat = args.category

    if cat:
        rows = db.execute("SELECT * FROM elo_ratings WHERE category=? ORDER BY rating DESC", (cat,)).fetchall()
        title = cat
    else:
        rows = db.execute(
            "SELECT model_id, 'all' as category, AVG(rating) as rating, SUM(wins) as wins, "
            "SUM(losses) as losses, SUM(draws) as draws "
            "FROM elo_ratings GROUP BY model_id ORDER BY rating DESC"
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
        ci = confidence_interval(r["wins"], r["losses"], r["draws"])
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
    """Show pairwise win rates."""
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
                    "SELECT COUNT(*) as n FROM comparisons c "
                    "JOIN responses r1 ON r1.comparison_id=c.id AND r1.model_id=? "
                    "JOIN responses r2 ON r2.comparison_id=c.id AND r2.model_id=? "
                    "WHERE c.winner_model=? AND c.status='voted'",
                    (m1, m2, m1),
                ).fetchone()["n"]
                total = db.execute(
                    "SELECT COUNT(*) as n FROM comparisons c "
                    "JOIN responses r1 ON r1.comparison_id=c.id AND r1.model_id=? "
                    "JOIN responses r2 ON r2.comparison_id=c.id AND r2.model_id=? "
                    "WHERE c.status IN ('voted','tie')",
                    (m1, m2),
                ).fetchone()["n"]
                print(f" {f'{wins}/{total}':>8}", end="")
        print()


def cmd_insights(args):
    """Analyze voting patterns."""
    db = get_db()
    total = db.execute("SELECT COUNT(*) as n FROM comparisons WHERE status IN ('voted','tie','ranked')").fetchone()["n"]
    if total < 5:
        print("Need at least 5 comparisons for insights.")
        return

    print(f"\n💡 Insights ({total} comparisons)\n")

    # Category dominance
    cat_winners = db.execute(
        "SELECT category, winner_model, COUNT(*) as n FROM comparisons "
        "WHERE status='voted' GROUP BY category, winner_model ORDER BY category, n DESC"
    ).fetchall()
    cat_map = {}
    for r in cat_winners:
        cat_map.setdefault(r["category"], []).append((r["winner_model"], r["n"]))
    for cat, winners in cat_map.items():
        if len(winners) > 1:
            top = winners[0]
            total_cat = sum(w[1] for w in winners)
            if top[1] / total_cat > 0.6:
                print(f"  📌 {top[0]} dominates in '{cat}' ({top[1]}/{total_cat} = {top[1] / total_cat:.0%})")

    # Length preference
    shorter_wins = longer_wins = 0
    for comp in db.execute("SELECT id, winner_model FROM comparisons WHERE status='voted'").fetchall():
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
            print(f"  📏 You prefer shorter responses ({shorter_wins} vs {longer_wins} longer wins)")
        elif longer_wins > shorter_wins * 1.5:
            print(f"  📏 You prefer longer/detailed responses ({longer_wins} vs {shorter_wins} shorter wins)")

    # Position bias
    bias = check_position_bias(db)
    if bias:
        print(f"  {bias}")

    # Low confidence
    low_conf = db.execute("SELECT COUNT(*) as n FROM elo_ratings WHERE wins+losses+draws < 5").fetchone()["n"]
    if low_conf:
        print(f"\n  ⚠️  {low_conf} model-category pairs with <5 comparisons (low confidence)")


def cmd_chart(args):
    """Show ASCII Elo trend chart."""
    render_chart(get_db(), args.category)


def cmd_replay(args):
    """Re-run a past comparison."""
    db = get_db()
    comp = db.execute(
        "SELECT * FROM comparisons WHERE id LIKE ? ORDER BY created_at DESC LIMIT 1", (f"{args.id}%",)
    ).fetchone()
    if not comp:
        print(f"Comparison '{args.id}' not found.")
        return
    print(f"Replaying: {comp['prompt'][:80]}")
    sys.argv = ["blind", "compare"] + (["--demo"] if args.demo else []) + ["-c", comp["category"], comp["prompt"]]
    from .cli import main

    main()


def cmd_history(args):
    """Show past comparisons."""
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
    """List categories with counts."""
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


def cmd_export(args):
    """Export all data to CSV."""
    db = get_db()
    out = args.output or "blind_export.csv"
    rows = db.execute(
        "SELECT c.created_at, c.category, c.prompt, c.winner_model, c.status, "
        "r.model_id, r.label, r.latency_ms, r.input_tokens, r.output_tokens, r.cost_usd, r.rank "
        "FROM comparisons c JOIN responses r ON r.comparison_id=c.id ORDER BY c.created_at DESC"
    ).fetchall()
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
    """Import from CSV."""
    db = get_db()
    count = 0
    with open(args.file) as f:
        for row in csv.DictReader(f):
            if row.get("winner") and row.get("model") and row.get("category"):
                update_elo(db, row["winner"], row["model"], row["category"])
                count += 1
    print(f"✓ Imported {count} records from {args.file}")


def cmd_reset(args):
    """Delete all data."""
    if not args.confirm:
        print("Delete all data? Use --confirm")
        return
    db = get_db()
    db.executescript(
        "DELETE FROM responses; DELETE FROM comparisons; DELETE FROM elo_ratings; DELETE FROM elo_history;"
    )
    print("✓ All data reset.")


def cmd_init(args):
    """Create default config file."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists() and not args.force:
        print(f"Config exists at {CONFIG_PATH}. Use --force to overwrite.")
        return
    with open(CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print(f"✓ Config created at {CONFIG_PATH}")


def cmd_config(args):
    """Show current config."""
    print(json.dumps(load_config(), indent=2))
