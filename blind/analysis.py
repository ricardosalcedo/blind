"""Analysis utilities: bias detection, categorization, insights."""

import os

from .config import CATEGORY_KEYWORDS


def auto_categorize(prompt):
    """Guess task category from prompt keywords."""
    prompt_lower = prompt.lower()
    scores = {cat: sum(1 for k in kw if k in prompt_lower) for cat, kw in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def check_position_bias(db):
    """Detect if user always picks the same position. Returns warning string or None."""
    rows = db.execute(
        "SELECT winner_position, COUNT(*) as n FROM comparisons "
        "WHERE status='voted' AND winner_position IS NOT NULL GROUP BY winner_position"
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


def get_available_models(config):
    """Return models that have credentials configured."""
    return [m for m in config["models"] if m.get("provider") == "bedrock" or os.environ.get(m.get("api_key_env", ""))]
