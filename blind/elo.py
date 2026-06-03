"""Elo rating system."""

import math


def expected_score(ra, rb):
    """Probability that player with rating `ra` beats player with rating `rb`."""
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def update_elo(db, winner_id, loser_id, category, k=32, draw=False):
    """Update Elo ratings and record history."""
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
        new_ra, new_rb = ra + k * (0.5 - ea), rb + k * (0.5 - eb)
        db.execute(
            "UPDATE elo_ratings SET rating=?, draws=draws+1 WHERE model_id=? AND category=?",
            (new_ra, winner_id, category),
        )
        db.execute(
            "UPDATE elo_ratings SET rating=?, draws=draws+1 WHERE model_id=? AND category=?",
            (new_rb, loser_id, category),
        )
    else:
        new_ra, new_rb = ra + k * (1 - ea), rb + k * (0 - eb)
        db.execute(
            "UPDATE elo_ratings SET rating=?, wins=wins+1 WHERE model_id=? AND category=?",
            (new_ra, winner_id, category),
        )
        db.execute(
            "UPDATE elo_ratings SET rating=?, losses=losses+1 WHERE model_id=? AND category=?",
            (new_rb, loser_id, category),
        )

    db.commit()

    # Record history
    for mid in (winner_id, loser_id):
        r = db.execute("SELECT rating FROM elo_ratings WHERE model_id=? AND category=?", (mid, category)).fetchone()
        db.execute(
            "INSERT INTO elo_history (model_id, category, rating) VALUES (?, ?, ?)", (mid, category, r["rating"])
        )
    db.commit()


def confidence_interval(wins, losses, draws):
    """Approximate 95% CI width. Returns None if insufficient data."""
    n = wins + losses + draws
    return 400 / math.sqrt(n) if n >= 5 else None
