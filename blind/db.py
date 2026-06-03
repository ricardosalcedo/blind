"""Database schema and access."""

import sqlite3

from .config import DB_PATH

SCHEMA = """
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
"""


def get_db():
    """Get a database connection, creating schema if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db
