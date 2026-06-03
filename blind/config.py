"""Configuration and constants."""

import json
import os
from pathlib import Path

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

# Cost per 1M tokens (input, output)
MODEL_COSTS = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
}

# Keywords for auto-categorization
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

DEMO_MODELS = [
    {"id": "model-alpha", "provider": "demo", "model": "alpha", "api_key_env": "_"},
    {"id": "model-beta", "provider": "demo", "model": "beta", "api_key_env": "_"},
    {"id": "model-gamma", "provider": "demo", "model": "gamma", "api_key_env": "_"},
]


def load_config():
    """Load user config, falling back to defaults."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return DEFAULT_CONFIG
