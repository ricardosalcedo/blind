# Blind

**Which LLM is actually better — for you?**

Blind sends your prompt to multiple models simultaneously, shows you the responses without labels, and asks you to pick the winner. Over time, it builds your personal model ranking using Elo scores, broken down by task category.

Chatbot Arena tells you what the crowd thinks. Blind tells you what **you** think.

## Why

- You pay for the most expensive model "just in case" when a cheaper one might be better for your tasks
- Benchmark scores don't reflect your actual use (writing style, code patterns, domain knowledge)
- Your preferences ≠ crowd preferences. A writer's ranking differs from a programmer's
- After 50 comparisons, you'll know exactly which model to use for what

## Install

```bash
pip install blind-llm
```

Or run directly:
```bash
git clone https://github.com/yourusername/blind
cd blind
pip install -e .
```

## Quick start

```bash
# Set API keys for the models you want to compare
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Initialize config (optional — defaults work)
blind init

# Run a comparison
blind compare "explain monads in simple terms"

# Tag by category
blind compare -c code "write a Python function to merge two sorted lists"

# See your rankings
blind stats
blind stats -c code
```

## How it works

```
You type a prompt
    → Blind sends it to N models (default 2)
    → Responses come back, shuffled and labeled A/B/C
    → You read them and pick the winner (or tie/skip)
    → Elo ratings update per model per category
    → Over time: your personal model ranking emerges
```

## Example session

```
$ blind compare -c writing "write a haiku about debugging"

══════════════════════════════════════════════════════════
Comparison: a3f21c | Category: writing
══════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────
  Response A  (1204ms)
──────────────────────────────────────────────────────────
Semicolons hide
in lines I've read forty times—
the bug was a typo.

──────────────────────────────────────────────────────────
  Response B  (890ms)
──────────────────────────────────────────────────────────
Stack trace unfurling,
hours lost to missing comma—
coffee grows cold now.

══════════════════════════════════════════════════════════
Which is better? [A/B] or [tie] or [skip]
> B

✓ You picked B.

🔍 Reveal:
  A = claude-sonnet
  B = gpt-4o
```

```
$ blind stats

📊 Rankings — All categories
  Model                    Elo    W    L    D
  ───────────────────────────────────────────
  gpt-4o                  1534   12    7    2
  claude-sonnet           1521   11    8    3
  gemini-pro              1445    5   13    1

  Total comparisons: 21
```

## Configuration

Config lives at `~/.blind/config.json`. Edit to add models:

```json
{
  "models": [
    {"id": "claude-sonnet", "provider": "anthropic", "model": "claude-sonnet-4-20250514", "api_key_env": "ANTHROPIC_API_KEY"},
    {"id": "gpt-4o", "provider": "openai", "model": "gpt-4o", "api_key_env": "OPENAI_API_KEY"},
    {"id": "local-llama", "provider": "openai-compatible", "model": "llama3", "api_key_env": "OLLAMA_KEY", "base_url": "http://localhost:11434/v1"}
  ]
}
```

Supported providers: `openai`, `anthropic`, `google`, `openai-compatible` (Ollama, LiteLLM, vLLM, etc.)

## Commands

| Command | Description |
|---------|-------------|
| `blind compare [prompt]` | Run a blind comparison |
| `blind stats [-c category]` | Show Elo rankings |
| `blind history [-n 20]` | Show past comparisons |
| `blind categories` | List categories with counts |
| `blind init` | Create default config |
| `blind config` | Show current config |

## Flags

- `-c, --category` — Tag comparison with a category (code, writing, research, math, etc.)
- `-n, --models` — Number of models per comparison (default: 2)

## Data

Everything stored locally in `~/.blind/blind.db` (SQLite). No telemetry, no cloud, no accounts.

## License

MIT
