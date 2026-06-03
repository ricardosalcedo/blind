"""CLI entry point and argument parsing."""

import argparse

from . import commands


def main():
    """Main entry point."""
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

    sub.add_parser("head2head", aliases=["h2h"], help="Pairwise win rates")
    sub.add_parser("insights", aliases=["i"], help="Voting pattern analysis")

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
    dispatch = {
        "compare": commands.cmd_compare,
        "c": commands.cmd_compare,
        "stats": commands.cmd_stats,
        "s": commands.cmd_stats,
        "head2head": commands.cmd_head2head,
        "h2h": commands.cmd_head2head,
        "insights": commands.cmd_insights,
        "i": commands.cmd_insights,
        "replay": commands.cmd_replay,
        "r": commands.cmd_replay,
        "history": commands.cmd_history,
        "h": commands.cmd_history,
        "categories": commands.cmd_categories,
        "cat": commands.cmd_categories,
        "chart": commands.cmd_chart,
        "export": commands.cmd_export,
        "import": commands.cmd_import_csv,
        "reset": commands.cmd_reset,
        "init": commands.cmd_init,
        "config": commands.cmd_config,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
