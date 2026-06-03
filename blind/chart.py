"""ASCII Elo trend chart."""


def render_chart(db, category=None):
    """Render an ASCII chart of Elo rating trends."""
    if category:
        rows = db.execute(
            "SELECT model_id, rating, recorded_at FROM elo_history WHERE category=? ORDER BY recorded_at",
            (category,),
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

    width = min(60, max(len(pts) for pts in series.values()))
    height = 15
    all_ratings = [r for pts in series.values() for r in pts]
    lo, hi = min(all_ratings) - 10, max(all_ratings) + 10
    span = hi - lo if hi > lo else 1

    symbols = "●○◆◇■□▲△"
    model_ids = sorted(series.keys())

    print(f"\n📈 Elo Trend{f' — {category}' if category else ''}")
    print(f"  {hi:.0f} ┐")

    # Build grid
    grid = [[" "] * width for _ in range(height)]
    for mi, mid in enumerate(model_ids):
        pts = series[mid]
        sampled = _resample(pts, width)
        sym = symbols[mi % len(symbols)]
        for x, rating in enumerate(sampled):
            y = max(0, min(height - 1, int((rating - lo) / span * (height - 1))))
            grid[height - 1 - y][x] = sym

    # Print
    for i, row in enumerate(grid):
        label = f"{(hi + lo) / 2:.0f} ┤" if i == height // 2 else "       │"
        print(f"  {label}{''.join(row)}")

    print(f"  {lo:.0f} ┘{'─' * width}")
    print(f"       {'oldest':<{width - 6}}{'latest':>6}")

    # Legend
    print("\n  Legend:")
    for mi, mid in enumerate(model_ids):
        sym = symbols[mi % len(symbols)]
        print(f"    {sym} {mid:<20} (current: {series[mid][-1]:.0f})")


def _resample(pts, width):
    """Downsample a series to fit the chart width."""
    if len(pts) <= width:
        return pts
    step = len(pts) / width
    return [pts[int(i * step)] for i in range(width)]
