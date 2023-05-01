import argparse
import math
from reexpastar.wa_star_cr import WeightedAStarCR, WeightedAStarNR, WeightedAStarAR, Grid


def parse_r_list(s: str):
    vals = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if tok in ("inf", "infinity", "∞"):
            vals.append(math.inf)
        else:
            vals.append(float(tok))
    return vals


def main():
    p = argparse.ArgumentParser(description="Ablate r over CR; compare with NR/AR")
    p.add_argument("--width", type=int, default=60)
    p.add_argument("--height", type=int, default=60)
    p.add_argument("--weight", type=float, default=1.5)
    p.add_argument("--r_values", type=str, default="inf,1.0,0.5,0.2,0.0")
    p.add_argument("--r_mode", type=str, default="abs", choices=["abs", "rel_edge", "rel_g"])
    args = p.parse_args()

    r_list = parse_r_list(args.r_values)
    walls = {(args.width // 2, y) for y in range(args.height)}
    gap_y = args.height // 3
    walls.remove((args.width // 2, gap_y))
    grid = Grid(args.width, args.height, walls=walls)
    start, goal = (0, 0), (args.width - 1, args.height - 1)
    h = grid.manhattan(goal)

    def is_goal(s):
        return s == goal

    def emit_row(name, eng):
        path, cost, st = eng.run_to_first_goal()
        row = dict(
            algo=name,
            w=eng.w,
            r=getattr(eng, "r", None),
            r_mode=getattr(eng, "r_mode", None),
            cost=cost,
            expansions=st.expansions,
            reopens=st.reopens,
            generated=st.generated,
            runtime_ms=st.runtime_ms,
            path_len=(len(path) if path else None),
        )
        return row

    rows = []
    rows.append(
        emit_row("NR", WeightedAStarNR(start, is_goal, grid.neighbors, h, weight=args.weight))
    )
    rows.append(
        emit_row("AR", WeightedAStarAR(start, is_goal, grid.neighbors, h, weight=args.weight))
    )
    for r in r_list:
        if math.isinf(r):
            continue
        rows.append(
            emit_row(
                f"CR(r={r})",
                WeightedAStarCR(
                    start, is_goal, grid.neighbors, h, weight=args.weight, r=r, r_mode=args.r_mode
                ),
            )
        )

    keys = [
        "algo",
        "w",
        "r",
        "r_mode",
        "cost",
        "expansions",
        "reopens",
        "generated",
        "runtime_ms",
        "path_len",
    ]
    print(",".join(keys))
    for row in rows:
        print(",".join(str(row[k]) for k in keys))


if __name__ == "__main__":
    main()
