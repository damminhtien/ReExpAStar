import argparse
from datetime import datetime
import math
import os
from typing import Any, cast

from reexpastar.ara_star import ARAStar
from reexpastar.core.types import GoalTest, HeuristicFn, NeighborFn
from reexpastar.scenarios import (
    scenario_geometric,
    scenario_grid_4,
    scenario_grid_8,
    scenario_grid_inconsistent,
    scenario_maze4,
    scenario_puzzle,
    scenario_terrain8,
)
from reexpastar.wa_star_cr import (
    WeightedAStarAR,
    WeightedAStarCR,
    WeightedAStarNR,
    WeightedAStarParams,
)


def run_one(_name: str, eng: WeightedAStarCR[Any]) -> dict[str, Any]:
    path, cost, st = eng.run_to_first_goal()
    return {
        "cost": cost,
        "expansions": st.expansions,
        "reopens": getattr(st, "reopens", 0),
        "generated": st.generated,
        "runtime_ms": st.runtime_ms,
        "path_len": (len(path) if path else None),
    }


def main():
    p = argparse.ArgumentParser(description="Diverse benchmark suite")
    p.add_argument("--weight", type=float, default=1.5)
    p.add_argument("--r_values", type=str, default="inf,0.5,0.2,0.0")
    p.add_argument("--r_mode", type=str, default="abs", choices=["abs", "rel_edge", "rel_g"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include_ara", action="store_true")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    scenarios = [
        scenario_grid_4(50, 50, density=0.15, seed=args.seed),
        scenario_grid_8(60, 60, density=0.20, seed=args.seed),
        scenario_grid_inconsistent(50, 50, density=0.10, seed=args.seed),
        scenario_terrain8(60, 60, seed=args.seed),
        scenario_maze4(51, 51, seed=args.seed),
        scenario_geometric(150, k=8, seed=args.seed),
        scenario_puzzle(steps=25, seed=args.seed),
    ]

    r_list = []
    for tok in args.r_values.split(","):
        t = tok.strip().lower()
        if t in ("inf", "infinity", "∞"):
            r_list.append(math.inf)
        else:
            r_list.append(float(t))

    keys = [
        "scenario",
        "kind",
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
    lines = [",".join(keys)]
    for sc in scenarios:
        goal_test = cast(GoalTest[Any], sc.is_goal)
        neighbors = cast(NeighborFn[Any], sc.neighbors)
        heuristic = cast(HeuristicFn[Any], sc.h)
        if args.include_ara:
            ara = ARAStar(sc.start, sc.is_goal, sc.neighbors, sc.h, w0=args.weight)
            hist, best_path = ara.run_schedule([args.weight, 1.2, 1.0])
            final_cost = hist[-1][1]
            lines.append(
                ",".join(
                    map(
                        str,
                        [
                            sc.name,
                            sc.meta["kind"],
                            "ARA*",
                            args.weight,
                            None,
                            None,
                            final_cost,
                            ara.stats.expansions,
                            0,
                            ara.stats.generated,
                            None,
                            (len(best_path) if best_path else None),
                        ],
                    )
                )
            )
        for r in r_list:
            if math.isinf(r):
                engines = [
                    (
                        "NR",
                        WeightedAStarNR[Any](
                            sc.start, goal_test, neighbors, heuristic, weight=args.weight
                        ),
                    )
                ]
            else:
                engines = [
                    (
                        "AR",
                        WeightedAStarAR[Any](
                            sc.start, goal_test, neighbors, heuristic, weight=args.weight
                        ),
                    ),
                    (
                        f"CR(r={r})",
                        WeightedAStarCR[Any](
                            sc.start,
                            goal_test,
                            neighbors,
                            heuristic,
                            params=WeightedAStarParams(
                                weight=args.weight,
                                r=r,
                                r_mode=args.r_mode,
                            ),
                        ),
                    ),
                ]
            for name, eng in engines:
                row = run_one(name, eng)
                out = {
                    "scenario": sc.name,
                    "kind": sc.meta["kind"],
                    "algo": name,
                    "w": args.weight,
                    "r": (None if name == "NR" else (0.0 if name == "AR" else r)),
                    "r_mode": (None if name == "NR" else ("abs" if name == "AR" else args.r_mode)),
                    **row,
                }
                lines.append(",".join(str(out[k]) for k in keys))

    out_path = args.out or os.path.join(
        "results", f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(out_path)
    print(lines[0])
    for ln in lines[1:15]:
        print(ln)


if __name__ == "__main__":
    main()
