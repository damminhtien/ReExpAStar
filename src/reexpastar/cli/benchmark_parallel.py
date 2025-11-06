import argparse
import csv
from datetime import datetime
import itertools
from multiprocessing import Pool, cpu_count
import os
from typing import Any, cast

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


def run_job(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    _, cfg = job
    sc = cfg["scenario_ctor"](**cfg["scenario_kwargs"])
    algo = cfg["algo"]
    weight = cfg["weight"]
    r_mode = cfg["r_mode"]
    goal_test = cast(GoalTest[Any], sc.is_goal)
    neighbors = cast(NeighborFn[Any], sc.neighbors)
    heuristic = cast(HeuristicFn[Any], sc.h)
    eng: WeightedAStarCR[Any]
    if algo == "NR":
        eng = WeightedAStarNR[Any](sc.start, goal_test, neighbors, heuristic, weight=weight)
    elif algo == "AR":
        eng = WeightedAStarAR[Any](sc.start, goal_test, neighbors, heuristic, weight=weight)
    else:
        r = cfg["r"]
        params = WeightedAStarParams(weight=weight, r=r, r_mode=r_mode)
        eng = WeightedAStarCR[Any](sc.start, goal_test, neighbors, heuristic, params=params)
    path, cost, st = eng.run_to_first_goal()
    return {
        "scenario": sc.name,
        "kind": sc.meta["kind"],
        "algo": algo,
        "w": weight,
        "r": cfg.get("r"),
        "r_mode": r_mode,
        "cost": cost,
        "expansions": st.expansions,
        "reopens": st.reopens,
        "generated": st.generated,
        "runtime_ms": st.runtime_ms,
        "path_len": (len(path) if path else None),
        "seed": cfg["scenario_kwargs"].get("seed", None),
    }


def main():
    p = argparse.ArgumentParser(description="Parallel benchmark across multiple seeds")
    p.add_argument("--weight", type=float, default=1.5)
    p.add_argument("--r_values", type=str, default="0.5,0.2")
    p.add_argument("--r_mode", type=str, default="abs", choices=["abs", "rel_edge", "rel_g"])
    p.add_argument("--seeds", type=str, default="0,1,2,3")
    p.add_argument("--jobs", type=int, default=0, help="number of processes (0 -> cpu_count)")
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    r_list = [float(x.strip()) for x in args.r_values.split(",") if x.strip() != ""]

    scenario_ctors = [
        (scenario_grid_4, {"width": 50, "height": 50, "density": 0.15}),
        (scenario_grid_8, {"width": 60, "height": 60, "density": 0.20}),
        (scenario_grid_inconsistent, {"width": 50, "height": 50, "density": 0.10}),
        (scenario_terrain8, {"width": 60, "height": 60}),
        (scenario_maze4, {"width": 51, "height": 51}),
        (scenario_geometric, {"n": 150, "k": 8}),
        (scenario_puzzle, {"steps": 25}),
    ]

    jobs: list[tuple[str, dict[str, Any]]] = []
    for (ctor, base_kwargs), seed in itertools.product(scenario_ctors, seeds):
        kwargs = dict(base_kwargs)
        kwargs["seed"] = seed
        for algo in ["NR", "AR"]:
            jobs.append(
                (
                    f"{ctor.__name__}-{algo}-seed{seed}",
                    {
                        "scenario_ctor": ctor,
                        "scenario_kwargs": kwargs,
                        "algo": algo,
                        "weight": args.weight,
                        "r_mode": args.r_mode,
                    },
                )
            )
        for r in r_list:
            jobs.append(
                (
                    f"{ctor.__name__}-CR{r}-seed{seed}",
                    {
                        "scenario_ctor": ctor,
                        "scenario_kwargs": kwargs,
                        "algo": "CR",
                        "weight": args.weight,
                        "r": r,
                        "r_mode": args.r_mode,
                    },
                )
            )

    procs = args.jobs or cpu_count()
    with Pool(processes=procs) as pool:
        rows = pool.map(run_job, jobs)

    # write aggregated CSV
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join(
        "results", f"benchmark_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
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
        "seed",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(out_path)


if __name__ == "__main__":
    main()
