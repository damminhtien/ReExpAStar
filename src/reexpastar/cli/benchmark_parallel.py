import argparse, math, os, itertools, csv
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Tuple, List
from reexpastar.wa_star_cr import WeightedAStarCR, WeightedAStarNR, WeightedAStarAR
from reexpastar.scenarios import (
    scenario_grid_4,
    scenario_grid_8,
    scenario_grid_inconsistent,
    scenario_geometric,
    scenario_puzzle,
    scenario_terrain8,
    scenario_maze4,
)


def run_job(job: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
    name, cfg = job
    sc = cfg["scenario_ctor"](**cfg["scenario_kwargs"])
    algo = cfg["algo"]
    weight = cfg["weight"]
    r_mode = cfg["r_mode"]
    if algo == "NR":
        eng = WeightedAStarNR(sc.start, sc.is_goal, sc.neighbors, sc.h, weight=weight)
    elif algo == "AR":
        eng = WeightedAStarAR(sc.start, sc.is_goal, sc.neighbors, sc.h, weight=weight)
    else:
        r = cfg["r"]
        eng = WeightedAStarCR(
            sc.start, sc.is_goal, sc.neighbors, sc.h, weight=weight, r=r, r_mode=r_mode
        )
    path, cost, st = eng.run_to_first_goal()
    return dict(
        scenario=sc.name,
        kind=sc.meta["kind"],
        algo=algo,
        w=weight,
        r=(cfg.get("r")),
        r_mode=r_mode,
        cost=cost,
        expansions=st.expansions,
        reopens=st.reopens,
        generated=st.generated,
        runtime_ms=st.runtime_ms,
        path_len=(len(path) if path else None),
        seed=cfg["scenario_kwargs"].get("seed", None),
    )


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
        (scenario_grid_4, dict(width=50, height=50, density=0.15)),
        (scenario_grid_8, dict(width=60, height=60, density=0.20)),
        (scenario_grid_inconsistent, dict(width=50, height=50, density=0.10)),
        (scenario_terrain8, dict(width=60, height=60)),
        (scenario_maze4, dict(width=51, height=51)),
        (scenario_geometric, dict(n=150, k=8)),
        (scenario_puzzle, dict(steps=25)),
    ]

    jobs: List[Tuple[str, Dict[str, Any]]] = []
    for (ctor, base_kwargs), seed in itertools.product(scenario_ctors, seeds):
        kwargs = dict(base_kwargs)
        kwargs["seed"] = seed
        for algo in ["NR", "AR"]:
            jobs.append(
                (
                    f"{ctor.__name__}-{algo}-seed{seed}",
                    dict(
                        scenario_ctor=ctor,
                        scenario_kwargs=kwargs,
                        algo=algo,
                        weight=args.weight,
                        r_mode=args.r_mode,
                    ),
                )
            )
        for r in r_list:
            jobs.append(
                (
                    f"{ctor.__name__}-CR{r}-seed{seed}",
                    dict(
                        scenario_ctor=ctor,
                        scenario_kwargs=kwargs,
                        algo="CR",
                        weight=args.weight,
                        r=r,
                        r_mode=args.r_mode,
                    ),
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
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(out_path)


if __name__ == "__main__":
    main()
