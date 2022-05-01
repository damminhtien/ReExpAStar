
import argparse, cProfile, pstats, io
from reexpastar.scenarios import scenario_grid_inconsistent
from reexpastar.wa_star_cr import WeightedAStarAR

def main():
    p = argparse.ArgumentParser(description="cProfile for a heavy scenario")
    p.add_argument('--weight', type=float, default=1.5)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    sc = scenario_grid_inconsistent(80, 80, density=0.12, seed=args.seed)
    eng = WeightedAStarAR(sc.start, sc.is_goal, sc.neighbors, sc.h, weight=args.weight)
    pr = cProfile.Profile()
    pr.enable()
    _, _, _ = eng.run_to_first_goal()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    print(s.getvalue())

if __name__ == "__main__":
    main()
