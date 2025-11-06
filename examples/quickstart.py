from typing import cast

from reexpastar.core.types import GoalTest, NeighborFn
from reexpastar.wa_star_cr import (
    Grid,
    WeightedAStarAR,
    WeightedAStarCR,
    WeightedAStarNR,
    WeightedAStarParams,
)

if __name__ == "__main__":
    g = Grid(30, 30, walls={(15, y) for y in range(30)} - {(15, 10)})
    start, goal = (0, 0), (29, 29)
    h = g.manhattan(goal)

    def is_goal(s: tuple[int, int]) -> bool:
        return s == goal

    goal_test = cast(GoalTest[tuple[int, int]], is_goal)
    neighbors_fn = cast(NeighborFn[tuple[int, int]], g.neighbors)

    for name, eng in [
        (
            "NR",
            WeightedAStarNR[tuple[int, int]](start, goal_test, neighbors_fn, h, weight=1.5),
        ),
        (
            "AR",
            WeightedAStarAR[tuple[int, int]](start, goal_test, neighbors_fn, h, weight=1.5),
        ),
        (
            "CR(r=0.3)",
            WeightedAStarCR[tuple[int, int]](
                start,
                goal_test,
                neighbors_fn,
                h,
                params=WeightedAStarParams(weight=1.5, r=0.3),
            ),
        ),
    ]:
        path, cost, st = eng.run_to_first_goal()
        print(f"{name}: cost={cost}, expansions={st.expansions}, reopens={st.reopens}")
