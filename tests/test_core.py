from collections.abc import Iterable
import math
from typing import Any, cast
import unittest

from reexpastar.core.types import GoalTest, HeuristicFn, NeighborFn
from reexpastar.scenarios import scenario_geometric, scenario_grid_inconsistent
from reexpastar.terrain_maze import (
    TerrainGrid,
    admissible_terrain_octile,
    generate_maze,
    generate_terrain,
)
from reexpastar.wa_star_cr import (
    Grid,
    WeightedAStarAR,
    WeightedAStarCR,
    WeightedAStarNR,
    WeightedAStarParams,
)


class TestWeightedAStarCR(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(10, 10, walls={(3, y) for y in range(10) if y != 5})
        self.start = (0, 0)
        self.goal = (9, 9)
        self.h = self.grid.manhattan(self.goal)
        self.h_fn = cast(HeuristicFn[tuple[int, int]], self.h)
        self.is_goal = cast(GoalTest[tuple[int, int]], lambda s: s == self.goal)
        self.neighbors = cast(NeighborFn[tuple[int, int]], self.grid.neighbors)

    def test_extremes_equivalence_consistent_h(self):
        engine_nr = WeightedAStarCR[tuple[int, int]](
            self.start,
            self.is_goal,
            self.neighbors,
            self.h_fn,
            params=WeightedAStarParams(weight=1.0, r=math.inf),
        )
        _, c_nr, _ = engine_nr.run_to_first_goal()
        engine_ar = WeightedAStarCR[tuple[int, int]](
            self.start,
            self.is_goal,
            self.neighbors,
            self.h_fn,
            params=WeightedAStarParams(weight=1.0, r=0.0),
        )
        _, c_ar, _ = engine_ar.run_to_first_goal()
        self.assertAlmostEqual(c_nr, c_ar)
        self.assertEqual(c_nr, 18.0)

    def test_weighted_bound_empirical(self):
        w = 1.5
        base = WeightedAStarCR[tuple[int, int]](
            self.start,
            self.is_goal,
            self.neighbors,
            self.h_fn,
            params=WeightedAStarParams(weight=1.0, r=0.0),
        )
        _, c_opt, _ = base.run_to_first_goal()
        for r in (0.0, math.inf):
            eng = WeightedAStarCR[tuple[int, int]](
                self.start,
                self.is_goal,
                self.neighbors,
                self.h_fn,
                params=WeightedAStarParams(weight=w, r=r),
            )
            _, c, _ = eng.run_to_first_goal()
            self.assertLessEqual(c, w * c_opt + 1e-9)

    def test_no_overwrite_without_improvement(self):
        graph = {"S": [("A", 10.0), ("B", 1.0)], "B": [("A", 11.0)], "A": []}

        def neigh(u: str) -> Iterable[tuple[str, float]]:
            yield from ((v, c) for v, c in graph.get(u, []))

        def h(_unused: str) -> float:
            return 0.0

        def is_goal(u: str) -> bool:
            return u == "A"

        neighbor_fn = cast(NeighborFn[str], neigh)
        heuristic_fn = cast(HeuristicFn[str], h)
        goal_test = cast(GoalTest[str], is_goal)

        eng = WeightedAStarCR[str](
            "S",
            goal_test,
            neighbor_fn,
            heuristic_fn,
            params=WeightedAStarParams(weight=1.0, r=0.0),
        )
        _, c, _ = eng.run_to_first_goal()
        self.assertEqual(c, 10.0)
        self.assertEqual(eng.parent["A"], "S")


class TestDiverseScenarios(unittest.TestCase):
    def test_diagonal_grid_octile(self):
        g = Grid(20, 20, walls=set())
        start, goal = (0, 0), (19, 19)
        h = g.octile(goal)

        def is_goal(s: tuple[int, int]) -> bool:
            return s == goal

        goal_test = cast(GoalTest[tuple[int, int]], is_goal)
        neighbors = cast(NeighborFn[tuple[int, int]], g.neighbors8)
        heuristic = cast(HeuristicFn[tuple[int, int]], h)

        eng = WeightedAStarAR[tuple[int, int]](start, goal_test, neighbors, heuristic, weight=1.0)
        _, c, _ = eng.run_to_first_goal()
        self.assertAlmostEqual(c, math.sqrt(2) * 19, places=6)

    def test_inconsistent_heuristic_triggers_reopens(self):
        sc = scenario_grid_inconsistent(40, 40, density=0.08, seed=123)
        goal_test = cast(GoalTest[Any], sc.is_goal)
        neighbors = cast(NeighborFn[Any], sc.neighbors)
        heuristic = cast(HeuristicFn[Any], sc.h)
        nr = WeightedAStarNR[Any](sc.start, goal_test, neighbors, heuristic, weight=1.5)
        _, _, st_nr = nr.run_to_first_goal()
        ar = WeightedAStarAR[Any](sc.start, goal_test, neighbors, heuristic, weight=1.5)
        _, _, st_ar = ar.run_to_first_goal()
        self.assertEqual(st_nr.reopens, 0)
        self.assertGreaterEqual(st_ar.reopens, 1)

    def test_geometric_graph_solution(self):
        sc = scenario_geometric(n=80, k=6, seed=2)
        goal_test = cast(GoalTest[Any], sc.is_goal)
        neighbors = cast(NeighborFn[Any], sc.neighbors)
        heuristic = cast(HeuristicFn[Any], sc.h)
        eng = WeightedAStarCR[Any](
            sc.start,
            goal_test,
            neighbors,
            heuristic,
            params=WeightedAStarParams(weight=1.2, r=0.2),
        )
        p, c, _ = eng.run_to_first_goal()
        self.assertIsNotNone(p)
        self.assertIsNotNone(c)
        self.assertGreater(len(p), 1)


class TestTerrainAndMaze(unittest.TestCase):
    def test_terrain_admissible_heuristic(self):
        width, height = 30, 30
        terr = generate_terrain(width, height, seed=7)
        tg = TerrainGrid(width, height, walls=set(), terrain=terr)
        start, goal = (0, 0), (width - 1, height - 1)
        h = admissible_terrain_octile(tg, goal)
        goal_test = cast(GoalTest[tuple[int, int]], lambda s: s == goal)
        neighbors = cast(NeighborFn[tuple[int, int]], tg.neighbors8)
        heuristic = cast(HeuristicFn[tuple[int, int]], h)
        eng = WeightedAStarCR[tuple[int, int]](
            start,
            goal_test,
            neighbors,
            heuristic,
            params=WeightedAStarParams(weight=1.0, r=0.0),
        )
        _, c, _ = eng.run_to_first_goal()
        self.assertIsNotNone(c)
        self.assertGreaterEqual(c, h(start) - 1e-9)

    def test_maze_reachability(self):
        g = generate_maze(31, 31, seed=5)
        start, goal = (0, 0), (30, 30)
        goal_test = cast(GoalTest[tuple[int, int]], lambda s: s == goal)
        neighbors = cast(NeighborFn[tuple[int, int]], g.neighbors4)
        heuristic = cast(HeuristicFn[tuple[int, int]], g.manhattan(goal))
        eng = WeightedAStarCR[tuple[int, int]](
            start,
            goal_test,
            neighbors,
            heuristic,
            params=WeightedAStarParams(weight=1.0, r=0.0),
        )
        p, c, _ = eng.run_to_first_goal()
        self.assertIsNotNone(c)
        self.assertGreater(len(p), 1)


if __name__ == "__main__":
    unittest.main()
