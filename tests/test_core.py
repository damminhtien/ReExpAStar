import math
import unittest
from reexpastar.wa_star_cr import WeightedAStarCR, WeightedAStarNR, WeightedAStarAR, Grid


class TestWeightedAStarCR(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(10, 10, walls={(3, y) for y in range(10) if y != 5})
        self.start = (0, 0)
        self.goal = (9, 9)
        self.h = self.grid.manhattan(self.goal)
        self.is_goal = lambda s: s == self.goal

    def test_extremes_equivalence_consistent_h(self):
        engine_nr = WeightedAStarCR(
            self.start, self.is_goal, self.grid.neighbors, self.h, weight=1.0, r=math.inf)
        _, c_nr, _ = engine_nr.run_to_first_goal()
        engine_ar = WeightedAStarCR(
            self.start, self.is_goal, self.grid.neighbors, self.h, weight=1.0, r=0.0)
        _, c_ar, _ = engine_ar.run_to_first_goal()
        self.assertAlmostEqual(c_nr, c_ar)
        self.assertEqual(c_nr, 18.0)

    def test_weighted_bound_empirical(self):
        w = 1.5
        base = WeightedAStarCR(self.start, self.is_goal,
                               self.grid.neighbors, self.h, weight=1.0, r=0.0)
        _, c_opt, _ = base.run_to_first_goal()
        for r in (0.0, math.inf):
            eng = WeightedAStarCR(self.start, self.is_goal,
                                  self.grid.neighbors, self.h, weight=w, r=r)
            _, c, _ = eng.run_to_first_goal()
            self.assertLessEqual(c, w * c_opt + 1e-9)

    def test_no_overwrite_without_improvement(self):
        graph = {'S': [('A', 10.0), ('B', 1.0)], 'B': [('A', 11.0)], 'A': []}

        def neigh(u: str):
            for v, c in graph.get(u, []):
                yield v, c

        def h(u: str) -> float: return 0.0
        def is_goal(u): return u == 'A'
        eng = WeightedAStarCR('S', is_goal, neigh, h, weight=1.0, r=0.0)
        _, c, _ = eng.run_to_first_goal()
        self.assertEqual(c, 10.0)
        self.assertEqual(eng.parent['A'], 'S')


class TestDiverseScenarios(unittest.TestCase):
    def test_diagonal_grid_octile(self):
        g = Grid(20, 20, walls=set())
        start, goal = (0, 0), (19, 19)
        h = g.octile(goal)
        def is_goal(s): return s == goal
        eng = WeightedAStarAR(start, is_goal, g.neighbors8, h, weight=1.0)
        _, c, _ = eng.run_to_first_goal()
        self.assertAlmostEqual(c, math.sqrt(2)*19, places=6)

    def test_inconsistent_heuristic_triggers_reopens(self):
        from reexpastar.scenarios import scenario_grid_inconsistent
        sc = scenario_grid_inconsistent(40, 40, density=0.08, seed=123)
        nr = WeightedAStarNR(sc.start, sc.is_goal,
                             sc.neighbors, sc.h, weight=1.5)
        _, _, st_nr = nr.run_to_first_goal()
        ar = WeightedAStarAR(sc.start, sc.is_goal,
                             sc.neighbors, sc.h, weight=1.5)
        _, _, st_ar = ar.run_to_first_goal()
        self.assertEqual(st_nr.reopens, 0)
        self.assertGreaterEqual(st_ar.reopens, 1)

    def test_geometric_graph_solution(self):
        from reexpastar.scenarios import scenario_geometric
        sc = scenario_geometric(n=80, k=6, seed=2)
        eng = WeightedAStarCR(sc.start, sc.is_goal,
                              sc.neighbors, sc.h, weight=1.2, r=0.2)
        p, c, _ = eng.run_to_first_goal()
        self.assertIsNotNone(p)
        self.assertIsNotNone(c)
        self.assertGreater(len(p), 1)


class TestTerrainAndMaze(unittest.TestCase):
    def test_terrain_admissible_heuristic(self):
        from reexpastar.terrain_maze import TerrainGrid, generate_terrain, admissible_terrain_octile
        width, height = 30, 30
        terr = generate_terrain(width, height, seed=7)
        tg = TerrainGrid(width, height, walls=set(), terrain=terr)
        start, goal = (0, 0), (width-1, height-1)
        h = admissible_terrain_octile(tg, goal)
        eng = WeightedAStarCR(start, lambda s: s == goal,
                              tg.neighbors8, h, weight=1.0, r=0.0)
        _, c, _ = eng.run_to_first_goal()
        self.assertIsNotNone(c)
        self.assertGreaterEqual(c, h(start) - 1e-9)

    def test_maze_reachability(self):
        from reexpastar.terrain_maze import generate_maze
        g = generate_maze(31, 31, seed=5)
        start, goal = (0, 0), (30, 30)
        eng = WeightedAStarCR(start, lambda s: s == goal,
                              g.neighbors4, g.manhattan(goal), weight=1.0, r=0.0)
        p, c, _ = eng.run_to_first_goal()
        self.assertIsNotNone(c)
        self.assertGreater(len(p), 1)


if __name__ == "__main__":
    unittest.main()
