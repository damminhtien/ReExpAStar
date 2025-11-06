from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
import random
from typing import Any, cast

from .core.types import GoalTest, HeuristicFn, NeighborFn
from .terrain_maze import (
    TerrainGrid,
    admissible_terrain_octile,
    generate_maze,
    generate_terrain,
)
from .wa_star_cr import Grid


@dataclass
class Scenario:
    name: str
    start: Any
    goal: Any
    neighbors: NeighborFn[Any]
    h: HeuristicFn[Any]
    is_goal: GoalTest[Any]
    meta: dict[str, Any]


def make_grid_obstacles(width: int, height: int, density: float, seed: int = 0) -> Grid:
    rng = random.Random(seed)
    walls = set()
    for x in range(width):
        for y in range(height):
            if rng.random() < density:
                walls.add((x, y))
    for p in [(0, 0), (width - 1, height - 1)]:
        walls.discard(p)
    return Grid(width, height, walls=walls)


def scenario_grid_4(width: int, height: int, density: float, seed: int = 0) -> Scenario:
    grid = make_grid_obstacles(width, height, density, seed)
    start, goal = (0, 0), (width - 1, height - 1)
    h = grid.manhattan(goal)
    return Scenario(
        name=f"grid4_{width}x{height}_d{density}_s{seed}",
        start=start,
        goal=goal,
        neighbors=cast(NeighborFn[Any], grid.neighbors4),
        h=cast(HeuristicFn[Any], h),
        is_goal=cast(GoalTest[Any], lambda s: s == goal),
        meta={"kind": "grid4", "density": density, "seed": seed},
    )


def scenario_grid_8(width: int, height: int, density: float, seed: int = 0) -> Scenario:
    grid = make_grid_obstacles(width, height, density, seed)
    start, goal = (0, 0), (width - 1, height - 1)
    h = grid.octile(goal)
    return Scenario(
        name=f"grid8_{width}x{height}_d{density}_s{seed}",
        start=start,
        goal=goal,
        neighbors=cast(NeighborFn[Any], grid.neighbors8),
        h=cast(HeuristicFn[Any], h),
        is_goal=cast(GoalTest[Any], lambda s: s == goal),
        meta={"kind": "grid8", "density": density, "seed": seed},
    )


def scenario_grid_inconsistent(width: int, height: int, density: float, seed: int = 0) -> Scenario:
    grid = make_grid_obstacles(width, height, density, seed)
    start, goal = (0, 0), (width - 1, height - 1)
    base_h = grid.manhattan(goal)

    def h(p: tuple[int, int]) -> float:
        bx = base_h(p)
        noise = 0.5 * math.sin((p[0] * 92821 + p[1] * 68917) * 0.01)
        return max(0.0, bx * (1.0 + noise))

    return Scenario(
        name=f"grid4_inconsistent_{width}x{height}_d{density}_s{seed}",
        start=start,
        goal=goal,
        neighbors=cast(NeighborFn[Any], grid.neighbors4),
        h=cast(HeuristicFn[Any], h),
        is_goal=cast(GoalTest[Any], lambda s: s == goal),
        meta={"kind": "grid4_inconsistent", "density": density, "seed": seed},
    )


def scenario_geometric(n: int, k: int, seed: int = 0) -> Scenario:
    rng = random.Random(seed)
    pts = [(rng.random(), rng.random()) for _ in range(n)]

    def dist(i, j):
        (x1, y1), (x2, y2) = pts[i], pts[j]
        return math.hypot(x1 - x2, y1 - y2)

    nbrs: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
    for i in range(n):
        dists = sorted((dist(i, j), j) for j in range(n) if j != i)[:k]
        for d, j in dists:
            nbrs[i].append((j, d))
            nbrs[j].append((i, d))
    start, goal = 0, n - 1

    def neighbors(i: int) -> Iterable[tuple[int, float]]:
        yield from nbrs[i]

    def h(i: int) -> float:
        (x1, y1), (x2, y2) = pts[i], pts[goal]
        return math.hypot(x1 - x2, y1 - y2)

    return Scenario(
        name=f"geom_n{n}_k{k}_s{seed}",
        start=start,
        goal=goal,
        neighbors=cast(NeighborFn[Any], neighbors),
        h=cast(HeuristicFn[Any], h),
        is_goal=cast(GoalTest[Any], lambda s: s == goal),
        meta={"kind": "geom", "n": n, "k": k, "seed": seed},
    )


# 8-puzzle
GOAL_8: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)
MOVES_8 = {
    0: [1, 3],
    1: [0, 2, 4],
    2: [1, 5],
    3: [0, 4, 6],
    4: [1, 3, 5, 7],
    5: [2, 4, 8],
    6: [3, 7],
    7: [4, 6, 8],
    8: [5, 7],
}


def puzzle_neighbors(state: tuple[int, ...]) -> Iterable[tuple[tuple[int, ...], float]]:
    z = state.index(0)
    for nz in MOVES_8[z]:
        lst = list(state)
        lst[z], lst[nz] = lst[nz], lst[z]
        yield tuple(lst), 1.0


def puzzle_h_manhattan(state: tuple[int, ...]) -> float:
    dist = 0
    for idx, val in enumerate(state):
        if val == 0:
            continue
        goal_idx = val - 1
        x, y = idx % 3, idx // 3
        gx, gy = goal_idx % 3, goal_idx // 3
        dist += abs(x - gx) + abs(y - gy)
    return float(dist)


def scramble_puzzle(steps: int, seed: int = 0) -> tuple[int, ...]:
    rng = random.Random(seed)
    s: tuple[int, ...] = GOAL_8
    for _ in range(steps):
        z = s.index(0)
        choices = MOVES_8[z]
        nz = rng.choice(choices)
        lst = list(s)
        lst[z], lst[nz] = lst[nz], lst[z]
        s = tuple(lst)
    return s


def scenario_puzzle(steps: int, seed: int = 0) -> Scenario:
    start = scramble_puzzle(steps, seed)
    goal = GOAL_8
    return Scenario(
        name=f"8p_{steps}_s{seed}",
        start=start,
        goal=goal,
        neighbors=cast(NeighborFn[Any], puzzle_neighbors),
        h=cast(HeuristicFn[Any], puzzle_h_manhattan),
        is_goal=cast(GoalTest[Any], lambda s: s == goal),
        meta={"kind": "8p", "steps": steps, "seed": seed},
    )


# Terrain & maze


def scenario_terrain8(width: int, height: int, seed: int = 0) -> Scenario:
    terr = generate_terrain(width, height, seed=seed)
    tg = TerrainGrid(width, height, walls=set(), terrain=terr)
    start, goal = (0, 0), (width - 1, height - 1)
    h = admissible_terrain_octile(tg, goal)
    return Scenario(
        name=f"terrain8_{width}x{height}_s{seed}",
        start=start,
        goal=goal,
        neighbors=cast(NeighborFn[Any], tg.neighbors8),
        h=cast(HeuristicFn[Any], h),
        is_goal=cast(GoalTest[Any], lambda s: s == goal),
        meta={"kind": "terrain8", "seed": seed},
    )


def scenario_maze4(width: int, height: int, seed: int = 0) -> Scenario:
    g = generate_maze(width, height, seed=seed)
    start, goal = (0, 0), (width - 1, height - 1)
    h = g.manhattan(goal)
    return Scenario(
        name=f"maze4_{width}x{height}_s{seed}",
        start=start,
        goal=goal,
        neighbors=cast(NeighborFn[Any], g.neighbors4),
        h=cast(HeuristicFn[Any], h),
        is_goal=cast(GoalTest[Any], lambda s: s == goal),
        meta={"kind": "maze4", "seed": seed},
    )
