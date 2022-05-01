
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Dict, Any, Optional, Callable
import random, math
from .wa_star_cr import Grid, HeuristicFn

@dataclass
class TerrainGrid(Grid):
    terrain: Dict[Tuple[int,int], float] = None
    def neighbors4(self, p: Tuple[int,int]) -> Iterable[Tuple[Tuple[int,int], float]]:
        x,y = p
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            q = (x+dx, y+dy)
            if self.in_bounds(q) and self.passable(q):
                cost = self.step * (self.terrain.get(q, 1.0) if self.terrain else 1.0)
                yield q, float(cost)
    def neighbors8(self, p: Tuple[int,int]) -> Iterable[Tuple[Tuple[int,int], float]]:
        x,y = p
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1), (1,1),(1,-1),(-1,1),(-1,-1)]:
            q = (x+dx, y+dy)
            if self.in_bounds(q) and self.passable(q):
                step = (self.step if dx==0 or dy==0 else math.sqrt(2)*self.step)
                cost = step * (self.terrain.get(q, 1.0) if self.terrain else 1.0)
                yield q, float(cost)

def generate_terrain(width: int, height: int, seed: int = 0, kinds: List[float] = [1.0, 1.5, 2.0, 3.0], p: List[float] = None) -> Dict[Tuple[int,int], float]:
    rng = random.Random(seed)
    if p is None:
        p = [0.55, 0.25, 0.15, 0.05]
    terr = {}
    for x in range(width):
        for y in range(height):
            val = rng.choices(kinds, weights=p, k=1)[0]
            terr[(x,y)] = float(val)
    return terr

def admissible_terrain_octile(grid: TerrainGrid, goal: Tuple[int,int]) -> HeuristicFn:
    min_c = 1.0
    if grid.terrain:
        min_c = min(grid.terrain.values())
        if min_c <= 0: min_c = 1e-6
    def h(p: Tuple[int,int]) -> float:
        dx = abs(p[0]-goal[0]); dy = abs(p[1]-goal[1])
        dmin, dmax = (dx if dx<dy else dy), (dx if dx>=dy else dy)
        return ((dmax - dmin) + math.sqrt(2)*dmin) * min_c
    return h

def generate_maze(width: int, height: int, seed: int = 0) -> Grid:
    rng = random.Random(seed)
    grid = Grid(width, height, walls=set())
    walls = {(x,y) for x in range(width) for y in range(height)}
    start = (0,0)
    stack = [start]; visited = {start}; walls.remove(start)
    def neighbors_cells(x,y):
        dirs = [(2,0),(-2,0),(0,2),(0,-2)]
        rng.shuffle(dirs)
        for dx,dy in dirs:
            nx, ny = x+dx, y+dy
            if 0 <= nx < width and 0 <= ny < height:
                yield (nx,ny), (x+dx//2, y+dy//2)
    while stack:
        cx, cy = stack[-1]; found = False
        for (nx,ny), between in neighbors_cells(cx,cy):
            if (nx,ny) not in visited:
                visited.add((nx,ny)); stack.append((nx,ny)); found = True
                walls.discard(between); walls.discard((nx,ny))
                break
        if not found: stack.pop()
    walls.discard((width-1, height-1))
    return Grid(width, height, walls=walls)
