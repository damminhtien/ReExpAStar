from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Generic
import heapq
import math
import time

from .core.types import S as State, NeighborFn, HeuristicFn, GoalTest


@dataclass(order=True)
class _OpenItem:
    key: Tuple[float, float]
    count: int
    state: State = field(compare=False)


class OpenList:
    """Binary heap with lazy decrease-key policy."""

    def __init__(self) -> None:
        self._heap: List[_OpenItem] = []
        self._best_key: Dict[State, Tuple[float, float]] = {}
        self._counter = 0

    def push(self, state: State, f: float, g: float) -> None:
        key = (f, g)
        best = self._best_key.get(state)
        if best is None or key < best:
            self._best_key[state] = key
            heapq.heappush(self._heap, _OpenItem(key, self._counter, state))
            self._counter += 1

    def pop(self) -> Optional[Tuple[State, float, float]]:
        while self._heap:
            item = heapq.heappop(self._heap)
            state = item.state
            key = self._best_key.get(state)
            if key is None:
                continue
            if key == item.key:
                f, g = key
                del self._best_key[state]
                return state, f, g
        return None

    def empty(self) -> bool:
        return not self._best_key

    def __contains__(self, s: State) -> bool:
        return s in self._best_key

    def get_key(self, s: State) -> Optional[Tuple[float, float]]:
        return self._best_key.get(s)


@dataclass
class SearchStats:
    expansions: int = 0
    reopens: int = 0
    generated: int = 0
    runtime_ms: float = 0.0


@dataclass
class ProposedUpdate:
    delta: float
    state: State
    new_g: float
    parent: Optional[State]


class WeightedAStarCR(Generic[State]):
    """Weighted A* with Conditional Re-expansion threshold r.

    r_mode:
      - 'abs': reopen if Δg > r
      - 'rel_edge': reopen if Δg > r * (edge cost)
      - 'rel_g': reopen if Δg > r * max(1, old_g)
    """

    def __init__(self, start: State, is_goal: GoalTest[State], neighbors: NeighborFn[State], h: HeuristicFn[State],
                 weight: float = 1.0, r: float = 0.0, r_mode: str = 'abs',
                 tie_break: str = 'g_low', max_expansions: Optional[int] = None, max_runtime_ms: Optional[float] = None,
                 log_every: Optional[int] = None, logger: Optional[Any] = None) -> None:
        assert weight >= 1.0, "weight must be ≥ 1"
        self.start = start
        self.is_goal = is_goal
        self.neighbors = neighbors
        self.h = h
        self.w = float(weight)
        self.r = float(r)
        self.r_mode = r_mode
        self.tie_break = tie_break
        self.max_expansions = max_expansions
        self.max_runtime_ms = max_runtime_ms
        self.log_every = log_every
        from .logging import get_logger as _get_logger
        self.logger = logger or _get_logger(__name__)
        self.g: Dict[State, float] = {start: 0.0}
        self.parent: Dict[State, Optional[State]] = {start: None}
        self.h_cache: Dict[State, float] = {start: h(start)}
        self.status: Dict[State, str] = {}
        self.open = OpenList()
        f0 = self._f(start)
        g0 = 0.0
        kb = g0 if self.tie_break == 'g_low' else -g0
        self.open.push(start, f0, kb)
        self.status[start] = 'OPEN'
        self.repairs: List[ProposedUpdate] = []
        self.stats = SearchStats()
        self.best_goal: Optional[State] = None

    def _h(self, s: State) -> float:
        if s not in self.h_cache:
            self.h_cache[s] = float(self.h(s))
        return self.h_cache[s]

    def _f(self, s: State) -> float:
        return self.g[s] + self.w * self._h(s)

    def _reconstruct(self, goal: State) -> List[State]:
        path = []
        cur: Optional[State] = goal
        while cur is not None:
            path.append(cur)
            cur = self.parent[cur]
        path.reverse()
        return path

    def _reopen_allowed(self, delta: float, edge_cost: float, old_g: float) -> bool:
        if self.r_mode == 'abs':
            return delta > self.r
        if self.r_mode == 'rel_edge':
            return delta > self.r * edge_cost
        if self.r_mode == 'rel_g':
            return delta > self.r * max(1.0, old_g if math.isfinite(old_g) else 1.0)
        return delta > self.r

    def search_until_goal(self) -> Tuple[Optional[List[State]], Optional[float]]:
        t0 = time.perf_counter()
        while True:
            # guards
            if self.max_runtime_ms is not None:
                if (time.perf_counter() - t0) * 1000.0 > self.max_runtime_ms:
                    self.logger.info('max_runtime_ms reached; stopping search')
                    self.stats.runtime_ms += (time.perf_counter() -
                                              t0) * 1000.0
                    return None, None
            popped = self.open.pop()
            if popped is None:
                self.stats.runtime_ms += (time.perf_counter() - t0) * 1000.0
                return None, None
            s, _, _ = popped
            if s not in self.g:
                continue
            self.status[s] = 'CLOSED'
            self.stats.expansions += 1
            if self.log_every and (self.stats.expansions % self.log_every == 0):
                self.logger.info(
                    'expansions=%(exp)d, generated=%(gen)d, reopens=%(reop)d',
                    {'exp': self.stats.expansions, 'gen': self.stats.generated, 'reop': self.stats.reopens})
            if self.max_expansions is not None and self.stats.expansions >= self.max_expansions:
                self.logger.info('max_expansions reached; stopping search')
                self.stats.runtime_ms += (time.perf_counter() - t0) * 1000.0
                return None, None
            if self.is_goal(s):
                self.best_goal = s
                self.stats.runtime_ms += (time.perf_counter() - t0) * 1000.0
                return self._reconstruct(s), self.g[s]
            g_s = self.g[s]
            for t, cost in self.neighbors(s):
                self.stats.generated += 1
                new_g = g_s + float(cost)
                old_g = self.g.get(t, math.inf)
                if new_g >= old_g:
                    continue
                delta = old_g - new_g
                if self.status.get(t) == 'OPEN':
                    self.g[t] = new_g
                    self.parent[t] = s
                    kb = new_g if self.tie_break == 'g_low' else -new_g
                    self.open.push(t, new_g + self.w * self._h(t), kb)
                elif self.status.get(t) == 'CLOSED':
                    if self._reopen_allowed(delta, float(cost), old_g):
                        self.g[t] = new_g
                        self.parent[t] = s
                        kb = new_g if self.tie_break == 'g_low' else -new_g
                        self.open.push(t, new_g + self.w * self._h(t), kb)
                        self.status[t] = 'OPEN'
                        self.stats.reopens += 1
                    else:
                        self.repairs.append(ProposedUpdate(
                            delta=delta, state=t, new_g=new_g, parent=s))
                else:
                    self.g[t] = new_g
                    self.parent[t] = s
                    self.status[t] = 'OPEN'
                    kb = new_g if self.tie_break == 'g_low' else -new_g
                    self.open.push(t, new_g + self.w * self._h(t), kb)

    def lower_r_and_repair(self, new_r: float) -> int:
        assert new_r <= self.r
        self.r = float(new_r)
        if not self.repairs:
            return 0
        self.repairs.sort(key=lambda u: u.delta, reverse=True)
        accepted = 0
        remaining: List[ProposedUpdate] = []
        for upd in self.repairs:
            cur_g = self.g.get(upd.state, math.inf)
            delta = cur_g - upd.new_g
            if upd.new_g < cur_g and self._reopen_allowed(delta, 1.0, cur_g):
                self.g[upd.state] = upd.new_g
                self.parent[upd.state] = upd.parent
                kb = self.g[upd.state] if self.tie_break == 'g_low' else - \
                    self.g[upd.state]
                self.open.push(upd.state, self._f(upd.state), kb)
                self.status[upd.state] = 'OPEN'
                accepted += 1
            else:
                new_delta = max(0.0, delta)
                if new_delta > 0.0:
                    remaining.append(ProposedUpdate(
                        delta=new_delta, state=upd.state, new_g=upd.new_g, parent=upd.parent))
        self.repairs = remaining
        return accepted

    def run_to_first_goal(self) -> Tuple[Optional[List[State]], Optional[float], SearchStats]:
        path, cost = self.search_until_goal()
        return path, cost, self.stats

    def run_ira_schedule(self, r_values: List[float]) -> Tuple[List[Tuple[float, float]], Optional[List[State]]]:
        history: List[Tuple[float, float]] = []
        best_cost = math.inf
        best_path: Optional[List[State]] = None
        p, c = self.search_until_goal()
        if p is not None and c is not None:
            best_cost = c
            best_path = p
        history.append((self.r, best_cost))
        for new_r in r_values[1:]:
            self.lower_r_and_repair(new_r)
            p, c = self.search_until_goal()
            if p is not None and c is not None and c < best_cost:
                best_cost = c
                best_path = p
            history.append((self.r, best_cost))
        return history, best_path


@dataclass
class Grid:
    width: int
    height: int
    walls: set[Tuple[int, int]] = field(default_factory=set)
    step: float = 1.0

    def in_bounds(self, p: Tuple[int, int]) -> bool:
        x, y = p
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, p: Tuple[int, int]) -> bool:
        return p not in self.walls

    def neighbors4(self, p: Tuple[int, int]) -> Iterable[Tuple[Tuple[int, int], float]]:
        x, y = p
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            q = (x+dx, y+dy)
            if self.in_bounds(q) and self.passable(q):
                yield q, self.step

    def neighbors8(self, p: Tuple[int, int]) -> Iterable[Tuple[Tuple[int, int], float]]:
        x, y = p
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            q = (x+dx, y+dy)
            if self.in_bounds(q) and self.passable(q):
                yield q, (self.step if dx == 0 or dy == 0 else math.sqrt(2)*self.step)

    def neighbors(self, p: Tuple[int, int]) -> Iterable[Tuple[Tuple[int, int], float]]:
        return self.neighbors4(p)

    def manhattan(self, goal: Tuple[int, int]) -> HeuristicFn:
        def h(p: Tuple[int, int]) -> float: 
            return abs(p[0] - goal[0]) + abs(p[1]-goal[1])
        return h

    def octile(self, goal: Tuple[int, int]) -> HeuristicFn:
        def h(p: Tuple[int, int]) -> float:
            dx = abs(p[0]-goal[0])
            dy = abs(p[1]-goal[1])
            dmin, dmax = (dx if dx < dy else dy), (dx if dx >= dy else dy)
            return (dmax - dmin) + math.sqrt(2)*dmin
        return h


class WeightedAStarNR(WeightedAStarCR[State]):
    def __init__(self, start: State, is_goal: GoalTest, neighbors: NeighborFn, h: HeuristicFn, weight: float = 1.0):
        super().__init__(start, is_goal, neighbors, h, weight=weight, r=math.inf)


class WeightedAStarAR(WeightedAStarCR[State]):
    def __init__(self, start: State, is_goal: GoalTest, neighbors: NeighborFn, h: HeuristicFn, weight: float = 1.0):
        super().__init__(start, is_goal, neighbors, h, weight=weight, r=0.0)


def compare_on_grid(width: int = 50, height: int = 50, weight: float = 1.5, r_cr: float = 0.2):
    grid = Grid(width, height, walls={(width//2, y)
                for y in range(height)} - {(width//2, height//3)})
    start, goal = (0, 0), (width-1, height-1)
    h = grid.manhattan(goal)
    def is_goal(s): return s == goal

    def run_engine(name: str, eng: WeightedAStarCR):
        path, cost, stats = eng.run_to_first_goal()
        return dict(algo=name, w=eng.w, r=getattr(eng, 'r', None), cost=cost,
                    expansions=stats.expansions, reopens=stats.reopens, generated=stats.generated,
                    runtime_ms=stats.runtime_ms, path_len=(len(path) if path else None))
    return [
        run_engine('NR', WeightedAStarNR(
            start, is_goal, grid.neighbors, h, weight=weight)),
        run_engine('AR', WeightedAStarAR(
            start, is_goal, grid.neighbors, h, weight=weight)),
        run_engine(f'CR(r={r_cr})', WeightedAStarCR(
            start, is_goal, grid.neighbors, h, weight=weight, r=r_cr)),
    ]
