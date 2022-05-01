
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple, TypeVar, Generic
import heapq, math

from .core.types import S as State, NeighborFn, HeuristicFn, GoalTest

@dataclass(order=True)
class _Item:
    key: Tuple[float, float]
    count: int
    state: State=field(compare=False)

class _Open:
    def __init__(self) -> None:
        self._heap: List[_Item] = []
        self._key: Dict[State, Tuple[float, float]] = {}
        self._ctr = 0
    def push(self, s: State, f: float, g: float):
        k = (f, g); b = self._key.get(s)
        if b is None or k < b:
            self._key[s] = k
            heapq.heappush(self._heap, _Item(k, self._ctr, s)); self._ctr += 1
    def pop(self):
        while self._heap:
            it = heapq.heappop(self._heap); s = it.state
            k = self._key.get(s)
            if k is None: continue
            if k == it.key:
                del self._key[s]; return s, k[0], k[1]
        return None
    def empty(self): return not self._key
    def __contains__(self, s): return s in self._key

@dataclass
class Stats:
    expansions: int = 0
    generated: int = 0

class ARAStar:
    """Anytime Repairing A* baseline with OPEN/CLOSED/INCONS reuse."""
    def __init__(self, start: State, is_goal: GoalTest, neighbors: NeighborFn, h: HeuristicFn, w0: float = 2.0) -> None:
        assert w0 >= 1.0
        self.s_start = start; self.is_goal = is_goal; self.neighbors = neighbors; self.h = h
        self.OPEN = _Open(); self.CLOSED: set[State] = set(); self.INCONS: set[State] = set()
        self.g: Dict[State, float] = {start: 0.0}; self.parent: Dict[State, Optional[State]] = {start: None}
        self.h_cache: Dict[State, float] = {start: h(start)}
        self.w = float(w0)
        self.OPEN.push(start, self._f(start), 0.0)
        self.stats = Stats()
        self.goal_found: Optional[State] = None
    def _h(self, s: State) -> float:
        if s not in self.h_cache: self.h_cache[s] = float(self.h(s))
        return self.h_cache[s]
    def _f(self, s: State) -> float:
        return self.g[s] + self.w * self._h(s)
    def _reconstruct(self, goal: State) -> List[State]:
        path = []; cur: Optional[State] = goal
        while cur is not None: path.append(cur); cur = self.parent[cur]
        path.reverse(); return path
    def improve_path(self) -> Tuple[Optional[List[State]], Optional[float]]:
        while not self.OPEN.empty():
            popped = self.OPEN.pop()
            if popped is None: break
            s, _, _ = popped
            if self.is_goal(s):
                self.goal_found = s
                return self._reconstruct(s), self.g[s]
            self.CLOSED.add(s)
            self.stats.expansions += 1
            g_s = self.g[s]
            for t, c in self.neighbors(s):
                self.stats.generated += 1
                new_g = g_s + float(c)
                if new_g < self.g.get(t, math.inf):
                    self.g[t] = new_g; self.parent[t] = s
                    if t not in self.CLOSED:
                        self.OPEN.push(t, self._f(t), self.g[t])
                    else:
                        self.INCONS.add(t)
        return None, None
    def update_keys(self):
        items = list(self.OPEN._key.items())
        self.OPEN._heap.clear(); self.OPEN._key.clear(); self.OPEN._ctr = 0
        for s, (f_old, g) in items:
            self.OPEN.push(s, self.g[s] + self.w * self._h(s), g)
    def repair_and_continue(self, new_w: float):
        assert 1.0 <= new_w <= self.w
        self.w = float(new_w)
        for s in self.INCONS:
            self.OPEN.push(s, self.g[s] + self.w * self._h(s), self.g[s])
        self.INCONS.clear()
        self.CLOSED.clear()
        self.update_keys()
    def run_schedule(self, w_values: List[float]) -> Tuple[List[Tuple[float, float]], Optional[List[State]]]:
        hist: List[Tuple[float, float]] = []
        best_cost = math.inf; best_path = None
        p, c = self.improve_path()
        if p is not None and c is not None:
            best_cost = c; best_path = p
        hist.append((self.w, best_cost))
        for nw in w_values[1:]:
            self.repair_and_continue(nw)
            p, c = self.improve_path()
            if p is not None and c is not None and c < best_cost:
                best_cost = c; best_path = p
            hist.append((self.w, best_cost))
        return hist, best_path
