from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from typing import Generic

from .core.types import GoalTest, HeuristicFn, NeighborFn, S as State


@dataclass(order=True)
class _Item(Generic[State]):
    key: tuple[float, float]
    count: int
    state: State = field(compare=False)


class _Open(Generic[State]):
    def __init__(self) -> None:
        self._heap: list[_Item[State]] = []
        self._key: dict[State, tuple[float, float]] = {}
        self._ctr = 0

    def push(self, s: State, f: float, g: float) -> None:
        k = (f, g)
        b = self._key.get(s)
        if b is None or k < b:
            self._key[s] = k
            heapq.heappush(self._heap, _Item(k, self._ctr, s))
            self._ctr += 1

    def pop(self) -> tuple[State, float, float] | None:
        while self._heap:
            it = heapq.heappop(self._heap)
            s = it.state
            k = self._key.get(s)
            if k is None:
                continue
            if k == it.key:
                del self._key[s]
                return s, k[0], k[1]
        return None

    def empty(self) -> bool:
        return not self._key

    def __contains__(self, s: State) -> bool:
        return s in self._key

    def clear(self) -> None:
        self._heap.clear()
        self._key.clear()
        self._ctr = 0

    def snapshot(self) -> list[tuple[State, tuple[float, float]]]:
        return list(self._key.items())


@dataclass
class Stats:
    expansions: int = 0
    generated: int = 0


class ARAStar(Generic[State]):  # pylint: disable=too-many-instance-attributes
    """Anytime Repairing A* baseline with OPEN/CLOSED/INCONS reuse."""

    def __init__(
        self,
        start: State,
        is_goal: GoalTest[State],
        neighbors: NeighborFn[State],
        h: HeuristicFn[State],
        *,
        w0: float = 2.0,
    ) -> None:
        assert w0 >= 1.0
        self.s_start = start
        self.is_goal = is_goal
        self.neighbors = neighbors
        self.h = h
        self.open_list: _Open[State] = _Open()
        self.closed: set[State] = set()
        self.inconsistent: set[State] = set()
        self.g: dict[State, float] = {start: 0.0}
        self.parent: dict[State, State | None] = {start: None}
        self.h_cache: dict[State, float] = {start: float(h(start))}
        self.w = float(w0)
        self.open_list.push(start, self._f(start), 0.0)
        self.stats = Stats()
        self.goal_found: State | None = None

    def _h(self, s: State) -> float:
        if s not in self.h_cache:
            self.h_cache[s] = float(self.h(s))
        return self.h_cache[s]

    def _f(self, s: State) -> float:
        return self.g[s] + self.w * self._h(s)

    def _reconstruct(self, goal: State) -> list[State]:
        path = []
        cur: State | None = goal
        while cur is not None:
            path.append(cur)
            cur = self.parent[cur]
        path.reverse()
        return path

    def improve_path(self) -> tuple[list[State] | None, float | None]:
        while not self.open_list.empty():
            popped = self.open_list.pop()
            if popped is None:
                break
            s, _, _ = popped
            if self.is_goal(s):
                self.goal_found = s
                return self._reconstruct(s), self.g[s]
            self.closed.add(s)
            self.stats.expansions += 1
            g_s = self.g[s]
            for t, c in self.neighbors(s):
                self.stats.generated += 1
                new_g = g_s + float(c)
                if new_g < self.g.get(t, math.inf):
                    self.g[t] = new_g
                    self.parent[t] = s
                    if t not in self.closed:
                        self.open_list.push(t, self._f(t), self.g[t])
                    else:
                        self.inconsistent.add(t)
        return None, None

    def update_keys(self) -> None:
        items = self.open_list.snapshot()
        self.open_list.clear()
        for s, (_, g_val) in items:
            self.open_list.push(s, self.g[s] + self.w * self._h(s), g_val)

    def repair_and_continue(self, new_w: float) -> None:
        assert 1.0 <= new_w <= self.w
        self.w = float(new_w)
        for s in self.inconsistent:
            self.open_list.push(s, self.g[s] + self.w * self._h(s), self.g[s])
        self.inconsistent.clear()
        self.closed.clear()
        self.update_keys()

    def run_schedule(
        self, w_values: list[float]
    ) -> tuple[list[tuple[float, float]], list[State] | None]:
        hist: list[tuple[float, float]] = []
        best_cost = math.inf
        best_path: list[State] | None = None
        p, c = self.improve_path()
        if p is not None and c is not None:
            best_cost = c
            best_path = p
        hist.append((self.w, best_cost))
        for nw in w_values[1:]:
            self.repair_and_continue(nw)
            p, c = self.improve_path()
            if p is not None and c is not None and c < best_cost:
                best_cost = c
                best_path = p
            hist.append((self.w, best_cost))
        return hist, best_path
