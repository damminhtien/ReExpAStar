
from __future__ import annotations
from typing import Protocol, Iterable, Tuple, TypeVar, Hashable, Callable

S = TypeVar("S", bound=Hashable)

class NeighborFn(Protocol[S]):
    def __call__(self, state: S) -> Iterable[Tuple[S, float]]: ...

class HeuristicFn(Protocol[S]):
    def __call__(self, state: S) -> float: ...

class GoalTest(Protocol[S]):
    def __call__(self, state: S) -> bool: ...
