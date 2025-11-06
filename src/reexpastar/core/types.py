from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Protocol, TypeVar

S = TypeVar("S", bound=Hashable)
_StateContra_contra = TypeVar("_StateContra_contra", bound=Hashable, contravariant=True)


class NeighborFn(Protocol[S]):
    def __call__(self, state: S) -> Iterable[tuple[S, float]]: ...


class HeuristicFn(Protocol[_StateContra_contra]):
    def __call__(self, state: _StateContra_contra) -> float: ...


class GoalTest(Protocol[_StateContra_contra]):
    def __call__(self, state: _StateContra_contra) -> bool: ...
