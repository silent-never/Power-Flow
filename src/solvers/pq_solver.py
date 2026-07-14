"""Fast-decoupled (PQ) solver scaffold."""

from .base_solver import BaseSolver


class FastDecoupledSolver(BaseSolver):
    """Fast-decoupled solver using the common solver interface."""

    def solve(self, grid):
        raise NotImplementedError("FastDecoupledSolver is not implemented yet")
