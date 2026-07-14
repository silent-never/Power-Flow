"""Tensor method solver scaffold."""

from .base_solver import BaseSolver


class TensorSolver(BaseSolver):
    """Tensor-method solver using the common solver interface."""

    def solve(self, grid):
        raise NotImplementedError("TensorSolver is not implemented yet")
