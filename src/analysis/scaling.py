"""潮流算法的系统规模指标与幂律增长趋势分析。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.grid import PowerGrid
from ..solvers.pq_solver import FastDecoupledSolver


@dataclass(frozen=True, slots=True)
class GridScaleMetrics:
    """一个电网算例的拓扑、状态变量和矩阵规模。"""

    node_count: int
    pq_count: int
    pv_count: int
    branch_count: int
    state_dimension: int
    jacobian_nnz: int
    jacobian_density: float
    b_prime_dimension: int
    b_prime_nnz: int
    b_double_prime_dimension: int
    b_double_prime_nnz: int
    dense_matrix_memory_bytes: int


@dataclass(frozen=True, slots=True)
class PowerLawFit:
    """幂律关系 y=C·x^p 的对数最小二乘拟合结果。"""

    exponent: float
    coefficient: float
    r_squared: float
    sample_count: int

    def predict(self, values: np.ndarray) -> np.ndarray:
        """按拟合结果预测给定规模对应的数值。"""
        return self.coefficient * np.asarray(values, dtype=float) ** self.exponent


def measure_grid_scale(grid: PowerGrid) -> GridScaleMetrics:
    """测量电网拓扑与当前潮流线性方程组的实际规模。"""
    working_grid = grid.clone()
    working_grid.get_mismatch()
    jacobian, _, _ = working_grid.get_jacobian()
    b_prime, _ = FastDecoupledSolver._build_b_prime(working_grid)
    b_double_prime, _ = FastDecoupledSolver._build_b_double_prime(working_grid)
    threshold = 1e-14
    jacobian_nnz = int(np.count_nonzero(np.abs(jacobian) > threshold))
    total_entries = int(jacobian.size)
    return GridScaleMetrics(
        node_count=working_grid.n,
        pq_count=int(np.count_nonzero(working_grid.bus_type == 1)),
        pv_count=int(np.count_nonzero(working_grid.bus_type == 2)),
        branch_count=len(working_grid.branches),
        state_dimension=int(jacobian.shape[0]),
        jacobian_nnz=jacobian_nnz,
        jacobian_density=(
            jacobian_nnz / total_entries if total_entries else 0.0
        ),
        b_prime_dimension=int(b_prime.shape[0]),
        b_prime_nnz=int(np.count_nonzero(np.abs(b_prime) > threshold)),
        b_double_prime_dimension=int(b_double_prime.shape[0]),
        b_double_prime_nnz=int(
            np.count_nonzero(np.abs(b_double_prime) > threshold)
        ),
        dense_matrix_memory_bytes=int(
            jacobian.nbytes + b_prime.nbytes + b_double_prime.nbytes
        ),
    )


def fit_power_law(
    sizes: np.ndarray,
    values: np.ndarray,
    minimum_size: float = 0.0,
) -> PowerLawFit:
    """在双对数坐标下拟合 y=C·x^p，并返回指数和决定系数。"""
    sizes = np.asarray(sizes, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = (
        np.isfinite(sizes)
        & np.isfinite(values)
        & (sizes > 0.0)
        & (values > 0.0)
        & (sizes >= minimum_size)
    )
    x = np.log(sizes[valid])
    y = np.log(values[valid])
    if len(x) < 2:
        return PowerLawFit(float("nan"), float("nan"), float("nan"), len(x))
    exponent, intercept = np.polyfit(x, y, 1)
    predicted = exponent * x + intercept
    residual_sum = float(np.sum((y - predicted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
    return PowerLawFit(
        exponent=float(exponent),
        coefficient=float(np.exp(intercept)),
        r_squared=r_squared,
        sample_count=len(x),
    )
