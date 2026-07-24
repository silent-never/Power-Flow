"""Newton--Raphson 潮流结果的跨算例与跨精度校核。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.grid import PowerGrid
from ..solvers.nr_solver import NewtonRaphsonSolver


@dataclass(frozen=True, slots=True)
class AccuracyValidationResult:
    """一次 NR 计算的最大偏差及收敛残差。"""

    label: str
    node_count: int
    tolerance: float
    success: bool
    iterations: int
    max_voltage_deviation: float
    max_angle_deviation_deg: float
    max_active_power_deviation: float
    max_reactive_non_pq_deviation: float
    max_active_mismatch: float
    max_reactive_mismatch: float

    @property
    def max_equation_mismatch(self) -> float:
        return max(self.max_active_mismatch, self.max_reactive_mismatch)


def _maximum(values: np.ndarray) -> float:
    """返回绝对值最大值；空数组按零处理。"""
    values = np.asarray(values, dtype=float)
    return float(np.max(np.abs(values))) if values.size else 0.0


def run_nr_accuracy_validation(
    base_grid: PowerGrid,
    *,
    label: str,
    tolerance: float,
    max_iterations: int,
    flat_start: bool = True,
) -> AccuracyValidationResult:
    """运行一次 NR，并计算标准结果偏差与潮流方程残差。"""
    grid = base_grid.clone()
    if flat_start:
        # 平坦启动仍须保留数据文件的参考母线相角，否则会给全部节点
        # 引入同一个刚性相角偏移，造成与 angle_final 比较时的伪偏差。
        slack_indices = np.flatnonzero(grid.bus_type == 3)
        reference_angle = (
            float(grid.theta[slack_indices[0]])
            if slack_indices.size
            else 0.0
        )
        grid.theta[:] = reference_angle
        grid.V[grid.bus_type == 1] = 1.0

    solver = NewtonRaphsonSolver(
        tol=tolerance,
        max_iter=max_iterations,
        verbose=False,
    )
    success, info = solver.solve(grid)

    # 刷新最终节点注入功率和方程失配。
    d_p, d_q = grid.get_mismatch()
    non_slack = grid.bus_type != 3
    pq_buses = grid.bus_type == 1
    non_pq = grid.bus_type != 1

    voltage_reference = np.asarray(
        [bus.get("v_final", 1.0) for bus in grid.buses],
        dtype=float,
    )
    angle_reference_deg = np.asarray(
        [bus.get("angle_final", bus.get("angle", 0.0)) for bus in grid.buses],
        dtype=float,
    )
    angle_calculated_deg = np.degrees(np.asarray(grid.theta, dtype=float))

    return AccuracyValidationResult(
        label=label,
        node_count=grid.n,
        tolerance=float(tolerance),
        success=bool(success),
        iterations=int(info.get("iterations", 0)),
        max_voltage_deviation=_maximum(grid.V - voltage_reference),
        max_angle_deviation_deg=_maximum(
            angle_calculated_deg - angle_reference_deg
        ),
        max_active_power_deviation=_maximum(grid.P_calc - grid.P_spec),
        max_reactive_non_pq_deviation=_maximum(
            grid.Q_calc[non_pq] - grid.Q_spec[non_pq]
        ),
        max_active_mismatch=_maximum(d_p[non_slack]),
        max_reactive_mismatch=_maximum(d_q[pq_buses]),
    )
