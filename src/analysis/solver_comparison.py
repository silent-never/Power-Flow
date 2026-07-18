"""标准 NR 法与快速解耦潮流法的性能对比流程。"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Callable

import numpy as np

from ..core.grid import PowerGrid
from ..solvers.base_solver import BaseSolver
from ..solvers.nr_solver import NewtonRaphsonSolver
from ..solvers.pq_solver import FastDecoupledSolver


@dataclass(frozen=True, slots=True)
class SolverBenchmark:
    """一个潮流算法的收敛与计时结果。"""

    name: str
    success: bool
    iterations: int
    final_error: float
    error_history: tuple[float, ...]
    warmup_time: float | None
    raw_elapsed_samples: tuple[float, ...]
    elapsed_samples: tuple[float, ...]
    excluded_elapsed_samples: tuple[float, ...]
    median_time: float
    mean_time: float
    standard_deviation: float
    timing_q1: float
    timing_q3: float
    timing_mad: float
    outlier_upper_bound: float
    final_grid: PowerGrid
    failure_reason: str = ""

    @property
    def timing_coefficient_of_variation(self) -> float:
        """返回正常耗时样本的变异系数，用于衡量计时稳定程度。"""
        if self.mean_time <= 0.0:
            return float("nan")
        return self.standard_deviation / self.mean_time

    @property
    def robust_timing_coefficient(self) -> float:
        """返回基于中位绝对偏差的稳健相对波动指标。"""
        if self.median_time <= 0.0:
            return float("nan")
        return 1.4826 * self.timing_mad / self.median_time

    @property
    def discarded_max_time(self) -> float | None:
        """兼容旧接口，返回被排除异常值中的最大值。"""
        if not self.excluded_elapsed_samples:
            return None
        return max(self.excluded_elapsed_samples)

    @property
    def observed_order(self) -> float:
        """返回误差序列最后一个可信的局部观测收敛阶。"""
        errors = np.asarray(self.error_history, dtype=float)
        orders: list[float] = []
        for index in range(1, len(errors) - 1):
            previous_error = errors[index - 1]
            current_error = errors[index]
            next_error = errors[index + 1]
            if not (
                np.isfinite(previous_error)
                and np.isfinite(current_error)
                and np.isfinite(next_error)
                and previous_error > current_error > next_error > 1e-14
            ):
                continue
            denominator = np.log(current_error / previous_error)
            if abs(denominator) <= 1e-12:
                continue
            order = np.log(next_error / current_error) / denominator
            if np.isfinite(order):
                orders.append(float(order))
        return orders[-1] if orders else float("nan")


@dataclass(frozen=True, slots=True)
class SolverComparisonResult:
    """NR 与快速解耦法的完整对比结果。"""

    nr: SolverBenchmark
    fast_decoupled: SolverBenchmark
    max_voltage_difference: float
    max_angle_difference_deg: float

    @property
    def speed_ratio(self) -> float:
        """返回 NR 中位耗时除以快速解耦法中位耗时。"""
        if self.fast_decoupled.median_time <= 0.0:
            return float("inf")
        return self.nr.median_time / self.fast_decoupled.median_time


def _prepare_grid(base_grid: PowerGrid, flat_start: bool) -> PowerGrid:
    """复制基准电网，并按需要设置统一的平坦初值。"""
    grid = base_grid.clone()
    if flat_start:
        grid.theta[:] = 0.0
        grid.V[grid.bus_type == 1] = 1.0
    return grid


def _final_mismatch(grid: PowerGrid) -> float:
    """计算参与迭代方程的最终最大功率不平衡量。"""
    d_p, d_q = grid.get_mismatch()
    non_slack = grid.bus_type != 3
    pq_buses = grid.bus_type == 1
    max_d_p = (
        float(np.max(np.abs(d_p[non_slack])))
        if np.any(non_slack)
        else 0.0
    )
    max_d_q = (
        float(np.max(np.abs(d_q[pq_buses])))
        if np.any(pq_buses)
        else 0.0
    )
    return max(max_d_p, max_d_q)


def _classify_timing_samples(
    samples: list[float],
) -> tuple[list[float], list[float], float]:
    """使用单侧 Tukey-IQR 规则识别会向上拖高均值的耗时异常值。

    运行时间受到进程调度、首次动态库调用和后台任务影响时，通常只会
    额外增加而不会变成负值，因此这里只识别右侧异常值。少于四个样本时
    四分位统计不稳定，此时保留全部样本。
    """
    if len(samples) < 4:
        return samples.copy(), [], float("inf")

    values = np.asarray(samples, dtype=float)
    q1, q3 = np.percentile(values, [25.0, 75.0])
    iqr = float(q3 - q1)
    upper_bound = float(q3 + 1.5 * iqr)
    retained = [value for value in samples if value <= upper_bound]
    excluded = [value for value in samples if value > upper_bound]

    # 极端情况下避免筛选规则将全部样本排除。
    if not retained:
        return samples.copy(), [], upper_bound
    return retained, excluded, upper_bound


def _benchmark_solver(
    name: str,
    solver_factory: Callable[[], BaseSolver],
    base_grid: PowerGrid,
    repeat_count: int,
    flat_start: bool,
) -> SolverBenchmark:
    """重复运行求解器，剔除最大耗时后统计中位数。"""
    elapsed_samples: list[float] = []
    warmup_time: float | None = None
    representative_grid = _prepare_grid(base_grid, flat_start)
    representative_info: dict = {
        "iterations": 0,
        "max_error_history": [],
        "time_elapsed": 0.0,
    }
    all_success = True
    failure_reason = ""

    for repeat_index in range(repeat_count):
        grid = _prepare_grid(base_grid, flat_start)
        try:
            success, info = solver_factory().solve(grid)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            success = False
            info = {
                "iterations": 0,
                "max_error_history": [],
                "time_elapsed": float("nan"),
            }
            failure_reason = f"{type(exc).__name__}: {exc}"

        elapsed = float(info.get("time_elapsed", float("nan")))
        if np.isfinite(elapsed) and elapsed >= 0.0:
            if repeat_index == 0:
                warmup_time = elapsed
            else:
                elapsed_samples.append(elapsed)
        all_success = all_success and bool(success)

        if repeat_index == 0:
            representative_grid = grid
            representative_info = info

        if not success and not failure_reason:
            failure_reason = "达到最大迭代次数后仍未收敛"

    raw_elapsed_samples = elapsed_samples.copy()
    elapsed_samples, excluded_samples, outlier_upper_bound = (
        _classify_timing_samples(raw_elapsed_samples)
    )

    final_error = _final_mismatch(representative_grid)
    median_time = (
        float(median(elapsed_samples))
        if elapsed_samples
        else float("nan")
    )
    mean_time = (
        float(np.mean(elapsed_samples))
        if elapsed_samples
        else float("nan")
    )
    standard_deviation = (
        float(np.std(elapsed_samples, ddof=1))
        if len(elapsed_samples) >= 2
        else 0.0
    )
    if elapsed_samples:
        timing_q1, timing_q3 = (
            float(value)
            for value in np.percentile(elapsed_samples, [25.0, 75.0])
        )
        timing_mad = float(
            np.median(np.abs(np.asarray(elapsed_samples) - median_time))
        )
    else:
        timing_q1 = timing_q3 = float("nan")
        timing_mad = float("nan")
    return SolverBenchmark(
        name=name,
        success=all_success,
        iterations=int(representative_info.get("iterations", 0)),
        final_error=final_error,
        error_history=tuple(
            float(value)
            for value in representative_info.get("max_error_history", ())
        ),
        warmup_time=warmup_time,
        raw_elapsed_samples=tuple(raw_elapsed_samples),
        elapsed_samples=tuple(elapsed_samples),
        excluded_elapsed_samples=tuple(excluded_samples),
        median_time=median_time,
        mean_time=mean_time,
        standard_deviation=standard_deviation,
        timing_q1=timing_q1,
        timing_q3=timing_q3,
        timing_mad=timing_mad,
        outlier_upper_bound=outlier_upper_bound,
        final_grid=representative_grid,
        failure_reason=failure_reason,
    )


def compare_power_flow_solvers(
    base_grid: PowerGrid,
    tolerance: float,
    nr_max_iterations: int,
    fast_decoupled_max_iterations: int,
    repeat_count: int = 5,
    flat_start: bool = True,
) -> SolverComparisonResult:
    """在相同初值和误差阈值下比较 NR 与快速解耦法。"""
    if repeat_count <= 1:
        raise ValueError("repeat_count 必须大于 1，以便统计重复运行耗时")

    nr = _benchmark_solver(
        name="Newton-Raphson",
        solver_factory=lambda: NewtonRaphsonSolver(
            tol=tolerance,
            max_iter=nr_max_iterations,
            verbose=False,
        ),
        base_grid=base_grid,
        repeat_count=repeat_count,
        flat_start=flat_start,
    )
    fast_decoupled = _benchmark_solver(
        name="Fast-Decoupled (XB)",
        solver_factory=lambda: FastDecoupledSolver(
            tol=tolerance,
            max_iter=fast_decoupled_max_iterations,
            verbose=False,
        ),
        base_grid=base_grid,
        repeat_count=repeat_count,
        flat_start=flat_start,
    )

    if nr.success and fast_decoupled.success:
        max_voltage_difference = float(
            np.max(np.abs(nr.final_grid.V - fast_decoupled.final_grid.V))
        )
        max_angle_difference_deg = float(
            np.degrees(
                np.max(
                    np.abs(
                        nr.final_grid.theta
                        - fast_decoupled.final_grid.theta
                    )
                )
            )
        )
    else:
        max_voltage_difference = float("nan")
        max_angle_difference_deg = float("nan")

    return SolverComparisonResult(
        nr=nr,
        fast_decoupled=fast_decoupled,
        max_voltage_difference=max_voltage_difference,
        max_angle_difference_deg=max_angle_difference_deg,
    )
