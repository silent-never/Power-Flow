"""临界负荷附近最优乘子法与非线性规划法的停滞分析。"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ..core.grid import PowerGrid
from ..cpf.predictor import apply_load_parameter, enforce_reactive_power_limits
from ..solvers.nonlinear_programming_solver import NonlinearProgrammingSolver
from ..solvers.nr_solver import NewtonRaphsonSolver
from ..solvers.optimal_multiplier_solver import OptimalMultiplierSolver
from ..solvers.pq_solver import FastDecoupledSolver


@dataclass(frozen=True, slots=True)
class CriticalStagnationPoint:
    """一个负荷倍率下某种算法的完整停滞诊断。"""

    load_multiplier: float
    solver_name: str
    success: bool
    iterations: int
    final_error: float
    final_objective: float
    terminal_control: float
    minimum_control: float
    q_limit_passes: int
    q_limited_buses: tuple[int, ...]
    failure_reason: str
    error_history: tuple[float, ...]
    objective_history: tuple[float, ...]
    control_history: tuple[float, ...]

    @property
    def status(self) -> str:
        """返回便于终端和图表显示的状态描述。"""
        if self.success:
            return "收敛"
        if self.solver_name == "Optimal Multiplier" and self.terminal_control <= 1e-6:
            return "乘子趋零停滞"
        if self.solver_name == "Nonlinear Programming":
            return "回溯步长耗尽"
        if np.isfinite(self.final_error):
            return "残差停滞/未收敛"
        return "数值失败"


@dataclass(frozen=True, slots=True)
class CriticalStagnationResult:
    """鼻点前后两种算法的停滞扫描结果。"""

    cpf_nose_multiplier: float
    enforce_q_limits: bool
    points: tuple[CriticalStagnationPoint, ...]

    def points_for(self, solver_name: str) -> tuple[CriticalStagnationPoint, ...]:
        """返回指定算法按负荷倍率排列的结果。"""
        return tuple(
            item for item in self.points if item.solver_name == solver_name
        )


@dataclass(frozen=True, slots=True)
class StagnationBoundary:
    """A refined convergence-to-stagnation bracket for one solver."""

    solver_name: str
    last_converged: CriticalStagnationPoint
    first_failed: CriticalStagnationPoint

    @property
    def bracket(self) -> tuple[float, float]:
        return (
            self.last_converged.load_multiplier,
            self.first_failed.load_multiplier,
        )

    @property
    def width(self) -> float:
        low, high = self.bracket
        return high - low


@dataclass(frozen=True, slots=True)
class AdvancedStagnationBoundaryResult:
    """Refined OM/NLP stagnation boundaries near the IEEE-118 nose point."""

    cpf_nose_multiplier: float
    tolerance: float
    refinement_tolerance: float
    enforce_q_limits: bool
    points: tuple[CriticalStagnationPoint, ...]
    boundaries: tuple[StagnationBoundary, ...]

    def points_for(self, solver_name: str) -> tuple[CriticalStagnationPoint, ...]:
        return tuple(
            sorted(
                (item for item in self.points if item.solver_name == solver_name),
                key=lambda item: item.load_multiplier,
            )
        )

    def boundary_for(self, solver_name: str) -> StagnationBoundary | None:
        return next(
            (item for item in self.boundaries if item.solver_name == solver_name),
            None,
        )


def _scaled_grid(base_grid: PowerGrid, multiplier: float) -> PowerGrid:
    """构造与 CPF 负荷增长定义一致的平坦启动电网。"""
    grid = PowerGrid(
        copy.deepcopy(base_grid.buses),
        copy.deepcopy(base_grid.branches),
        base_grid.base_mva,
    )
    grid.theta[:] = 0.0
    grid.V[grid.bus_type == 1] = 1.0
    apply_load_parameter(grid, multiplier - 1.0)
    return grid


def _final_error(grid: PowerGrid) -> float:
    """计算当前有效潮流方程的最大绝对失配。"""
    d_p, d_q = grid.get_mismatch()
    values = np.concatenate(
        (d_p[grid.bus_type != 3], d_q[grid.bus_type == 1])
    )
    if values.size == 0 or np.any(~np.isfinite(values)):
        return float("nan")
    return float(np.max(np.abs(values)))


def _solve_point(
    base_grid: PowerGrid,
    multiplier: float,
    solver_name: str,
    tolerance: float,
    max_iterations: int,
    enforce_q_limits: bool,
) -> CriticalStagnationPoint:
    """运行一个倍率点，并合并多轮 PV 转 PQ 后的诊断轨迹。"""
    grid = _scaled_grid(base_grid, multiplier)
    error_history: list[float] = []
    objective_history: list[float] = []
    control_history: list[float] = []
    limited_buses: set[int] = set()
    total_iterations = 0
    terminal_control = float("nan")
    failure_reason = ""
    success = False
    pass_count = 0

    for _ in range(grid.n + 1):
        pass_count += 1
        if solver_name == "Newton-Raphson":
            solver = NewtonRaphsonSolver(
                tol=tolerance,
                max_iter=max_iterations,
                verbose=False,
            )
        elif solver_name == "Fast-Decoupled (XB)":
            solver = FastDecoupledSolver(
                tol=tolerance,
                max_iter=max_iterations,
                verbose=False,
            )
        elif solver_name == "Optimal Multiplier":
            solver = OptimalMultiplierSolver(
                tol=tolerance,
                max_iter=max_iterations,
                verbose=False,
            )
        else:
            solver = NonlinearProgrammingSolver(
                tol=tolerance,
                max_iter=max_iterations,
                verbose=False,
            )
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                success, info = solver.solve(grid)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            success = False
            info = {"failure_reason": f"{type(exc).__name__}: {exc}"}

        total_iterations += int(info.get("iterations", 0))
        error_history.extend(float(value) for value in info.get("max_error_history", ()))
        objective_history.extend(float(value) for value in info.get("objective_history", ()))
        if solver_name == "Optimal Multiplier":
            controls = [float(value) for value in info.get("multiplier_history", ())]
            terminal_control = controls[-1] if controls else float("nan")
        elif solver_name == "Nonlinear Programming":
            controls = [float(value) for value in info.get("step_length_history", ())]
            terminal_control = float(info.get("terminal_step_length", float("nan")))
            if not success and np.isfinite(terminal_control):
                controls.append(terminal_control)
        else:
            controls = []
            terminal_control = float("nan")
        control_history.extend(controls)
        failure_reason = str(info.get("failure_reason", ""))

        if not success or not enforce_q_limits:
            break
        switched = enforce_reactive_power_limits(grid, multiplier - 1.0)
        limited_buses.update(item[0] for item in switched)
        if not switched:
            break

    final_error = _final_error(grid)
    final_objective = (
        0.5 * float(np.sum(np.square(error_history[-1:])))
        if not objective_history and np.isfinite(final_error)
        else (objective_history[-1] if objective_history else float("nan"))
    )
    finite_controls = [value for value in control_history if np.isfinite(value)]
    minimum_control = min(finite_controls) if finite_controls else float("nan")
    return CriticalStagnationPoint(
        load_multiplier=float(multiplier),
        solver_name=solver_name,
        success=bool(success),
        iterations=total_iterations,
        final_error=final_error,
        final_objective=float(final_objective),
        terminal_control=terminal_control,
        minimum_control=minimum_control,
        q_limit_passes=pass_count,
        q_limited_buses=tuple(sorted(limited_buses)),
        failure_reason=failure_reason,
        error_history=tuple(error_history),
        objective_history=tuple(objective_history),
        control_history=tuple(control_history),
    )


def run_critical_stagnation_analysis(
    base_grid: PowerGrid,
    load_multipliers: tuple[float, ...],
    cpf_nose_multiplier: float,
    tolerance: float,
    nr_max_iterations: int = 20,
    fast_decoupled_max_iterations: int = 100,
    advanced_max_iterations: int = 60,
    enforce_q_limits: bool = True,
) -> CriticalStagnationResult:
    """扫描鼻点前后负荷倍率并诊断四种方法的停滞行为。"""
    if not load_multipliers or any(value <= 0.0 for value in load_multipliers):
        raise ValueError("load_multipliers 必须包含正数")
    if cpf_nose_multiplier <= 0.0:
        raise ValueError("cpf_nose_multiplier 必须大于 0")
    if tolerance <= 0.0 or min(
        nr_max_iterations,
        fast_decoupled_max_iterations,
        advanced_max_iterations,
    ) <= 0:
        raise ValueError("收敛精度和最大迭代次数必须为正数")

    points = []
    solver_settings = (
        ("Newton-Raphson", nr_max_iterations),
        ("Fast-Decoupled (XB)", fast_decoupled_max_iterations),
        ("Optimal Multiplier", advanced_max_iterations),
        ("Nonlinear Programming", advanced_max_iterations),
    )
    for multiplier in sorted(set(float(value) for value in load_multipliers)):
        for solver_name, max_iterations in solver_settings:
            points.append(
                _solve_point(
                    base_grid,
                    multiplier,
                    solver_name,
                    tolerance,
                    max_iterations,
                    enforce_q_limits,
                )
            )
    return CriticalStagnationResult(
        cpf_nose_multiplier=float(cpf_nose_multiplier),
        enforce_q_limits=bool(enforce_q_limits),
        points=tuple(points),
    )


def run_advanced_stagnation_boundary_analysis(
    base_grid: PowerGrid,
    load_multipliers: tuple[float, ...],
    cpf_nose_multiplier: float,
    tolerance: float,
    max_iterations: int = 60,
    enforce_q_limits: bool = True,
    refinement_tolerance: float = 1e-5,
) -> AdvancedStagnationBoundaryResult:
    """Locate OM/NLP convergence-failure brackets by coarse scan and bisection.

    A point counts as converged only when the solver reports success *and* the
    independently recomputed power-flow residual is below ``tolerance``.  This
    prevents a vanishing multiplier or line-search step from being mistaken for
    a physical power-flow solution.
    """
    if len(load_multipliers) < 2 or any(value <= 0.0 for value in load_multipliers):
        raise ValueError("load_multipliers must contain at least two positive values")
    if cpf_nose_multiplier <= 0.0 or tolerance <= 0.0:
        raise ValueError("cpf_nose_multiplier and tolerance must be positive")
    if max_iterations <= 0 or refinement_tolerance <= 0.0:
        raise ValueError("iteration and refinement limits must be positive")

    solver_names = ("Optimal Multiplier", "Nonlinear Programming")
    points: list[CriticalStagnationPoint] = []

    def solve(multiplier: float, solver_name: str) -> CriticalStagnationPoint:
        point = _solve_point(
            base_grid,
            multiplier,
            solver_name,
            tolerance,
            max_iterations,
            enforce_q_limits,
        )
        points.append(point)
        return point

    def is_strictly_converged(point: CriticalStagnationPoint) -> bool:
        return (
            point.success
            and np.isfinite(point.final_error)
            and point.final_error < tolerance
        )

    coarse_values = sorted(set(float(value) for value in load_multipliers))
    coarse: dict[str, list[CriticalStagnationPoint]] = {
        solver_name: [] for solver_name in solver_names
    }
    for multiplier in coarse_values:
        for solver_name in solver_names:
            coarse[solver_name].append(solve(multiplier, solver_name))

    boundaries: list[StagnationBoundary] = []
    for solver_name in solver_names:
        converged = [
            item for item in coarse[solver_name] if is_strictly_converged(item)
        ]
        if not converged:
            continue
        low_point = max(converged, key=lambda item: item.load_multiplier)
        failed = [
            item
            for item in coarse[solver_name]
            if item.load_multiplier > low_point.load_multiplier
            and not is_strictly_converged(item)
        ]
        if not failed:
            continue
        high_point = min(failed, key=lambda item: item.load_multiplier)

        while high_point.load_multiplier - low_point.load_multiplier > refinement_tolerance:
            middle = 0.5 * (
                low_point.load_multiplier + high_point.load_multiplier
            )
            middle_point = solve(middle, solver_name)
            if is_strictly_converged(middle_point):
                low_point = middle_point
            else:
                high_point = middle_point

        boundaries.append(
            StagnationBoundary(
                solver_name=solver_name,
                last_converged=low_point,
                first_failed=high_point,
            )
        )

    return AdvancedStagnationBoundaryResult(
        cpf_nose_multiplier=float(cpf_nose_multiplier),
        tolerance=float(tolerance),
        refinement_tolerance=float(refinement_tolerance),
        enforce_q_limits=bool(enforce_q_limits),
        points=tuple(points),
        boundaries=tuple(boundaries),
    )
