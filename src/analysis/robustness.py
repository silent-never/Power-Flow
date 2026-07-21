"""NR 与快速解耦潮流法的鲁棒性测试流程。"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Callable

import numpy as np

from ..core.grid import PowerGrid
from ..cpf.predictor import apply_load_parameter, enforce_reactive_power_limits
from ..solvers.base_solver import BaseSolver
from ..solvers.nr_solver import NewtonRaphsonSolver
from ..solvers.pq_solver import FastDecoupledSolver


@dataclass(frozen=True, slots=True)
class RobustnessTrial:
    """一次指定扰动下的潮流求解结果。"""

    category: str
    scenario: str
    parameter: float
    solver_name: str
    success: bool
    iterations: int
    final_error: float
    failure_mode: str = ""


@dataclass(frozen=True, slots=True)
class RobustnessSummary:
    """同一场景下多次随机试验的汇总结果。"""

    category: str
    scenario: str
    parameter: float
    solver_name: str
    trial_count: int
    success_count: int
    success_rate: float
    mean_iterations: float
    median_final_error: float
    failure_modes: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class SolverRobustnessResult:
    """两种潮流算法的全部鲁棒性试验及汇总结果。"""

    trials: tuple[RobustnessTrial, ...]
    summaries: tuple[RobustnessSummary, ...]

    def max_converged_load_multiplier(self, solver_name: str) -> float:
        """返回指定算法在测试点中能够收敛的最大负荷倍率。"""
        values = [
            item.parameter
            for item in self.summaries
            if item.category == "load"
            and item.solver_name == solver_name
            and item.success_rate > 0.0
        ]
        return max(values) if values else float("nan")

    def load_convergence_bracket(
        self,
        solver_name: str,
    ) -> tuple[float, float] | None:
        """返回指定算法最后收敛点与其后的首次失败点。"""
        load_items = [
            item
            for item in self.summaries
            if item.category == "load" and item.solver_name == solver_name
        ]
        converged = [item.parameter for item in load_items if item.success_rate > 0.0]
        if not converged:
            return None
        low = max(converged)
        failed = [
            item.parameter
            for item in load_items
            if item.success_rate == 0.0 and item.parameter > low
        ]
        if not failed:
            return None
        return low, min(failed)


def _flat_start(grid: PowerGrid) -> None:
    """设置统一平坦初值，同时保留 PV 与平衡节点的给定电压。"""
    grid.theta[:] = 0.0
    grid.V[grid.bus_type == 1] = 1.0


def _final_mismatch(grid: PowerGrid) -> float:
    """计算参与潮流方程的最终最大功率不平衡量。"""
    d_p, d_q = grid.get_mismatch()
    non_slack = grid.bus_type != 3
    pq_buses = grid.bus_type == 1
    max_p = float(np.max(np.abs(d_p[non_slack]))) if np.any(non_slack) else 0.0
    max_q = float(np.max(np.abs(d_q[pq_buses]))) if np.any(pq_buses) else 0.0
    return max(max_p, max_q)


def _classify_failure(
    error_history: list[float],
    finite_state: bool,
    positive_voltage: bool,
) -> str:
    """根据状态和残差轨迹区分主要失败形式。"""
    if not finite_state:
        return "非有限状态"
    if not positive_voltage:
        return "非正电压"
    errors = np.asarray(error_history, dtype=float)
    errors = errors[np.isfinite(errors) & (errors > 0.0)]
    if len(errors) < 2:
        return "未收敛"
    if errors[-1] > max(errors[0] * 10.0, 1.0):
        return "残差发散"
    if len(errors) >= 5:
        recent = errors[-5:]
        changes = np.diff(recent)
        direction_changes = np.count_nonzero(changes[:-1] * changes[1:] < 0.0)
        if direction_changes >= 2 and np.max(recent) / np.min(recent) > 2.0:
            return "残差振荡"
        if recent[-1] / recent[0] > 0.8:
            return "残差停滞"
    return "达到迭代上限"


def _solve_trial(
    grid: PowerGrid,
    solver_name: str,
    solver_factory: Callable[[], BaseSolver],
    category: str,
    scenario: str,
    parameter: float,
    load_lambda: float | None = None,
    enforce_q_limits: bool = False,
) -> RobustnessTrial:
    """在独立电网副本上运行一次求解并执行严格有效性检查。"""
    working_grid = grid.clone()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            total_iterations = 0
            combined_history: list[float] = []
            info: dict = {}
            for _ in range(working_grid.n + 1):
                success, info = solver_factory().solve(working_grid)
                total_iterations += int(info.get("iterations", 0))
                combined_history.extend(
                    float(value)
                    for value in info.get("max_error_history", [])
                )
                if not success or not enforce_q_limits or load_lambda is None:
                    break
                switched = enforce_reactive_power_limits(
                    working_grid,
                    load_lambda,
                )
                if not switched:
                    break
        finite_state = bool(
            np.all(np.isfinite(working_grid.V))
            and np.all(np.isfinite(working_grid.theta))
        )
        positive_voltage = bool(finite_state and np.all(working_grid.V > 0.0))
        success = bool(success and finite_state and positive_voltage)
        final_error = _final_mismatch(working_grid) if finite_state else float("nan")
        failure_mode = "" if success else _classify_failure(
            combined_history,
            finite_state,
            positive_voltage,
        )
        iterations = total_iterations
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        success = False
        iterations = 0
        final_error = float("nan")
        failure_mode = type(exc).__name__

    return RobustnessTrial(
        category=category,
        scenario=scenario,
        parameter=parameter,
        solver_name=solver_name,
        success=success,
        iterations=iterations,
        final_error=final_error,
        failure_mode=failure_mode,
    )


def _scaled_load_grid(base_grid: PowerGrid, multiplier: float) -> PowerGrid:
    """使用与 CPF 相同的负荷参数定义构造独立电网。"""
    grid = PowerGrid(
        copy.deepcopy(base_grid.buses),
        copy.deepcopy(base_grid.branches),
        base_grid.base_mva,
    )
    _flat_start(grid)
    apply_load_parameter(grid, multiplier - 1.0)
    return grid


def _scaled_resistance_grid(base_grid: PowerGrid, multiplier: float) -> PowerGrid:
    """放大支路电阻以构造更高 R/X 比的测试电网。"""
    branches = copy.deepcopy(base_grid.branches)
    for branch in branches:
        branch["r"] = float(branch.get("r", 0.0)) * multiplier
    grid = PowerGrid(copy.deepcopy(base_grid.buses), branches, base_grid.base_mva)
    _flat_start(grid)
    return grid


def _summarize(trials: list[RobustnessTrial]) -> tuple[RobustnessSummary, ...]:
    """按照场景和算法汇总成功率、迭代次数及失败形式。"""
    grouped: dict[tuple[str, str, float, str], list[RobustnessTrial]] = {}
    for trial in trials:
        key = (trial.category, trial.scenario, trial.parameter, trial.solver_name)
        grouped.setdefault(key, []).append(trial)

    summaries: list[RobustnessSummary] = []
    for (category, scenario, parameter, solver_name), group in grouped.items():
        successful = [item for item in group if item.success]
        failures = Counter(item.failure_mode for item in group if not item.success)
        summaries.append(
            RobustnessSummary(
                category=category,
                scenario=scenario,
                parameter=parameter,
                solver_name=solver_name,
                trial_count=len(group),
                success_count=len(successful),
                success_rate=len(successful) / len(group),
                mean_iterations=(
                    float(np.mean([item.iterations for item in successful]))
                    if successful
                    else float("nan")
                ),
                median_final_error=(
                    float(median(item.final_error for item in successful))
                    if successful
                    else float("nan")
                ),
                failure_modes=tuple(sorted(failures.items())),
            )
        )
    initial = [item for item in summaries if item.category == "initial"]
    solver_order = {"Newton-Raphson": 0, "Fast-Decoupled (XB)": 1}
    load = sorted(
        (item for item in summaries if item.category == "load"),
        key=lambda item: (item.parameter, solver_order[item.solver_name]),
    )
    resistance = sorted(
        (item for item in summaries if item.category == "resistance"),
        key=lambda item: (item.parameter, solver_order[item.solver_name]),
    )
    return tuple(initial + load + resistance)


def run_solver_robustness(
    base_grid: PowerGrid,
    tolerance: float,
    nr_max_iterations: int,
    fast_decoupled_max_iterations: int,
    random_trials: int,
    voltage_perturbations: tuple[float, ...],
    angle_perturbations_deg: tuple[float, ...],
    load_multipliers: tuple[float, ...],
    resistance_multipliers: tuple[float, ...],
    random_seed: int = 2026,
    enforce_q_limits: bool = False,
    load_refinement_tolerance: float = 1e-5,
) -> SolverRobustnessResult:
    """执行初值、负荷水平和 R/X 比三类鲁棒性测试。"""
    if random_trials <= 0:
        raise ValueError("random_trials 必须为正整数")
    if len(voltage_perturbations) != len(angle_perturbations_deg):
        raise ValueError("电压与相角扰动等级数量必须一致")
    if load_refinement_tolerance <= 0.0:
        raise ValueError("load_refinement_tolerance 必须为正数")

    solver_factories: tuple[tuple[str, Callable[[], BaseSolver]], ...] = (
        (
            "Newton-Raphson",
            lambda: NewtonRaphsonSolver(
                tol=tolerance,
                max_iter=nr_max_iterations,
                verbose=False,
            ),
        ),
        (
            "Fast-Decoupled (XB)",
            lambda: FastDecoupledSolver(
                tol=tolerance,
                max_iter=fast_decoupled_max_iterations,
                verbose=False,
            ),
        ),
    )
    trials: list[RobustnessTrial] = []

    # 给定初值和平坦初值各执行一次，作为随机扰动测试的参照。
    initial_grids = [
        ("数据给定初值", 0.0, base_grid.clone()),
        ("平坦初值", 0.0, base_grid.clone()),
    ]
    _flat_start(initial_grids[1][2])
    for scenario, parameter, grid in initial_grids:
        for solver_name, factory in solver_factories:
            trials.append(
                _solve_trial(grid, solver_name, factory, "initial", scenario, parameter)
            )

    rng = np.random.default_rng(random_seed)
    pq_buses = base_grid.bus_type == 1
    non_slack = base_grid.bus_type != 3
    for voltage_ratio, angle_deg in zip(
        voltage_perturbations,
        angle_perturbations_deg,
    ):
        scenario = f"随机 ±{voltage_ratio:.0%}V / ±{angle_deg:g}°"
        for _ in range(random_trials):
            grid = base_grid.clone()
            _flat_start(grid)
            grid.V[pq_buses] += rng.uniform(
                -voltage_ratio,
                voltage_ratio,
                size=int(np.count_nonzero(pq_buses)),
            )
            grid.theta[non_slack] += np.radians(
                rng.uniform(
                    -angle_deg,
                    angle_deg,
                    size=int(np.count_nonzero(non_slack)),
                )
            )
            for solver_name, factory in solver_factories:
                trials.append(
                    _solve_trial(
                        grid,
                        solver_name,
                        factory,
                        "initial",
                        scenario,
                        voltage_ratio,
                    )
                )

    load_trials: dict[str, list[RobustnessTrial]] = {
        solver_name: [] for solver_name, _ in solver_factories
    }

    def solve_load_point(
        multiplier: float,
        solver_name: str,
        factory: Callable[[], BaseSolver],
    ) -> RobustnessTrial:
        grid = _scaled_load_grid(base_grid, multiplier)
        multiplier_text = f"{multiplier:.6f}".rstrip("0").rstrip(".")
        return _solve_trial(
            grid,
            solver_name,
            factory,
            "load",
            f"负荷 {multiplier_text}×",
            multiplier,
            load_lambda=multiplier - 1.0,
            enforce_q_limits=enforce_q_limits,
        )

    for multiplier in sorted(set(load_multipliers)):
        for solver_name, factory in solver_factories:
            trial = solve_load_point(multiplier, solver_name, factory)
            trials.append(trial)
            load_trials[solver_name].append(trial)

    # 每种算法独立细化自己的收敛—失败区间，避免粗步长掩盖边界差异。
    for solver_name, factory in solver_factories:
        solver_load_trials = load_trials[solver_name]
        converged = [item.parameter for item in solver_load_trials if item.success]
        if not converged:
            continue
        low = max(converged)
        failed = [
            item.parameter
            for item in solver_load_trials
            if not item.success and item.parameter > low
        ]
        if not failed:
            continue
        high = min(failed)
        while high - low > load_refinement_tolerance:
            middle = (low + high) / 2.0
            trial = solve_load_point(middle, solver_name, factory)
            trials.append(trial)
            load_trials[solver_name].append(trial)
            if trial.success:
                low = middle
            else:
                high = middle

    for multiplier in resistance_multipliers:
        grid = _scaled_resistance_grid(base_grid, multiplier)
        scenario = f"电阻 {multiplier:.1f}×"
        for solver_name, factory in solver_factories:
            trials.append(
                _solve_trial(
                    grid,
                    solver_name,
                    factory,
                    "resistance",
                    scenario,
                    multiplier,
                )
            )

    return SolverRobustnessResult(
        trials=tuple(trials),
        summaries=_summarize(trials),
    )
