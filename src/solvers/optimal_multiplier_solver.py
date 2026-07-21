"""沿牛顿方向进行一维最优化的最优乘子潮流求解器。"""

from __future__ import annotations

import time

import numpy as np

from ._nonlinear_utils import (
    active_mismatch,
    mismatch_objective,
    set_trial_state,
    split_direction,
)
from .base_solver import BaseSolver


class OptimalMultiplierSolver(BaseSolver):
    """通过最小化真实潮流失配范数选择每一步的牛顿乘子。

    本实现采用数值一维搜索，而不是依赖特定坐标形式的三次多项式
    闭式解，因此可以直接沿用项目现有的极坐标和相对电压增量定义。
    """

    def __init__(
        self,
        tol: float = 1e-6,
        max_iter: int = 30,
        verbose: bool = True,
        multiplier_max: float = 2.0,
        search_tolerance: float = 1e-3,
        max_search_iter: int = 18,
    ) -> None:
        super().__init__(tol=tol, max_iter=max_iter, verbose=verbose)
        if multiplier_max <= 0.0:
            raise ValueError("multiplier_max 必须大于 0")
        if search_tolerance <= 0.0:
            raise ValueError("search_tolerance 必须大于 0")
        self.multiplier_max = float(multiplier_max)
        self.search_tolerance = float(search_tolerance)
        self.max_search_iter = int(max_search_iter)

    def _find_optimal_multiplier(
        self,
        grid,
        base_voltage: np.ndarray,
        base_angle: np.ndarray,
        voltage_direction: np.ndarray,
        angle_direction: np.ndarray,
        theta_idx: np.ndarray,
        v_idx: np.ndarray,
    ) -> tuple[float, float, int]:
        """用粗搜索和黄金分割搜索求取失配范数最小的乘子。"""
        evaluations = 0

        def evaluate(multiplier: float) -> float:
            nonlocal evaluations
            evaluations += 1
            valid = set_trial_state(
                grid,
                base_voltage,
                base_angle,
                voltage_direction,
                angle_direction,
                multiplier,
            )
            if not valid:
                return float("inf")
            return mismatch_objective(grid, theta_idx, v_idx)

        coarse_points = np.linspace(0.0, self.multiplier_max, 9)
        if 1.0 < self.multiplier_max and not np.any(np.isclose(coarse_points, 1.0)):
            coarse_points = np.sort(np.append(coarse_points, 1.0))
        coarse_values = np.asarray([evaluate(float(point)) for point in coarse_points])
        best_index = int(np.argmin(coarse_values))
        best_multiplier = float(coarse_points[best_index])
        best_objective = float(coarse_values[best_index])

        left_index = max(0, best_index - 1)
        right_index = min(len(coarse_points) - 1, best_index + 1)
        left = float(coarse_points[left_index])
        right = float(coarse_points[right_index])
        golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = right - golden_ratio * (right - left)
        x2 = left + golden_ratio * (right - left)
        f1 = evaluate(x1)
        f2 = evaluate(x2)

        for _ in range(self.max_search_iter):
            if right - left <= self.search_tolerance:
                break
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - golden_ratio * (right - left)
                f1 = evaluate(x1)
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + golden_ratio * (right - left)
                f2 = evaluate(x2)

        candidates = (
            (best_multiplier, best_objective),
            (x1, f1),
            (x2, f2),
        )
        best_multiplier, best_objective = min(candidates, key=lambda item: item[1])
        set_trial_state(
            grid,
            base_voltage,
            base_angle,
            voltage_direction,
            angle_direction,
            float(best_multiplier),
        )
        grid.get_mismatch()
        return float(best_multiplier), float(best_objective), evaluations

    def solve(self, grid):
        """求解交流潮流并返回收敛状态和迭代信息。"""
        start_time = time.perf_counter()
        info = {
            "iterations": 0,
            "max_error_history": [],
            "objective_history": [],
            "multiplier_history": [],
            "search_evaluations": [],
            "time_elapsed": 0.0,
            "failure_reason": "",
            "timing_breakdown": {
                "mismatch_time": 0.0,
                "matrix_build_time": 0.0,
                "factorization_time": 0.0,
                "linear_solve_time": 0.0,
                "state_update_time": 0.0,
                "line_search_time": 0.0,
            },
        }
        timing = info["timing_breakdown"]

        for iteration in range(self.max_iter):
            phase_start = time.perf_counter()
            mismatch, theta_idx, v_idx = active_mismatch(grid)
            timing["mismatch_time"] += time.perf_counter() - phase_start
            max_error = float(np.max(np.abs(mismatch))) if mismatch.size else 0.0
            objective = 0.5 * float(mismatch @ mismatch)
            info["max_error_history"].append(max_error)
            info["objective_history"].append(objective)

            if self.verbose:
                print(
                    f"最优乘子法迭代 {iteration:2d}: "
                    f"最大失配={max_error:.6e}, 目标函数={objective:.6e}"
                )
            if max_error < self.tol:
                info["iterations"] = iteration
                info["time_elapsed"] = time.perf_counter() - start_time
                return True, info

            phase_start = time.perf_counter()
            jacobian, _, _ = grid.get_jacobian()
            timing["matrix_build_time"] += time.perf_counter() - phase_start
            try:
                phase_start = time.perf_counter()
                direction = -np.linalg.solve(jacobian, mismatch)
                timing["linear_solve_time"] += time.perf_counter() - phase_start
            except np.linalg.LinAlgError as exc:
                info["iterations"] = iteration
                info["failure_reason"] = f"雅可比矩阵求解失败：{exc}"
                info["time_elapsed"] = time.perf_counter() - start_time
                return False, info

            voltage_direction, angle_direction = split_direction(
                grid, direction, theta_idx, v_idx
            )
            base_voltage = grid.V.copy()
            base_angle = grid.theta.copy()
            phase_start = time.perf_counter()
            multiplier, new_objective, evaluations = self._find_optimal_multiplier(
                grid,
                base_voltage,
                base_angle,
                voltage_direction,
                angle_direction,
                theta_idx,
                v_idx,
            )
            timing["line_search_time"] += time.perf_counter() - phase_start
            info["multiplier_history"].append(multiplier)
            info["search_evaluations"].append(evaluations)

            if multiplier <= 1e-8 or new_objective >= objective * (1.0 - 1e-12):
                grid.V[:] = base_voltage
                grid.theta[:] = base_angle
                info["iterations"] = iteration + 1
                info["failure_reason"] = "最优乘子接近零，目标函数无法继续下降"
                info["time_elapsed"] = time.perf_counter() - start_time
                return False, info

        info["iterations"] = self.max_iter
        info["failure_reason"] = "达到最大迭代次数"
        info["time_elapsed"] = time.perf_counter() - start_time
        return False, info
