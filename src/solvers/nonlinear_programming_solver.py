"""基于非线性最小二乘规划的交流潮流求解器。"""

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


class NonlinearProgrammingSolver(BaseSolver):
    """用全局化牛顿法最小化潮流失配平方和。

    目标函数为 ``0.5 * ||F(x)||²``。算法优先计算不形成 ``J.T @ J``
    的牛顿方向，并用 Armijo 回溯确保目标函数下降；雅可比奇异或方向
    不满足下降条件时，才使用 Levenberg-Marquardt 正则化方向。

    该方法用于寻找潮流可行点，不等同于包含发电成本、线路容量等运行
    约束的最优潮流（OPF）。
    """

    def __init__(
        self,
        tol: float = 1e-6,
        max_iter: int = 50,
        verbose: bool = True,
        initial_damping: float = 1e-6,
        damping_increase: float = 10.0,
        backtracking_factor: float = 0.5,
        armijo_constant: float = 1e-4,
        max_trial_steps: int = 12,
        step_tolerance: float = 1e-10,
    ) -> None:
        super().__init__(tol=tol, max_iter=max_iter, verbose=verbose)
        if initial_damping <= 0.0:
            raise ValueError("initial_damping 必须大于 0")
        if damping_increase <= 1.0:
            raise ValueError("damping_increase 必须大于 1")
        if not 0.0 < backtracking_factor < 1.0:
            raise ValueError("backtracking_factor 必须位于 (0, 1) 内")
        if not 0.0 < armijo_constant < 1.0:
            raise ValueError("armijo_constant 必须位于 (0, 1) 内")
        self.initial_damping = float(initial_damping)
        self.damping_increase = float(damping_increase)
        self.backtracking_factor = float(backtracking_factor)
        self.armijo_constant = float(armijo_constant)
        self.max_trial_steps = int(max_trial_steps)
        self.step_tolerance = float(step_tolerance)

    def _regularized_direction(
        self,
        jacobian: np.ndarray,
        mismatch: np.ndarray,
        damping: float,
    ) -> tuple[np.ndarray, float]:
        """在牛顿方向不可用时计算正则化最小二乘方向。"""
        normal_matrix = jacobian.T @ jacobian
        gradient = jacobian.T @ mismatch
        scale = np.maximum(np.diag(normal_matrix), 1.0)
        current_damping = damping
        for _ in range(self.max_trial_steps):
            try:
                direction = np.linalg.solve(
                    normal_matrix + current_damping * np.diag(scale),
                    -gradient,
                )
            except np.linalg.LinAlgError:
                current_damping *= self.damping_increase
                continue
            if float(gradient @ direction) < 0.0:
                return direction, current_damping
            current_damping *= self.damping_increase
        raise np.linalg.LinAlgError("无法构造下降方向")

    def solve(self, grid):
        """求解交流潮流并返回收敛状态和优化过程信息。"""
        start_time = time.perf_counter()
        damping = self.initial_damping
        info = {
            "iterations": 0,
            "max_error_history": [],
            "objective_history": [],
            "step_length_history": [],
            "trial_step_length_history": [],
            "terminal_step_length": float("nan"),
            "damping_history": [],
            "regularized_steps": 0,
            "rejected_steps": 0,
            "time_elapsed": 0.0,
            "failure_reason": "",
            "timing_breakdown": {
                "mismatch_time": 0.0,
                "matrix_build_time": 0.0,
                "factorization_time": 0.0,
                "linear_solve_time": 0.0,
                "state_update_time": 0.0,
                "objective_evaluation_time": 0.0,
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
                    f"非线性规划法迭代 {iteration:2d}: "
                    f"最大失配={max_error:.6e}, 目标函数={objective:.6e}"
                )
            if max_error < self.tol:
                info["iterations"] = iteration
                info["time_elapsed"] = time.perf_counter() - start_time
                return True, info

            phase_start = time.perf_counter()
            jacobian, _, _ = grid.get_jacobian()
            gradient = jacobian.T @ mismatch
            timing["matrix_build_time"] += time.perf_counter() - phase_start

            used_regularization = False
            try:
                phase_start = time.perf_counter()
                direction = np.linalg.solve(jacobian, -mismatch)
                timing["linear_solve_time"] += time.perf_counter() - phase_start
                directional_derivative = float(gradient @ direction)
                if not np.all(np.isfinite(direction)) or directional_derivative >= 0.0:
                    raise np.linalg.LinAlgError("牛顿方向不是下降方向")
            except np.linalg.LinAlgError:
                phase_start = time.perf_counter()
                try:
                    direction, damping = self._regularized_direction(
                        jacobian, mismatch, damping
                    )
                except np.linalg.LinAlgError as exc:
                    info["iterations"] = iteration
                    info["failure_reason"] = f"优化方向求解失败：{exc}"
                    info["time_elapsed"] = time.perf_counter() - start_time
                    return False, info
                timing["linear_solve_time"] += time.perf_counter() - phase_start
                directional_derivative = float(gradient @ direction)
                used_regularization = True
                info["regularized_steps"] += 1

            voltage_direction, angle_direction = split_direction(
                grid, direction, theta_idx, v_idx
            )
            base_voltage = grid.V.copy()
            base_angle = grid.theta.copy()
            step_length = 1.0
            accepted = False

            for trial_index in range(self.max_trial_steps):
                info["trial_step_length_history"].append(step_length)
                info["terminal_step_length"] = step_length
                valid = set_trial_state(
                    grid,
                    base_voltage,
                    base_angle,
                    voltage_direction,
                    angle_direction,
                    step_length,
                )
                phase_start = time.perf_counter()
                trial_objective = (
                    mismatch_objective(grid, theta_idx, v_idx)
                    if valid
                    else float("inf")
                )
                timing["objective_evaluation_time"] += time.perf_counter() - phase_start
                armijo_bound = (
                    objective
                    + self.armijo_constant
                    * step_length
                    * directional_derivative
                )
                if trial_objective <= armijo_bound:
                    accepted = True
                    break
                grid.V[:] = base_voltage
                grid.theta[:] = base_angle
                step_length *= self.backtracking_factor
                info["rejected_steps"] += 1

            if not accepted:
                grid.V[:] = base_voltage
                grid.theta[:] = base_angle
                info["iterations"] = iteration + 1
                info["failure_reason"] = "回溯搜索无法找到使目标函数下降的步长"
                info["time_elapsed"] = time.perf_counter() - start_time
                return False, info

            info["step_length_history"].append(step_length)
            info["damping_history"].append(damping if used_regularization else 0.0)
            step_norm = step_length * float(np.linalg.norm(direction, ord=np.inf))
            if step_norm < self.step_tolerance and max_error >= self.tol:
                info["iterations"] = iteration + 1
                info["failure_reason"] = "优化步长过小但潮流失配尚未收敛"
                info["time_elapsed"] = time.perf_counter() - start_time
                return False, info

        info["iterations"] = self.max_iter
        info["failure_reason"] = "达到最大迭代次数"
        info["time_elapsed"] = time.perf_counter() - start_time
        return False, info
