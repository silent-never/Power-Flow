"""连续潮流预测点的 Newton 校正。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.grid import PowerGrid
from .parameter import (
    ABSOLUTE_VP_ANGLE_PARAMETERIZATION,
    LOCAL_VOLTAGE_PARAMETERIZATION,
    NATURAL_PARAMETERIZATION,
    PSEUDO_ARCLENGTH_PARAMETERIZATION,
    TANGENT_ANGLE_PARAMETERIZATION,
    CPFParameterState,
)
from .predictor import (
    PredictorResult,
    apply_load_parameter,
    build_arclength_weights,
    build_lambda_derivative,
    enforce_reactive_power_limits,
    absolute_vp_angle,
)


@dataclass(frozen=True, slots=True)
class CorrectorResult:
    """一次 CPF 校正步的结果。"""

    grid: PowerGrid
    lambda_value: float
    converged: bool
    iterations: int
    max_error_history: tuple[float, ...]
    constraint_history: tuple[float, ...]
    reason: str = ""

    @property
    def load_multiplier(self) -> float:
        """返回校正点对应的统一负荷倍率。"""
        return 1.0 + self.lambda_value

    @property
    def final_error(self) -> float:
        """返回最后一次校正迭代的最大误差。"""
        if not self.max_error_history:
            return float("nan")
        return self.max_error_history[-1]


def _reduced_mismatch(
    d_p: np.ndarray,
    d_q: np.ndarray,
    theta_indices: list[int],
    voltage_indices: list[int],
) -> np.ndarray:
    """按照雅可比矩阵的方程顺序组装降维不平衡向量。"""
    return np.concatenate([d_p[theta_indices], d_q[voltage_indices]])


def _parameter_constraint(
    grid: PowerGrid,
    lambda_value: float,
    predictor: PredictorResult,
    theta_indices: list[int],
    voltage_indices: list[int],
) -> float:
    """计算当前点相对于预测超平面的伪弧长约束误差。"""
    delta_theta = grid.theta - predictor.grid.theta
    delta_v_over_v = np.zeros(grid.n)
    delta_v_over_v[voltage_indices] = (
        grid.V[voltage_indices] - predictor.grid.V[voltage_indices]
    ) / predictor.grid.V[voltage_indices]
    delta_lambda = lambda_value - predictor.lambda_value

    tangent = predictor.tangent
    tangent_vector = tangent.reduced_vector(theta_indices, voltage_indices)
    displacement = np.concatenate(
        [
            delta_theta[theta_indices],
            delta_v_over_v[voltage_indices],
            np.array([delta_lambda]),
        ]
    )
    weights = build_arclength_weights(theta_indices, voltage_indices)
    return float(np.dot(weights * tangent_vector, displacement))


def _tangent_angle_components(
    predictor: PredictorResult,
    parameter_state: CPFParameterState,
) -> tuple[int, float, float]:
    """返回监测母线索引和归一化 P–V 切线分量。"""
    bus_index = parameter_state.tangent_angle_bus_index
    if bus_index is None:
        raise ValueError("P–V 切线角母线尚未确定")
    tangent_p = predictor.tangent.d_lambda
    tangent_v = (
        predictor.grid.V[bus_index]
        * predictor.tangent.d_v_over_v[bus_index]
    )
    tangent_norm = np.hypot(tangent_p, tangent_v)
    if tangent_norm <= 0.0 or not np.isfinite(tangent_norm):
        raise ValueError("预测 P–V 切线方向无效")
    return bus_index, tangent_p / tangent_norm, tangent_v / tangent_norm


def _tangent_angle_constraint(
    grid: PowerGrid,
    lambda_value: float,
    predictor: PredictorResult,
    parameter_state: CPFParameterState,
) -> float:
    """计算当前点相对于预测点 P–V 切线法面的约束误差。"""
    bus_index, tangent_p, tangent_v = _tangent_angle_components(
        predictor,
        parameter_state,
    )
    delta_p = lambda_value - predictor.lambda_value
    delta_v = grid.V[bus_index] - predictor.grid.V[bus_index]
    return float(tangent_p * delta_p + tangent_v * delta_v)


def _local_voltage_constraint(
    grid: PowerGrid,
    predictor: PredictorResult,
    parameter_state: CPFParameterState,
) -> float:
    """计算选定母线电压与预测目标电压之间的约束误差。"""
    bus_index = parameter_state.local_voltage_bus_index
    if bus_index is None:
        raise ValueError("局部电压参数化母线尚未确定")
    return float(grid.V[bus_index] - predictor.grid.V[bus_index])


def _absolute_vp_angle_constraint(
    grid: PowerGrid,
    lambda_value: float,
    predictor: PredictorResult,
    parameter_state: CPFParameterState,
) -> float:
    """计算绝对角 α=atan2(V, P/P0) 与目标角之间的误差。"""
    bus_index = parameter_state.absolute_vp_angle_bus_index
    if bus_index is None:
        raise ValueError("绝对 V/P 角母线尚未确定")
    if predictor.parameter_target is None:
        raise ValueError("绝对 V/P 角目标值尚未确定")
    current_angle = absolute_vp_angle(grid, lambda_value, bus_index)
    return current_angle - predictor.parameter_target


def _solve_natural_correction(
    jacobian: np.ndarray,
    mismatch: np.ndarray,
) -> tuple[np.ndarray, float]:
    """求解固定 λ 条件下的普通 Newton 校正量。"""
    state_correction = np.linalg.solve(jacobian, -mismatch)
    return state_correction, 0.0


def _solve_pseudo_arclength_correction(
    grid: PowerGrid,
    jacobian: np.ndarray,
    mismatch: np.ndarray,
    constraint: float,
    predictor: PredictorResult,
    theta_indices: list[int],
    voltage_indices: list[int],
) -> tuple[np.ndarray, float]:
    """求解同时修正状态变量和 λ 的增广 Newton 方程。"""
    lambda_derivative = build_lambda_derivative(
        grid,
        theta_indices,
        voltage_indices,
    )
    tangent = predictor.tangent.reduced_vector(
        theta_indices,
        voltage_indices,
    )
    weights = build_arclength_weights(theta_indices, voltage_indices)

    variable_count = jacobian.shape[0]
    augmented_matrix = np.zeros(
        (variable_count + 1, variable_count + 1),
        dtype=float,
    )
    augmented_matrix[:variable_count, :variable_count] = jacobian
    augmented_matrix[:variable_count, -1] = lambda_derivative
    augmented_matrix[-1, :] = weights * tangent

    right_hand_side = -np.concatenate(
        [mismatch, np.array([constraint], dtype=float)]
    )
    correction = np.linalg.solve(augmented_matrix, right_hand_side)
    return correction[:-1], float(correction[-1])


def _solve_tangent_angle_correction(
    grid: PowerGrid,
    jacobian: np.ndarray,
    mismatch: np.ndarray,
    constraint: float,
    predictor: PredictorResult,
    parameter_state: CPFParameterState,
    theta_indices: list[int],
    voltage_indices: list[int],
) -> tuple[np.ndarray, float]:
    """求解带 P–V 切线法面约束的增广 Newton 方程。"""
    bus_index, tangent_p, tangent_v = _tangent_angle_components(
        predictor,
        parameter_state,
    )
    if bus_index not in voltage_indices:
        raise ValueError("P–V 切线角母线不是 PQ 母线")

    lambda_derivative = build_lambda_derivative(
        grid,
        theta_indices,
        voltage_indices,
    )
    variable_count = jacobian.shape[0]
    augmented_matrix = np.zeros(
        (variable_count + 1, variable_count + 1),
        dtype=float,
    )
    augmented_matrix[:variable_count, :variable_count] = jacobian
    augmented_matrix[:variable_count, -1] = lambda_derivative
    voltage_position = len(theta_indices) + voltage_indices.index(bus_index)
    augmented_matrix[-1, voltage_position] = tangent_v * grid.V[bus_index]
    augmented_matrix[-1, -1] = tangent_p

    right_hand_side = -np.concatenate(
        [mismatch, np.array([constraint], dtype=float)]
    )
    correction = np.linalg.solve(augmented_matrix, right_hand_side)
    return correction[:-1], float(correction[-1])


def _solve_local_voltage_correction(
    grid: PowerGrid,
    jacobian: np.ndarray,
    mismatch: np.ndarray,
    constraint: float,
    parameter_state: CPFParameterState,
    theta_indices: list[int],
    voltage_indices: list[int],
) -> tuple[np.ndarray, float]:
    """求解固定局部母线电压的增广 Newton 校正方程。"""
    bus_index = parameter_state.local_voltage_bus_index
    if bus_index is None or bus_index not in voltage_indices:
        raise ValueError("局部电压参数化母线不是 PQ 母线")
    lambda_derivative = build_lambda_derivative(
        grid,
        theta_indices,
        voltage_indices,
    )
    variable_count = jacobian.shape[0]
    augmented_matrix = np.zeros(
        (variable_count + 1, variable_count + 1), dtype=float
    )
    augmented_matrix[:variable_count, :variable_count] = jacobian
    augmented_matrix[:variable_count, -1] = lambda_derivative
    voltage_position = len(theta_indices) + voltage_indices.index(bus_index)
    augmented_matrix[-1, voltage_position] = grid.V[bus_index]
    right_hand_side = -np.concatenate(
        [mismatch, np.array([constraint], dtype=float)]
    )
    correction = np.linalg.solve(augmented_matrix, right_hand_side)
    return correction[:-1], float(correction[-1])


def _solve_absolute_vp_angle_correction(
    grid: PowerGrid,
    lambda_value: float,
    jacobian: np.ndarray,
    mismatch: np.ndarray,
    constraint: float,
    parameter_state: CPFParameterState,
    theta_indices: list[int],
    voltage_indices: list[int],
) -> tuple[np.ndarray, float]:
    """求解带绝对 V/P 角约束的增广 Newton 方程。"""
    bus_index = parameter_state.absolute_vp_angle_bus_index
    if bus_index is None or bus_index not in voltage_indices:
        raise ValueError("绝对 V/P 角母线不是 PQ 母线")

    lambda_derivative = build_lambda_derivative(
        grid,
        theta_indices,
        voltage_indices,
    )
    load_multiplier = 1.0 + lambda_value
    voltage = grid.V[bus_index]
    denominator = load_multiplier**2 + voltage**2
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("绝对 V/P 角的坐标无效")

    variable_count = jacobian.shape[0]
    augmented_matrix = np.zeros(
        (variable_count + 1, variable_count + 1),
        dtype=float,
    )
    augmented_matrix[:variable_count, :variable_count] = jacobian
    augmented_matrix[:variable_count, -1] = lambda_derivative
    voltage_position = len(theta_indices) + voltage_indices.index(bus_index)
    augmented_matrix[-1, voltage_position] = (
        load_multiplier * voltage / denominator
    )
    augmented_matrix[-1, -1] = -voltage / denominator

    right_hand_side = -np.concatenate(
        [mismatch, np.array([constraint], dtype=float)]
    )
    correction = np.linalg.solve(augmented_matrix, right_hand_side)
    return correction[:-1], float(correction[-1])


def _apply_correction(
    grid: PowerGrid,
    state_correction: np.ndarray,
    theta_indices: list[int],
    voltage_indices: list[int],
) -> None:
    """将降维 Newton 校正量应用到完整电网状态。"""
    theta_count = len(theta_indices)
    d_theta = np.zeros(grid.n)
    d_v_over_v = np.zeros(grid.n)
    d_theta[theta_indices] = state_correction[:theta_count]
    d_v_over_v[voltage_indices] = state_correction[theta_count:]
    grid.update_state(d_v_over_v, d_theta)


def apply_corrector(
    predictor: PredictorResult,
    parameter_state: CPFParameterState,
) -> CorrectorResult:
    """使用预测点作为初值执行 CPF 校正迭代。

    自然参数化保持预测 λ 不变，只校正电压和相角；伪弧长参数化
    使用预测切向量建立超平面约束，同时校正电压、相角和 λ。
    传入的预测结果不会被修改。
    """
    settings = parameter_state.settings
    parameterization = parameter_state.active_parameterization
    grid = predictor.grid.clone()
    lambda_value = predictor.lambda_value
    error_history: list[float] = []
    constraint_history: list[float] = []

    for iteration in range(settings.max_corrector_iters + 1):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                d_p, d_q = grid.get_mismatch()
                if settings.enforce_q_limits:
                    switched_buses = enforce_reactive_power_limits(
                        grid,
                        lambda_value,
                    )
                else:
                    switched_buses = ()
                if switched_buses:
                    d_p, d_q = grid.get_mismatch()
                jacobian, theta_indices, voltage_indices = grid.get_jacobian()
                mismatch = _reduced_mismatch(
                    d_p,
                    d_q,
                    theta_indices,
                    voltage_indices,
                )

                if parameterization == NATURAL_PARAMETERIZATION:
                    constraint = 0.0
                elif parameterization == LOCAL_VOLTAGE_PARAMETERIZATION:
                    constraint = _local_voltage_constraint(
                        grid,
                        predictor,
                        parameter_state,
                    )
                elif parameterization == PSEUDO_ARCLENGTH_PARAMETERIZATION:
                    constraint = _parameter_constraint(
                        grid,
                        lambda_value,
                        predictor,
                        theta_indices,
                        voltage_indices,
                    )
                elif parameterization == TANGENT_ANGLE_PARAMETERIZATION:
                    constraint = _tangent_angle_constraint(
                        grid,
                        lambda_value,
                        predictor,
                        parameter_state,
                    )
                elif parameterization == ABSOLUTE_VP_ANGLE_PARAMETERIZATION:
                    constraint = _absolute_vp_angle_constraint(
                        grid,
                        lambda_value,
                        predictor,
                        parameter_state,
                    )
                else:
                    raise ValueError(
                        f"不支持的 CPF 参数化方式: {parameterization}"
                    )

                max_mismatch = (
                    float(np.max(np.abs(mismatch)))
                    if mismatch.size
                    else 0.0
                )
                max_error = max(max_mismatch, abs(constraint))
                error_history.append(max_error)
                constraint_history.append(constraint)

                if max_error < settings.corrector_tol:
                    return CorrectorResult(
                        grid=grid,
                        lambda_value=lambda_value,
                        converged=True,
                        iterations=iteration,
                        max_error_history=tuple(error_history),
                        constraint_history=tuple(constraint_history),
                    )

                if iteration >= settings.max_corrector_iters:
                    break

                if parameterization == NATURAL_PARAMETERIZATION:
                    state_correction, lambda_correction = (
                        _solve_natural_correction(jacobian, mismatch)
                    )
                elif parameterization == LOCAL_VOLTAGE_PARAMETERIZATION:
                    state_correction, lambda_correction = (
                        _solve_local_voltage_correction(
                            grid,
                            jacobian,
                            mismatch,
                            constraint,
                            parameter_state,
                            theta_indices,
                            voltage_indices,
                        )
                    )
                elif parameterization == PSEUDO_ARCLENGTH_PARAMETERIZATION:
                    state_correction, lambda_correction = (
                        _solve_pseudo_arclength_correction(
                            grid,
                            jacobian,
                            mismatch,
                            constraint,
                            predictor,
                            theta_indices,
                            voltage_indices,
                        )
                    )
                elif parameterization == TANGENT_ANGLE_PARAMETERIZATION:
                    state_correction, lambda_correction = (
                        _solve_tangent_angle_correction(
                            grid,
                            jacobian,
                            mismatch,
                            constraint,
                            predictor,
                            parameter_state,
                            theta_indices,
                            voltage_indices,
                        )
                    )
                elif parameterization == ABSOLUTE_VP_ANGLE_PARAMETERIZATION:
                    state_correction, lambda_correction = (
                        _solve_absolute_vp_angle_correction(
                            grid,
                            lambda_value,
                            jacobian,
                            mismatch,
                            constraint,
                            parameter_state,
                            theta_indices,
                            voltage_indices,
                        )
                    )
                else:
                    raise ValueError(
                        f"不支持的 CPF 参数化方式: {parameterization}"
                    )

                _apply_correction(
                    grid,
                    state_correction,
                    theta_indices,
                    voltage_indices,
                )
                lambda_value += lambda_correction
                if not settings.contains_lambda(lambda_value):
                    return CorrectorResult(
                        grid=grid,
                        lambda_value=lambda_value,
                        converged=False,
                        iterations=iteration + 1,
                        max_error_history=tuple(error_history),
                        constraint_history=tuple(constraint_history),
                        reason="校正后的 λ 超出配置允许范围",
                    )
                apply_load_parameter(grid, lambda_value)

        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            return CorrectorResult(
                grid=grid,
                lambda_value=lambda_value,
                converged=False,
                iterations=iteration,
                max_error_history=tuple(error_history),
                constraint_history=tuple(constraint_history),
                reason=f"{type(exc).__name__}: {exc}",
            )

    return CorrectorResult(
        grid=grid,
        lambda_value=lambda_value,
        converged=False,
        iterations=settings.max_corrector_iters,
        max_error_history=tuple(error_history),
        constraint_history=tuple(constraint_history),
        reason="达到最大校正迭代次数后仍未收敛",
    )
