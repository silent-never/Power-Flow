"""连续潮流预测步的切向量计算与状态外推。"""

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


@dataclass(frozen=True, slots=True)
class CPFTangent:
    """CPF 曲线上一个点的完整切向量。"""

    d_theta: np.ndarray
    d_v_over_v: np.ndarray
    d_lambda: float

    def reduced_vector(
        self,
        theta_indices: list[int],
        voltage_indices: list[int],
    ) -> np.ndarray:
        """按照雅可比矩阵的变量顺序返回降维切向量。"""
        return np.concatenate(
            [
                self.d_theta[theta_indices],
                self.d_v_over_v[voltage_indices],
                np.array([self.d_lambda], dtype=float),
            ]
        )

    @property
    def norm(self) -> float:
        """返回完整切向量的二范数。"""
        vector = np.concatenate(
            [self.d_theta, self.d_v_over_v, np.array([self.d_lambda])]
        )
        return float(np.linalg.norm(vector))


@dataclass(frozen=True, slots=True)
class PredictorResult:
    """一次 CPF 预测步的输出结果。"""

    grid: PowerGrid
    lambda_value: float
    tangent: CPFTangent
    step_size: float
    parameter_target: float | None = None

    @property
    def load_multiplier(self) -> float:
        """返回预测点对应的统一负荷倍率。"""
        return 1.0 + self.lambda_value


def load_growth_components(bus: dict) -> tuple[float, float]:
    """返回母线参与 CPF 增长的有功、无功负荷分量。

    负有功负荷表示固定等值注入，不随 λ 放大，其配套无功也保持
    不变。正有功负荷的 P/Q 按原功率因数同步增长；纯无功正负荷
    同样参与增长，而纯无功补偿保持固定。
    """
    p_load = float(bus.get("load_mw", 0.0))
    q_load = float(bus.get("load_mvar", 0.0))
    if p_load < 0.0 or (p_load == 0.0 and q_load <= 0.0):
        return 0.0, 0.0
    return p_load, q_load


def build_arclength_weights(
    theta_indices: list[int],
    voltage_indices: list[int],
) -> np.ndarray:
    """构造与系统规模无关的伪弧长内积权重。

    相角组和电压组分别使用组内均方值，避免节点数增加时仅因
    状态变量更多就压低 λ 分量；λ 本身保持单位权重。
    """
    theta_weight = 1.0 / max(len(theta_indices), 1)
    voltage_weight = 1.0 / max(len(voltage_indices), 1)
    return np.concatenate(
        [
            np.full(len(theta_indices), theta_weight),
            np.full(len(voltage_indices), voltage_weight),
            np.ones(1),
        ]
    )


def apply_load_parameter(grid: PowerGrid, lambda_value: float) -> None:
    """根据 λ 更新电网的有功和无功给定值。

    ``grid.buses`` 中的负荷始终保存基准工况数据，只有 ``P_spec`` 和
    ``Q_spec`` 随 λ 改变，避免连续放大已经缩放过的负荷。负有功
    负荷按固定等值注入处理，不参与增长。
    """
    multiplier = 1.0 + float(lambda_value)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("负荷倍率 1 + λ 必须是大于 0 的有限数")

    q_limits = getattr(grid, "cpf_q_limit_mvar", {})
    for bus in grid.buses:
        index = grid.idx_map[bus["number"]]
        p_gen = float(bus.get("gen_mw", 0.0))
        q_gen = float(
            q_limits.get(index, bus.get("gen_mvar", 0.0))
        )
        p_load = float(bus.get("load_mw", 0.0))
        q_load = float(bus.get("load_mvar", 0.0))
        p_growth, q_growth = load_growth_components(bus)

        grid.P_spec[index] = (
            p_gen - p_load - float(lambda_value) * p_growth
        ) / grid.base_mva
        grid.Q_spec[index] = (
            q_gen - q_load - float(lambda_value) * q_growth
        ) / grid.base_mva


def enforce_reactive_power_limits(
    grid: PowerGrid,
    lambda_value: float,
    tolerance_mvar: float = 1e-6,
) -> tuple[tuple[int, float, str], ...]:
    """检查 PV 母线无功限值，并将越限母线单向转换为 PQ。

    返回值包含母线编号、固定无功出力和触发的上下限方向。已经
    转换的母线不会自动恢复为 PV，以避免临界点附近反复切换。
    """
    grid.get_mismatch()
    q_limits = dict(getattr(grid, "cpf_q_limit_mvar", {}))
    switched: list[tuple[int, float, str]] = []

    for bus in grid.buses:
        index = grid.idx_map[bus["number"]]
        if grid.bus_type[index] != 2:
            continue

        q_min = float(bus.get("min_q_v", 0.0))
        q_max = float(bus.get("max_q_v", 0.0))
        if not np.isfinite(q_min) or not np.isfinite(q_max) or q_min > q_max:
            continue

        _, q_growth = load_growth_components(bus)
        q_load = float(bus.get("load_mvar", 0.0))
        actual_q_load = q_load + float(lambda_value) * q_growth
        required_q_generation = (
            grid.Q_calc[index] * grid.base_mva + actual_q_load
        )

        if required_q_generation > q_max + tolerance_mvar:
            fixed_q = q_max
            limit_side = "上限"
        elif required_q_generation < q_min - tolerance_mvar:
            fixed_q = q_min
            limit_side = "下限"
        else:
            continue

        grid.bus_type[index] = 1
        q_limits[index] = fixed_q
        switched.append((int(bus["number"]), fixed_q, limit_side))

    if switched:
        grid.cpf_q_limit_mvar = q_limits
        apply_load_parameter(grid, lambda_value)

    return tuple(switched)


def build_lambda_derivative(
    grid: PowerGrid,
    theta_indices: list[int],
    voltage_indices: list[int],
) -> np.ndarray:
    """构造潮流不平衡量对 λ 的偏导列向量。

    当前采用正负荷等比例增长、负有功负荷固定的方向，因此：

        ∂P_spec/∂λ = -P_growth/base_mva
        ∂Q_spec/∂λ = -Q_growth/base_mva
    """
    d_p_d_lambda = np.zeros(grid.n)
    d_q_d_lambda = np.zeros(grid.n)

    for bus in grid.buses:
        index = grid.idx_map[bus["number"]]
        p_growth, q_growth = load_growth_components(bus)
        d_p_d_lambda[index] = -p_growth / grid.base_mva
        d_q_d_lambda[index] = -q_growth / grid.base_mva

    return np.concatenate(
        [d_p_d_lambda[theta_indices], d_q_d_lambda[voltage_indices]]
    )


def _expand_tangent(
    grid: PowerGrid,
    reduced_state_tangent: np.ndarray,
    d_lambda: float,
    theta_indices: list[int],
    voltage_indices: list[int],
) -> CPFTangent:
    """将降维求解结果展开为与全部母线对应的切向量。"""
    theta_count = len(theta_indices)
    d_theta = np.zeros(grid.n)
    d_v_over_v = np.zeros(grid.n)
    d_theta[theta_indices] = reduced_state_tangent[:theta_count]
    d_v_over_v[voltage_indices] = reduced_state_tangent[theta_count:]
    return CPFTangent(d_theta, d_v_over_v, float(d_lambda))


def _natural_tangent(
    grid: PowerGrid,
    jacobian: np.ndarray,
    lambda_derivative: np.ndarray,
    direction: int,
    theta_indices: list[int],
    voltage_indices: list[int],
) -> CPFTangent:
    """使用自然参数化计算以 λ 为参数的切向量。"""
    d_lambda = float(direction)
    state_tangent = np.linalg.solve(
        jacobian,
        -lambda_derivative * d_lambda,
    )
    return _expand_tangent(
        grid,
        state_tangent,
        d_lambda,
        theta_indices,
        voltage_indices,
    )


def _pseudo_arclength_tangent(
    grid: PowerGrid,
    jacobian: np.ndarray,
    lambda_derivative: np.ndarray,
    direction: int,
    theta_indices: list[int],
    voltage_indices: list[int],
    previous_tangent: CPFTangent | None,
) -> CPFTangent:
    """使用上一切向量约束计算归一化伪弧长切向量。"""
    weights = build_arclength_weights(theta_indices, voltage_indices)
    if previous_tangent is None:
        natural = _natural_tangent(
            grid,
            jacobian,
            lambda_derivative,
            direction,
            theta_indices,
            voltage_indices,
        )
        augmented = natural.reduced_vector(theta_indices, voltage_indices)
    else:
        previous = previous_tangent.reduced_vector(
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
        augmented_matrix[-1, :] = weights * previous

        right_hand_side = np.zeros(variable_count + 1)
        right_hand_side[-1] = 1.0
        augmented = np.linalg.solve(augmented_matrix, right_hand_side)

        if np.dot(weights * augmented, previous) < 0.0:
            augmented *= -1.0

    tangent_norm = np.sqrt(np.dot(weights, augmented * augmented))
    if not np.isfinite(tangent_norm) or tangent_norm <= 0.0:
        raise ValueError("无法对 CPF 切向量进行归一化")
    augmented /= tangent_norm

    return _expand_tangent(
        grid,
        augmented[:-1],
        augmented[-1],
        theta_indices,
        voltage_indices,
    )


def _resolve_tangent_angle_bus_index(
    grid: PowerGrid,
    parameter_state: CPFParameterState,
    voltage_indices: list[int],
    natural_tangent: CPFTangent,
) -> int:
    """解析并保存 P–V 切线角所使用的 PQ 母线索引。"""
    if parameter_state.tangent_angle_bus_index is not None:
        bus_index = parameter_state.tangent_angle_bus_index
    elif parameter_state.settings.tangent_angle_bus > 0:
        bus_number = parameter_state.settings.tangent_angle_bus
        if bus_number not in grid.idx_map:
            raise ValueError(f"找不到 P–V 切线角母线: {bus_number}")
        bus_index = grid.idx_map[bus_number]
    else:
        sensitivities = np.abs(
            grid.V[voltage_indices]
            * natural_tangent.d_v_over_v[voltage_indices]
        )
        if sensitivities.size == 0:
            raise ValueError("系统中没有可用于 P–V 切线角的 PQ 母线")
        bus_index = voltage_indices[int(np.argmax(sensitivities))]

    if bus_index not in voltage_indices:
        bus_number = int(grid.buses[bus_index]["number"])
        raise ValueError(
            f"P–V 切线角母线 {bus_number} 必须是 PQ 母线"
        )
    parameter_state.tangent_angle_bus_index = bus_index
    return bus_index


def _resolve_local_voltage_bus_index(
    grid: PowerGrid,
    parameter_state: CPFParameterState,
    voltage_indices: list[int],
    natural_tangent: CPFTangent,
) -> int:
    """解析并保存局部电压参数化使用的 PQ 母线索引。"""
    if parameter_state.local_voltage_bus_index is not None:
        bus_index = parameter_state.local_voltage_bus_index
    elif parameter_state.settings.local_voltage_bus > 0:
        bus_number = parameter_state.settings.local_voltage_bus
        if bus_number not in grid.idx_map:
            raise ValueError(f"找不到局部电压参数化母线: {bus_number}")
        bus_index = grid.idx_map[bus_number]
    else:
        sensitivities = np.abs(
            grid.V[voltage_indices]
            * natural_tangent.d_v_over_v[voltage_indices]
        )
        if sensitivities.size == 0:
            raise ValueError("系统中没有可用于局部参数化的 PQ 母线")
        bus_index = voltage_indices[int(np.argmax(sensitivities))]

    if bus_index not in voltage_indices:
        bus_number = int(grid.buses[bus_index]["number"])
        raise ValueError(f"局部参数化母线 {bus_number} 必须是 PQ 母线")
    parameter_state.local_voltage_bus_index = bus_index
    return bus_index


def _local_voltage_tangent(
    grid: PowerGrid,
    jacobian: np.ndarray,
    lambda_derivative: np.ndarray,
    parameter_state: CPFParameterState,
    theta_indices: list[int],
    voltage_indices: list[int],
) -> CPFTangent:
    """以选定 PQ 母线的电压幅值为局部延拓参数计算切向量。"""
    natural = _natural_tangent(
        grid,
        jacobian,
        lambda_derivative,
        parameter_state.direction,
        theta_indices,
        voltage_indices,
    )
    bus_index = _resolve_local_voltage_bus_index(
        grid,
        parameter_state,
        voltage_indices,
        natural,
    )
    variable_count = jacobian.shape[0]
    augmented_matrix = np.zeros(
        (variable_count + 1, variable_count + 1), dtype=float
    )
    augmented_matrix[:variable_count, :variable_count] = jacobian
    augmented_matrix[:variable_count, -1] = lambda_derivative
    voltage_position = len(theta_indices) + voltage_indices.index(bus_index)
    # 潮流状态采用 dV/V；乘以当前 V 后，末行固定实际电压变化率 dV/ds。
    augmented_matrix[-1, voltage_position] = grid.V[bus_index]
    right_hand_side = np.zeros(variable_count + 1)
    right_hand_side[-1] = -float(parameter_state.direction)
    augmented = np.linalg.solve(augmented_matrix, right_hand_side)
    return _expand_tangent(
        grid,
        augmented[:-1],
        augmented[-1],
        theta_indices,
        voltage_indices,
    )


def _resolve_absolute_vp_angle_bus_index(
    grid: PowerGrid,
    parameter_state: CPFParameterState,
    voltage_indices: list[int],
    natural_tangent: CPFTangent,
) -> int:
    """解析绝对 V/P 角参数化使用的 PQ 母线索引。"""
    if parameter_state.absolute_vp_angle_bus_index is not None:
        bus_index = parameter_state.absolute_vp_angle_bus_index
    elif parameter_state.settings.absolute_vp_angle_bus > 0:
        bus_number = parameter_state.settings.absolute_vp_angle_bus
        if bus_number not in grid.idx_map:
            raise ValueError(f"找不到绝对 V/P 角母线: {bus_number}")
        bus_index = grid.idx_map[bus_number]
    else:
        sensitivities = np.abs(
            grid.V[voltage_indices]
            * natural_tangent.d_v_over_v[voltage_indices]
        )
        if sensitivities.size == 0:
            raise ValueError("系统中没有可用于绝对 V/P 角的 PQ 母线")
        bus_index = voltage_indices[int(np.argmax(sensitivities))]

    if bus_index not in voltage_indices:
        bus_number = int(grid.buses[bus_index]["number"])
        raise ValueError(f"绝对 V/P 角母线 {bus_number} 必须是 PQ 母线")
    parameter_state.absolute_vp_angle_bus_index = bus_index
    return bus_index


def absolute_vp_angle(
    grid: PowerGrid,
    lambda_value: float,
    bus_index: int,
) -> float:
    """计算 atan2(V, P/P0)，其中 P/P0 等于统一负荷倍率。"""
    load_multiplier = 1.0 + float(lambda_value)
    return float(np.arctan2(grid.V[bus_index], load_multiplier))


def _absolute_vp_angle_tangent(
    grid: PowerGrid,
    jacobian: np.ndarray,
    lambda_derivative: np.ndarray,
    parameter_state: CPFParameterState,
    theta_indices: list[int],
    voltage_indices: list[int],
) -> CPFTangent:
    """令绝对角 α=atan2(V, P/P0) 等速变化并求解 CPF 切向量。"""
    natural = _natural_tangent(
        grid,
        jacobian,
        lambda_derivative,
        parameter_state.direction,
        theta_indices,
        voltage_indices,
    )
    bus_index = _resolve_absolute_vp_angle_bus_index(
        grid,
        parameter_state,
        voltage_indices,
        natural,
    )

    load_multiplier = 1.0 + parameter_state.lambda_value
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

    # 状态变量采用相对电压增量 dV/V，因此角度对该变量的偏导
    # 需要在 dα/dV 的基础上再乘当前电压 V。
    augmented_matrix[-1, voltage_position] = (
        load_multiplier * voltage / denominator
    )
    augmented_matrix[-1, -1] = -voltage / denominator

    right_hand_side = np.zeros(variable_count + 1)
    # 初始方向为 +1 时负荷增加、绝对 V/P 角减小。
    right_hand_side[-1] = -float(parameter_state.direction)
    augmented = np.linalg.solve(augmented_matrix, right_hand_side)
    return _expand_tangent(
        grid,
        augmented[:-1],
        augmented[-1],
        theta_indices,
        voltage_indices,
    )


def _tangent_angle_tangent(
    grid: PowerGrid,
    jacobian: np.ndarray,
    lambda_derivative: np.ndarray,
    parameter_state: CPFParameterState,
    theta_indices: list[int],
    voltage_indices: list[int],
    previous_tangent: CPFTangent | None,
) -> CPFTangent:
    """在归一化负荷 P/P0–V 平面中计算单位切线方向。"""
    if previous_tangent is None:
        natural = _natural_tangent(
            grid,
            jacobian,
            lambda_derivative,
            parameter_state.direction,
            theta_indices,
            voltage_indices,
        )
        bus_index = _resolve_tangent_angle_bus_index(
            grid,
            parameter_state,
            voltage_indices,
            natural,
        )
        augmented = natural.reduced_vector(theta_indices, voltage_indices)
    else:
        bus_index = parameter_state.tangent_angle_bus_index
        if bus_index is None or bus_index not in voltage_indices:
            raise ValueError("P–V 切线角母线未确定或不再是 PQ 母线")

        previous_p = previous_tangent.d_lambda
        previous_v = (
            grid.V[bus_index]
            * previous_tangent.d_v_over_v[bus_index]
        )
        previous_norm = np.hypot(previous_p, previous_v)
        if previous_norm <= 0.0 or not np.isfinite(previous_norm):
            raise ValueError("上一 P–V 切线方向无效")
        previous_p /= previous_norm
        previous_v /= previous_norm

        variable_count = jacobian.shape[0]
        augmented_matrix = np.zeros(
            (variable_count + 1, variable_count + 1),
            dtype=float,
        )
        augmented_matrix[:variable_count, :variable_count] = jacobian
        augmented_matrix[:variable_count, -1] = lambda_derivative
        voltage_position = (
            len(theta_indices) + voltage_indices.index(bus_index)
        )
        augmented_matrix[-1, voltage_position] = (
            previous_v * grid.V[bus_index]
        )
        augmented_matrix[-1, -1] = previous_p

        right_hand_side = np.zeros(variable_count + 1)
        right_hand_side[-1] = 1.0
        augmented = np.linalg.solve(augmented_matrix, right_hand_side)

        current_p = augmented[-1]
        current_v = grid.V[bus_index] * augmented[voltage_position]
        if previous_p * current_p + previous_v * current_v < 0.0:
            augmented *= -1.0

    voltage_position = len(theta_indices) + voltage_indices.index(bus_index)
    tangent_p = augmented[-1]
    tangent_v = grid.V[bus_index] * augmented[voltage_position]
    tangent_norm = np.hypot(tangent_p, tangent_v)
    if tangent_norm <= 0.0 or not np.isfinite(tangent_norm):
        raise ValueError("无法归一化 P–V 切线角方向")
    augmented /= tangent_norm

    return _expand_tangent(
        grid,
        augmented[:-1],
        augmented[-1],
        theta_indices,
        voltage_indices,
    )


def compute_tangent(
    grid: PowerGrid,
    parameter_state: CPFParameterState,
    previous_tangent: CPFTangent | None = None,
) -> CPFTangent:
    """在当前已收敛潮流点计算 CPF 切向量。"""
    grid.get_mismatch()
    jacobian, theta_indices, voltage_indices = grid.get_jacobian()
    lambda_derivative = build_lambda_derivative(
        grid,
        theta_indices,
        voltage_indices,
    )

    parameterization = parameter_state.active_parameterization
    if parameterization == NATURAL_PARAMETERIZATION:
        return _natural_tangent(
            grid,
            jacobian,
            lambda_derivative,
            parameter_state.direction,
            theta_indices,
            voltage_indices,
        )
    if parameterization == LOCAL_VOLTAGE_PARAMETERIZATION:
        return _local_voltage_tangent(
            grid,
            jacobian,
            lambda_derivative,
            parameter_state,
            theta_indices,
            voltage_indices,
        )
    if parameterization == PSEUDO_ARCLENGTH_PARAMETERIZATION:
        return _pseudo_arclength_tangent(
            grid,
            jacobian,
            lambda_derivative,
            parameter_state.direction,
            theta_indices,
            voltage_indices,
            previous_tangent,
        )
    if parameterization == TANGENT_ANGLE_PARAMETERIZATION:
        return _tangent_angle_tangent(
            grid,
            jacobian,
            lambda_derivative,
            parameter_state,
            theta_indices,
            voltage_indices,
            previous_tangent,
        )
    if parameterization == ABSOLUTE_VP_ANGLE_PARAMETERIZATION:
        return _absolute_vp_angle_tangent(
            grid,
            jacobian,
            lambda_derivative,
            parameter_state,
            theta_indices,
            voltage_indices,
        )
    raise ValueError(f"不支持的 CPF 参数化方式: {parameterization}")


def compute_predictor(
    grid: PowerGrid,
    parameter_state: CPFParameterState,
    previous_tangent: CPFTangent | None = None,
) -> PredictorResult:
    """计算切向量并生成下一步预测点。

    当前已收敛电网和参数状态不会被修改，预测结果保存在新的电网副本中。
    """
    tangent = compute_tangent(grid, parameter_state, previous_tangent)
    step_size = parameter_state.step_size
    parameter_target = None
    if (
        parameter_state.active_parameterization
        == ABSOLUTE_VP_ANGLE_PARAMETERIZATION
    ):
        bus_index = parameter_state.absolute_vp_angle_bus_index
        if bus_index is None:
            raise ValueError("绝对 V/P 角母线尚未确定")
        current_angle = absolute_vp_angle(
            grid,
            parameter_state.lambda_value,
            bus_index,
        )
        parameter_target = (
            current_angle
            - parameter_state.direction * step_size
        )
    predicted_lambda = (
        parameter_state.lambda_value + step_size * tangent.d_lambda
    )
    if not parameter_state.settings.contains_lambda(predicted_lambda):
        raise ValueError("预测点 λ 超出配置允许范围")

    predicted_grid = grid.clone()
    predicted_grid.update_state(
        tangent.d_v_over_v * step_size,
        tangent.d_theta * step_size,
    )
    apply_load_parameter(predicted_grid, predicted_lambda)

    return PredictorResult(
        grid=predicted_grid,
        lambda_value=predicted_lambda,
        tangent=tangent,
        step_size=step_size,
        parameter_target=parameter_target,
    )
