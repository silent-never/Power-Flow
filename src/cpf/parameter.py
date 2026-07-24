"""连续潮流的延拓参数与运行状态管理。

本模块统一采用以下负荷增长定义：

    负荷倍率 = 1 + λ

因此，λ = 0 表示基准负荷，λ = 0.2 表示负荷增长 20%。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


NATURAL_PARAMETERIZATION = "natural"
LOCAL_VOLTAGE_PARAMETERIZATION = "local_voltage"
PSEUDO_ARCLENGTH_PARAMETERIZATION = "pseudo_arclength"
TANGENT_ANGLE_PARAMETERIZATION = "tangent_angle"
ABSOLUTE_VP_ANGLE_PARAMETERIZATION = "absolute_vp_angle"
SUPPORTED_PARAMETERIZATIONS = frozenset(
    {
        NATURAL_PARAMETERIZATION,
        LOCAL_VOLTAGE_PARAMETERIZATION,
        PSEUDO_ARCLENGTH_PARAMETERIZATION,
        TANGENT_ANGLE_PARAMETERIZATION,
        ABSOLUTE_VP_ANGLE_PARAMETERIZATION,
    }
)


@dataclass(frozen=True, slots=True)
class CPFParameter:
    """CPF 的固定配置参数。

    该类只保存一次 CPF 任务期间不应改变的配置。
    当前的 λ、当前步长和步数由 ``CPFParameterState`` 管理。
    """

    lambda_init: float = 0.0
    lambda_min: float = -0.99
    lambda_max: float = 2.0

    step_size: float = 0.02
    min_step: float = 1e-4
    max_step: float = 0.1
    step_increase_factor: float = 1.25
    step_decrease_factor: float = 0.5
    fast_convergence_iters: int = 4
    slow_convergence_iters: int = 15
    max_step_retries: int = 8

    corrector_tol: float = 1e-8
    max_corrector_iters: int = 15
    max_steps: int = 500
    post_nose_steps: int | None = 20

    parameterization: str = NATURAL_PARAMETERIZATION
    initial_direction: int = 1
    enforce_q_limits: bool = True
    local_voltage_bus: int = 0
    tangent_angle_bus: int = 0
    tangent_angle_refinement: bool = True
    tangent_angle_refinement_cos_threshold: float = 0.35
    tangent_angle_refinement_min_step_ratio: float = 0.02
    tangent_angle_full_state_cos_threshold: float = 0.20
    tangent_angle_stop_at_second_lambda_turn: bool = True
    tangent_angle_pseudo_fallback: bool = True
    absolute_vp_angle_bus: int = 0
    absolute_vp_angle_pseudo_fallback: bool = True

    def __post_init__(self) -> None:
        """在创建配置时立即检查参数合法性。"""
        finite_values = {
            "lambda_init": self.lambda_init,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "step_size": self.step_size,
            "min_step": self.min_step,
            "max_step": self.max_step,
            "step_increase_factor": self.step_increase_factor,
            "step_decrease_factor": self.step_decrease_factor,
            "corrector_tol": self.corrector_tol,
            "tangent_angle_refinement_cos_threshold": (
                self.tangent_angle_refinement_cos_threshold
            ),
            "tangent_angle_refinement_min_step_ratio": (
                self.tangent_angle_refinement_min_step_ratio
            ),
            "tangent_angle_full_state_cos_threshold": (
                self.tangent_angle_full_state_cos_threshold
            ),
        }
        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} 必须是有限数")

        if self.lambda_min >= self.lambda_max:
            raise ValueError("lambda_min 必须小于 lambda_max")
        if not self.lambda_min <= self.lambda_init <= self.lambda_max:
            raise ValueError("lambda_init 必须位于 lambda_min 与 lambda_max 之间")
        if 1.0 + self.lambda_min <= 0.0:
            raise ValueError("lambda_min 必须保证负荷倍率 1 + λ 大于 0")

        if self.min_step <= 0.0:
            raise ValueError("min_step 必须大于 0")
        if self.max_step < self.min_step:
            raise ValueError("max_step 不能小于 min_step")
        if not self.min_step <= self.step_size <= self.max_step:
            raise ValueError("step_size 必须位于 min_step 与 max_step 之间")
        if self.step_increase_factor <= 1.0:
            raise ValueError("step_increase_factor 必须大于 1")
        if not 0.0 < self.step_decrease_factor < 1.0:
            raise ValueError("step_decrease_factor 必须位于 0 与 1 之间")
        if self.fast_convergence_iters < 0:
            raise ValueError("fast_convergence_iters 不能小于 0")
        if self.slow_convergence_iters <= self.fast_convergence_iters:
            raise ValueError(
                "slow_convergence_iters 必须大于 fast_convergence_iters"
            )
        if self.max_step_retries < 0:
            raise ValueError("max_step_retries 不能小于 0")

        if self.corrector_tol <= 0.0:
            raise ValueError("corrector_tol 必须大于 0")
        if self.max_corrector_iters <= 0:
            raise ValueError("max_corrector_iters 必须大于 0")
        if self.max_steps <= 0:
            raise ValueError("max_steps 必须大于 0")
        if self.post_nose_steps is not None and self.post_nose_steps <= 0:
            raise ValueError("post_nose_steps 必须大于 0 或设为 None")

        if self.parameterization not in SUPPORTED_PARAMETERIZATIONS:
            supported = ", ".join(sorted(SUPPORTED_PARAMETERIZATIONS))
            raise ValueError(f"parameterization 仅支持: {supported}")
        if self.initial_direction not in (-1, 1):
            raise ValueError("initial_direction 只能是 -1 或 1")
        if not isinstance(self.enforce_q_limits, bool):
            raise ValueError("enforce_q_limits 必须是布尔值")
        if self.tangent_angle_bus < 0:
            raise ValueError("tangent_angle_bus 不能小于 0")
        if self.local_voltage_bus < 0:
            raise ValueError("local_voltage_bus 不能小于 0")
        if self.absolute_vp_angle_bus < 0:
            raise ValueError("absolute_vp_angle_bus 不能小于 0")
        if not isinstance(self.tangent_angle_refinement, bool):
            raise ValueError("tangent_angle_refinement 必须是布尔值")
        if not isinstance(self.tangent_angle_pseudo_fallback, bool):
            raise ValueError("tangent_angle_pseudo_fallback 必须是布尔值")
        if not isinstance(
            self.tangent_angle_stop_at_second_lambda_turn,
            bool,
        ):
            raise ValueError(
                "tangent_angle_stop_at_second_lambda_turn 必须是布尔值"
            )
        if not isinstance(self.absolute_vp_angle_pseudo_fallback, bool):
            raise ValueError("absolute_vp_angle_pseudo_fallback 必须是布尔值")
        if not 0.0 < self.tangent_angle_refinement_cos_threshold <= 1.0:
            raise ValueError(
                "tangent_angle_refinement_cos_threshold 必须位于 0 和 1 之间"
            )
        if not 0.0 < self.tangent_angle_refinement_min_step_ratio <= 1.0:
            raise ValueError(
                "tangent_angle_refinement_min_step_ratio 必须位于 0 和 1 之间"
            )
        if not 0.0 <= self.tangent_angle_full_state_cos_threshold <= 1.0:
            raise ValueError(
                "tangent_angle_full_state_cos_threshold 必须位于 0 和 1 之间"
            )

    def load_multiplier(self, lambda_value: float | None = None) -> float:
        """将延拓参数 λ 转换为统一负荷倍率。"""
        value = self.lambda_init if lambda_value is None else float(lambda_value)
        if not math.isfinite(value):
            raise ValueError("lambda_value 必须是有限数")

        multiplier = 1.0 + value
        if multiplier <= 0.0:
            raise ValueError("负荷倍率 1 + λ 必须大于 0")
        return multiplier

    def lambda_from_multiplier(self, multiplier: float) -> float:
        """将统一负荷倍率转换为延拓参数 λ。"""
        value = float(multiplier)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("multiplier 必须是大于 0 的有限数")
        return value - 1.0

    def clamp_step(self, step_size: float) -> float:
        """将给定步长限制在配置的最小与最大步长之间。"""
        value = float(step_size)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("step_size 必须是大于 0 的有限数")
        return min(max(value, self.min_step), self.max_step)

    def contains_lambda(self, lambda_value: float) -> bool:
        """判断给定 λ 是否位于允许范围内。"""
        value = float(lambda_value)
        return math.isfinite(value) and self.lambda_min <= value <= self.lambda_max

    def create_state(self) -> "CPFParameterState":
        """使用当前配置创建 CPF 运行状态。"""
        return CPFParameterState(
            settings=self,
            lambda_value=self.lambda_init,
            step_size=self.step_size,
            direction=self.initial_direction,
        )


@dataclass(slots=True)
class CPFParameterState:
    """CPF 运行过程中会变化的延拓参数状态。"""

    settings: CPFParameter
    lambda_value: float
    step_size: float
    direction: int = 1
    step_index: int = 0
    retry_count: int = 0
    local_voltage_bus_index: int | None = None
    tangent_angle_bus_index: int | None = None
    absolute_vp_angle_bus_index: int | None = None
    active_parameterization: str | None = None

    def __post_init__(self) -> None:
        """检查运行状态，并将步长约束到允许范围。"""
        if not self.settings.contains_lambda(self.lambda_value):
            raise ValueError("lambda_value 超出配置的 λ 范围")
        if self.direction not in (-1, 1):
            raise ValueError("direction 只能是 -1 或 1")
        if self.step_index < 0:
            raise ValueError("step_index 不能小于 0")
        if self.retry_count < 0:
            raise ValueError("retry_count 不能小于 0")
        if self.active_parameterization is None:
            self.active_parameterization = self.settings.parameterization
        if self.active_parameterization not in SUPPORTED_PARAMETERIZATIONS:
            raise ValueError("active_parameterization 不受支持")
        self.step_size = self.settings.clamp_step(self.step_size)

    @property
    def load_multiplier(self) -> float:
        """返回当前 λ 对应的统一负荷倍率。"""
        return self.settings.load_multiplier(self.lambda_value)

    @property
    def signed_step(self) -> float:
        """返回带延拓方向的有符号步长。"""
        return self.direction * self.step_size

    @property
    def reached_step_limit(self) -> bool:
        """判断是否已达到允许的最大延拓步数。"""
        return self.step_index >= self.settings.max_steps

    def set_step_size(self, step_size: float) -> None:
        """更新步长，并自动应用上下限约束。"""
        self.step_size = self.settings.clamp_step(step_size)

    def reverse_direction(self) -> None:
        """反转延拓方向，用于越过鼻点后跟踪下半支。"""
        self.direction *= -1

    def accept(self, lambda_value: float) -> None:
        """接受一个已收敛的校正点，并推进步数。"""
        value = float(lambda_value)
        if not self.settings.contains_lambda(value):
            raise ValueError("新的 lambda_value 超出配置的 λ 范围")
        self.lambda_value = value
        self.step_index += 1
        self.retry_count = 0
