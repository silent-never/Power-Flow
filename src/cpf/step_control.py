"""连续潮流的自适应步长控制。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .corrector import CorrectorResult
from .parameter import CPFParameterState


STEP_INCREASED = "increase"
STEP_KEPT = "keep"
STEP_DECREASED = "decrease"
STEP_RETRY = "retry"
STEP_STOP = "stop"


@dataclass(frozen=True, slots=True)
class StepControlResult:
    """一次步长调整的决策结果。"""

    previous_step: float
    new_step: float
    action: str
    should_retry: bool
    retry_count: int
    reason: str

    @property
    def changed(self) -> bool:
        """判断本次决策是否改变了步长。"""
        return not math.isclose(
            self.previous_step,
            self.new_step,
            rel_tol=0.0,
            abs_tol=1e-15,
        )


def _successful_step(
    parameter_state: CPFParameterState,
    corrector_result: CorrectorResult,
) -> StepControlResult:
    """根据校正迭代次数调整下一个成功步的步长。"""
    settings = parameter_state.settings
    previous_step = parameter_state.step_size
    parameter_state.retry_count = 0

    if corrector_result.iterations <= settings.fast_convergence_iters:
        candidate = previous_step * settings.step_increase_factor
        new_step = settings.clamp_step(candidate)
        parameter_state.set_step_size(new_step)
        if math.isclose(new_step, previous_step, rel_tol=0.0, abs_tol=1e-15):
            action = STEP_KEPT
            reason = "校正收敛较快，但步长已达上限"
        else:
            action = STEP_INCREASED
            reason = "校正收敛较快，增大下一步步长"

    elif corrector_result.iterations >= settings.slow_convergence_iters:
        candidate = previous_step * settings.step_decrease_factor
        new_step = settings.clamp_step(candidate)
        parameter_state.set_step_size(new_step)
        if math.isclose(new_step, previous_step, rel_tol=0.0, abs_tol=1e-15):
            action = STEP_KEPT
            reason = "校正收敛较慢，但步长已达下限"
        else:
            action = STEP_DECREASED
            reason = "校正收敛较慢，减小下一步步长"

    else:
        new_step = previous_step
        action = STEP_KEPT
        reason = "校正迭代次数适中，保持当前步长"

    return StepControlResult(
        previous_step=previous_step,
        new_step=new_step,
        action=action,
        should_retry=False,
        retry_count=parameter_state.retry_count,
        reason=reason,
    )


def _failed_step(
    parameter_state: CPFParameterState,
    corrector_result: CorrectorResult,
) -> StepControlResult:
    """在校正失败后减小步长，并决定是否重试。"""
    settings = parameter_state.settings
    previous_step = parameter_state.step_size
    parameter_state.retry_count += 1
    candidate = previous_step * settings.step_decrease_factor
    new_step = settings.clamp_step(candidate)
    parameter_state.set_step_size(new_step)

    already_at_minimum = math.isclose(
        previous_step,
        settings.min_step,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    retries_exhausted = parameter_state.retry_count > settings.max_step_retries
    should_retry = not already_at_minimum and not retries_exhausted

    if retries_exhausted:
        action = STEP_STOP
        reason = (
            f"校正失败且已超过最大重试次数 "
            f"{settings.max_step_retries}"
        )
    elif already_at_minimum:
        action = STEP_STOP
        reason = "校正失败且当前已是最小步长"
    else:
        action = STEP_RETRY
        detail = corrector_result.reason or "校正未收敛"
        reason = f"{detail}；减小步长后重试当前延拓步"

    return StepControlResult(
        previous_step=previous_step,
        new_step=new_step,
        action=action,
        should_retry=should_retry,
        retry_count=parameter_state.retry_count,
        reason=reason,
    )


def adjust_step(
    parameter_state: CPFParameterState,
    corrector_result: CorrectorResult,
) -> StepControlResult:
    """根据校正结果调整 CPF 步长。

    该函数会直接更新 ``parameter_state.step_size`` 和失败
    重试计数，同时返回一个不可变的决策记录供日志使用。
    """
    if corrector_result.converged:
        return _successful_step(parameter_state, corrector_result)
    return _failed_step(parameter_state, corrector_result)
