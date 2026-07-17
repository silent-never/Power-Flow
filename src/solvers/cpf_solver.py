"""连续潮流预测—校正主求解器。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..core.grid import PowerGrid
from ..cpf.corrector import CorrectorResult, apply_corrector
from ..cpf.parameter import (
    ABSOLUTE_VP_ANGLE_PARAMETERIZATION,
    NATURAL_PARAMETERIZATION,
    PSEUDO_ARCLENGTH_PARAMETERIZATION,
    TANGENT_ANGLE_PARAMETERIZATION,
    CPFParameter,
    CPFParameterState,
)
from ..cpf.predictor import (
    CPFTangent,
    PredictorResult,
    apply_load_parameter,
    compute_predictor,
    enforce_reactive_power_limits,
    absolute_vp_angle,
)
from ..cpf.step_control import StepControlResult, adjust_step
from .base_solver import BaseSolver
from .nr_solver import NewtonRaphsonSolver


@dataclass(frozen=True, slots=True)
class CPFPoint:
    """CPF 曲线上一个已经收敛的运行点。"""

    step_index: int
    lambda_value: float
    load_multiplier: float
    voltage: np.ndarray
    theta: np.ndarray
    p_calc: np.ndarray
    q_calc: np.ndarray
    min_voltage: float
    min_voltage_bus: int
    min_pq_voltage: float
    min_pq_voltage_bus: int
    max_pq_voltage_drop_ratio: float
    max_pq_voltage_drop_bus: int
    q_limited_buses: tuple[int, ...]
    pv_tangent_angle_deg: float | None
    absolute_vp_angle_deg: float | None
    step_size: float
    corrector_iterations: int
    final_error: float
    jacobian_condition_number: float
    tangent: CPFTangent | None = None


@dataclass(frozen=True, slots=True)
class CPFResult:
    """一次完整 CPF 计算的轨迹与终止信息。"""

    points: tuple[CPFPoint, ...]
    step_decisions: tuple[StepControlResult, ...]
    success: bool
    stop_reason: str
    time_elapsed: float
    failed_attempts: int
    final_grid: PowerGrid
    parameterization: str
    enforce_q_limits: bool
    tangent_angle_bus: int | None
    absolute_vp_angle_bus: int | None
    used_pseudo_fallback: bool
    detected_nose_index: int | None

    @property
    def grid(self) -> PowerGrid:
        """返回最终电网，兼容通用结果报告接口。"""
        return self.final_grid

    @property
    def nose_point(self) -> CPFPoint:
        """返回首次检测到的鼻点；未跨越鼻点时返回最大 λ 点。"""
        if not self.points:
            raise ValueError("CPF 结果中没有可用运行点")
        if self.detected_nose_index is not None:
            return self.points[self.detected_nose_index]
        return max(self.points, key=lambda point: point.lambda_value)

    @property
    def max_lambda(self) -> float:
        """返回首次鼻点对应的 λ，保留原有属性名以兼容调用方。"""
        return self.nose_point.lambda_value

    @property
    def max_load_multiplier(self) -> float:
        """返回首次鼻点对应的统一负荷倍率。"""
        return self.nose_point.load_multiplier


def _build_point(
    grid: PowerGrid,
    reference_voltage: np.ndarray,
    step_index: int,
    lambda_value: float,
    step_size: float,
    corrector_iterations: int,
    final_error: float,
    tangent: CPFTangent | None,
    tangent_angle_bus_index: int | None,
    absolute_vp_angle_bus_index: int | None = None,
) -> CPFPoint:
    """从已收敛电网创建不可变的 CPF 轨迹点。"""
    grid.get_mismatch()
    jacobian, _, _ = grid.get_jacobian()
    try:
        condition_number = float(np.linalg.cond(jacobian))
    except np.linalg.LinAlgError:
        condition_number = float("inf")

    min_voltage_index = int(np.argmin(grid.V))
    pq_indices = np.flatnonzero(grid.bus_type == 1)
    if pq_indices.size == 0:
        pq_indices = np.arange(grid.n)

    min_pq_index = int(pq_indices[np.argmin(grid.V[pq_indices])])
    relative_drop = (
        reference_voltage[pq_indices] - grid.V[pq_indices]
    ) / reference_voltage[pq_indices]
    drop_position = int(np.argmax(relative_drop))
    drop_index = int(pq_indices[drop_position])
    max_drop = max(float(relative_drop[drop_position]), 0.0)
    q_limited_indices = getattr(grid, "cpf_q_limit_mvar", {})
    q_limited_buses = tuple(
        sorted(int(grid.buses[index]["number"]) for index in q_limited_indices)
    )
    if tangent is None or tangent_angle_bus_index is None:
        tangent_angle_deg = None
    else:
        tangent_dv = (
            grid.V[tangent_angle_bus_index]
            * tangent.d_v_over_v[tangent_angle_bus_index]
        )
        tangent_angle_deg = float(
            np.degrees(np.arctan2(tangent_dv, tangent.d_lambda))
        )
    if absolute_vp_angle_bus_index is None:
        absolute_angle_deg = None
    else:
        absolute_angle_deg = float(
            np.degrees(
                absolute_vp_angle(
                    grid,
                    lambda_value,
                    absolute_vp_angle_bus_index,
                )
            )
        )
    return CPFPoint(
        step_index=step_index,
        lambda_value=lambda_value,
        load_multiplier=1.0 + lambda_value,
        voltage=grid.V.copy(),
        theta=grid.theta.copy(),
        p_calc=grid.P_calc.copy(),
        q_calc=grid.Q_calc.copy(),
        min_voltage=float(grid.V[min_voltage_index]),
        min_voltage_bus=int(grid.buses[min_voltage_index]["number"]),
        min_pq_voltage=float(grid.V[min_pq_index]),
        min_pq_voltage_bus=int(grid.buses[min_pq_index]["number"]),
        max_pq_voltage_drop_ratio=max_drop,
        max_pq_voltage_drop_bus=(
            int(grid.buses[drop_index]["number"])
            if max_drop > 0.0
            else 0
        ),
        q_limited_buses=q_limited_buses,
        pv_tangent_angle_deg=tangent_angle_deg,
        absolute_vp_angle_deg=absolute_angle_deg,
        step_size=step_size,
        corrector_iterations=corrector_iterations,
        final_error=final_error,
        jacobian_condition_number=condition_number,
        tangent=tangent,
    )


def _failed_corrector_result(
    grid: PowerGrid,
    lambda_value: float,
    reason: str,
) -> CorrectorResult:
    """将预测阶段异常转换成统一的校正失败结果。"""
    return CorrectorResult(
        grid=grid,
        lambda_value=lambda_value,
        converged=False,
        iterations=0,
        max_error_history=(),
        constraint_history=(),
        reason=reason,
    )


class ContinuationSolver(BaseSolver):
    """组织基准潮流、预测、校正和步长控制的 CPF 求解器。"""

    def __init__(self, params: CPFParameter | None = None, **kwargs):
        super().__init__(**kwargs)
        self.params = params or CPFParameter()

    def _can_fallback_to_pseudo(
        self,
        parameter_state: CPFParameterState,
    ) -> bool:
        """判断当前角度参数化失败后是否允许伪弧长接管。"""
        if (
            parameter_state.active_parameterization
            == TANGENT_ANGLE_PARAMETERIZATION
        ):
            return self.params.tangent_angle_pseudo_fallback
        if (
            parameter_state.active_parameterization
            == ABSOLUTE_VP_ANGLE_PARAMETERIZATION
        ):
            return self.params.absolute_vp_angle_pseudo_fallback
        return False

    @staticmethod
    def _detect_first_nose_index(
        points: list[CPFPoint],
    ) -> int | None:
        """在最近三个收敛点中检测首次 λ 局部极大点。"""
        if len(points) < 3:
            return None

        previous_delta = (
            points[-2].lambda_value - points[-3].lambda_value
        )
        current_delta = (
            points[-1].lambda_value - points[-2].lambda_value
        )
        scale = max(
            1.0,
            abs(points[-3].lambda_value),
            abs(points[-2].lambda_value),
            abs(points[-1].lambda_value),
        )
        direction_tolerance = 1e-12 * scale
        if (
            previous_delta > direction_tolerance
            and current_delta < -direction_tolerance
        ):
            return len(points) - 2
        return None

    def _refine_angle_step(
        self,
        grid: PowerGrid,
        parameter_state: CPFParameterState,
        predictor: PredictorResult,
    ) -> StepControlResult | None:
        """在 P–V 切线接近竖直时限制角度参数化预测步长。"""
        settings = parameter_state.settings
        if (
            parameter_state.active_parameterization
            not in {
                TANGENT_ANGLE_PARAMETERIZATION,
                ABSOLUTE_VP_ANGLE_PARAMETERIZATION,
            }
            or not settings.tangent_angle_refinement
        ):
            return None

        if (
            parameter_state.active_parameterization
            == TANGENT_ANGLE_PARAMETERIZATION
        ):
            bus_index = parameter_state.tangent_angle_bus_index
        else:
            bus_index = parameter_state.absolute_vp_angle_bus_index
        if bus_index is None:
            return None
        tangent_p = predictor.tangent.d_lambda
        tangent_v = (
            grid.V[bus_index]
            * predictor.tangent.d_v_over_v[bus_index]
        )
        tangent_norm = np.hypot(tangent_p, tangent_v)
        if tangent_norm <= 0.0 or not np.isfinite(tangent_norm):
            return None

        abs_cos_phi = abs(tangent_p) / tangent_norm
        threshold = settings.tangent_angle_refinement_cos_threshold
        if abs_cos_phi >= threshold:
            return None

        local_ratio = max(
            settings.tangent_angle_refinement_min_step_ratio,
            abs_cos_phi / threshold,
        )
        target_step = max(
            settings.min_step,
            settings.max_step * local_ratio,
        )
        previous_step = parameter_state.step_size
        if target_step >= previous_step - 1e-15:
            return None

        parameter_state.set_step_size(target_step)
        phi_deg = float(np.degrees(np.arctan2(tangent_v, tangent_p)))
        return StepControlResult(
            previous_step=previous_step,
            new_step=parameter_state.step_size,
            action="decrease",
            should_retry=True,
            retry_count=parameter_state.retry_count,
            reason=(
                f"P–V 切线角 φ={phi_deg:.4f}° 接近鼻点，"
                f"将预测步长缩小到 {parameter_state.step_size:.6f}"
            ),
        )

    def _print_point(self, point: CPFPoint) -> None:
        """输出一个简洁的 CPF 收敛点摘要。"""
        if not self.verbose:
            return
        print(
            f"CPF {point.step_index:3d}: "
            f"λ={point.lambda_value:+.6f}, "
            f"倍率={point.load_multiplier:.6f}, "
            f"PQ最低电压={point.min_pq_voltage:.6f} "
            f"(母线 {point.min_pq_voltage_bus}), "
            f"PQ最大压降={point.max_pq_voltage_drop_ratio:.3%} "
            f"(母线 {point.max_pq_voltage_drop_bus or '无'}), "
            f"校正={point.corrector_iterations:2d} 次, "
            f"步长={point.step_size:.6f}"
        )
        if point.q_limited_buses:
            buses = ", ".join(str(bus) for bus in point.q_limited_buses)
            print(f"           已因无功越限转为 PQ 的母线: {buses}")
        if point.pv_tangent_angle_deg is not None:
            print(f"           P–V 切线角 φ={point.pv_tangent_angle_deg:.4f}°")
        if point.absolute_vp_angle_deg is not None:
            print(
                f"           绝对 V/P 角 α="
                f"{point.absolute_vp_angle_deg:.4f}°"
            )

    def _finish(
        self,
        points: list[CPFPoint],
        decisions: list[StepControlResult],
        success: bool,
        stop_reason: str,
        start_time: float,
        failed_attempts: int,
        final_grid: PowerGrid,
    ) -> tuple[bool, CPFResult]:
        """创建最终结果并输出终止摘要。"""
        result = CPFResult(
            points=tuple(points),
            step_decisions=tuple(decisions),
            success=success,
            stop_reason=stop_reason,
            time_elapsed=time.perf_counter() - start_time,
            failed_attempts=failed_attempts,
            final_grid=final_grid,
            parameterization=self.params.parameterization,
            enforce_q_limits=self.params.enforce_q_limits,
            tangent_angle_bus=(
                int(
                    final_grid.buses[
                        self._active_tangent_angle_bus_index
                    ]["number"]
                )
                if self._active_tangent_angle_bus_index is not None
                else None
            ),
            absolute_vp_angle_bus=(
                int(
                    final_grid.buses[
                        self._active_absolute_vp_angle_bus_index
                    ]["number"]
                )
                if self._active_absolute_vp_angle_bus_index is not None
                else None
            ),
            used_pseudo_fallback=self._used_pseudo_fallback,
            detected_nose_index=self._detected_nose_index,
        )
        if self.verbose:
            status = "完成" if success else "停止"
            print(
                f"CPF {status}: {stop_reason}；"
                f"收敛点={len(points)}, 失败尝试={failed_attempts}, "
                f"耗时={result.time_elapsed:.4f} 秒"
            )
            if points:
                nose = result.nose_point
                print(
                    f"当前最大负荷倍率={nose.load_multiplier:.6f}, "
                    f"λ={nose.lambda_value:.6f}, "
                    f"PQ最低电压={nose.min_pq_voltage:.6f}, "
                    f"PQ最大压降={nose.max_pq_voltage_drop_ratio:.3%}"
                )
        return success, result

    def solve(self, grid: PowerGrid) -> tuple[bool, CPFResult]:
        """执行完整 CPF 预测—校正循环。

        输入电网不会被修改。求解器首先在 ``lambda_init`` 处求取基准
        潮流，然后不断生成预测点、执行校正、调整步长并接受收敛点。
        """
        start_time = time.perf_counter()
        parameter_state = self.params.create_state()
        self._active_tangent_angle_bus_index: int | None = None
        self._active_absolute_vp_angle_bus_index: int | None = None
        self._used_pseudo_fallback = False
        self._detected_nose_index: int | None = None
        current_grid = grid.clone()
        apply_load_parameter(current_grid, parameter_state.lambda_value)

        if self.verbose:
            print(
                f"开始 CPF 计算，参数化方式={self.params.parameterization}, "
                f"初始 λ={parameter_state.lambda_value:.6f}"
            )

        initial_solver = NewtonRaphsonSolver(
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=False,
        )
        try:
            initial_success, initial_info = initial_solver.solve(current_grid)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            initial_success = False
            initial_info = {
                "iterations": 0,
                "max_error_history": (),
            }
            initial_reason = f"基准潮流异常: {type(exc).__name__}: {exc}"
        else:
            initial_reason = "基准潮流未收敛"

        if not initial_success:
            return self._finish(
                points=[],
                decisions=[],
                success=False,
                stop_reason=initial_reason,
                start_time=start_time,
                failed_attempts=1,
                final_grid=current_grid,
            )

        # 基准点也必须满足发电机无功限值；每次转换后重新求解，
        # 直到没有新的 PV 母线越限。
        q_limit_passes = (
            range(current_grid.n)
            if self.params.enforce_q_limits
            else range(0)
        )
        for _ in q_limit_passes:
            switched = enforce_reactive_power_limits(
                current_grid,
                parameter_state.lambda_value,
            )
            if not switched:
                break
            if self.verbose:
                for bus_number, fixed_q, limit_side in switched:
                    print(
                        f"CPF 无功越限: 母线 {bus_number} 触及{limit_side}，"
                        f"固定 Qg={fixed_q:.6f} Mvar 并转换为 PQ"
                    )
            try:
                initial_success, initial_info = initial_solver.solve(current_grid)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
                initial_success = False
                initial_reason = (
                    f"PV 转 PQ 后基准潮流异常: {type(exc).__name__}: {exc}"
                )
            else:
                initial_reason = "PV 转 PQ 后基准潮流未收敛"
            if not initial_success:
                return self._finish(
                    points=[],
                    decisions=[],
                    success=False,
                    stop_reason=initial_reason,
                    start_time=start_time,
                    failed_attempts=1,
                    final_grid=current_grid,
                )

        initial_errors = initial_info.get("max_error_history", ())
        initial_error = float(initial_errors[-1]) if initial_errors else 0.0
        points = [
            _build_point(
                current_grid,
                reference_voltage=current_grid.V.copy(),
                step_index=0,
                lambda_value=parameter_state.lambda_value,
                step_size=0.0,
                corrector_iterations=int(initial_info.get("iterations", 0)),
                final_error=initial_error,
                tangent=None,
                tangent_angle_bus_index=None,
            )
        ]
        self._print_point(points[-1])
        reference_voltage = points[0].voltage

        decisions: list[StepControlResult] = []
        failed_attempts = 0
        previous_tangent: CPFTangent | None = None

        while not parameter_state.reached_step_limit:
            if parameter_state.active_parameterization == NATURAL_PARAMETERIZATION:
                next_lambda = (
                    parameter_state.lambda_value + parameter_state.signed_step
                )
                if not self.params.contains_lambda(next_lambda):
                    return self._finish(
                        points,
                        decisions,
                        True,
                        "已到达 λ 配置边界",
                        start_time,
                        failed_attempts,
                        current_grid,
                    )

            predictor: PredictorResult
            try:
                predictor = compute_predictor(
                    current_grid,
                    parameter_state,
                    previous_tangent,
                )
                self._active_tangent_angle_bus_index = (
                    parameter_state.tangent_angle_bus_index
                )
                self._active_absolute_vp_angle_bus_index = (
                    parameter_state.absolute_vp_angle_bus_index
                )
            except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
                failed_attempts += 1
                failed = _failed_corrector_result(
                    current_grid,
                    parameter_state.lambda_value,
                    f"预测失败: {type(exc).__name__}: {exc}",
                )
                decision = adjust_step(parameter_state, failed)
                decisions.append(decision)
                if self.verbose:
                    print(f"CPF 步骤重试: {decision.reason}")
                if decision.should_retry:
                    continue
                if self._can_fallback_to_pseudo(parameter_state):
                    parameter_state.active_parameterization = (
                        PSEUDO_ARCLENGTH_PARAMETERIZATION
                    )
                    parameter_state.retry_count = 0
                    self._used_pseudo_fallback = True
                    fallback = StepControlResult(
                        previous_step=parameter_state.step_size,
                        new_step=parameter_state.step_size,
                        action="fallback",
                        should_retry=True,
                        retry_count=0,
                        reason=(
                            "角度投影在最小步长下仍无法预测，"
                            "切换为伪弧长参数化加密并跨越鼻点"
                        ),
                    )
                    decisions.append(fallback)
                    if self.verbose:
                        print(f"CPF 参数化切换: {fallback.reason}")
                    continue
                return self._finish(
                    points,
                    decisions,
                    False,
                    decision.reason,
                    start_time,
                    failed_attempts,
                    current_grid,
                )

            refinement = self._refine_angle_step(
                current_grid,
                parameter_state,
                predictor,
            )
            if refinement is not None:
                decisions.append(refinement)
                if self.verbose:
                    print(f"CPF 鼻点加密: {refinement.reason}")
                continue

            corrector = apply_corrector(predictor, parameter_state)
            decision = adjust_step(parameter_state, corrector)
            decisions.append(decision)

            if not corrector.converged:
                failed_attempts += 1
                if self.verbose:
                    print(f"CPF 步骤重试: {decision.reason}")
                if decision.should_retry:
                    continue
                if self._can_fallback_to_pseudo(parameter_state):
                    parameter_state.active_parameterization = (
                        PSEUDO_ARCLENGTH_PARAMETERIZATION
                    )
                    parameter_state.retry_count = 0
                    self._used_pseudo_fallback = True
                    fallback = StepControlResult(
                        previous_step=parameter_state.step_size,
                        new_step=parameter_state.step_size,
                        action="fallback",
                        should_retry=True,
                        retry_count=0,
                        reason=(
                            "角度投影在最小步长下仍无法校正，"
                            "切换为伪弧长参数化加密并跨越鼻点"
                        ),
                    )
                    decisions.append(fallback)
                    if self.verbose:
                        print(f"CPF 参数化切换: {fallback.reason}")
                    continue
                return self._finish(
                    points,
                    decisions,
                    False,
                    decision.reason,
                    start_time,
                    failed_attempts,
                    current_grid,
                )

            parameter_state.accept(corrector.lambda_value)
            current_grid = corrector.grid
            previous_tangent = predictor.tangent
            point = _build_point(
                current_grid,
                reference_voltage=reference_voltage,
                step_index=parameter_state.step_index,
                lambda_value=parameter_state.lambda_value,
                step_size=predictor.step_size,
                corrector_iterations=corrector.iterations,
                final_error=corrector.final_error,
                tangent=predictor.tangent,
                tangent_angle_bus_index=parameter_state.tangent_angle_bus_index,
                absolute_vp_angle_bus_index=(
                    parameter_state.absolute_vp_angle_bus_index
                ),
            )
            points.append(point)
            self._print_point(point)

            if self._detected_nose_index is None:
                self._detected_nose_index = self._detect_first_nose_index(
                    points
                )

            if (
                self.params.parameterization
                in {
                    PSEUDO_ARCLENGTH_PARAMETERIZATION,
                    TANGENT_ANGLE_PARAMETERIZATION,
                    ABSOLUTE_VP_ANGLE_PARAMETERIZATION,
                }
                and self.params.post_nose_steps is not None
            ):
                nose_index = self._detected_nose_index
                accepted_after_nose = (
                    len(points) - 1 - nose_index
                    if nose_index is not None
                    else 0
                )
                if (
                    nose_index is not None
                    and accepted_after_nose >= self.params.post_nose_steps
                ):
                    return self._finish(
                        points,
                        decisions,
                        True,
                        f"已越过鼻点并继续跟踪 "
                        f"{accepted_after_nose} 个收敛点",
                        start_time,
                        failed_attempts,
                        current_grid,
                    )

        return self._finish(
            points,
            decisions,
            True,
            "已达到最大 CPF 步数",
            start_time,
            failed_attempts,
            current_grid,
        )
