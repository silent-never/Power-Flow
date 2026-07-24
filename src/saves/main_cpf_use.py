"""Power-Flow 项目的 CPF 连续潮流入口。"""

import csv
from dataclasses import replace
from pathlib import Path

from src.core.grid import PowerGrid
from src.cpf.parameter import (
    ABSOLUTE_VP_ANGLE_PARAMETERIZATION,
    LOCAL_VOLTAGE_PARAMETERIZATION,
    NATURAL_PARAMETERIZATION,
    PSEUDO_ARCLENGTH_PARAMETERIZATION,
    TANGENT_ANGLE_PARAMETERIZATION,
    CPFParameter,
)
from src.io.parser import parse_dat_txt
from src.solvers.cpf_solver import ContinuationSolver
from src.utils.config import load_config
from src.visualization.cpf_plotter import CPFPlotter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PARAMETERIZATION_LABELS = {
    NATURAL_PARAMETERIZATION: "自然参数化",
    LOCAL_VOLTAGE_PARAMETERIZATION: "局部电压参数化",
    PSEUDO_ARCLENGTH_PARAMETERIZATION: "伪弧长参数化",
    TANGENT_ANGLE_PARAMETERIZATION: "切线角参数化",
    ABSOLUTE_VP_ANGLE_PARAMETERIZATION: "绝对 V/P 角参数化",
}


def create_cpf_parameters(config):
    """将全局配置转换为 CPF 专用参数。"""
    return CPFParameter(
        lambda_init=config.cpf_lambda_init,
        lambda_min=config.cpf_lambda_min,
        lambda_max=config.cpf_lambda_max,
        step_size=config.cpf_step_size,
        min_step=config.cpf_min_step,
        max_step=config.cpf_max_step,
        step_increase_factor=config.cpf_step_increase_factor,
        step_decrease_factor=config.cpf_step_decrease_factor,
        fast_convergence_iters=config.cpf_fast_convergence_iters,
        slow_convergence_iters=config.cpf_slow_convergence_iters,
        max_step_retries=config.cpf_max_step_retries,
        corrector_tol=config.cpf_corrector_tolerance,
        max_corrector_iters=config.cpf_max_corrector_iterations,
        max_steps=config.cpf_max_steps,
        post_nose_steps=config.cpf_post_nose_steps,
        parameterization=config.cpf_parameterization,
        initial_direction=config.cpf_initial_direction,
        enforce_q_limits=config.cpf_enforce_q_limits,
        local_voltage_bus=config.cpf_local_voltage_bus,
        tangent_angle_bus=config.cpf_tangent_angle_bus,
        tangent_angle_refinement=config.cpf_tangent_angle_refinement,
        tangent_angle_refinement_cos_threshold=(
            config.cpf_tangent_angle_refinement_cos_threshold
        ),
        tangent_angle_refinement_min_step_ratio=(
            config.cpf_tangent_angle_refinement_min_step_ratio
        ),
        tangent_angle_full_state_cos_threshold=(
            config.cpf_tangent_angle_full_state_cos_threshold
        ),
        tangent_angle_stop_at_second_lambda_turn=(
            config.cpf_tangent_angle_stop_at_second_lambda_turn
        ),
        tangent_angle_pseudo_fallback=(
            config.cpf_tangent_angle_pseudo_fallback
        ),
        absolute_vp_angle_bus=config.cpf_absolute_vp_angle_bus,
        absolute_vp_angle_pseudo_fallback=(
            config.cpf_absolute_vp_angle_pseudo_fallback
        ),
    )


def print_cpf_summary(cpf_success, cpf_result, grid):
    """输出 CPF 计算状态和鼻点摘要。"""
    print("\n" + "=" * 64)
    print("                    CPF 连续潮流结果")
    print("-" * 64)
    print(f"计算状态       : {'成功' if cpf_success else '未完成'}")
    print(f"停止原因       : {cpf_result.stop_reason}")
    print(f"参数化方式     : {cpf_result.parameterization}")
    print(
        f"PV→PQ 转换    : "
        f"{'开启' if cpf_result.enforce_q_limits else '关闭'}"
    )
    if cpf_result.tangent_angle_bus is not None:
        print(f"P–V 切线角母线 : {cpf_result.tangent_angle_bus}")
    if cpf_result.local_voltage_bus is not None:
        print(f"局部参数化母线 : {cpf_result.local_voltage_bus}")
    if cpf_result.absolute_vp_angle_bus is not None:
        print(f"绝对 V/P 角母线: {cpf_result.absolute_vp_angle_bus}")
    print(
        f"伪弧长自动接管 : "
        f"{'是' if cpf_result.used_pseudo_fallback else '否'}"
    )
    print(f"收敛轨迹点数   : {len(cpf_result.points)}")
    print(f"失败尝试次数   : {cpf_result.failed_attempts}")
    print(f"总耗时         : {cpf_result.time_elapsed:.4f} 秒")

    if cpf_result.points:
        nose = cpf_result.nose_point
        base_load_mw = sum(
            float(bus.get("load_mw", 0.0)) for bus in grid.buses
        )
        base_load_mvar = sum(
            float(bus.get("load_mvar", 0.0)) for bus in grid.buses
        )
        print("-" * 64)
        print(f"鼻点所在步数   : {nose.step_index}")
        print(f"鼻点 λ         : {nose.lambda_value:.8f}")
        print(f"鼻点负荷倍率   : {nose.load_multiplier:.8f}")
        print(
            f"鼻点总负荷     : "
            f"{base_load_mw * nose.load_multiplier:.2f} MW / "
            f"{base_load_mvar * nose.load_multiplier:.2f} Mvar"
        )
        print(
            f"鼻点 PQ 最低电压: {nose.min_pq_voltage:.6f} p.u. "
            f"(母线 {nose.min_pq_voltage_bus})"
        )
        print(
            f"鼻点 PQ 最大压降: "
            f"{nose.max_pq_voltage_drop_ratio:.4%} "
            f"(母线 {nose.max_pq_voltage_drop_bus or '无'})"
        )
        limited_buses = nose.q_limited_buses
        print(
            f"鼻点无功越限母线: "
            f"{', '.join(map(str, limited_buses)) if limited_buses else '无'}"
        )
        if nose.pv_tangent_angle_deg is not None:
            print(f"鼻点 P–V 切线角: {nose.pv_tangent_angle_deg:.4f}°")
        if nose.absolute_vp_angle_deg is not None:
            print(f"鼻点绝对 V/P 角: {nose.absolute_vp_angle_deg:.4f}°")
        print(f"鼻点雅可比条件数: {nose.jacobian_condition_number:.6e}")
    print("=" * 64)


def run_cpf(grid, config):
    """运行连续潮流并返回求解结果。"""
    solver = ContinuationSolver(
        params=create_cpf_parameters(config),
        tol=config.cpf_corrector_tolerance,
        max_iter=config.max_iterations,
        verbose=config.cpf_verbose,
    )
    return solver.solve(grid)


def _case_sort_key(path: Path) -> int:
    """从 IEEE 算例文件名中提取节点数。"""
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits) if digits else 0


def run_all_cpf_cases(config) -> None:
    """遍历 data 中全部 IEEE 算例，每个系统仅保存一张 P--V 曲线。"""
    case_paths = sorted(
        (PROJECT_ROOT / "data").glob("*ieee.txt"),
        key=_case_sort_key,
    )
    if not case_paths:
        raise FileNotFoundError("data 目录中没有找到 *ieee.txt 算例")

    # 不同算例的母线编号集合不同，批处理统一自动选择敏感 PQ 母线。
    batch_config = replace(
        config,
        cpf_local_voltage_bus=0,
        cpf_tangent_angle_bus=0,
        cpf_absolute_vp_angle_bus=0,
        cpf_monitor_bus=0,
        cpf_verbose=False,
    )
    parameterization_label = PARAMETERIZATION_LABELS[
        batch_config.cpf_parameterization
    ]
    output_root = (
        PROJECT_ROOT
        / config.plot_dir
        / "cpf_all_cases_pv_curve"
        / batch_config.cpf_parameterization
    )
    summary_rows = []

    print("=" * 72)
    print(f"全部 IEEE 节点系统 CPF 批处理：{parameterization_label}")
    print(f"算例数量：{len(case_paths)}")
    print(f"图片根目录：{output_root}")
    print("=" * 72)

    for case_index, case_path in enumerate(case_paths, start=1):
        case_number = _case_sort_key(case_path)
        case_label = f"IEEE_{case_number:03d}"
        print(
            f"[{case_index:02d}/{len(case_paths):02d}] "
            f"正在计算 {case_label}: {case_path.name}"
        )
        try:
            buses, branches = parse_dat_txt(case_path)
            grid = PowerGrid(
                buses,
                branches,
                base_mva=batch_config.base_mva,
            )
            cpf_success, cpf_result = run_cpf(grid, batch_config)
            if cpf_result.points:
                output_files = (
                    CPFPlotter(output_root).plot_pv_curve(
                        cpf_result,
                        save_name=f"{case_label}_CPF_01_PV_Curve.png",
                    ),
                )
                nose = cpf_result.nose_point
                nose_multiplier = nose.load_multiplier
                nose_voltage = nose.min_pq_voltage
            else:
                output_files = ()
                nose_multiplier = float("nan")
                nose_voltage = float("nan")
            summary_rows.append(
                {
                    "case": case_label,
                    "file": case_path.name,
                    "nodes": len(buses),
                    "success": cpf_success,
                    "accepted_points": len(cpf_result.points),
                    "failed_attempts": cpf_result.failed_attempts,
                    "nose_detected": (
                        cpf_result.detected_nose_index is not None
                    ),
                    "nose_load_multiplier": nose_multiplier,
                    "nose_min_pq_voltage": nose_voltage,
                    "used_pseudo_fallback": (
                        cpf_result.used_pseudo_fallback
                    ),
                    "elapsed_seconds": cpf_result.time_elapsed,
                    "image_count": len(output_files),
                    "stop_reason": cpf_result.stop_reason,
                    "error": "",
                }
            )
            print(
                f"  状态={'成功' if cpf_success else '未完成'}，"
                f"轨迹点={len(cpf_result.points)}，"
                f"鼻点倍率={nose_multiplier:.8f}，"
                f"图片={len(output_files)}"
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "case": case_label,
                    "file": case_path.name,
                    "nodes": case_number,
                    "success": False,
                    "accepted_points": 0,
                    "failed_attempts": 0,
                    "nose_detected": False,
                    "nose_load_multiplier": float("nan"),
                    "nose_min_pq_voltage": float("nan"),
                    "used_pseudo_fallback": False,
                    "elapsed_seconds": float("nan"),
                    "image_count": 0,
                    "stop_reason": "批处理异常",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  [失败] {type(exc).__name__}: {exc}")

    summary_path = (
        PROJECT_ROOT
        / "output"
        / "tables"
        / (
            "CPF_All_Cases_PV_Curve_"
            f"{batch_config.cpf_parameterization}_Summary.csv"
        )
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=summary_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    successful_count = sum(bool(row["success"]) for row in summary_rows)
    image_count = sum(int(row["image_count"]) for row in summary_rows)
    print("=" * 72)
    print(
        f"批处理结束：{successful_count}/{len(summary_rows)} 个算例完成，"
        f"共生成 {image_count} 张图片"
    )
    print(f"汇总表：{summary_path}")
    print(f"图片根目录：{output_root}")
    print("=" * 72)


def main():
    """读取配置并使用其中指定的参数化方式执行 CPF。"""
    config = load_config(PROJECT_ROOT / "config.yaml")
    if config.cpf_all_cases:
        run_all_cpf_cases(config)
        return
    file_path = PROJECT_ROOT / config.data_file
    parameterization_label = PARAMETERIZATION_LABELS[config.cpf_parameterization]

    print("=" * 64)
    print("                    CPF 连续潮流")
    print("=" * 64)
    print(f"参数化方式       : {parameterization_label}")
    configured_bus = {
        LOCAL_VOLTAGE_PARAMETERIZATION: config.cpf_local_voltage_bus,
        TANGENT_ANGLE_PARAMETERIZATION: config.cpf_tangent_angle_bus,
        ABSOLUTE_VP_ANGLE_PARAMETERIZATION: config.cpf_absolute_vp_angle_bus,
    }.get(config.cpf_parameterization)
    if configured_bus is not None:
        bus_text = (
            str(configured_bus)
            if configured_bus > 0
            else "自动选择最敏感 PQ 母线"
        )
        print(f"参数化参考母线   : {bus_text}")
    if config.cpf_parameterization == LOCAL_VOLTAGE_PARAMETERIZATION:
        print("预测方向         : 所选母线电压降低方向")

    if not file_path.exists():
        print(f"[错误] 找不到数据文件: {file_path}")
        return

    print(f"\n[1/4] 正在解析测试系统: {file_path}")
    buses, branches = parse_dat_txt(file_path)
    print(f"      已读取 {len(buses)} 条母线、{len(branches)} 条支路")
    bus_numbers = {int(bus["number"]) for bus in buses}
    if configured_bus is not None and configured_bus > 0 and configured_bus not in bus_numbers:
        raise ValueError(
            f"{parameterization_label}参考母线 {configured_bus} 不在当前 "
            f"{len(buses)} 节点算例中；请填写有效 PQ 母线号或设为 0 自动选择"
        )
    if config.cpf_monitor_bus > 0 and config.cpf_monitor_bus not in bus_numbers:
        raise ValueError(
            f"绘图监视母线 {config.cpf_monitor_bus} 不在当前 "
            f"{len(buses)} 节点算例中；请填写有效母线号或设为 0 自动选择"
        )

    print("\n[2/4] 正在建立电网模型 ...")
    grid = PowerGrid(buses, branches, base_mva=config.base_mva)

    if not config.run_cpf:
        print("\n[停止] config.yaml 中的 run_cpf 为 false")
        return

    print("\n[3/4] 正在执行 CPF 预测—校正计算 ...")
    cpf_success, cpf_result = run_cpf(grid, config)
    print_cpf_summary(cpf_success, cpf_result, grid)

    if cpf_result.points:
        print("\n[4/4] 正在绘制 CPF 分析曲线 ...")
        plotter = CPFPlotter(PROJECT_ROOT / config.plot_dir)
        monitor_bus = config.cpf_monitor_bus or None
        output_files = plotter.plot_all(cpf_result, bus_number=monitor_bus)
        print(f">>> 已生成 {len(output_files)} 张 CPF 图像")
        print(f">>> 输出目录: {PROJECT_ROOT / config.plot_dir}")
    else:
        print("\n[4/4] 没有收敛轨迹点，跳过 CPF 绘图")


if __name__ == "__main__":
    main()
