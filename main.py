"""标准 NR 法与快速解耦潮流法的批量对比入口。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis.solver_comparison import (
    SolverBenchmark,
    SolverComparisonResult,
    compare_power_flow_solvers,
)
from src.analysis.critical_stagnation import (
    CriticalStagnationResult,
    run_critical_stagnation_analysis,
)
from src.analysis.robustness import (
    SolverRobustnessResult,
    run_solver_robustness,
)
from src.analysis.scaling import (
    GridScaleMetrics,
    fit_power_law,
    measure_grid_scale,
)
from src.core.grid import PowerGrid
from src.io.parser import parse_dat_txt
from src.utils.config import load_config
from src.visualization.pf_plotter import PFPlotter


PROJECT_ROOT = Path(__file__).resolve().parent


def _format_time(seconds: float) -> str:
    """根据耗时大小选择便于阅读的显示单位。"""
    if not np.isfinite(seconds):
        return "无有效数据"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3f} μs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds:.6f} s"


def _print_benchmark(benchmark: SolverBenchmark) -> None:
    """打印一个算法的收敛和重复计时结果。"""
    print(f"算法             : {benchmark.name}")
    print(f"收敛状态         : {'成功' if benchmark.success else '失败'}")
    print(f"迭代次数         : {benchmark.iterations}")
    print(f"观测收敛阶       : {benchmark.observed_order:.4f}")
    print(f"最终最大不平衡量 : {benchmark.final_error:.6e}")
    print(f"中位求解耗时     : {_format_time(benchmark.median_time)}")
    print(f"平均求解耗时     : {_format_time(benchmark.mean_time)}")
    print(f"耗时标准差       : {_format_time(benchmark.standard_deviation)}")
    print(
        f"耗时变异系数     : "
        f"{benchmark.timing_coefficient_of_variation:.2%}"
    )
    print(
        f"稳健相对波动     : "
        f"{benchmark.robust_timing_coefficient:.2%}"
    )
    print(
        f"耗时四分位区间   : [{_format_time(benchmark.timing_q1)}, "
        f"{_format_time(benchmark.timing_q3)}]"
    )
    samples = ", ".join(
        _format_time(value) for value in benchmark.elapsed_samples
    )
    print(f"纳入统计的耗时   : {samples or '无有效数据'}")
    excluded = ", ".join(
        _format_time(value)
        for value in benchmark.excluded_elapsed_samples
    )
    print(f"IQR排除的异常耗时: {excluded or '无'}")
    timing = benchmark.timing_breakdown
    print("平均耗时组成     :")
    print(f"  功率不平衡量   : {_format_time(timing.mismatch_time)}")
    print(f"  矩阵构造       : {_format_time(timing.matrix_build_time)}")
    print(f"  矩阵分解/预处理: {_format_time(timing.factorization_time)}")
    print(f"  线性求解/回代  : {_format_time(timing.linear_solve_time)}")
    print(f"  状态更新       : {_format_time(timing.state_update_time)}")
    print(f"  其余开销       : {_format_time(timing.other_time)}")
    if benchmark.failure_reason:
        print(f"失败原因         : {benchmark.failure_reason}")


def print_comparison_summary(
    comparison: SolverComparisonResult,
    repeat_count: int,
    flat_start: bool,
) -> None:
    """输出单个测试系统的详细对比摘要。"""
    print("\n" + "=" * 72)
    print("                   潮流算法收敛与速度对比")
    print("-" * 72)
    print(f"统一初值         : {'平坦启动' if flat_start else '数据文件给定值'}")
    print(f"总运行次数       : {repeat_count}")
    print(f"正式计时次数     : {repeat_count - 1}")
    print("异常值识别规则   : 耗时 > Q3 + 1.5 × IQR")
    print("-" * 72)
    _print_benchmark(comparison.nr)
    print("-" * 72)
    _print_benchmark(comparison.fast_decoupled)
    print("-" * 72)
    if comparison.nr.success and comparison.fast_decoupled.success:
        print(
            f"两种算法最大电压差: "
            f"{comparison.max_voltage_difference:.6e} p.u."
        )
        print(
            f"两种算法最大相角差: "
            f"{comparison.max_angle_difference_deg:.6e}°"
        )
        ratio = comparison.speed_ratio
        if ratio >= 1.0:
            print(f"速度关系         : 快速解耦法约快 {ratio:.3f} 倍")
        else:
            print(f"速度关系         : NR 约快 {1.0 / ratio:.3f} 倍")
    else:
        print("解一致性比较     : 至少一种算法未收敛，无法比较")
    print("=" * 72)


def _case_sort_key(path: Path) -> int:
    """从 IEEE 文件名提取节点数用于排序。"""
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits) if digits else 0


def _case_label(path: Path, node_count: int) -> str:
    """生成图表和终端中使用的测试系统标签。"""
    number = _case_sort_key(path)
    return str(number or node_count)


def _resolve_tolerance_values(config) -> tuple[float, ...]:
    """解析精度扫描配置，并保证基础精度只出现一次。"""
    values = [
        float(value.strip())
        for value in config.comparison_tolerance_values.replace("，", ",").split(",")
        if value.strip()
    ]
    values.append(float(config.tolerance))
    # 数值越大代表要求越宽松，因此按降序先执行低精度。
    return tuple(sorted(set(values), reverse=True))


def _parse_float_values(value: str) -> tuple[float, ...]:
    """解析配置中的逗号分隔浮点数。"""
    return tuple(
        float(item.strip())
        for item in value.replace("，", ",").split(",")
        if item.strip()
    )


def _print_tolerance_summary(
    label: str,
    tolerance_results: list[tuple[float, SolverComparisonResult]],
) -> None:
    """输出一个算例在不同收敛精度下的对比结果。"""
    print(f"\n{label} 节点系统不同精度对比")
    print("精度       算法   状态  迭代  最终残差    中位耗时")
    print("-" * 68)
    for tolerance, comparison in tolerance_results:
        for index, (short_name, benchmark) in enumerate(
            (("NR", comparison.nr), ("FDLF", comparison.fast_decoupled))
        ):
            tolerance_text = f"{tolerance:>8.0e}" if index == 0 else " " * 8
            print(
                f"{tolerance_text}  {short_name:<5}  "
                f"{'成功' if benchmark.success else '失败':>4}  "
                f"{benchmark.iterations:>4}  {benchmark.final_error:>10.3e}  "
                f"{_format_time(benchmark.median_time):>11}"
            )
    print("-" * 68)


def _print_robustness_summary(
    label: str,
    result: SolverRobustnessResult,
) -> None:
    """输出初值、负荷倍率和线路电阻扰动的鲁棒性摘要。"""
    print(f"\n{label} 节点系统鲁棒性测试")
    print(
        "场景                  算法                 成功/总数  "
        "成功率   平均迭代  中位最终残差  失败形式"
    )
    print("-" * 112)
    for summary in result.summaries:
        solver_label = (
            "NR"
            if summary.solver_name == "Newton-Raphson"
            else "FDLF"
        )
        failure_text = ", ".join(
            f"{name}:{count}" for name, count in summary.failure_modes
        ) or "无"
        iteration_text = (
            f"{summary.mean_iterations:.2f}"
            if np.isfinite(summary.mean_iterations)
            else "--"
        )
        error_text = (
            f"{summary.median_final_error:.3e}"
            if np.isfinite(summary.median_final_error)
            else "--"
        )
        print(
            f"{summary.scenario:<20}  {solver_label:<18}  "
            f"{summary.success_count:>3}/{summary.trial_count:<3}  "
            f"{summary.success_rate:>6.1%}  {iteration_text:>8}  "
            f"{error_text:>12}  {failure_text}"
        )
    print("-" * 112)
    for solver_name, short_name in (
        ("Newton-Raphson", "NR"),
        ("Fast-Decoupled (XB)", "FDLF"),
    ):
        bracket = result.load_convergence_bracket(solver_name)
        if bracket is None:
            print(f"{short_name} 未形成收敛—失败边界区间")
            continue
        low, high = bracket
        print(
            f"{short_name} 负荷收敛边界: "
            f"[{low:.6f}, {high:.6f}]，区间宽度 {high - low:.2e}"
        )


def _print_critical_stagnation_summary(
    label: str,
    result: CriticalStagnationResult,
) -> None:
    """输出鼻点前后两种全局化方法的停滞诊断。"""
    print(f"\n{label} 节点系统临界负荷停滞分析")
    print(
        f"CPF参考鼻点={result.cpf_nose_multiplier:.6f}，"
        f"PV转PQ={'开启' if result.enforce_q_limits else '关闭'}"
    )
    print("倍率      算法  状态            总迭代  最终残差    终止μ/α      PV-PQ轮数")
    print("-" * 88)
    short_names = {
        "Newton-Raphson": "NR",
        "Fast-Decoupled (XB)": "FDLF",
        "Optimal Multiplier": "OM",
        "Nonlinear Programming": "NLP",
    }
    for point in result.points:
        short_name = short_names[point.solver_name]
        control_text = (
            f"{point.terminal_control:.3e}"
            if np.isfinite(point.terminal_control)
            else "--"
        )
        print(
            f"{point.load_multiplier:>8.4f}  {short_name:<4}  "
            f"{point.status:<14}  {point.iterations:>6}  "
            f"{point.final_error:>10.3e}  {control_text:>11}  "
            f"{point.q_limit_passes:>8}"
        )
    print("-" * 88)


def _print_scaling_summary(
    results: list[tuple[str, GridScaleMetrics, SolverComparisonResult]],
    minimum_nodes: int,
) -> None:
    """输出矩阵规模、等效单次迭代耗时和幂律增长指数。"""
    print("\n系统规模增长趋势")
    print(
        "系统  节点  支路  PQ  PV  状态维数  J非零元  J密度  "
        "矩阵内存  NR每迭代  FD每迭代"
    )
    print("-" * 116)
    for label, metrics, comparison in results:
        nr_per_iteration = comparison.nr.median_time / max(
            comparison.nr.iterations,
            1,
        )
        fd_per_iteration = comparison.fast_decoupled.median_time / max(
            comparison.fast_decoupled.iterations,
            1,
        )
        print(
            f"{label:>4}  {metrics.node_count:>4}  "
            f"{metrics.branch_count:>4}  {metrics.pq_count:>3}  "
            f"{metrics.pv_count:>3}  {metrics.state_dimension:>8}  "
            f"{metrics.jacobian_nnz:>7}  {metrics.jacobian_density:>6.2%}  "
            f"{metrics.dense_matrix_memory_bytes / 1024.0:>8.1f}KB  "
            f"{_format_time(nr_per_iteration):>10}  "
            f"{_format_time(fd_per_iteration):>10}"
        )
    print("-" * 116)

    fitted = [item for item in results if item[1].node_count >= minimum_nodes]
    sizes = np.asarray([item[1].state_dimension for item in fitted], dtype=float)
    for solver_name, getter in (
        ("NR", lambda comparison: comparison.nr),
        ("FDLF", lambda comparison: comparison.fast_decoupled),
    ):
        total_times = np.asarray(
            [getter(item[2]).median_time for item in fitted],
            dtype=float,
        )
        per_iteration_times = np.asarray(
            [
                getter(item[2]).median_time
                / max(getter(item[2]).iterations, 1)
                for item in fitted
            ],
            dtype=float,
        )
        total_fit = fit_power_law(sizes, total_times)
        iteration_fit = fit_power_law(sizes, per_iteration_times)
        print(
            f"{solver_name}（节点数≥{minimum_nodes}）: "
            f"总耗时 ∝ 状态维数^{total_fit.exponent:.3f} "
            f"(R2={total_fit.r_squared:.3f})；"
            f"每迭代耗时 ∝ 状态维数^{iteration_fit.exponent:.3f} "
            f"(R2={iteration_fit.r_squared:.3f})"
        )


def _resolve_case_files(config) -> list[Path]:
    """根据配置返回单个或全部 IEEE 测试文件。"""
    if config.comparison_all_cases:
        return sorted(
            (PROJECT_ROOT / "data").glob("*ieee.txt"),
            key=_case_sort_key,
        )
    return [PROJECT_ROOT / config.data_file]


def _print_batch_header() -> None:
    """打印批量结果表头。"""
    print(
        "系统  节点  NR状态  NR迭代  NR阶数  NR中位耗时  "
        "FD状态  FD迭代  FD阶数  FD中位耗时"
    )
    print("-" * 104)


def _print_batch_row(
    label: str,
    node_count: int,
    comparison: SolverComparisonResult,
) -> None:
    """打印一个测试系统的紧凑批量结果。"""
    nr = comparison.nr
    fd = comparison.fast_decoupled
    print(
        f"{label:>4}  {node_count:>4}  "
        f"{'成功' if nr.success else '失败':>6}  "
        f"{nr.iterations:>6}  {nr.observed_order:>6.3f}  "
        f"{_format_time(nr.median_time):>11}  "
        f"{'成功' if fd.success else '失败':>6}  "
        f"{fd.iterations:>6}  {fd.observed_order:>6.3f}  "
        f"{_format_time(fd.median_time):>11}"
    )


def main() -> None:
    """执行单个或全部 IEEE 系统的算法对比并绘图。"""
    config = load_config(PROJECT_ROOT / "config.yaml")
    case_files = _resolve_case_files(config)

    print("=" * 72)
    print("              Newton-Raphson 与快速解耦潮流对比")
    print("=" * 72)
    if not case_files:
        print("[错误] data 目录中没有找到 *ieee.txt 测试文件")
        return

    missing_files = [path for path in case_files if not path.exists()]
    if missing_files:
        print(f"[错误] 找不到数据文件: {missing_files[0]}")
        return

    print(f"测试系统数量     : {len(case_files)}")
    print(
        f"统一初值         : "
        f"{'平坦启动' if config.comparison_flat_start else '数据文件给定值'}"
    )
    print(f"每种算法总运行数 : {config.comparison_repeat_count}")
    print(f"每种算法正式计时 : {config.comparison_repeat_count - 1}")
    print("异常值识别规则     : 耗时 > Q3 + 1.5 × IQR")
    if config.comparison_run_tolerance_sweep:
        values = ", ".join(
            f"{value:.0e}" for value in _resolve_tolerance_values(config)
        )
        print(f"详细算例精度扫描   : {values}")
        print(
            f"每个精度正式计时   : "
            f"{config.comparison_tolerance_repeat_count - 1} 次"
        )
    if config.comparison_run_robustness:
        print(
            f"随机初值鲁棒性试验 : "
            f"每个扰动等级 {config.robustness_random_trials} 次"
        )
    if config.comparison_run_scaling_analysis:
        print(
            f"规模趋势拟合范围   : "
            f"节点数 ≥ {config.comparison_scaling_min_nodes}"
        )
    if config.comparison_run_critical_stagnation:
        print(
            f"临界停滞扫描范围   : "
            f"{config.critical_stagnation_load_multipliers}"
        )

    plotter = PFPlotter(save_dir=PROJECT_ROOT / config.plot_dir)
    case_results: list[tuple[str, int, SolverComparisonResult]] = []
    scaling_results: list[
        tuple[str, GridScaleMetrics, SolverComparisonResult]
    ] = []
    if config.comparison_all_cases:
        print("\n")
        _print_batch_header()

    for case_index, file_path in enumerate(case_files, start=1):
        buses, branches = parse_dat_txt(file_path)
        node_count = len(buses)
        label = _case_label(file_path, node_count)
        base_grid = PowerGrid(buses, branches, base_mva=config.base_mva)
        scale_metrics = measure_grid_scale(base_grid)
        comparison = compare_power_flow_solvers(
            base_grid=base_grid,
            tolerance=config.tolerance,
            nr_max_iterations=config.max_iterations,
            fast_decoupled_max_iterations=(
                config.comparison_fast_decoupled_max_iterations
            ),
            repeat_count=config.comparison_repeat_count,
            flat_start=config.comparison_flat_start,
        )
        case_results.append((label, node_count, comparison))
        scaling_results.append((label, scale_metrics, comparison))

        if config.comparison_all_cases:
            _print_batch_row(label, node_count, comparison)
        else:
            print(f"\n测试文件: {file_path}")
            print_comparison_summary(
                comparison,
                repeat_count=config.comparison_repeat_count,
                flat_start=config.comparison_flat_start,
            )

        is_detail_case = (
            not config.comparison_all_cases
            or node_count == config.comparison_detail_case
        )
        if is_detail_case:
            plotter.plot_solver_comparison(
                comparison,
                save_name=f"Solver_{label}_NR_FDLF_Comparison.png",
            )
            if config.comparison_run_tolerance_sweep:
                tolerance_results = []
                for tolerance in _resolve_tolerance_values(config):
                    if (
                        config.comparison_tolerance_repeat_count
                        == config.comparison_repeat_count
                        and np.isclose(
                            tolerance,
                            config.tolerance,
                            rtol=1e-12,
                            atol=0.0,
                        )
                    ):
                        tolerance_comparison = comparison
                    else:
                        tolerance_comparison = compare_power_flow_solvers(
                            base_grid=base_grid,
                            tolerance=tolerance,
                            nr_max_iterations=config.max_iterations,
                            fast_decoupled_max_iterations=(
                                config.comparison_fast_decoupled_max_iterations
                            ),
                            repeat_count=(
                                config.comparison_tolerance_repeat_count
                            ),
                            flat_start=config.comparison_flat_start,
                        )
                    tolerance_results.append(
                        (tolerance, tolerance_comparison)
                    )
                _print_tolerance_summary(label, tolerance_results)
                plotter.plot_solver_tolerance_comparison(
                    tolerance_results,
                    save_name=(
                        f"Solver_{label}_Tolerance_Comparison.png"
                    ),
                )
            if config.comparison_run_robustness:
                robustness_result = run_solver_robustness(
                    base_grid=base_grid,
                    tolerance=config.tolerance,
                    nr_max_iterations=config.max_iterations,
                    fast_decoupled_max_iterations=(
                        config.comparison_fast_decoupled_max_iterations
                    ),
                    random_trials=config.robustness_random_trials,
                    voltage_perturbations=_parse_float_values(
                        config.robustness_voltage_perturbations
                    ),
                    angle_perturbations_deg=_parse_float_values(
                        config.robustness_angle_perturbations_deg
                    ),
                    load_multipliers=_parse_float_values(
                        config.robustness_load_multipliers
                    ),
                    resistance_multipliers=_parse_float_values(
                        config.robustness_resistance_multipliers
                    ),
                    random_seed=config.robustness_random_seed,
                    enforce_q_limits=config.cpf_enforce_q_limits,
                    load_refinement_tolerance=(
                        config.robustness_load_refinement_tolerance
                    ),
                )
                _print_robustness_summary(label, robustness_result)
                plotter.plot_solver_robustness(
                    robustness_result,
                    save_name=f"Solver_{label}_Robustness.png",
                )
            if config.comparison_run_critical_stagnation:
                stagnation_result = run_critical_stagnation_analysis(
                    base_grid=base_grid,
                    load_multipliers=_parse_float_values(
                        config.critical_stagnation_load_multipliers
                    ),
                    cpf_nose_multiplier=(
                        config.critical_stagnation_cpf_nose_multiplier
                    ),
                    tolerance=config.tolerance,
                    nr_max_iterations=config.max_iterations,
                    fast_decoupled_max_iterations=(
                        config.comparison_fast_decoupled_max_iterations
                    ),
                    advanced_max_iterations=(
                        config.critical_stagnation_max_iterations
                    ),
                    enforce_q_limits=(
                        config.critical_stagnation_enforce_q_limits
                    ),
                )
                _print_critical_stagnation_summary(label, stagnation_result)
                plotter.plot_critical_stagnation(
                    stagnation_result,
                    save_name=f"Solver_{label}_Critical_Stagnation.png",
                )

    if config.comparison_all_cases:
        print("-" * 104)
        converged_count = sum(
            result.nr.success and result.fast_decoupled.success
            for _, _, result in case_results
        )
        print(
            f"完成 {len(case_results)} 个系统，其中两种算法均收敛 "
            f"{converged_count} 个。"
        )
        plotter.plot_solver_batch_summary(case_results)
        if config.comparison_run_scaling_analysis:
            _print_scaling_summary(
                scaling_results,
                config.comparison_scaling_min_nodes,
            )
            plotter.plot_solver_scaling_trend(
                scaling_results,
                minimum_nodes=config.comparison_scaling_min_nodes,
            )

    print(f"\n>>> 图像输出目录: {PROJECT_ROOT / config.plot_dir}")


if __name__ == "__main__":
    main()
