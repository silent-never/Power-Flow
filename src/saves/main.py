"""标准 NR 法与快速解耦潮流法的批量对比入口。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis.solver_comparison import (
    SolverBenchmark,
    SolverComparisonResult,
    compare_power_flow_solvers,
)
from src.analysis.robustness import (
    SolverRobustnessResult,
    run_solver_robustness,
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
        for value in config.comparison_tolerance_values.split(",")
        if value.strip()
    ]
    values.append(float(config.tolerance))
    # 数值越大代表要求越宽松，因此按降序先执行低精度。
    return tuple(sorted(set(values), reverse=True))


def _parse_float_values(value: str) -> tuple[float, ...]:
    """解析配置中的逗号分隔浮点数。"""
    return tuple(
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    )


def _print_tolerance_summary(
    label: str,
    tolerance_results: list[tuple[float, SolverComparisonResult]],
) -> None:
    """输出一个算例在不同收敛精度下的对比结果。"""
    print(f"\n{label} 节点系统不同精度对比")
    print(
        "精度       NR迭代  NR最终残差  NR中位耗时  "
        "FD迭代  FD最终残差  FD中位耗时  FD加速比"
    )
    print("-" * 100)
    for tolerance, comparison in tolerance_results:
        nr = comparison.nr
        fd = comparison.fast_decoupled
        print(
            f"{tolerance:>8.0e}  "
            f"{nr.iterations:>6}  {nr.final_error:>10.3e}  "
            f"{_format_time(nr.median_time):>11}  "
            f"{fd.iterations:>6}  {fd.final_error:>10.3e}  "
            f"{_format_time(fd.median_time):>11}  "
            f"{comparison.speed_ratio:>8.3f}×"
        )
    print("-" * 100)


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
        maximum = result.max_converged_load_multiplier(solver_name)
        text = f"{maximum:.2f}×" if np.isfinite(maximum) else "无收敛点"
        print(f"{short_name} 测试范围内最大收敛负荷倍率: {text}")


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

    plotter = PFPlotter(save_dir=PROJECT_ROOT / config.plot_dir)
    case_results: list[tuple[str, int, SolverComparisonResult]] = []
    if config.comparison_all_cases:
        print("\n")
        _print_batch_header()

    for case_index, file_path in enumerate(case_files, start=1):
        buses, branches = parse_dat_txt(file_path)
        node_count = len(buses)
        label = _case_label(file_path, node_count)
        base_grid = PowerGrid(buses, branches, base_mva=config.base_mva)
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
                )
                _print_robustness_summary(label, robustness_result)
                plotter.plot_solver_robustness(
                    robustness_result,
                    save_name=f"Solver_{label}_Robustness.png",
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

    print(f"\n>>> 图像输出目录: {PROJECT_ROOT / config.plot_dir}")


if __name__ == "__main__":
    main()
