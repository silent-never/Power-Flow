"""标准 NR 法与快速解耦潮流法的批量对比入口。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis.solver_comparison import (
    SolverBenchmark,
    SolverComparisonResult,
    compare_power_flow_solvers,
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

        if (
            not config.comparison_all_cases
            or node_count == config.comparison_detail_case
        ):
            plotter.plot_solver_comparison(
                comparison,
                save_name=f"Solver_{label}_NR_FDLF_Comparison.png",
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
