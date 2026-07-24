"""IEEE-118 最优乘子法与非线性规划法的停滞边界分析入口。"""

from __future__ import annotations

import csv
from pathlib import Path

from src.analysis.critical_stagnation import (
    AdvancedStagnationBoundaryResult,
    run_advanced_stagnation_boundary_analysis,
)
from src.core.grid import PowerGrid
from src.io.parser import parse_dat_txt
from src.utils.config import load_config
from src.visualization.pf_plotter import PFPlotter


PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_float_values(raw: str) -> tuple[float, ...]:
    return tuple(
        float(item.strip())
        for item in raw.replace("，", ",").split(",")
        if item.strip()
    )


def _short_name(solver_name: str) -> str:
    return "OM" if solver_name == "Optimal Multiplier" else "NLP"


def _failure_type(point) -> str:
    """根据停止控制量和求解器消息区分停滞机制。"""
    reason = point.failure_reason
    if point.solver_name == "Optimal Multiplier":
        if point.terminal_control <= 1e-6 or point.minimum_control <= 1e-6:
            return "非零残差停滞（最优乘子趋零）"
        return "最优乘子法未收敛"
    if "回溯" in reason or "Armijo" in reason:
        return "非零残差停滞（回溯步长耗尽）"
    if "步长" in reason:
        return "非零残差停滞（步长过小）"
    return "非线性规划法未收敛"


def _print_summary(result: AdvancedStagnationBoundaryResult) -> None:
    print("\n" + "=" * 88)
    print("IEEE 118：最优乘子法与非线性规划法的收敛失败/残差停滞边界")
    print(f"严格收敛判据：求解器成功且最大功率失配 < {result.tolerance:.1e}")
    print(f"二分区间宽度要求：{result.refinement_tolerance:.1e}")
    print(f"CPF 鼻点参考负荷倍数：{result.cpf_nose_multiplier:.6f}")
    print("-" * 88)
    for solver_name in ("Optimal Multiplier", "Nonlinear Programming"):
        boundary = result.boundary_for(solver_name)
        if boundary is None:
            print(f"{_short_name(solver_name)}：当前扫描范围没有形成‘收敛—失败’夹逼区间。")
            continue
        low = boundary.last_converged
        high = boundary.first_failed
        print(f"{_short_name(solver_name)} 最后严格收敛点：{low.load_multiplier:.8f}×，"
              f"残差={low.final_error:.3e}，迭代={low.iterations}")
        print(f"{_short_name(solver_name)} 首个失败/停滞点：{high.load_multiplier:.8f}×，"
              f"残差={high.final_error:.3e}，迭代={high.iterations}")
        print(f"  类型：{_failure_type(high)}")
        if high.failure_reason:
            print(f"  求解器停止信息：{high.failure_reason}")
        print(
            f"  边界区间：[{boundary.bracket[0]:.8f}, "
            f"{boundary.bracket[1]:.8f}]，宽度={boundary.width:.3e}"
        )
    print("=" * 88)


def _write_points_csv(
    result: AdvancedStagnationBoundaryResult,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "solver",
                "load_multiplier",
                "strictly_converged",
                "iterations",
                "final_residual",
                "terminal_control",
                "minimum_control",
                "failure_reason",
            ]
        )
        for point in sorted(
            result.points,
            key=lambda item: (item.solver_name, item.load_multiplier),
        ):
            writer.writerow(
                [
                    point.solver_name,
                    f"{point.load_multiplier:.10f}",
                    point.success and point.final_error < result.tolerance,
                    point.iterations,
                    f"{point.final_error:.12e}",
                    f"{point.terminal_control:.12e}",
                    f"{point.minimum_control:.12e}",
                    point.failure_reason,
                ]
            )
    return output_path


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    data_path = PROJECT_ROOT / config.data_file
    buses, branches = parse_dat_txt(data_path)
    base_grid = PowerGrid(buses, branches, base_mva=config.base_mva)
    if base_grid.n != 118:
        raise ValueError(
            f"本分析要求 IEEE 118 节点数据，当前文件为 {base_grid.n} 节点：{data_path}"
        )

    result = run_advanced_stagnation_boundary_analysis(
        base_grid=base_grid,
        load_multipliers=_parse_float_values(
            config.critical_stagnation_load_multipliers
        ),
        cpf_nose_multiplier=config.critical_stagnation_cpf_nose_multiplier,
        tolerance=config.tolerance,
        max_iterations=config.critical_stagnation_max_iterations,
        enforce_q_limits=config.critical_stagnation_enforce_q_limits,
        refinement_tolerance=config.critical_stagnation_refinement_tolerance,
    )
    _print_summary(result)

    plotter = PFPlotter(str(PROJECT_ROOT / config.plot_dir))
    figure_path = plotter.plot_advanced_stagnation_boundaries(result)
    csv_path = _write_points_csv(
        result,
        PROJECT_ROOT
        / "output"
        / "tables"
        / "IEEE_118_OM_NLP_Stagnation_Boundaries.csv",
    )
    print(f"边界图：{figure_path}")
    print(f"扫描明细：{csv_path}")


if __name__ == "__main__":
    main()
