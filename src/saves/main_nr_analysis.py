"""生成 NR 跨算例、跨收敛精度的偏差校核图和明细表。"""

from __future__ import annotations

import csv
from pathlib import Path

from src.analysis.accuracy_validation import run_nr_accuracy_validation
from src.core.grid import PowerGrid
from src.io.parser import parse_dat_txt
from src.utils.config import load_config
from src.visualization.pf_plotter import PFPlotter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parse_tolerances(raw: str) -> list[float]:
    return [
        float(value.strip())
        for value in raw.replace("，", ",").split(",")
        if value.strip()
    ]


def _write_csv(path: Path, results) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "label",
        "node_count",
        "tolerance",
        "success",
        "iterations",
        "max_voltage_deviation",
        "max_angle_deviation_deg",
        "max_active_power_deviation",
        "max_reactive_non_pq_deviation",
        "max_active_mismatch",
        "max_reactive_mismatch",
        "max_equation_mismatch",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({field: getattr(item, field) for field in fields})


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    case_files = sorted((PROJECT_ROOT / "data").glob("*ieee.txt"))
    if not case_files:
        raise FileNotFoundError("data 目录中没有 *ieee.txt 算例")

    case_results = []
    detail_grid = None
    for path in case_files:
        buses, branches = parse_dat_txt(path)
        grid = PowerGrid(buses, branches, base_mva=config.base_mva)
        label = path.stem.removesuffix("ieee").lstrip("0") or "0"
        result = run_nr_accuracy_validation(
            grid,
            label=label,
            tolerance=config.tolerance,
            max_iterations=config.max_iterations,
            flat_start=config.comparison_flat_start,
        )
        case_results.append(result)
        print(
            f"IEEE {result.node_count:>3}: "
            f"{'收敛' if result.success else '失败'}, "
            f"迭代 {result.iterations:>2}, "
            f"最大方程失配 {result.max_equation_mismatch:.3e}"
        )
        if result.node_count == config.comparison_detail_case:
            detail_grid = grid

    case_results.sort(key=lambda item: item.node_count)
    if detail_grid is None:
        raise ValueError(
            f"未找到 {config.comparison_detail_case} 节点详细算例"
        )

    tolerance_results = [
        run_nr_accuracy_validation(
            detail_grid,
            label=str(config.comparison_detail_case),
            tolerance=tolerance,
            max_iterations=config.max_iterations,
            flat_start=config.comparison_flat_start,
        )
        for tolerance in _parse_tolerances(config.comparison_tolerance_values)
    ]

    plotter = PFPlotter(save_dir=PROJECT_ROOT / config.plot_dir)
    plotter.plot_accuracy_by_case(case_results)
    plotter.plot_accuracy_by_tolerance(tolerance_results)

    table_dir = PROJECT_ROOT / "output" / "tables"
    _write_csv(table_dir / "IEEE_All_Cases_Accuracy.csv", case_results)
    _write_csv(
        table_dir / "IEEE_118_Tolerance_Accuracy.csv",
        tolerance_results,
    )

    failed_cases = [item.node_count for item in case_results if not item.success]
    failed_tolerances = [
        item.tolerance
        for item in tolerance_results
        if not item.success or item.max_equation_mismatch >= item.tolerance
    ]
    print(f"\n全部算例收敛失败项: {failed_cases or '无'}")
    print(f"118 节点精度不达标项: {failed_tolerances or '无'}")
    print(f"图像目录: {PROJECT_ROOT / config.plot_dir}")
    print(f"数据目录: {table_dir}")


if __name__ == "__main__":
    main()
