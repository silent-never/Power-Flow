"""IEEE-118 五种 CPF 参数化方式的统一对比入口。"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.core.grid import PowerGrid
from src.cpf.parameter import (
    ABSOLUTE_VP_ANGLE_PARAMETERIZATION,
    LOCAL_VOLTAGE_PARAMETERIZATION,
    NATURAL_PARAMETERIZATION,
    PSEUDO_ARCLENGTH_PARAMETERIZATION,
    TANGENT_ANGLE_PARAMETERIZATION,
)
from src.io.parser import parse_dat_txt
from src.saves.main_cpf_use import create_cpf_parameters
from src.solvers.cpf_solver import CPFResult, ContinuationSolver
from src.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHODS = (
    (NATURAL_PARAMETERIZATION, "自然参数化", "#1f77b4", "o"),
    (LOCAL_VOLTAGE_PARAMETERIZATION, "局部电压参数化", "#ff7f0e", "s"),
    (PSEUDO_ARCLENGTH_PARAMETERIZATION, "伪弧长参数化", "#2ca02c", "^"),
    (TANGENT_ANGLE_PARAMETERIZATION, "切线角参数化", "#d62728", "D"),
    (ABSOLUTE_VP_ANGLE_PARAMETERIZATION, "绝对 V/P 角参数化", "#9467bd", "v"),
)


def _setup_matplotlib() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["mathtext.fontset"] = "dejavusans"


def _monitor_bus(results: dict[str, CPFResult], configured_bus: int) -> int:
    if configured_bus > 0:
        return configured_bus
    reference = results[PSEUDO_ARCLENGTH_PARAMETERIZATION]
    pq_indices = np.flatnonzero(reference.final_grid.bus_type == 1)
    base_voltage = reference.points[0].voltage[pq_indices]
    nose_voltage = reference.nose_point.voltage[pq_indices]
    index = int(pq_indices[np.argmax(1.0 - nose_voltage / base_voltage)])
    return int(reference.final_grid.buses[index]["number"])


def _plot_comparison(
    results: dict[str, CPFResult],
    monitor_bus: int,
    output_path: Path,
) -> Path:
    _setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    pv_axis, zoom_axis, step_axis, iteration_axis = axes.flat
    nose_multipliers = []

    for key, label, color, marker in METHODS:
        result = results[key]
        bus_index = result.final_grid.idx_map[monitor_bus]
        multipliers = np.asarray(
            [point.load_multiplier for point in result.points], dtype=float
        )
        voltages = np.asarray(
            [point.voltage[bus_index] for point in result.points], dtype=float
        )
        point_indices = np.arange(len(result.points))
        steps = np.asarray([point.step_size for point in result.points])
        iterations = np.asarray(
            [point.corrector_iterations for point in result.points]
        )
        linestyle = "-" if result.success else "--"
        legend_label = label + ("" if result.success else "（提前停止）")
        pv_axis.plot(
            multipliers,
            voltages,
            color=color,
            marker=marker,
            markevery=max(1, len(result.points) // 18),
            markersize=4,
            linewidth=1.8,
            linestyle=linestyle,
            label=legend_label,
        )
        zoom_axis.plot(
            multipliers,
            voltages,
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.5,
            linestyle=linestyle,
            label=label,
        )
        step_axis.plot(
            point_indices[1:],
            steps[1:],
            color=color,
            linewidth=1.7,
            label=label,
        )
        iteration_axis.plot(
            point_indices[1:],
            iterations[1:],
            color=color,
            linewidth=1.5,
            label=label,
        )
        if result.detected_nose_index is not None:
            nose = result.nose_point
            nose_multipliers.append(nose.load_multiplier)
            nose_voltage = nose.voltage[bus_index]
            for axis in (pv_axis, zoom_axis):
                axis.scatter(
                    nose.load_multiplier,
                    nose_voltage,
                    color=color,
                    marker="*",
                    s=100,
                    zorder=6,
                )

    reference_nose = (
        float(np.median(nose_multipliers)) if nose_multipliers else 1.0
    )
    zoom_axis.set_xlim(reference_nose - 0.025, reference_nose + 0.012)
    zoom_voltages = []
    for result in results.values():
        bus_index = result.final_grid.idx_map[monitor_bus]
        zoom_voltages.extend(
            point.voltage[bus_index]
            for point in result.points
            if reference_nose - 0.025
            <= point.load_multiplier
            <= reference_nose + 0.012
        )
    if zoom_voltages:
        lower, upper = min(zoom_voltages), max(zoom_voltages)
        margin = max(0.005, 0.12 * (upper - lower))
        zoom_axis.set_ylim(lower - margin, upper + margin)

    pv_axis.set_title(f"母线 {monitor_bus} 的完整 P--V 轨迹", fontsize=14)
    zoom_axis.set_title("鼻点邻域放大（星号为检测鼻点）", fontsize=14)
    step_axis.set_title("预测步长随已接受点变化", fontsize=14)
    iteration_axis.set_title("每个接受点的校正迭代次数", fontsize=14)
    for axis in (pv_axis, zoom_axis):
        axis.set_xlabel("统一负荷倍率", fontsize=11)
        axis.set_ylabel("电压幅值 (p.u.)", fontsize=11)
    step_axis.set_xlabel("已接受 CPF 点序号", fontsize=11)
    step_axis.set_ylabel("预测步长", fontsize=11)
    iteration_axis.set_xlabel("已接受 CPF 点序号", fontsize=11)
    iteration_axis.set_ylabel("校正迭代次数", fontsize=11)
    for axis in axes.flat:
        axis.grid(True, which="both", linestyle="--", alpha=0.45)
        axis.legend(fontsize=8, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_summary(
    results: dict[str, CPFResult],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = {key: label for key, label, _, _ in METHODS}
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "parameterization",
                "label",
                "success",
                "accepted_points",
                "failed_attempts",
                "nose_detected",
                "nose_load_multiplier",
                "nose_min_pq_voltage",
                "post_nose_points",
                "used_pseudo_fallback",
                "elapsed_seconds",
                "stop_reason",
            ]
        )
        for key, _, _, _ in METHODS:
            result = results[key]
            nose = result.nose_point
            post_nose = (
                len(result.points) - 1 - result.detected_nose_index
                if result.detected_nose_index is not None
                else 0
            )
            writer.writerow(
                [
                    key,
                    labels[key],
                    result.success,
                    len(result.points),
                    result.failed_attempts,
                    result.detected_nose_index is not None,
                    f"{nose.load_multiplier:.10f}",
                    f"{nose.min_pq_voltage:.10f}",
                    post_nose,
                    result.used_pseudo_fallback,
                    f"{result.time_elapsed:.6f}",
                    result.stop_reason,
                ]
            )
    return output_path


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    buses, branches = parse_dat_txt(PROJECT_ROOT / config.data_file)
    if len(buses) != 118:
        raise ValueError("参数化对比当前要求使用 IEEE 118 节点算例")
    base_grid = PowerGrid(buses, branches, base_mva=config.base_mva)
    base_parameters = create_cpf_parameters(config)
    results: dict[str, CPFResult] = {}

    for key, label, _, _ in METHODS:
        print(f"[CPF 参数化对比] 正在运行：{label}")
        parameters = replace(base_parameters, parameterization=key)
        solver = ContinuationSolver(
            params=parameters,
            tol=config.cpf_corrector_tolerance,
            max_iter=config.max_iterations,
            verbose=False,
        )
        _, result = solver.solve(base_grid)
        results[key] = result
        print(
            f"  成功={result.success}，点数={len(result.points)}，"
            f"鼻点倍率={result.nose_point.load_multiplier:.8f}，"
            f"失败尝试={result.failed_attempts}，"
            f"伪弧长接管={result.used_pseudo_fallback}"
        )

    monitor_bus = _monitor_bus(results, config.cpf_monitor_bus)
    figure_path = _plot_comparison(
        results,
        monitor_bus,
        PROJECT_ROOT / config.plot_dir / "CPF_118_Parameterization_Comparison.png",
    )
    summary_path = _write_summary(
        results,
        PROJECT_ROOT / "output" / "tables" / "CPF_118_Parameterization_Summary.csv",
    )
    print(f"[CPF 参数化对比] 监视母线：{monitor_bus}")
    print(f"[CPF 参数化对比] 图像：{figure_path}")
    print(f"[CPF 参数化对比] 汇总：{summary_path}")


if __name__ == "__main__":
    main()
