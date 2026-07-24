"""IEEE 145 节点五种 CPF 参数化方式的重复计时。"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

from src.saves.main_cpf_use import create_cpf_parameters
from src.core.grid import PowerGrid
from src.cpf.parameter import (
    ABSOLUTE_VP_ANGLE_PARAMETERIZATION,
    LOCAL_VOLTAGE_PARAMETERIZATION,
    NATURAL_PARAMETERIZATION,
    PSEUDO_ARCLENGTH_PARAMETERIZATION,
    TANGENT_ANGLE_PARAMETERIZATION,
)
from src.io.parser import parse_dat_txt
from src.solvers.cpf_solver import ContinuationSolver
from src.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPEAT_COUNT = 10
METHODS = (
    (NATURAL_PARAMETERIZATION, "自然参数化"),
    (LOCAL_VOLTAGE_PARAMETERIZATION, "局部电压参数化"),
    (PSEUDO_ARCLENGTH_PARAMETERIZATION, "伪弧长参数化"),
    (TANGENT_ANGLE_PARAMETERIZATION, "切线角参数化"),
    (ABSOLUTE_VP_ANGLE_PARAMETERIZATION, "绝对 V/P 角参数化"),
)


def _solve(base_grid: PowerGrid, config, parameterization: str):
    parameters = replace(
        create_cpf_parameters(config),
        parameterization=parameterization,
        local_voltage_bus=0,
        tangent_angle_bus=0,
        absolute_vp_angle_bus=0,
    )
    solver = ContinuationSolver(
        params=parameters,
        tol=config.cpf_corrector_tolerance,
        max_iter=config.max_iterations,
        verbose=False,
    )
    return solver.solve(base_grid)[1]


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    buses, branches = parse_dat_txt(PROJECT_ROOT / "data" / "145ieee.txt")
    base_grid = PowerGrid(buses, branches, base_mva=config.base_mva)

    sample_path = (
        PROJECT_ROOT
        / "output"
        / "tables"
        / "CPF_145_Parameterization_Timing_Samples.csv"
    )
    samples = {parameterization: [] for parameterization, _ in METHODS}
    if sample_path.exists():
        with sample_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                samples[row["parameterization"]].append(
                    float(row["elapsed_seconds"])
                )

    # 没有历史样本时先预热；中断后重启则直接从已落盘样本继续。
    for parameterization, label in METHODS:
        if not samples[parameterization]:
            print(f"[预热] {label}")
            _solve(base_grid, config, parameterization)

    last_results = {}
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not sample_path.exists() or sample_path.stat().st_size == 0
    with sample_path.open("a", encoding="utf-8", newline="") as sample_handle:
        sample_writer = csv.writer(sample_handle)
        if needs_header:
            sample_writer.writerow(
                ["parameterization", "run_index", "elapsed_seconds"]
            )
        # 按轮次交错运行，减弱机器温度和后台负载随时间变化造成的顺序偏差。
        while any(len(values) < REPEAT_COUNT for values in samples.values()):
            repeat_index = min(len(values) for values in samples.values())
            print(f"[计时] 补充第 {repeat_index + 1}/{REPEAT_COUNT} 轮")
            made_progress = False
            for parameterization, label in METHODS:
                if len(samples[parameterization]) >= REPEAT_COUNT:
                    continue
                result = _solve(base_grid, config, parameterization)
                samples[parameterization].append(result.time_elapsed)
                last_results[parameterization] = result
                sample_writer.writerow(
                    [
                        parameterization,
                        len(samples[parameterization]),
                        f"{result.time_elapsed:.9f}",
                    ]
                )
                sample_handle.flush()
                made_progress = True
                print(
                    f"  {label}: {result.time_elapsed:.6f} s，"
                    f"点数={len(result.points)}，完成={result.success}"
                )
            if not made_progress:
                break

    output_path = (
        PROJECT_ROOT
        / "output"
        / "tables"
        / "CPF_145_Parameterization_Timing_10_Runs.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "parameterization",
                "label",
                "repeat_count",
                "mean_seconds",
                "standard_deviation_seconds",
                "min_seconds",
                "max_seconds",
                "accepted_points",
                "success",
                "used_pseudo_fallback",
            ]
        )
        for parameterization, label in METHODS:
            values = samples[parameterization]
            result = last_results.get(parameterization)
            if result is None:
                result = _solve(base_grid, config, parameterization)
            writer.writerow(
                [
                    parameterization,
                    label,
                    REPEAT_COUNT,
                    f"{mean(values):.9f}",
                    f"{stdev(values):.9f}",
                    f"{min(values):.9f}",
                    f"{max(values):.9f}",
                    len(result.points),
                    result.success,
                    result.used_pseudo_fallback,
                ]
            )

    print(f"[完成] 计时结果：{output_path}")


if __name__ == "__main__":
    main()
