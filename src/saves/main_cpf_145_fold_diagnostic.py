"""诊断 IEEE 145 节点 CPF 极限点及母线 123 的投影敏感度。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.saves.main_cpf_use import create_cpf_parameters
from src.core.grid import PowerGrid
from src.cpf.parameter import (
    ABSOLUTE_VP_ANGLE_PARAMETERIZATION,
    PSEUDO_ARCLENGTH_PARAMETERIZATION,
)
from src.cpf.predictor import apply_load_parameter, build_lambda_derivative
from src.io.parser import parse_dat_txt
from src.solvers.cpf_solver import CPFPoint, ContinuationSolver
from src.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITOR_BUS = 123
SUPPORTED_PARAMETERIZATIONS = (
    PSEUDO_ARCLENGTH_PARAMETERIZATION,
    ABSOLUTE_VP_ANGLE_PARAMETERIZATION,
)


def _grid_at_point(
    buses: list[dict],
    branches: list[dict],
    base_mva: float,
    point: CPFPoint,
    final_q_limits: dict[int, float],
) -> PowerGrid:
    grid = PowerGrid(buses, branches, base_mva=base_mva)
    active_q_limits: dict[int, float] = {}
    for bus_number in point.q_limited_buses:
        index = grid.idx_map[bus_number]
        grid.bus_type[index] = 1
        active_q_limits[index] = final_q_limits[index]
    grid.cpf_q_limit_mvar = active_q_limits
    apply_load_parameter(grid, point.lambda_value)
    grid.V = point.voltage.copy()
    grid.theta = point.theta.copy()
    grid.get_mismatch()
    return grid


def _reduced_mismatch(grid: PowerGrid) -> tuple[np.ndarray, list[int], list[int]]:
    d_p, d_q = grid.get_mismatch()
    _, theta_indices, voltage_indices = grid.get_jacobian()
    return (
        np.concatenate([d_p[theta_indices], d_q[voltage_indices]]),
        theta_indices,
        voltage_indices,
    )


def _directional_second_derivative(
    grid: PowerGrid,
    direction: np.ndarray,
    theta_indices: list[int],
    voltage_indices: list[int],
    epsilon: float,
) -> np.ndarray:
    theta_count = len(theta_indices)
    base_mismatch, _, _ = _reduced_mismatch(grid)
    values = []
    for sign in (-1.0, 1.0):
        perturbed = grid.clone()
        perturbed.theta[theta_indices] += (
            sign * epsilon * direction[:theta_count]
        )
        perturbed.V[voltage_indices] *= (
            1.0 + sign * epsilon * direction[theta_count:]
        )
        mismatch, _, _ = _reduced_mismatch(perturbed)
        values.append(mismatch)
    return (values[1] - 2.0 * base_mismatch + values[0]) / epsilon**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parameterization",
        choices=SUPPORTED_PARAMETERIZATIONS,
        default=PSEUDO_ARCLENGTH_PARAMETERIZATION,
    )
    arguments = parser.parse_args()
    parameterization = arguments.parameterization

    config = load_config(PROJECT_ROOT / "config.yaml")
    buses, branches = parse_dat_txt(PROJECT_ROOT / "data" / "145ieee.txt")
    base_grid = PowerGrid(buses, branches, base_mva=config.base_mva)
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
    success, result = solver.solve(base_grid)
    if not success or result.detected_nose_index is None:
        raise RuntimeError(f"CPF 未可靠越过鼻点：{result.stop_reason}")

    nose_index = result.detected_nose_index
    nose = result.points[nose_index]
    final_q_limits = dict(
        getattr(result.final_grid, "cpf_q_limit_mvar", {})
    )
    nose_grid = _grid_at_point(
        buses,
        branches,
        config.base_mva,
        nose,
        final_q_limits,
    )
    jacobian, theta_indices, voltage_indices = nose_grid.get_jacobian()
    left_vectors, singular_values, right_vectors_h = np.linalg.svd(jacobian)
    left_null = left_vectors[:, -1]
    right_null = right_vectors_h[-1]
    lambda_derivative = build_lambda_derivative(
        nose_grid,
        theta_indices,
        voltage_indices,
    )

    sigma_min = float(singular_values[-1])
    sigma_next = float(singular_values[-2])
    sigma_max = float(singular_values[0])
    transversality = float(abs(left_null @ lambda_derivative))
    transversality_relative = transversality / float(
        np.linalg.norm(lambda_derivative)
    )
    second_1 = _directional_second_derivative(
        nose_grid,
        right_null,
        theta_indices,
        voltage_indices,
        1e-4,
    )
    second_2 = _directional_second_derivative(
        nose_grid,
        right_null,
        theta_indices,
        voltage_indices,
        5e-5,
    )
    curvature_1 = float(abs(left_null @ second_1))
    curvature_2 = float(abs(left_null @ second_2))

    theta_count = len(theta_indices)
    voltage_null = np.zeros(nose_grid.n)
    voltage_null[voltage_indices] = right_null[theta_count:]
    monitor_index = nose_grid.idx_map[MONITOR_BUS]
    pq_bus_numbers = np.asarray(
        [int(nose_grid.buses[index]["number"]) for index in voltage_indices]
    )
    pq_components = np.abs(voltage_null[voltage_indices])
    order = np.argsort(pq_components)[::-1]
    monitor_position = voltage_indices.index(monitor_index)
    monitor_component = float(pq_components[monitor_position])
    monitor_rank = int(np.flatnonzero(order == monitor_position)[0]) + 1
    max_component = float(pq_components[order[0]])

    previous_point = result.points[nose_index - 1]
    next_point = result.points[nose_index + 1]
    central_voltage_change = np.abs(
        next_point.voltage - previous_point.voltage
    )
    pq_central_change = central_voltage_change[voltage_indices]
    monitor_central_change = float(central_voltage_change[monitor_index])
    central_rank = (
        int(
            np.flatnonzero(
                np.argsort(pq_central_change)[::-1] == monitor_position
            )[0]
        )
        + 1
    )

    base_condition = result.points[0].jacobian_condition_number
    nose_condition = nose.jacobian_condition_number
    lambda_before = nose.lambda_value - previous_point.lambda_value
    lambda_after = next_point.lambda_value - nose.lambda_value

    top_rows = []
    for rank, position in enumerate(order[:10], start=1):
        top_rows.append(
            (
                rank,
                int(pq_bus_numbers[position]),
                float(pq_components[position]),
                float(pq_components[position] / max_component),
            )
        )

    output_path = (
        PROJECT_ROOT
        / "output"
        / "tables"
        / f"CPF_145_Fold_Diagnostic_{parameterization}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        ("cpf_success", success, ""),
        ("nose_load_multiplier", nose.load_multiplier, ""),
        ("lambda_increment_before_nose", lambda_before, ""),
        ("lambda_increment_after_nose", lambda_after, ""),
        ("base_jacobian_condition", base_condition, ""),
        ("nose_jacobian_condition", nose_condition, ""),
        ("nose_sigma_max", sigma_max, ""),
        ("nose_sigma_next_smallest", sigma_next, ""),
        ("nose_sigma_min", sigma_min, ""),
        ("sigma_next_to_min_ratio", sigma_next / sigma_min, ""),
        ("transversality_abs", transversality, ""),
        ("transversality_relative", transversality_relative, ""),
        ("fold_curvature_eps_1e-4", curvature_1, ""),
        ("fold_curvature_eps_5e-5", curvature_2, ""),
        ("monitor_bus", MONITOR_BUS, ""),
        ("monitor_null_component", monitor_component, ""),
        ("monitor_null_component_to_max", monitor_component / max_component, ""),
        ("monitor_null_component_rank", monitor_rank, len(voltage_indices)),
        ("monitor_central_voltage_change", monitor_central_change, ""),
        (
            "monitor_central_change_to_max",
            monitor_central_change / float(np.max(pq_central_change)),
            "",
        ),
        ("monitor_central_change_rank", central_rank, len(voltage_indices)),
    )
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "note"])
        writer.writerows(rows)

    top_path = (
        PROJECT_ROOT
        / "output"
        / "tables"
        / f"CPF_145_Critical_Direction_{parameterization}_Top10.csv"
    )
    with top_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "bus", "abs_component", "relative_to_max"])
        writer.writerows(top_rows)

    print(f"参数化：{parameterization}")
    print(f"CPF 成功：{success}")
    print(f"鼻点负荷倍率：{nose.load_multiplier:.10f}")
    print(f"鼻点前后 Δλ：{lambda_before:.6e}, {lambda_after:.6e}")
    print(f"雅可比条件数（基准/鼻点）：{base_condition:.6e}, {nose_condition:.6e}")
    print(
        f"鼻点奇异值 σmax/σ2min/σmin："
        f"{sigma_max:.6e}, {sigma_next:.6e}, {sigma_min:.6e}"
    )
    print(
        f"横截性 |u'Fλ|/||Fλ||：{transversality_relative:.6e}"
    )
    print(
        f"二阶非退化量（ε=1e-4/5e-5）："
        f"{curvature_1:.6e}, {curvature_2:.6e}"
    )
    print(
        f"母线 {MONITOR_BUS} 临界电压分量：最大值的 "
        f"{monitor_component / max_component:.3%}，"
        f"排名 {monitor_rank}/{len(voltage_indices)}"
    )
    print(
        f"母线 {MONITOR_BUS} 鼻点两侧电压变化：最大值的 "
        f"{monitor_central_change / float(np.max(pq_central_change)):.3%}，"
        f"排名 {central_rank}/{len(voltage_indices)}"
    )
    print(f"诊断汇总：{output_path}")
    print(f"临界电压方向：{top_path}")


if __name__ == "__main__":
    main()
