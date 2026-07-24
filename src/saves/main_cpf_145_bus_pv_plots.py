"""绘制 IEEE 145 节点系统中代表母线的 P--V 曲线。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.saves.main_cpf_use import create_cpf_parameters
from src.core.grid import PowerGrid
from src.cpf.parameter import ABSOLUTE_VP_ANGLE_PARAMETERIZATION
from src.io.parser import parse_dat_txt
from src.solvers.cpf_solver import ContinuationSolver
from src.utils.config import load_config
from src.visualization.cpf_plotter import CPFPlotter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 绝对 V/P 角加密鼻点处，临界右奇异向量电压分量排名靠前的母线；
# 母线 123 保留为报告当前监视母线，便于横向对照。
MONITOR_BUSES = (138, 120, 123, 125, 127, 92)


def _plot_comparison(result, output_path: Path) -> Path:
    multipliers = np.asarray(
        [point.load_multiplier for point in result.points],
        dtype=float,
    )
    nose_index = (
        result.detected_nose_index
        if result.detected_nose_index is not None
        else int(np.argmax(multipliers))
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8.5),
        constrained_layout=True,
    )
    for axis, bus_number in zip(axes.flat, MONITOR_BUSES):
        bus_index = result.final_grid.idx_map[bus_number]
        voltages = np.asarray(
            [point.voltage[bus_index] for point in result.points],
            dtype=float,
        )
        axis.plot(
            multipliers,
            voltages,
            color="#1f77b4",
            linewidth=1.8,
            marker="o",
            markersize=2.8,
        )
        axis.scatter(
            multipliers[nose_index],
            voltages[nose_index],
            color="#d62728",
            marker="*",
            s=90,
            zorder=5,
            label="检测鼻点",
        )
        axis.set_title(f"母线 {bus_number}", fontsize=13)
        axis.set_xlabel("统一负荷倍率")
        axis.set_ylabel("电压幅值 (p.u.)")
        axis.grid(True, linestyle="--", alpha=0.45)
        axis.legend(fontsize=8, loc="best")
    figure.suptitle(
        "IEEE 145 节点系统代表母线的 P--V 曲线"
        "（绝对 V/P 角参数化）",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    buses, branches = parse_dat_txt(PROJECT_ROOT / "data" / "145ieee.txt")
    base_grid = PowerGrid(buses, branches, base_mva=config.base_mva)
    parameters = replace(
        create_cpf_parameters(config),
        parameterization=ABSOLUTE_VP_ANGLE_PARAMETERIZATION,
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

    output_dir = PROJECT_ROOT / "output" / "plots" / "CPF" / "IEEE_145"
    plotter = CPFPlotter(output_dir)
    paths = []
    for bus_number in MONITOR_BUSES:
        paths.append(
            plotter.plot_pv_curve(
                result,
                bus_number=bus_number,
                save_name=(
                    "absolute_vp_angle_IEEE_145_"
                    f"Bus_{bus_number:03d}_PV_Curve.png"
                ),
            )
        )
    comparison_path = _plot_comparison(
        result,
        output_dir
        / "absolute_vp_angle_IEEE_145_Selected_Buses_PV_Comparison.png",
    )

    print(f"CPF 鼻点倍率：{result.nose_point.load_multiplier:.10f}")
    for path in paths:
        print(f"单母线图：{path}")
    print(f"六节点对比图：{comparison_path}")


if __name__ == "__main__":
    main()
