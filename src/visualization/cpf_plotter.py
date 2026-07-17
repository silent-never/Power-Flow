"""连续潮流结果的专用可视化工具。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..solvers.cpf_solver import CPFResult


def _setup_matplotlib_environment() -> None:
    """设置中文字体和数学符号显示。"""
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["mathtext.fontset"] = "dejavusans"


_setup_matplotlib_environment()


class CPFPlotter:
    """绘制 PV 曲线、条件数和步长控制等 CPF 结果。"""

    def __init__(self, save_dir: str | Path = "output/plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save_figure(self, figure, save_name: str) -> Path:
        """保存并关闭图像。"""
        output_path = self.save_dir / save_name
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        print(f"[CPF 绘图] 图像已保存至: {output_path}")
        return output_path

    @staticmethod
    def _nose_index(result: CPFResult) -> int:
        """返回求解器锁定的首次鼻点索引。"""
        if result.detected_nose_index is not None:
            return result.detected_nose_index
        return max(
            range(len(result.points)),
            key=lambda index: result.points[index].lambda_value,
        )

    @staticmethod
    def _resolve_bus(result: CPFResult, bus_number: int | None) -> tuple[int, int]:
        """确定 PV 曲线监视母线及其内部索引。

        未指定母线时，自动选择鼻点相对基准电压下降最大的 PQ 母线，
        避免固定低电压的 PV 母线干扰判断。
        """
        if bus_number is None:
            pq_indices = np.flatnonzero(result.final_grid.bus_type == 1)
            if pq_indices.size == 0:
                raise ValueError("当前系统中没有可用的 PQ 母线")
            reference_voltage = result.points[0].voltage[pq_indices]
            nose_voltage = result.nose_point.voltage[pq_indices]
            relative_drop = 1.0 - nose_voltage / reference_voltage
            bus_index = int(
                pq_indices[int(np.argmax(relative_drop))]
            )
            selected_bus = int(result.final_grid.buses[bus_index]["number"])
        else:
            selected_bus = int(bus_number)
        if selected_bus not in result.final_grid.idx_map:
            raise ValueError(f"找不到 CPF 监视母线: {selected_bus}")
        return selected_bus, result.final_grid.idx_map[selected_bus]

    def plot_pv_curve(
        self,
        result: CPFResult,
        bus_number: int | None = None,
        save_name: str = "CPF_01_PV_Curve.png",
    ) -> Path:
        """绘制指定母线的负荷倍率—电压 PV 曲线。"""
        selected_bus, bus_index = self._resolve_bus(result, bus_number)
        multipliers = np.array(
            [point.load_multiplier for point in result.points]
        )
        voltages = np.array(
            [point.voltage[bus_index] for point in result.points]
        )
        nose_index = self._nose_index(result)

        figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
        axis.plot(
            multipliers,
            voltages,
            color="#1f77b4",
            linewidth=2,
            marker="o",
            markersize=3,
            label=f"母线 {selected_bus}",
        )
        axis.scatter(
            multipliers[nose_index],
            voltages[nose_index],
            color="#d62728",
            s=80,
            zorder=5,
            label="鼻点",
        )
        axis.annotate(
            f"倍率={multipliers[nose_index]:.4f}\n"
            f"V={voltages[nose_index]:.4f}",
            (multipliers[nose_index], voltages[nose_index]),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=9,
        )
        axis.set_title(f"母线 {selected_bus} 的 PV 曲线", fontsize=15)
        axis.set_xlabel("统一负荷倍率", fontsize=12)
        axis.set_ylabel("电压幅值 (p.u.)", fontsize=12)
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend(loc="best")
        return self._save_figure(figure, save_name)

    def plot_min_voltage(
        self,
        result: CPFResult,
        save_name: str = "CPF_02_Min_Voltage.png",
    ) -> Path:
        """绘制 PQ 母线相对基准点的最大电压下降比例。"""
        steps = np.array([point.step_index for point in result.points])
        voltage_drops = 100.0 * np.array(
            [
                point.max_pq_voltage_drop_ratio
                for point in result.points
            ]
        )
        nose_index = self._nose_index(result)

        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        axis.plot(steps, voltage_drops, color="#2ca02c", linewidth=2)
        axis.scatter(
            steps[nose_index],
            voltage_drops[nose_index],
            color="#d62728",
            s=70,
            label=(
                f"鼻点：母线 "
                f"{result.nose_point.max_pq_voltage_drop_bus or '无'}"
            ),
            zorder=5,
        )
        axis.set_title("CPF 过程中的 PQ 母线最大相对压降", fontsize=15)
        axis.set_xlabel("CPF 步数", fontsize=12)
        axis.set_ylabel("相对基准电压下降 (%)", fontsize=12)
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend(loc="best")
        return self._save_figure(figure, save_name)

    def plot_condition_number(
        self,
        result: CPFResult,
        save_name: str = "CPF_03_Jacobian_Condition.png",
    ) -> Path:
        """绘制雅可比矩阵条件数随 CPF 步数的变化。"""
        steps = np.array([point.step_index for point in result.points])
        conditions = np.array(
            [point.jacobian_condition_number for point in result.points]
        )
        valid = np.isfinite(conditions) & (conditions > 0.0)

        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        axis.semilogy(
            steps[valid],
            conditions[valid],
            color="#9467bd",
            linewidth=2,
        )
        axis.set_title("CPF 雅可比矩阵条件数", fontsize=15)
        axis.set_xlabel("CPF 步数", fontsize=12)
        axis.set_ylabel("条件数（对数坐标）", fontsize=12)
        axis.grid(True, which="both", linestyle="--", alpha=0.5)
        return self._save_figure(figure, save_name)

    def plot_step_control(
        self,
        result: CPFResult,
        save_name: str = "CPF_04_Step_Control.png",
    ) -> Path:
        """绘制预测步长和校正迭代次数。"""
        points = result.points[1:]
        steps = np.array([point.step_index for point in points])
        step_sizes = np.array([point.step_size for point in points])
        iterations = np.array([point.corrector_iterations for point in points])

        figure, step_axis = plt.subplots(
            figsize=(9, 5),
            constrained_layout=True,
        )
        iteration_axis = step_axis.twinx()
        step_axis.plot(
            steps,
            step_sizes,
            color="#ff7f0e",
            linewidth=2,
            label="预测步长",
        )
        iteration_axis.plot(
            steps,
            iterations,
            color="#1f77b4",
            linewidth=1.5,
            alpha=0.8,
            label="校正迭代次数",
        )
        step_axis.set_title("CPF 自适应步长与校正迭代", fontsize=15)
        step_axis.set_xlabel("CPF 步数", fontsize=12)
        step_axis.set_ylabel("预测步长", color="#ff7f0e", fontsize=12)
        iteration_axis.set_ylabel(
            "校正迭代次数",
            color="#1f77b4",
            fontsize=12,
        )
        step_axis.grid(True, linestyle="--", alpha=0.4)
        lines = step_axis.lines + iteration_axis.lines
        labels = [line.get_label() for line in lines]
        step_axis.legend(lines, labels, loc="best")
        return self._save_figure(figure, save_name)

    def plot_condition_with_voltage_axis(
        self,
        result: CPFResult,
        bus_number: int | None = None,
        save_name: str = "CPF_05_Condition_Voltage.png",
    ) -> Path:
        """绘制等间距步数—条件数曲线，并在顶部标注对应电压。

        底部横轴按照 CPF 收敛点等间距排列并显示实际步数；顶部横轴
        在相同位置显示监视母线的电压幅值，用于观察雅可比矩阵趋于
        奇异时电压所处的水平。
        """
        selected_bus, bus_index = self._resolve_bus(result, bus_number)
        point_count = len(result.points)
        positions = np.arange(point_count)
        steps = np.array([point.step_index for point in result.points])
        voltages = np.array(
            [point.voltage[bus_index] for point in result.points]
        )
        conditions = np.array(
            [point.jacobian_condition_number for point in result.points]
        )
        valid = np.isfinite(conditions) & (conditions > 0.0)
        if not np.any(valid):
            raise ValueError("CPF 结果中没有可绘制的正有限条件数")

        nose_index = self._nose_index(result)
        figure, step_axis = plt.subplots(
            figsize=(11, 6),
            constrained_layout=True,
        )
        voltage_axis = step_axis.twiny()

        step_axis.semilogy(
            positions[valid],
            conditions[valid],
            color="#9467bd",
            linewidth=2,
            marker="o",
            markersize=3,
            label="NR 雅可比矩阵条件数",
        )
        if valid[nose_index]:
            step_axis.scatter(
                positions[nose_index],
                conditions[nose_index],
                color="#d62728",
                s=80,
                zorder=5,
                label="鼻点",
            )
        step_axis.axvline(
            positions[nose_index],
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
        )

        tick_count = min(point_count, 12)
        tick_positions = np.unique(
            np.linspace(0, point_count - 1, tick_count, dtype=int)
        )
        step_axis.set_xticks(tick_positions)
        step_axis.set_xticklabels(steps[tick_positions])
        step_axis.set_xlim(-0.5, point_count - 0.5)
        step_axis.set_title(
            "NR 雅可比矩阵条件数与电压幅值对应关系",
            fontsize=15,
            pad=14,
        )
        step_axis.set_xlabel("CPF 步数（等间距）", fontsize=12)
        step_axis.set_ylabel("雅可比矩阵条件数（对数坐标）", fontsize=12)
        step_axis.grid(True, which="both", linestyle="--", alpha=0.45)
        step_axis.legend(loc="best")

        voltage_axis.set_xlim(step_axis.get_xlim())
        voltage_axis.set_xticks(tick_positions)
        voltage_axis.set_xticklabels(
            [f"{voltages[index]:.4f}" for index in tick_positions]
        )
        voltage_axis.set_xlabel(
            f"母线 {selected_bus} 电压幅值 (p.u.)",
            fontsize=12,
            labelpad=10,
        )

        return self._save_figure(figure, save_name)

    def plot_all(
        self,
        result: CPFResult,
        bus_number: int | None = None,
    ) -> tuple[Path, ...]:
        """生成全部 CPF 标准图像。"""
        if not result.points:
            return ()
        return (
            self.plot_pv_curve(result, bus_number=bus_number),
            self.plot_min_voltage(result),
            self.plot_condition_number(result),
            self.plot_step_control(result),
            self.plot_condition_with_voltage_axis(
                result,
                bus_number=bus_number,
            ),
        )


def plot_pv_curve(
    result: CPFResult,
    output_path: str | Path | None = None,
    bus_number: int | None = None,
) -> Path:
    """兼容函数式调用方式绘制单条 PV 曲线。"""
    path = Path(output_path or "CPF_01_PV_Curve.png")
    plotter = CPFPlotter(path.parent)
    return plotter.plot_pv_curve(
        result,
        bus_number=bus_number,
        save_name=path.name,
    )
