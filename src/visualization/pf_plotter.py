import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

from ..analysis.scaling import fit_power_law


def _setup_matplotlib_environment():
    """
    解决中文显示问题：强制使用 Windows 自带的微软雅黑字体
    """
    # 1. 解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

    # 2. (关键修改) 不搞复杂的自动查找了，直接指定微软雅黑和黑体
    #    您的Windows电脑里肯定有这两个字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    
    # 3. 数学符号保持正常
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    plt.rcParams['mathtext.default'] = 'regular'

    # 4. 图像排版设置
    plt.rcParams['figure.autolayout'] = False

_setup_matplotlib_environment()



class PFPlotter:
    def __init__(self, save_dir="output/plots"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # 新增通用辅助函数，避免四张对比图重复写逻辑
    def _save_figure(self, fig, save_name):
        file_path = os.path.join(self.save_dir, save_name)
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[绘图] 图像已保存至: {file_path}")
        return file_path

    def _set_x_ticks(self, ax, x_values, x_labels=None):
        """按等间距位置设置横轴刻度，并可显示真实节点编号。"""
        x_values = np.asarray(x_values)
        n = len(x_values)
        if n == 0:
            return

        x_labels = x_values if x_labels is None else np.asarray(x_labels)
        step = 1 if n <= 12 else max(1, int(np.ceil(n / 12)))
        tick_idx = np.arange(0, n, step)

        # 保证最后一个节点编号能够显示
        if tick_idx[-1] != n - 1:
            tick_idx = np.append(tick_idx, n - 1)

        ax.set_xticks(x_values[tick_idx])
        ax.set_xticklabels(x_labels[tick_idx])

    def _finalize_axes(self, ax, title, xlabel, ylabel, x_values, x_labels=None):
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel(xlabel, fontsize=12, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', labelsize=10)
        self._set_x_ticks(ax, x_values, x_labels)

        # 自动留白，防止点贴边
        ax.margins(x=0.04, y=0.10)

    def _mark_deviation_points(
        self, ax, x, computed, reference, tol, unit="", label_prefix="超限点", point_labels=None
    ):
        """
        高亮偏差超限点，并标注最大偏差点
        """
        x = np.asarray(x)
        point_labels = x if point_labels is None else np.asarray(point_labels)
        computed = np.asarray(computed, dtype=float)
        reference = np.asarray(reference, dtype=float)

        diff = computed - reference
        abs_diff = np.abs(diff)

        if len(abs_diff) == 0:
            return diff

        max_idx = int(np.argmax(abs_diff))
        max_dev = abs_diff[max_idx]

        # 超限点：abs(diff) > tol
        outlier_idx = np.where(abs_diff > tol)[0] if tol is not None else np.array([max_idx])

        if outlier_idx.size > 0:
            ax.scatter(
                x[outlier_idx],
                computed[outlier_idx],
                s=80,
                facecolors='none',
                edgecolors='#d62728',
                linewidths=2.0,
                zorder=5,
                label=f"{label_prefix}(|Δ|>{tol}{unit})" if tol is not None else "最大偏差点"
            )

        # 单独标出最大偏差点
        ax.scatter(
            [x[max_idx]],
            [computed[max_idx]],
            s=120,
            marker='*',
            color='#d62728',
            zorder=6,
            label=f"最大偏差点 |Δ|={max_dev:.4g}{unit}"
        )

        # 文本标注
        ax.annotate(
            f"max Δ={max_dev:.4g}{unit}\nBus {point_labels[max_idx]}",
            xy=(x[max_idx], computed[max_idx]),
            xytext=(0, 18),
            textcoords='offset points',
            ha='center',
            fontsize=9,
            color='#d62728',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#d62728', alpha=0.85)
        )

        # 给 y 轴自动留一点上下边界
        y_all = np.concatenate([computed, reference])
        y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
        span = y_max - y_min
        if span <= 1e-12:
            span = 1.0
        ax.set_ylim(y_min - 0.08 * span, y_max + 0.12 * span)

        return diff

    def _plot_comparison_chart(
        self,
        x,
        computed,
        reference,
        title,
        xlabel,
        ylabel,
        save_name,
        tol,
        computed_label="计算值",
        reference_label="标准值",
        unit="",
        x_labels=None
    ):
        x = np.asarray(x)
        computed = np.asarray(computed, dtype=float)
        reference = np.asarray(reference, dtype=float)

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

        ax.plot(
            x, computed,
            marker='o', markersize=5.5,
            linewidth=2.0,
            color='#1f77b4',
            label=computed_label
        )
        ax.plot(
            x, reference,
            marker='s', markersize=5.0,
            linewidth=1.8,
            linestyle='--',
            color='#ff7f0e',
            label=reference_label
        )

        self._mark_deviation_points(
            ax, x, computed, reference, tol, unit=unit, point_labels=x_labels
        )

        self._finalize_axes(ax, title, xlabel, ylabel, x, x_labels=x_labels)

        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        return self._save_figure(fig, save_name)

    def plot_all_comparisons(
        self,
        grid,
        v_tol=0.02,
        angle_tol=2.0,
        p_tol=0.02,
        q_tol=0.02,
        prefix="IEEE",
    ):
        """
        一次性绘制四张对比图：
        1) 电压：V vs v_final
        2) 相角：theta vs angle_final
        3) 有功：P_calc vs P_spec
        4) 非PQ节点无功：Q_calc vs Q_spec
        """
        bus_numbers = np.array([bus['number'] for bus in grid.buses], dtype=int)
        # 实际绘图坐标采用等间距序号，节点编号只作为横轴标签显示
        bus_positions = np.arange(len(bus_numbers))

        # 1. 电压对比
        v_ref = np.array([bus.get('v_final', 1.0) for bus in grid.buses], dtype=float)
        self._plot_comparison_chart(
            x=bus_positions,
            x_labels=bus_numbers,
            computed=np.asarray(grid.V, dtype=float),
            reference=v_ref,
            title="节点电压幅值对比：计算值 vs 数据标准结果",
            xlabel="节点编号",
            ylabel="电压幅值 (p.u.)",
            save_name=f"{prefix}_01_Voltage_Comparison.png",
            tol=v_tol,
            computed_label="计算电压 V",
            reference_label="数据给定 v_final",
            unit=" p.u."
        )

        # 2. 相角对比（统一使用度）
        theta_calc_deg = np.degrees(np.asarray(grid.theta, dtype=float))
        theta_ref_deg = np.array(
            [bus.get('angle_final', bus.get('angle', 0.0)) for bus in grid.buses],
            dtype=float
        )
        self._plot_comparison_chart(
            x=bus_positions,
            x_labels=bus_numbers,
            computed=theta_calc_deg,
            reference=theta_ref_deg,
            title="节点相角对比：计算值 vs 数据标准结果",
            xlabel="节点编号",
            ylabel="相角 (deg)",
            save_name=f"{prefix}_02_Angle_Comparison.png",
            tol=angle_tol,
            computed_label="计算相角 θ",
            reference_label="数据给定 angle_final",
            unit=" deg"
        )

        # 3. 有功对比：使用标幺值 P_calc vs P_spec
        self._plot_comparison_chart(
            x=bus_positions,
            x_labels=bus_numbers,
            computed=np.asarray(grid.P_calc, dtype=float),
            reference=np.asarray(grid.P_spec, dtype=float),
            title="有功功率对比：计算值 vs 标准给定值",
            xlabel="节点编号",
            ylabel="有功功率 (p.u.)",
            save_name=f"{prefix}_03_Active_Power_Comparison.png",
            tol=p_tol,
            computed_label="计算有功 P_calc",
            reference_label="标准给定 P_spec",
            unit=" p.u."
        )

        # 4. 无功对比：只画非PQ节点（PV + Slack）
        non_pq_mask = np.asarray(grid.bus_type) != 1
        non_pq_bus_numbers = bus_numbers[non_pq_mask]
        non_pq_positions = np.arange(len(non_pq_bus_numbers))

        q_calc_non_pq = np.asarray(grid.Q_calc, dtype=float)[non_pq_mask]
        q_spec_non_pq = np.asarray(grid.Q_spec, dtype=float)[non_pq_mask]

        if len(non_pq_bus_numbers) > 0:
            self._plot_comparison_chart(
                x=non_pq_positions,
                x_labels=non_pq_bus_numbers,
                computed=q_calc_non_pq,
                reference=q_spec_non_pq,
                title="非PQ节点无功对比：计算值 vs 标准给定值",
                xlabel="节点编号（非PQ节点）",
                ylabel="无功功率 (p.u.)",
                save_name=f"{prefix}_04_Reactive_NonPQ_Comparison.png",
                tol=q_tol,
                computed_label="计算无功 Q_calc",
                reference_label="标准给定 Q_spec",
                unit=" p.u."
            )
        else:
            print("[绘图] 当前系统没有非PQ节点，跳过无功对比图。")

    def plot_deviation_outliers_summary(
        self,
        grid,
        v_tol=0.02,
        angle_tol=2.0,
        p_tol=0.02,
        q_tol=0.02,
        save_name="IEEE_Deviation_Outliers_Summary.png",
    ):
        """
        将 V、theta、P、Q 四类偏差超限节点绘制在同一张 2×2 柱状图中。

        偏差定义：
            deviation = computed - reference

        仅绘制满足 abs(deviation) > tolerance 的节点。
        Q 偏差仅统计非 PQ 节点（PV 与 Slack）。
        """
        bus_numbers = np.array(
            [bus["number"] for bus in grid.buses],
            dtype=int
        )

        v_reference = np.array(
            [bus.get("v_final", 1.0) for bus in grid.buses],
            dtype=float
        )
        v_diff = np.asarray(grid.V, dtype=float) - v_reference

        theta_reference = np.array(
            [
                bus.get("angle_final", bus.get("angle", 0.0))
                for bus in grid.buses
            ],
            dtype=float
        )
        theta_diff = (
            np.degrees(np.asarray(grid.theta, dtype=float))
            - theta_reference
        )

        # =====================================================
        # 平衡节点 PQ 功率偏差
        # =====================================================
        slack_mask = np.asarray(grid.bus_type) == 3
        slack_bus_numbers = bus_numbers[slack_mask]
        if np.any(slack_mask):
            slack_p_diff = (
                np.asarray(grid.P_calc, dtype=float)[slack_mask]
                -
                np.asarray(grid.P_spec, dtype=float)[slack_mask]
            )

            slack_q_diff = (
                np.asarray(grid.Q_calc, dtype=float)[slack_mask]
                -
                np.asarray(grid.Q_spec, dtype=float)[slack_mask]
            )
            slack_labels = np.concatenate(
                [
                    np.array(
                        [
                            f"Bus {x}-P"
                            for x in slack_bus_numbers
                        ]
                    ),
                    np.array(
                        [
                            f"Bus {x}-Q"
                            for x in slack_bus_numbers
                        ]
                    )
                ]
            )
            slack_pq_diff = np.concatenate(
                [
                    slack_p_diff,
                    slack_q_diff
                ]
            )
        else:
            slack_labels = np.array([])
            slack_pq_diff = np.array([])


        non_pq_mask = np.asarray(grid.bus_type) != 1
        q_bus_numbers = bus_numbers[non_pq_mask]
        q_diff = (
            np.asarray(grid.Q_calc, dtype=float)[non_pq_mask]
            - np.asarray(grid.Q_spec, dtype=float)[non_pq_mask]
        )

        datasets = [
            (
                "电压偏差超限节点",
                bus_numbers,
                v_diff,
                v_tol,
                "ΔV (p.u.)",
                "p.u."
            ),

            (
                "相角偏差超限节点",
                bus_numbers,
                theta_diff,
                angle_tol,
                "Δθ (deg)",
                "deg"
            ),

            (
                "平衡节点PQ功率偏差",
                slack_labels,
                slack_pq_diff,
                p_tol,
                "ΔP / ΔQ (p.u.)",
                "p.u."
            ),

            (
                "非PQ节点无功偏差超限节点",
                q_bus_numbers,
                q_diff,
                q_tol,
                "ΔQ (p.u.)",
                "p.u."
            ),
        ]

        fig, axes = plt.subplots(
            2, 2, figsize=(16, 10), constrained_layout=True
        )
        axes = axes.ravel()

        total_outliers = 0

        for ax, dataset in zip(axes, datasets):
            title, numbers, diff, tol, ylabel, unit = dataset
            numbers = np.asarray(numbers)
            diff = np.asarray(diff, dtype=float)

            outlier_idx = np.where(np.abs(diff) > tol)[0]
            total_outliers += len(outlier_idx)

            ax.set_title(
                f"{title}（{len(outlier_idx)}个）",
                fontsize=13,
                pad=10
            )
            ax.set_xlabel("节点编号", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.axhline(0.0, color="black", linewidth=0.9)

            if len(outlier_idx) == 0:
                ax.text(
                    0.5, 0.5,
                    "无偏差超限节点",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14
                )
                ax.set_xticks([])
                continue

            outlier_numbers = numbers[outlier_idx]
            outlier_diff = diff[outlier_idx]
            x_positions = np.arange(len(outlier_idx))

            colors = np.where(
                outlier_diff >= 0,
                "#d62728",
                "#1f77b4"
            )

            bars = ax.bar(
                x_positions,
                outlier_diff,
                color=colors,
                alpha=0.85,
                width=0.72
            )

            ax.axhline(
                tol,
                color="#ff7f0e",
                linestyle="--",
                linewidth=1.2,
                label=f"+阈值 {tol:g} {unit}"
            )
            ax.axhline(
                -tol,
                color="#ff7f0e",
                linestyle="--",
                linewidth=1.2,
                label=f"-阈值 {-tol:g} {unit}"
            )

            n = len(outlier_idx)
            step = 1 if n <= 20 else max(1, int(np.ceil(n / 15)))
            tick_idx = np.arange(0, n, step)

            if tick_idx[-1] != n - 1:
                tick_idx = np.append(tick_idx, n - 1)

            ax.set_xticks(x_positions[tick_idx])
            ax.set_xticklabels(
                outlier_numbers[tick_idx],
                rotation=45,
                ha="right"
            )

            if n <= 20:
                label_indices = np.arange(n)
            else:
                label_indices = np.argsort(np.abs(outlier_diff))[-20:]

            y_span = float(np.ptp(outlier_diff))
            if y_span <= 1e-12:
                y_span = max(float(np.max(np.abs(outlier_diff))), 1.0)
            offset = 0.02 * y_span

            for idx in label_indices:
                value = outlier_diff[idx]
                bar = bars[idx]
                y_text = value + offset if value >= 0 else value - offset
                va = "bottom" if value >= 0 else "top"

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_text,
                    f"{value:.4g}",
                    ha="center",
                    va=va,
                    fontsize=8,
                    rotation=90 if n > 12 else 0
                )

            ax.legend(fontsize=8, loc="best", framealpha=0.9)
            ax.margins(x=0.02, y=0.15)

        fig.suptitle(
            f"V、θ、P、Q 偏差超限汇总（共 {total_outliers} 个超限点）",
            fontsize=16
        )

        return self._save_figure(fig, save_name)


    # 保留原有电压分布图，兼容旧调用方式
    def plot_voltage_profile(self, grid, save_name="voltage_profile.png"):
        """绘制系统节点电压幅值分布图（仅计算值）"""
        bus_numbers = np.array([bus['number'] for bus in grid.buses], dtype=int)
        bus_positions = np.arange(len(bus_numbers))

        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax.plot(bus_positions, grid.V, marker='o', linestyle='-', color='#1f77b4', label="计算电压 (V)")
        ax.axhline(1.05, color='red', linestyle='--', alpha=0.7, label="上限 (1.05)")
        ax.axhline(0.95, color='green', linestyle='--', alpha=0.7, label="下限 (0.95)")

        self._finalize_axes(
            ax, "电力系统节点电压幅值分布", "节点编号", "电压幅值 (pu)",
            bus_positions, x_labels=bus_numbers
        )
        ax.legend(fontsize=10, loc='best', framealpha=0.9)

        self._save_figure(fig, save_name)

    # 收敛曲线绘图
    def plot_convergence(self, info, algo_name="Newton-Raphson", save_name="convergence.png"):
        if 'max_error_history' not in info or not info['max_error_history']:
            return

        errors = info['max_error_history']
        iterations = range(1, len(errors) + 1)

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(iterations, errors, marker='s', color='#ff7f0e', linewidth=2)
        ax.set_yscale('log')

        ax.set_title(f"{algo_name} 算法收敛过程", fontsize=14, pad=10)
        ax.set_xlabel("迭代次数", fontsize=12)
        ax.set_ylabel("最大不平衡量 (Max Error)", fontsize=12)
        ax.grid(True, which="both", linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', labelsize=10)

        self._save_figure(fig, save_name)

    def plot_solver_comparison(
        self,
        comparison,
        save_name="Solver_NR_FDLF_Comparison.png",
    ):
        """绘制误差、观测收敛阶和重复计时分布。"""
        benchmarks = [comparison.nr, comparison.fast_decoupled]
        short_names = ["NR", "快速解耦"]
        colors = ["#1f77b4", "#ff7f0e"]
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(15, 10),
            constrained_layout=True,
        )

        (
            convergence_axis,
            order_axis,
            timing_axis,
            breakdown_axis,
        ) = axes.flat
        for benchmark, color in zip(benchmarks, colors):
            errors = np.asarray(benchmark.error_history, dtype=float)
            valid = np.isfinite(errors) & (errors > 0.0)
            iterations = np.arange(1, len(errors) + 1)
            if np.any(valid):
                convergence_axis.semilogy(
                    iterations[valid],
                    errors[valid],
                    marker="o",
                    markersize=4,
                    linewidth=2,
                    color=color,
                    label=(
                        f"{benchmark.name} "
                        f"({benchmark.iterations} 次)"
                    ),
                )

            # 使用连续三个误差点估算局部观测收敛阶：
            # p_k = ln(e_{k+1}/e_k) / ln(e_k/e_{k-1})。
            order_iterations = []
            observed_orders = []
            for index in range(1, len(errors) - 1):
                previous_error = errors[index - 1]
                current_error = errors[index]
                next_error = errors[index + 1]
                if not (
                    np.isfinite(previous_error)
                    and np.isfinite(current_error)
                    and np.isfinite(next_error)
                    and previous_error > current_error > next_error > 1e-14
                ):
                    continue
                denominator = np.log(current_error / previous_error)
                if abs(denominator) <= 1e-12:
                    continue
                observed_order = (
                    np.log(next_error / current_error) / denominator
                )
                if np.isfinite(observed_order):
                    order_iterations.append(index + 2)
                    observed_orders.append(observed_order)

            if observed_orders:
                order_axis.plot(
                    order_iterations,
                    observed_orders,
                    marker="o",
                    markersize=5,
                    linewidth=2,
                    color=color,
                    label=benchmark.name,
                )

        convergence_axis.set_title("算法收敛过程", fontsize=14)
        convergence_axis.set_xlabel("迭代次数", fontsize=11)
        convergence_axis.set_ylabel("最大功率不平衡量", fontsize=11)
        convergence_axis.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.5,
        )
        convergence_axis.legend(loc="best", fontsize=9)

        order_axis.axhline(
            1.0,
            color="#2ca02c",
            linestyle="--",
            linewidth=1.5,
            label="线性收敛 p=1",
        )
        order_axis.axhline(
            2.0,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label="平方收敛 p=2",
        )
        order_axis.set_title("逐步观测收敛阶", fontsize=14)
        order_axis.set_xlabel("迭代次数", fontsize=11)
        order_axis.set_ylabel("观测阶 p", fontsize=11)
        order_axis.set_ylim(0.0, 2.6)
        order_axis.grid(True, linestyle="--", alpha=0.5)
        order_axis.legend(loc="best", fontsize=8)

        timing_samples = [
            1000.0 * np.asarray(item.elapsed_samples, dtype=float)
            for item in benchmarks
        ]
        boxplot = timing_axis.boxplot(
            timing_samples,
            tick_labels=short_names,
            patch_artist=True,
            showmeans=True,
        )
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        for position, (benchmark, samples, color) in enumerate(
            zip(benchmarks, timing_samples, colors),
            start=1,
        ):
            timing_axis.scatter(
                np.full(len(samples), position),
                samples,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            excluded_ms = (
                1000.0
                * np.asarray(
                    benchmark.excluded_elapsed_samples,
                    dtype=float,
                )
            )
            if len(excluded_ms) > 0:
                timing_axis.scatter(
                    np.full(len(excluded_ms), position),
                    excluded_ms,
                    marker="x",
                    s=110,
                    color="#d62728",
                    linewidth=2.5,
                    zorder=7,
                    label=(
                        "IQR 识别的右侧异常值"
                        if position == 1
                        else None
                    ),
                )
                excluded_text = ", ".join(
                    f"{value:.3f}"
                    for value in excluded_ms
                )
                timing_axis.annotate(
                    f"排除 {excluded_text} ms",
                    xy=(position, float(np.max(excluded_ms))),
                    xytext=(8, -14),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="#d62728",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "edgecolor": "#d62728",
                        "alpha": 0.85,
                    },
                )
            timing_axis.text(
                position,
                1.02,
                (
                    f"中位数 {benchmark.median_time * 1000.0:.3f} ms\n"
                    f"均值 {benchmark.mean_time * 1000.0:.3f} ms"
                ),
                transform=timing_axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.9,
                },
                zorder=6,
            )

        timing_axis.set_title(
            "重复运行耗时分布（右侧异常值按 IQR 排除）",
            fontsize=14,
            pad=40,
        )
        timing_axis.set_yscale("log")
        timing_axis.set_ylabel("求解耗时 (ms，对数坐标)", fontsize=11)
        timing_axis.grid(
            axis="y",
            which="both",
            linestyle="--",
            alpha=0.5,
        )

        phase_names = [
            "功率不平衡量",
            "矩阵构造",
            "矩阵分解/预处理",
            "线性求解/回代",
            "状态更新",
            "其余开销",
        ]
        phase_colors = [
            "#4c78a8",
            "#f58518",
            "#e45756",
            "#72b7b2",
            "#54a24b",
            "#b8b8b8",
        ]
        breakdown_axis.set_title("平均总耗时组成", fontsize=14, pad=12)
        breakdown_axis.axis("off")
        pie_axes = [
            breakdown_axis.inset_axes([0.00, 0.13, 0.48, 0.78]),
            breakdown_axis.inset_axes([0.52, 0.13, 0.48, 0.78]),
        ]
        legend_wedges = None
        for pie_axis, benchmark, short_name in zip(
            pie_axes,
            benchmarks,
            short_names,
        ):
            breakdown = benchmark.timing_breakdown
            values = np.asarray(
                [
                    breakdown.mismatch_time,
                    breakdown.matrix_build_time,
                    breakdown.factorization_time,
                    breakdown.linear_solve_time,
                    breakdown.state_update_time,
                    breakdown.other_time,
                ],
                dtype=float,
            )

            def show_percentage(percentage):
                return f"{percentage:.1f}%" if percentage >= 2.0 else ""

            wedges, _, _ = pie_axis.pie(
                values,
                colors=phase_colors,
                startangle=90,
                counterclock=False,
                autopct=show_percentage,
                pctdistance=0.68,
                textprops={"fontsize": 8},
                wedgeprops={"linewidth": 0.7, "edgecolor": "white"},
            )
            legend_wedges = wedges
            pie_axis.set_title(
                f"{short_name}\n均值 {benchmark.mean_time * 1000.0:.3f} ms",
                fontsize=10,
            )

        breakdown_axis.legend(
            legend_wedges,
            phase_names,
            fontsize=8,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=3,
        )
        return self._save_figure(fig, save_name)

    def plot_solver_tolerance_comparison(
        self,
        tolerance_results,
        save_name="Solver_Tolerance_Comparison.png",
    ):
        """绘制 NR 与快速解耦法在不同收敛精度下的性能变化。"""
        if not tolerance_results:
            return None

        tolerances = np.asarray(
            [item[0] for item in tolerance_results],
            dtype=float,
        )
        comparisons = [item[1] for item in tolerance_results]
        positions = np.arange(len(comparisons))
        labels = [f"{value:.0e}" for value in tolerances]
        nr_iterations = np.asarray(
            [item.nr.iterations if item.nr.success else np.nan for item in comparisons],
            dtype=float,
        )
        fd_iterations = np.asarray(
            [
                item.fast_decoupled.iterations
                if item.fast_decoupled.success
                else np.nan
                for item in comparisons
            ],
            dtype=float,
        )
        nr_times = 1000.0 * np.asarray(
            [item.nr.median_time for item in comparisons],
            dtype=float,
        )
        fd_times = 1000.0 * np.asarray(
            [item.fast_decoupled.median_time for item in comparisons],
            dtype=float,
        )
        nr_errors = np.asarray(
            [item.nr.final_error for item in comparisons],
            dtype=float,
        )
        fd_errors = np.asarray(
            [item.fast_decoupled.final_error for item in comparisons],
            dtype=float,
        )
        speed_ratios = np.asarray(
            [item.speed_ratio for item in comparisons], dtype=float
        )

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 10),
            constrained_layout=True,
        )
        iteration_axis, timing_axis, error_axis, ratio_axis = axes.flat
        colors = ["#1f77b4", "#ff7f0e"]

        iteration_axis.plot(
            positions,
            nr_iterations,
            marker="o",
            linewidth=2,
            color=colors[0],
            label="Newton-Raphson",
        )
        iteration_axis.plot(
            positions,
            fd_iterations,
            marker="o",
            linewidth=2,
            color=colors[1],
            label="Fast-Decoupled (XB)",
        )
        iteration_axis.set_title("不同精度下的迭代次数", fontsize=14)
        iteration_axis.set_ylabel("迭代次数", fontsize=11)
        iteration_axis.grid(True, linestyle="--", alpha=0.5)
        iteration_axis.legend(fontsize=9)

        timing_axis.semilogy(
            positions,
            nr_times,
            marker="o",
            linewidth=2,
            color=colors[0],
            label="Newton-Raphson",
        )
        timing_axis.semilogy(
            positions,
            fd_times,
            marker="o",
            linewidth=2,
            color=colors[1],
            label="Fast-Decoupled (XB)",
        )
        timing_axis.set_title("不同精度下的中位求解耗时", fontsize=14)
        timing_axis.set_ylabel("中位耗时 (ms，对数坐标)", fontsize=11)
        timing_axis.grid(True, which="both", linestyle="--", alpha=0.5)
        timing_axis.legend(fontsize=9)

        error_axis.semilogy(
            positions,
            nr_errors,
            marker="o",
            linewidth=2,
            color=colors[0],
            label="NR 最终残差",
        )
        error_axis.semilogy(
            positions,
            fd_errors,
            marker="o",
            linewidth=2,
            color=colors[1],
            label="FDLF 最终残差",
        )
        error_axis.semilogy(
            positions,
            tolerances,
            marker="x",
            linestyle="--",
            linewidth=1.5,
            color="#666666",
            label="目标精度",
        )
        error_axis.set_title("目标精度与最终功率残差", fontsize=14)
        error_axis.set_ylabel("最大功率不平衡量", fontsize=11)
        error_axis.grid(True, which="both", linestyle="--", alpha=0.5)
        error_axis.legend(fontsize=9)

        ratio_axis.bar(
            positions,
            speed_ratios,
            width=0.6,
            color="#54a24b",
            alpha=0.85,
        )
        ratio_axis.axhline(
            1.0,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label="两种算法耗时相同",
        )
        ratio_axis.set_title("快速解耦法相对加速比", fontsize=14)
        ratio_axis.set_ylabel("NR中位耗时 / FDLF中位耗时", fontsize=11)
        ratio_axis.grid(axis="y", linestyle="--", alpha=0.5)
        ratio_axis.legend(fontsize=9)

        for axis in axes.flat:
            axis.set_xticks(positions)
            axis.set_xticklabels(labels)
            axis.set_xlabel("收敛精度（向右逐渐严格）", fontsize=11)

        return self._save_figure(fig, save_name)

    def plot_solver_robustness(
        self,
        result,
        save_name="Solver_Robustness.png",
    ):
        """绘制初值扰动、负荷倍率和线路 R/X 扰动的鲁棒性结果。"""
        solver_names = ["Newton-Raphson", "Fast-Decoupled (XB)"]
        solver_labels = ["NR", "快速解耦"]
        colors = ["#1f77b4", "#ff7f0e"]
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(15, 10),
            constrained_layout=True,
        )
        initial_rate_axis, initial_iteration_axis, load_axis, resistance_axis = (
            axes.flat
        )

        def category_data(category, solver_name):
            return [
                item
                for item in result.summaries
                if item.category == category and item.solver_name == solver_name
            ]

        def grouped_bars(axis, category, value_name, title, ylabel):
            first = category_data(category, solver_names[0])
            labels = [item.scenario for item in first]
            positions = np.arange(len(labels), dtype=float)
            width = 0.38
            for solver_index, (solver_name, solver_label, color) in enumerate(
                zip(solver_names, solver_labels, colors)
            ):
                summaries = category_data(category, solver_name)
                values = np.asarray(
                    [getattr(item, value_name) for item in summaries],
                    dtype=float,
                )
                offsets = positions + (solver_index - 0.5) * width
                bars = axis.bar(
                    offsets,
                    np.nan_to_num(values, nan=0.0),
                    width=width,
                    color=color,
                    alpha=0.85,
                    label=solver_label,
                )
                for bar, value, summary in zip(bars, values, summaries):
                    if value_name == "success_rate":
                        text = f"{value:.0%}"
                    elif np.isfinite(value):
                        text = f"{value:.1f}"
                    else:
                        text = "失败"
                    axis.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                        text,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="#d62728" if not summary.success_count else "#333333",
                    )
            axis.set_title(title, fontsize=14)
            axis.set_ylabel(ylabel, fontsize=11)
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=20, ha="right")
            axis.grid(axis="y", linestyle="--", alpha=0.5)
            axis.legend(fontsize=9)

        grouped_bars(
            initial_rate_axis,
            "initial",
            "success_rate",
            "不同初值下的收敛成功率",
            "成功率",
        )
        initial_rate_axis.set_ylim(0.0, 1.12)
        initial_rate_axis.yaxis.set_major_formatter(PercentFormatter(1.0))

        grouped_bars(
            initial_iteration_axis,
            "initial",
            "mean_iterations",
            "不同初值下的平均迭代次数",
            "成功试验平均迭代次数",
        )
        for solver_name, solver_label, color in zip(
            solver_names,
            solver_labels,
            colors,
        ):
            summaries = category_data("load", solver_name)
            parameters = np.asarray(
                [item.parameter for item in summaries],
                dtype=float,
            )
            success_rates = np.asarray(
                [item.success_rate for item in summaries],
                dtype=float,
            )
            load_axis.plot(
                parameters,
                success_rates,
                marker="o",
                markersize=5,
                linewidth=2,
                color=color,
                label=solver_label,
            )
            bracket = result.load_convergence_bracket(solver_name)
            if bracket is not None:
                low, high = bracket
                load_axis.axvspan(low, high, color=color, alpha=0.16)
                load_axis.axvline(
                    (low + high) / 2.0,
                    color=color,
                    linestyle="--",
                    linewidth=1.3,
                )
                load_axis.annotate(
                    f"{solver_label}: [{low:.6f}, {high:.6f}]",
                    xy=((low + high) / 2.0, 0.48),
                    xytext=(4, 0),
                    textcoords="offset points",
                    rotation=90,
                    ha="left",
                    va="center",
                    fontsize=8,
                    color=color,
                )
        load_axis.set_title("鼻点附近负荷倍率收敛边界", fontsize=14)
        load_axis.set_xlabel("负荷倍率", fontsize=11)
        load_axis.set_ylabel("收敛成功率", fontsize=11)
        load_axis.set_ylim(0.0, 1.12)
        load_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        load_axis.grid(True, linestyle="--", alpha=0.5)
        load_axis.legend(fontsize=9, loc="best")
        grouped_bars(
            resistance_axis,
            "resistance",
            "mean_iterations",
            "线路电阻倍率（R/X）平均迭代次数",
            "成功试验平均迭代次数",
        )

        return self._save_figure(fig, save_name)

    def plot_critical_stagnation(
        self,
        result,
        save_name="Solver_Critical_Stagnation.png",
    ):
        """绘制鼻点前后四种潮流算法的停滞特征。"""
        solver_specs = (
            ("Newton-Raphson", "NR", "#1f77b4", "o"),
            ("Fast-Decoupled (XB)", "FDLF", "#ff7f0e", "^"),
            ("Optimal Multiplier", "最优乘子", "#2ca02c", "s"),
            ("Nonlinear Programming", "非线性规划", "#9467bd", "D"),
        )
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(15, 10),
            constrained_layout=True,
        )
        residual_axis, iteration_axis, history_axis, control_axis = axes.flat

        selected_points = []
        for solver_name, label, color, marker in solver_specs:
            points = list(result.points_for(solver_name))
            multipliers = np.asarray(
                [item.load_multiplier for item in points], dtype=float
            )
            residuals = np.asarray(
                [max(item.final_error, 1e-16) for item in points], dtype=float
            )
            # 极端发散值只在图上截顶，避免压缩其他方法的非零停滞残差。
            displayed_residuals = np.minimum(residuals, 1e3)
            iterations = np.asarray([item.iterations for item in points])
            terminal_controls = np.asarray(
                [
                    max(item.terminal_control, 1e-8)
                    if np.isfinite(item.terminal_control)
                    else np.nan
                    for item in points
                ],
                dtype=float,
            )
            residual_axis.semilogy(
                multipliers,
                displayed_residuals,
                marker=marker,
                linewidth=2,
                color=color,
                label=label,
            )
            failed = np.asarray([not item.success for item in points])
            if np.any(failed):
                residual_axis.scatter(
                    multipliers[failed],
                    displayed_residuals[failed],
                    marker="x",
                    s=100,
                    linewidth=2.2,
                    color="#d62728",
                    zorder=5,
                )
            iteration_axis.plot(
                multipliers,
                iterations,
                marker=marker,
                linewidth=2,
                color=color,
                label=label,
            )
            if solver_name in {"Optimal Multiplier", "Nonlinear Programming"}:
                control_axis.semilogy(
                    multipliers,
                    terminal_controls,
                    marker=marker,
                    linewidth=2,
                    color=color,
                    label=(
                        "终止乘子 μ"
                        if solver_name == "Optimal Multiplier"
                        else "终止步长 α"
                    ),
                )

            successful = [item for item in points if item.success]
            failures = [item for item in points if not item.success]
            if successful:
                selected_points.append((max(successful, key=lambda item: item.load_multiplier), color, "--"))
            if failures:
                selected_points.append((min(failures, key=lambda item: item.load_multiplier), color, "-"))

        short_names = {
            "Newton-Raphson": "NR",
            "Fast-Decoupled (XB)": "FDLF",
            "Optimal Multiplier": "OM",
            "Nonlinear Programming": "NLP",
        }
        for point, color, linestyle in selected_points:
            errors = np.asarray(point.error_history, dtype=float)
            valid = np.isfinite(errors) & (errors > 0.0)
            history_axis.semilogy(
                np.arange(1, len(errors) + 1)[valid],
                errors[valid],
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=(
                    f"{short_names[point.solver_name]} "
                    f"{point.load_multiplier:.4f}× {point.status}"
                ),
            )

        for axis in (residual_axis, iteration_axis, control_axis):
            axis.axvline(
                result.cpf_nose_multiplier,
                color="#d62728",
                linestyle="--",
                linewidth=1.6,
                label="CPF鼻点" if axis is residual_axis else None,
            )
            axis.set_xlabel("统一负荷倍率", fontsize=11)
            axis.grid(True, which="both", linestyle="--", alpha=0.5)

        residual_axis.axhline(
            1e-6,
            color="#666666",
            linestyle=":",
            linewidth=1.4,
            label="收敛阈值 1e-6",
        )
        residual_axis.set_title("鼻点前后的最终潮流残差", fontsize=14)
        residual_axis.set_ylabel("最终最大功率失配", fontsize=11)
        residual_axis.set_ylim(1e-14, 2e3)
        residual_axis.text(
            0.99,
            0.02,
            "超过 1e3 的发散残差按 1e3 截顶显示",
            transform=residual_axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#666666",
        )
        residual_axis.legend(fontsize=8)

        iteration_axis.set_title("总迭代次数（包含PV转PQ后的重求解）", fontsize=14)
        iteration_axis.set_ylabel("总迭代次数", fontsize=11)
        iteration_axis.legend(fontsize=9)

        history_axis.set_title("最后收敛点与首次失败点的残差轨迹", fontsize=14)
        history_axis.set_xlabel("累计迭代序号", fontsize=11)
        history_axis.set_ylabel("最大功率失配", fontsize=11)
        history_axis.grid(True, which="both", linestyle="--", alpha=0.5)
        history_axis.legend(fontsize=8)

        control_axis.axhline(
            1.0,
            color="#666666",
            linestyle=":",
            linewidth=1.3,
            label="完整步长",
        )
        control_axis.set_title("失败时乘子或回溯步长趋零", fontsize=14)
        control_axis.set_ylabel("终止控制量（对数坐标）", fontsize=11)
        control_axis.legend(fontsize=8)

        return self._save_figure(fig, save_name)

    def plot_solver_scaling_trend(
        self,
        scaling_results,
        minimum_nodes=30,
        save_name="Solver_Scaling_Trend.png",
    ):
        """绘制系统规模、矩阵稀疏度和求解耗时的增长趋势。"""
        if len(scaling_results) < 2:
            return None
        ordered = sorted(scaling_results, key=lambda item: item[1].node_count)
        metrics = [item[1] for item in ordered]
        comparisons = [item[2] for item in ordered]
        nodes = np.asarray([item.node_count for item in metrics], dtype=float)
        branches = np.asarray([item.branch_count for item in metrics], dtype=float)
        pq_counts = np.asarray([item.pq_count for item in metrics], dtype=float)
        state_dimensions = np.asarray(
            [item.state_dimension for item in metrics],
            dtype=float,
        )
        jacobian_nnz = np.asarray(
            [item.jacobian_nnz for item in metrics],
            dtype=float,
        )
        b_prime_nnz = np.asarray(
            [item.b_prime_nnz for item in metrics],
            dtype=float,
        )
        b_double_prime_nnz = np.asarray(
            [item.b_double_prime_nnz for item in metrics],
            dtype=float,
        )
        solver_specs = (
            ("NR", lambda item: item.nr, "#1f77b4"),
            ("FDLF", lambda item: item.fast_decoupled, "#ff7f0e"),
        )
        total_time_series = []
        iteration_time_series = []
        for name, getter, color in solver_specs:
            times = 1000.0 * np.asarray(
                [getter(item).median_time for item in comparisons],
                dtype=float,
            )
            per_iteration = times / np.maximum(
                [getter(item).iterations for item in comparisons],
                1,
            )
            total_time_series.append((name, times, color))
            iteration_time_series.append((name, per_iteration, color))

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(15, 10),
            constrained_layout=True,
        )
        topology_axis, matrix_axis, total_axis, iteration_axis = axes.flat

        topology_axis.loglog(
            nodes, branches, marker="o", linewidth=2, label="支路数"
        )
        topology_axis.loglog(
            nodes, state_dimensions, marker="s", linewidth=2, label="状态维数"
        )
        topology_axis.loglog(
            nodes, pq_counts, marker="^", linewidth=2, label="PQ节点数"
        )
        topology_axis.set_title("拓扑与状态变量规模", fontsize=14)
        topology_axis.set_xlabel("节点数", fontsize=11)
        topology_axis.set_ylabel("数量（对数坐标）", fontsize=11)
        topology_axis.grid(True, which="both", linestyle="--", alpha=0.5)
        topology_axis.legend(fontsize=9)

        matrix_axis.loglog(
            state_dimensions,
            jacobian_nnz,
            marker="o",
            linewidth=2,
            label="雅可比矩阵非零元",
        )
        matrix_axis.loglog(
            state_dimensions,
            b_prime_nnz,
            marker="s",
            linewidth=2,
            label="B' 非零元",
        )
        matrix_axis.loglog(
            state_dimensions,
            b_double_prime_nnz,
            marker="^",
            linewidth=2,
            label="B'' 非零元",
        )
        matrix_axis.set_title("线性方程组非零元增长", fontsize=14)
        matrix_axis.set_xlabel("潮流状态维数", fontsize=11)
        matrix_axis.set_ylabel("非零元数量（对数坐标）", fontsize=11)
        matrix_axis.grid(True, which="both", linestyle="--", alpha=0.5)
        matrix_axis.legend(fontsize=9)

        fit_mask = nodes >= minimum_nodes
        fit_sizes = state_dimensions[fit_mask]

        def plot_time_fit(axis, series, title, ylabel):
            fit_x = np.geomspace(np.min(fit_sizes), np.max(fit_sizes), 100)
            for name, values, color in series:
                axis.loglog(
                    state_dimensions,
                    values,
                    marker="o",
                    linewidth=1.8,
                    color=color,
                    label=f"{name} 实测",
                )
                fitted = fit_power_law(fit_sizes, values[fit_mask])
                axis.loglog(
                    fit_x,
                    fitted.predict(fit_x),
                    linestyle="--",
                    color=color,
                    label=(
                        f"{name}拟合 p={fitted.exponent:.2f}, "
                        f"R2={fitted.r_squared:.2f}"
                    ),
                )
            axis.set_title(title, fontsize=14)
            axis.set_xlabel("潮流状态维数", fontsize=11)
            axis.set_ylabel(ylabel, fontsize=11)
            axis.grid(True, which="both", linestyle="--", alpha=0.5)
            axis.legend(fontsize=8)

        plot_time_fit(
            total_axis,
            total_time_series,
            f"总耗时增长（拟合节点数 ≥ {minimum_nodes}）",
            "中位总耗时 (ms，对数坐标)",
        )
        plot_time_fit(
            iteration_axis,
            iteration_time_series,
            f"等效单次迭代耗时（拟合节点数 ≥ {minimum_nodes}）",
            "中位总耗时/迭代次数 (ms，对数坐标)",
        )

        return self._save_figure(fig, save_name)

    def plot_solver_batch_summary(
        self,
        case_results,
        save_name="Solver_All_Cases_Summary.png",
    ):
        """汇总绘制全部测试系统的迭代、观测阶和中位耗时。"""
        if not case_results:
            return None

        labels = [item[0] for item in case_results]
        positions = np.arange(len(case_results))
        solver_specs = (
            ("NR", lambda item: item.nr, "#1f77b4"),
            ("FDLF", lambda item: item.fast_decoupled, "#ff7f0e"),
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(19, 5.5),
            constrained_layout=True,
        )
        iteration_axis, order_axis, timing_axis = axes
        width = 0.38
        for solver_index, (name, getter, color) in enumerate(solver_specs):
            benchmarks = [getter(item[2]) for item in case_results]
            offset = positions + (solver_index - 0.5) * width
            iteration_axis.bar(
                offset,
                [item.iterations for item in benchmarks],
                width=width,
                color=color,
                label=name,
            )
            order_axis.plot(
                positions,
                [item.observed_order for item in benchmarks],
                marker="o",
                linewidth=1.8,
                color=color,
                label=name,
            )
            timing_axis.semilogy(
                positions,
                [1000.0 * item.median_time for item in benchmarks],
                marker="o",
                linewidth=1.8,
                color=color,
                label=name,
            )
        iteration_axis.set_title("全部算例迭代次数", fontsize=14)
        iteration_axis.set_ylabel("迭代次数", fontsize=11)
        iteration_axis.grid(axis="y", linestyle="--", alpha=0.5)
        iteration_axis.legend(fontsize=9)

        order_axis.axhline(
            1.0,
            color="#2ca02c",
            linestyle="--",
            linewidth=1.3,
            label="p=1",
        )
        order_axis.axhline(
            2.0,
            color="#d62728",
            linestyle="--",
            linewidth=1.3,
            label="p=2",
        )
        order_axis.set_title("全部算例观测收敛阶", fontsize=14)
        order_axis.set_ylabel("观测阶 p", fontsize=11)
        order_axis.set_ylim(0.0, 2.7)
        order_axis.grid(True, linestyle="--", alpha=0.5)
        order_axis.legend(fontsize=8, loc="best")

        timing_axis.set_title("全部算例中位求解耗时", fontsize=14)
        timing_axis.set_ylabel("求解耗时 (ms，对数坐标)", fontsize=11)
        timing_axis.grid(True, which="both", linestyle="--", alpha=0.5)
        timing_axis.legend(fontsize=9)

        for axis in axes:
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=45, ha="right")
            axis.set_xlabel("IEEE 测试系统节点数", fontsize=11)

        return self._save_figure(fig, save_name)

    def plot_loadability_curve(
        self,
        result,
        save_name="IEEE_07_Loadability_Scan.png",
    ):
        """绘制统一负荷倍率与系统最低电压的关系。"""
        converged = [point for point in result.points if point.converged]
        if not converged:
            return None

        multipliers = [point.multiplier for point in converged]
        min_voltages = [point.min_voltage for point in converged]
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.plot(
            multipliers,
            min_voltages,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color="#1f77b4",
            label="收敛点的最低母线电压",
        )

        bracket = result.collapse_bracket
        if bracket is not None:
            stable, failed = bracket
            ax.axvspan(
                stable,
                failed,
                color="#d62728",
                alpha=0.2,
                label="数值崩溃区间",
            )

        ax.set_title("统一负荷倍率扫描", fontsize=14, pad=10)
        ax.set_xlabel("负荷倍率", fontsize=12)
        ax.set_ylabel("全网最低电压 (p.u.)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="best")
        return self._save_figure(fig, save_name)
