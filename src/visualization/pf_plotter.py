import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


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
        colors = ["#1f77b4", "#ff7f0e"]
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 5),
            constrained_layout=True,
        )

        convergence_axis, order_axis, timing_axis = axes
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
            tick_labels=[item.name for item in benchmarks],
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
        nr_iterations = np.array(
            [item[2].nr.iterations for item in case_results],
            dtype=float,
        )
        fd_iterations = np.array(
            [item[2].fast_decoupled.iterations for item in case_results],
            dtype=float,
        )
        nr_orders = np.array(
            [item[2].nr.observed_order for item in case_results],
            dtype=float,
        )
        fd_orders = np.array(
            [
                item[2].fast_decoupled.observed_order
                for item in case_results
            ],
            dtype=float,
        )
        nr_times = 1000.0 * np.array(
            [item[2].nr.median_time for item in case_results],
            dtype=float,
        )
        fd_times = 1000.0 * np.array(
            [
                item[2].fast_decoupled.median_time
                for item in case_results
            ],
            dtype=float,
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(19, 5.5),
            constrained_layout=True,
        )
        iteration_axis, order_axis, timing_axis = axes
        width = 0.38

        iteration_axis.bar(
            positions - width / 2,
            nr_iterations,
            width=width,
            color="#1f77b4",
            label="Newton-Raphson",
        )
        iteration_axis.bar(
            positions + width / 2,
            fd_iterations,
            width=width,
            color="#ff7f0e",
            label="Fast-Decoupled (XB)",
        )
        iteration_axis.set_title("全部算例迭代次数", fontsize=14)
        iteration_axis.set_ylabel("迭代次数", fontsize=11)
        iteration_axis.grid(axis="y", linestyle="--", alpha=0.5)
        iteration_axis.legend(fontsize=9)

        order_axis.plot(
            positions,
            nr_orders,
            marker="o",
            linewidth=2,
            color="#1f77b4",
            label="Newton-Raphson",
        )
        order_axis.plot(
            positions,
            fd_orders,
            marker="o",
            linewidth=2,
            color="#ff7f0e",
            label="Fast-Decoupled (XB)",
        )
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

        timing_axis.semilogy(
            positions,
            nr_times,
            marker="o",
            linewidth=2,
            color="#1f77b4",
            label="Newton-Raphson",
        )
        timing_axis.semilogy(
            positions,
            fd_times,
            marker="o",
            linewidth=2,
            color="#ff7f0e",
            label="Fast-Decoupled (XB)",
        )
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
