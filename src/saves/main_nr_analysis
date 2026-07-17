# main.py
from pathlib import Path

# 导入自定义模块
from src.analysis.loadability import scan_loadability
from src.io.parser import parse_dat_txt
from src.core.grid import PowerGrid
from src.solvers.nr_solver import NewtonRaphsonSolver
from src.io.pf_reporter import ResultReporter
from src.utils.config import load_config
from src.visualization.pf_plotter import PFPlotter

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    config = load_config(PROJECT_ROOT / "config.yaml")
    
    print("=" * 60)
    print("                 Power_Flow - 潮流计算与可视化工具")
    
    # 1. 设定测试用例
    file_path = PROJECT_ROOT / config.data_file
    
    if not file_path.exists():
        print(f"[错误] 找不到数据文件: {file_path}")
        return

    # 2. IO层：解析数据
    print(f"\n[1/5] 正在解析数据文件: {file_path} ...")
    buses, branches = parse_dat_txt(file_path)
    print(f"      -> 成功读取 {len(buses)} 条母线，{len(branches)} 条支路")

    # 3. 核心层：建立电网模型 (初始化 V, theta, Y_bus 等)
    print("\n[2/5] 正在构建系统拓扑与数学模型 ...")
    grid = PowerGrid(buses, branches, base_mva=config.base_mva)

    # 4. 求解器层：牛顿-拉夫逊迭代计算
    print("\n[3/5] 开始启动牛顿-拉夫逊求解器 ...")
    # 这里可以轻松修改精度和最大迭代次数
    if config.algorithm != "nr":
        raise ValueError(f"当前入口尚不支持求解器: {config.algorithm}")
    solver = NewtonRaphsonSolver(
        tol=config.tolerance,
        max_iter=config.max_iterations,
    )
    
    # 核心计算
    success, info = solver.solve(grid)

    # 5. IO层：结果输出与后处理
    print("\n[4/5] 正在生成结果报表 ...")
    reporter = ResultReporter(grid)
    
    if success:
        # 打印节点结果表
        reporter.print_node_results(info)
        # 执行电压越限检测 (比如上限 1.05, 下限 0.95)
        # reporter.check_voltage_thresholds(v_max=1.05, v_min=0.95)
        
        # 绘图模块
        print("\n[6/6] 正在绘制可视化图像 ...")
        plotter = PFPlotter(save_dir=PROJECT_ROOT / config.plot_dir)

        # 四类对比图：
        # 1) 电压 vs v_final
        # 2) 相角 vs angle_final
        # 3) 有功 vs P_spec
        # 4) 非PQ节点无功 vs Q_spec
        plotter.plot_all_comparisons(
            grid,
            v_tol=0.02,      # 电压偏差阈值
            angle_tol=2.0,   # 相角偏差阈值（deg）
            p_tol=0.02,      # 有功偏差阈值（p.u.）
            q_tol=0.02,      # 无功偏差阈值（p.u.）
            prefix="IEEE"
        )
        # 5) 与标准数据的偏差
        plotter.plot_deviation_outliers_summary(
            grid,
            v_tol=0.02,
            angle_tol=2.0,
            p_tol=0.02,
            q_tol=0.02,
            save_name="IEEE_05_Deviation_Outliers_Summary.png"
        )
        # 6) 画收敛特性曲线
        plotter.plot_convergence(info, algo_name="Newton-Raphson", save_name="IEEE_06_Convergence.png")

        if config.run_loadability_scan:
            print("\n[负荷扫描] 开始统一放大全部母线的 P/Q 负荷 ...")
            loadability = scan_loadability(
                grid,
                start=config.load_multiplier_start,
                stop=config.load_multiplier_stop,
                step=config.load_multiplier_step,
                refinement_tolerance=config.load_refinement_tolerance,
                tolerance=config.tolerance,
                max_iterations=config.load_scan_max_iterations,
            )
            plotter.plot_loadability_curve(loadability)

            bracket = loadability.collapse_bracket
            if bracket is None:
                print(
                    f">>> 扫描至 {config.load_multiplier_stop:.3f} 倍仍未发现不收敛点，"
                    "请增大 load_multiplier_stop。"
                )
            else:
                stable, failed = bracket
                stable_point = max(
                    (point for point in loadability.points if point.converged),
                    key=lambda point: point.multiplier,
                )
                print(
                    f">>> 最后收敛倍率: {stable:.4f} "
                    f"(总负荷约 {loadability.base_load_mw * stable:.2f} MW / "
                    f"{loadability.base_load_mvar * stable:.2f} Mvar)"
                )
                print(
                    f">>> 此时最低电压: {stable_point.min_voltage:.4f} p.u. "
                    f"(母线 {stable_point.min_voltage_bus})"
                )
                print(
                    f">>> 首个不收敛倍率: {failed:.4f} "
                    f"(总负荷约 {loadability.base_load_mw * failed:.2f} MW / "
                    f"{loadability.base_load_mvar * failed:.2f} Mvar)"
                )
                print(
                    ">>> 该区间是普通潮流的数值崩溃边界，以后可用 CPF 精确追踪鼻点。"
                )

        print(f"\n>>> 所有计算与绘图任务已完成！请查看 {config.plot_dir} 文件夹。")
        
    else:
        # 如果发散（病态潮流），也会保存当时的误差曲线用于分析
        print("\n[警告] 潮流计算未能收敛！正在生成发散特征分析图...")
        plotter = PFPlotter(save_dir=PROJECT_ROOT / config.plot_dir)
        plotter.plot_convergence(info, algo_name="Newton-Raphson_Diverged", save_name="Divergence_Trend.png")
        print(">>> 建议检查系统的病态程度（如条件数）或尝试其他求解器（如张量法）。")


if __name__ == "__main__":
    main()
