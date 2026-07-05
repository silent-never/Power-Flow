# main.py
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入我们精心设计的各个模块
from src.io.parser import parse_dat_txt
from src.core.grid import PowerGrid
from src.solvers.nr_solver import NewtonRaphsonSolver
from src.io.reporter import ResultReporter
from src.io.plotter import GridPlotter

def main():
    print("=" * 60)
    print("                 Power_Flow - 潮流计算与可视化工具")
    
    # 1. 设定测试用例（你可以随时换成 009ieee.txt 或 039ieee.txt）
    file_path = "data/039ieee.txt"  # 请确保 data 文件夹下有这个文件
    
    if not os.path.exists(file_path):
        print(f"[错误] 找不到数据文件: {file_path}")
        return

    # 2. IO层：解析数据
    print(f"\n[1/5] 正在解析数据文件: {file_path} ...")
    buses, branches = parse_dat_txt(file_path)
    print(f"      -> 成功读取 {len(buses)} 条母线，{len(branches)} 条支路")

    # 3. 核心层：建立电网模型 (初始化 V, theta, Y_bus 等)
    print("\n[2/5] 正在构建系统拓扑与数学模型 ...")
    grid = PowerGrid(buses, branches, base_mva=100.0)

    # 4. 求解器层：牛顿-拉夫逊迭代计算
    print("\n[3/5] 开始启动牛顿-拉夫逊求解器 ...")
    # 这里可以轻松修改精度和最大迭代次数
    solver = NewtonRaphsonSolver(tol=1e-6, max_iter=20) 
    
    # 核心计算就这一行！
    success, info = solver.solve(grid)

    # 5. IO层：结果输出与后处理
    print("\n[4/5] 正在生成结果报表 ...")
    reporter = ResultReporter(grid)
    
    if success:
        # 打印节点结果表
        reporter.print_node_results(info)
        # 执行电压越限检测 (比如上限 1.05, 下限 0.95)
        reporter.check_voltage_thresholds(v_max=1.05, v_min=0.95)
        
        # 绘图模块
        print("\n[5/5] 正在绘制可视化图像 ...")
        plotter = GridPlotter(save_dir="output/plots")
        # 画电压分布图
        plotter.plot_voltage_profile(grid, save_name="IEEE39_Voltage_Profile.png")
        # 画收敛特性曲线（平方收敛的完美体现）
        plotter.plot_convergence(info, algo_name="Newton-Raphson", save_name="IEEE39_Convergence.png")
        
        print("\n>>> 所有计算与绘图任务已圆满完成！请查看 output/plots 文件夹。")
        
    else:
        # 如果发散（病态潮流），也会保存当时的误差曲线用于分析
        print("\n[警告] 潮流计算未能收敛！正在生成发散特征分析图...")
        plotter = GridPlotter(save_dir="output/plots")
        plotter.plot_convergence(info, algo_name="Newton-Raphson_Diverged", save_name="Divergence_Trend.png")
        print(">>> 建议检查系统的病态程度（如条件数）或尝试其他求解器（如张量法）。")

if __name__ == "__main__":
    main()