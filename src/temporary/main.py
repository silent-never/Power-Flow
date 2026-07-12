import numpy as np

from src.read_data import parse_dat_txt
from src.creat_y_bus import build_y_bus
from src.creat_y_bus import print_matrix
from src.power_mismatch import calc_power
from src.power_mismatch import power_mismatch
from src.calculate_jacobian import build_jacobian
from src.Newton_Raphson import newton_raphson
from src.plot_results import plot_results
from src.threshold_check import check_threshold

import matplotlib.pyplot as plt
import os

clc_precision = 1e-6
file_path = "data/003ieee.txt"
thresholds = {
    'V': 0.001,          # 电压偏差不超过 pu
    'theta_deg': 0.01,   # 相角偏差不超过 度
    'P': 0.001,          # 有功偏差不超过 pu
    'Q': 0.001           # 无功偏差不超过 pu
}

def main(clc_precision,file_path):

    # 1. 读取数据
    buses, branches = parse_dat_txt(file_path)
    print(f"读取到 {len(buses)} 条母线，{len(branches)} 条支路")

    # 2. 构建导纳矩阵并打印
    Y, idx_map = build_y_bus(buses, branches, base_mva=100.0)
    print("Ybus 矩阵构建完成，形状:", Y.shape)
    print_matrix(Y, precision=4, title="节点导纳矩阵 Y")

    # 3. 执行牛顿-拉夫逊潮流计算
    V, theta_deg, P_calc, Q_calc, it = newton_raphson(
        buses, branches, Y,
        base_mva=100.0,
        tol=clc_precision,
        max_iter=20
    )

    # 4. 输出结果
    if V is not None:
        print(f"\n===== 潮流计算结果 ===== 精度：{clc_precision}")
        print(f"{'节点':<6}{'V(pu)':>10}{'θ(deg)':>12}{'P_calc(pu)':>14}{'Q_calc(pu)':>14}")

        for bus in buses:
            bus_num = bus['number']
            idx = idx_map[bus_num]
            print(f"{bus_num:<6}{V[idx]:>10.4f}{theta_deg[idx]:>12.4f}"
                  f"{P_calc[idx]:>14.4f}{Q_calc[idx]:>14.4f}")

    else:
        print("潮流计算不收敛或出错")

    # 5. 生成图表
    plot_results(buses, idx_map, V, theta_deg, P_calc, Q_calc, save_dir ="plots")

    # 6.打印超差报告
    check_threshold(buses, idx_map, V, theta_deg, P_calc, Q_calc,
                    thresholds=thresholds, return_details=False)



if __name__ == "__main__":
    main(clc_precision, file_path)