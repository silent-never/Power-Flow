import numpy as np
from src.creat_y_bus import build_y_bus
from src.creat_y_bus import print_matrix
from src.power_mismatch import power_mismatch
from src.calculate_jacobian import build_jacobian
from src.calculate_jacobian import build_jacobian_PQ

def newton_raphson(buses, branches, Y, base_mva=100.0, tol=1e-6, max_iter=20):
    """
    牛顿-拉夫逊法潮流计算主函数

    参数:
        buses (list of dict): 母线数据列表
        branches (list of dict): 支路数据列表
        Y (ndarray): 节点导纳矩阵（复数）
        base_mva (float): 基准容量 (MVA)
        tol (float): 收敛精度
        max_iter (int): 最大迭代次数

    返回:
        tuple: (V, theta_deg, P_calc, Q_calc)
    """
    n = len(buses)

    # --- 初始化数组 ---
    bus_type = np.zeros(n, dtype=int)   # 1=PQ, 2=PV, 3=平衡
    V0 = np.zeros(n)
    theta0 = np.zeros(n)
    P_spec = np.zeros(n)
    Q_spec = np.zeros(n)

    # 建立节点编号到索引的映射（假设编号连续从1开始）
    idx_map = {bus['number']: i for i, bus in enumerate(buses)}
    # --- 首先找到平衡节点并提取其相角 ---
    slack_angle_deg = 0.0  # 默认相角
    slack_bus_idx = None

    for bus in buses:
        idx = idx_map[bus['number']]
        raw_type = bus.get('type', 1)
        if raw_type == 3:  # 平衡节点
            bus_type[idx] = 3
            slack_bus_idx = idx
            slack_angle_deg = bus.get('angle', 0.0)
            print(f"平衡节点 (Bus {bus['number']}) 相角初值: {slack_angle_deg}°")
        elif raw_type == 2:
            bus_type[idx] = 2
        else:
            bus_type[idx] = 1

    # --- 从解析数据中提取初值和给定值 ---
    for bus in buses:
        idx = idx_map[bus['number']]

        # 电压初值
        V0[idx] = bus['v_desired']

        # 相角初值：相对于平衡节点的相角差（度转弧度）
        node_angle_deg = bus.get('angle', 0.0)
        relative_angle_deg = node_angle_deg - slack_angle_deg
        theta0[idx] = np.radians(relative_angle_deg)

        # 注入功率（发电 - 负荷），标幺值
        P_spec[idx] = (bus.get('gen_mw', 0.0) - bus.get('load_mw', 0.0)) / base_mva
        Q_spec[idx] = (bus.get('gen_mvar', 0.0) - bus.get('load_mvar', 0.0)) / base_mva

    # --- 初始化迭代变量 ---
    V = V0.copy()
    theta = theta0.copy()

    print("开始牛顿-拉夫逊迭代...")
    for it in range(max_iter):
        # 计算功率不平衡量和当前功率值
        dP, dQ, P_calc, Q_calc = power_mismatch(V, theta, Y, bus_type, P_spec, Q_spec)

        max_dP = np.max(np.abs(dP))
        max_dQ = np.max(np.abs(dQ))
        print(f"Iter {it:2d}: max|dP| = {max_dP:.6e}, max|dQ| = {max_dQ:.6e}")

        if max_dP < tol and max_dQ < tol:
            print("k=", it, "收敛成功！")
            break

        # 形成雅可比矩阵
        J, theta_idx, v_idx = build_jacobian(V, theta, Y, bus_type, P_calc, Q_calc)

        # 构建右端项 b（顺序：先所有非平衡节点的 dP，再所有 PQ 节点的 dQ）
        dW = np.concatenate([dP[theta_idx], dQ[v_idx]])

        # 求解 J * dx = -dW
        try:
            dx = np.linalg.solve(J, -dW)
        except np.linalg.LinAlgError:
            print("雅可比矩阵奇异，迭代失败")
            return None, None, None, None

        # 更新相角
        for pos, i in enumerate(theta_idx):
            theta[i] += dx[pos]

        # 更新电压幅值（PQ 节点）
        start = len(theta_idx)
        for pos, i in enumerate(v_idx):
            V[i] *= (1 + dx[start + pos])

    # 最终功率已经由最后一次的 power_mismatch 计算得到
    theta_deg_relative = np.degrees(theta)               # 相对相角
    theta_deg_absolute = theta_deg_relative + slack_angle_deg  # 绝对相角
    return V, theta_deg_absolute, P_calc, Q_calc, it

def newton_raphson_PQ(buses, branches, Y, base_mva=100.0, tol=1e-6, max_iter=20):
    """
    牛顿-拉夫逊法潮流计算主函数

    参数:
        buses (list of dict): 母线数据列表
        branches (list of dict): 支路数据列表
        Y (ndarray): 节点导纳矩阵（复数）
        base_mva (float): 基准容量 (MVA)
        tol (float): 收敛精度
        max_iter (int): 最大迭代次数

    返回:
        tuple: (V, theta_deg, P_calc, Q_calc)
    """
    n = len(buses)

    # --- 初始化数组 ---
    bus_type = np.zeros(n, dtype=int)   # 1=PQ, 2=PV, 3=平衡
    V0 = np.zeros(n)
    theta0 = np.zeros(n)
    P_spec = np.zeros(n)
    Q_spec = np.zeros(n)

    # 建立节点编号到索引的映射（假设编号连续从1开始）
    idx_map = {bus['number']: i for i, bus in enumerate(buses)}
    # --- 首先找到平衡节点并提取其相角 ---
    slack_angle_deg = 0.0  # 默认相角
    slack_bus_idx = None

    for bus in buses:
        idx = idx_map[bus['number']]
        raw_type = bus.get('type', 1)
        if raw_type == 3:  # 平衡节点
            bus_type[idx] = 3
            slack_bus_idx = idx
            slack_angle_deg = bus.get('angle', 0.0)
            print(f"平衡节点 (Bus {bus['number']}) 相角初值: {slack_angle_deg}°")
        elif raw_type == 2:
            bus_type[idx] = 2
        else:
            bus_type[idx] = 1

    # --- 从解析数据中提取初值和给定值 ---
    for bus in buses:
        idx = idx_map[bus['number']]

        # 电压初值
        V0[idx] = bus['v_desired']

        # 相角初值：相对于平衡节点的相角差（度转弧度）
        node_angle_deg = bus.get('angle', 0.0)
        relative_angle_deg = node_angle_deg - slack_angle_deg
        theta0[idx] = np.radians(relative_angle_deg)

        # 注入功率（发电 - 负荷），标幺值
        P_spec[idx] = (bus.get('gen_mw', 0.0) - bus.get('load_mw', 0.0)) / base_mva
        Q_spec[idx] = (bus.get('gen_mvar', 0.0) - bus.get('load_mvar', 0.0)) / base_mva

    # --- 初始化迭代变量 ---
    V = V0.copy()
    theta = theta0.copy()

    print("开始牛顿-拉夫逊迭代...")
    for it in range(max_iter):
        # 计算功率不平衡量和当前功率值
        dP, dQ, P_calc, Q_calc = power_mismatch(V, theta, Y, bus_type, P_spec, Q_spec)

        max_dP = np.max(np.abs(dP))
        max_dQ = np.max(np.abs(dQ))
        print(f"Iter {it:2d}: max|dP| = {max_dP:.6e}, max|dQ| = {max_dQ:.6e}")

        if max_dP < tol and max_dQ < tol:
            print("k=", it, "收敛成功！")
            break

        # 形成雅可比矩阵
        J, theta_idx, v_idx = build_jacobian_PQ(V, theta, Y, bus_type, P_calc, Q_calc)

        # 构建右端项 b（顺序：先所有非平衡节点的 dP，再所有 PQ 节点的 dQ）
        dW = np.concatenate([dP[theta_idx], dQ[v_idx]])

        # 求解 J * dx = -dW
        try:
            dx = np.linalg.solve(J, -dW)
        except np.linalg.LinAlgError:
            print("雅可比矩阵奇异，迭代失败")
            return None, None, None, None

        # 更新相角
        for pos, i in enumerate(theta_idx):
            theta[i] += dx[pos]

        # 更新电压幅值（PQ 节点）
        start = len(theta_idx)
        for pos, i in enumerate(v_idx):
            V[i] *= (1 + dx[start + pos])

    # 最终功率已经由最后一次的 power_mismatch 计算得到
    theta_deg_relative = np.degrees(theta)               # 相对相角
    theta_deg_absolute = theta_deg_relative + slack_angle_deg  # 绝对相角
    return V, theta_deg_absolute, P_calc, Q_calc, it
