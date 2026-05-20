import numpy as np

def calc_power(V, theta, Y):
    """
    计算所有节点的有功和无功注入功率。

    参数:
        V (ndarray): 节点电压幅值，形状 (n,)，单位 pu
        theta (ndarray): 节点相角，形状 (n,)，单位弧度
        Y (ndarray): 节点导纳矩阵，形状 (n, n)，复数类型

    返回:
        tuple: (P, Q)
            P (ndarray): 有功功率，形状 (n,)
            Q (ndarray): 无功功率，形状 (n,)
    """
    n = len(V)
    P = np.zeros(n)
    Q = np.zeros(n)

    # 遍历所有节点对，累加计算功率
    for i in range(n):
        for j in range(n):
            G = Y[i][j].real          # 电导
            B = Y[i][j].imag          # 电纳
            delta = theta[i] - theta[j]  # 相角差
            # 有功功率累加项
            P[i] += V[i] * V[j] * (G * np.cos(delta) + B * np.sin(delta))
            # 无功功率累加项
            Q[i] += V[i] * V[j] * (G * np.sin(delta) - B * np.cos(delta))

    return P, Q

def power_mismatch(V, theta, Y, bus_type, P_spec, Q_spec):
    """
    计算节点功率不平衡量。
    有功不平衡量 dP = 给定有功 P_spec - 计算有功 P_calc
    无功不平衡量 dQ = 给定无功 Q_spec - 计算无功 Q_calc

    平衡节点和 PV 节点的不平衡量会根据节点类型进行归零：
        - 平衡节点：dP 和 dQ 均置零（不参与迭代）
        - PV 节点：dQ 置零（无功不参与迭代）
        - PQ 节点：dP 和 dQ 均保留

    参数:
        V (ndarray): 节点电压幅值，形状 (n,)，单位 pu
        theta (ndarray): 节点相角，形状 (n,)，单位弧度
        Y (ndarray): 节点导纳矩阵，形状 (n, n)，复数类型
        bus_type (ndarray): 节点类型数组，形状 (n,)，编码规则：
            1 = PQ 节点
            2 = PV 节点
            3 = 平衡节点
        P_spec (ndarray): 节点有功注入给定值，形状 (n,)，单位 pu
        Q_spec (ndarray): 节点无功注入给定值，形状 (n,)，单位 pu

    返回:
        tuple: (dP, dQ)
            dP (ndarray): 有功不平衡量，形状 (n,)
            dQ (ndarray): 无功不平衡量，形状 (n,)
    """
    n = len(V)                              # 节点数

    # 计算当前功率注入值
    P_calc, Q_calc = calc_power(V, theta, Y)

    # 计算不平衡量（给定值 - 计算值）
    dP = P_spec - P_calc
    dQ = Q_spec - Q_calc

    # 根据节点类型，将不需要参与迭代的不平衡量置零
    for i in range(n):
        if bus_type[i] == 3:                # 平衡节点
            dP[i] = 0.0
            dQ[i] = 0.0
        elif bus_type[i] == 2:              # PV 节点，无功不参与迭代
            dQ[i] = 0.0
        # PQ 节点（bus_type[i] == 1）不做任何处理，保留 dP 和 dQ

    return dP, dQ ,P_calc, Q_calc