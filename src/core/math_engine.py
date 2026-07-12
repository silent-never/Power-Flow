import numpy as np

import numpy as np

def build_y_bus(buses, branches, idx_map):
    """
    构建节点导纳矩阵 Y_bus 
    """
    n = len(buses)
    Y = np.zeros((n, n), dtype=complex)

    # 1. 处理并联元件（对地导纳 G_shunt + j B_shunt）
    for bus in buses:
        i = idx_map[bus['number']]
        G_shunt = bus.get('g_shunt', 0.0)
        B_shunt = bus.get('b_shunt', 0.0)
        Y[i][i] += G_shunt + 1j * B_shunt

    # 2. 处理支路（串联阻抗与对地充电电容、变压器变比）
    for br in branches:
        i = idx_map[br['from_bus']]
        j = idx_map[br['to_bus']]
        
        r = br.get('r', 0.0)
        x = br.get('x', 0.0)
        b = br.get('b', 0.0)             # 线路充电电纳
        br_type = br.get('type', 1)      # 1=线路, 2=变压器
        tap = br.get('final_ratio', 1.0) # 变压器变比（标幺值）
        
        # 串联导纳
        z = r + 1j * x
        if abs(z) < 1e-12: # 防止除零异常
            y_series = 1e8 
        else:
            y_series = 1.0 / z
            
        # 填充非对角元和对角元
        if br_type == 2 and tap != 1.0:
            if tap == 0:
                print(f"Warning: Transformer branch {br['from_bus']}->{br['to_bus']} has tap=0, set to 1")
                tap = 1.0
            # 变压器模型：变比放在 from_bus 侧
            Y[i][i] += y_series / (tap * tap)
            Y[j][j] += y_series
            Y[i][j] -= y_series / tap
            Y[j][i] -= y_series / tap
        else:
            # 普通线路模型
            Y[i][i] += y_series
            Y[j][j] += y_series
            Y[i][j] -= y_series
            Y[j][i] -= y_series
            
        # 处理充电电纳（平分到两端）
        if b != 0.0:
            Y[i][i] += 1j * b / 2.0
            Y[j][j] += 1j * b / 2.0

    return Y

def calc_mismatch(V, theta, Y, bus_type, P_spec, Q_spec):
    """
    计算功率不平衡量
    """
    n = len(V)
    P_calc = np.zeros(n)
    Q_calc = np.zeros(n)

    # 遍历计算注入功率
    for i in range(n):
        for j in range(n):
            G = Y[i][j].real
            B = Y[i][j].imag
            delta = theta[i] - theta[j]
            P_calc[i] += V[i] * V[j] * (G * np.cos(delta) + B * np.sin(delta))
            Q_calc[i] += V[i] * V[j] * (G * np.sin(delta) - B * np.cos(delta))

    dP = P_spec - P_calc
    dQ = Q_spec - Q_calc

    # 针对不同节点类型，屏蔽不需要迭代的不平衡量
    for i in range(n):
        if bus_type[i] == 3:      # 平衡节点 (Slack)
            dP[i] = 0.0
            dQ[i] = 0.0
        elif bus_type[i] == 2:    # PV节点
            dQ[i] = 0.0           # 无功不参与迭代

    return dP, dQ, P_calc, Q_calc

def build_jacobian(V, theta, Y, bus_type, P_calc, Q_calc):
    """
    构建牛顿-拉夫逊法的雅可比矩阵。
    """
    n = len(V)

    # 确定未知量索引
    theta_idx = [i for i in range(n) if bus_type[i] != 3] # 非平衡节点
    v_idx = [i for i in range(n) if bus_type[i] == 1]     # 仅 PQ 节点

    m = len(theta_idx) + len(v_idx)
    J = np.zeros((m, m))

    # 构建基础导数矩阵 H, N, K, L
    H = np.zeros((n, n))
    N = np.zeros((n, n))
    K = np.zeros((n, n))
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                G = Y[i][j].real
                B = Y[i][j].imag
                delta = theta[i] - theta[j]
                
                # 非对角元
                H[i][j] = -V[i] * V[j] * (G * np.sin(delta) - B * np.cos(delta))
                N[i][j] = -V[i] * V[j] * (G * np.cos(delta) + B * np.sin(delta))
                K[i][j] = -N[i][j]      # =  V[i]*V[j]*(G*cos + B*sin)
                L[i][j] = H[i][j]       # = -V[i]*V[j]*(G*sin - B*cos)
                
        # 对角元 (利用已计算的 P_calc 和 Q_calc 简化表达式)
        Gii = Y[i][i].real
        Bii = Y[i][i].imag
        H[i][i] =  Q_calc[i] + Bii * V[i] * V[i]
        N[i][i] = -P_calc[i] - Gii * V[i] * V[i]
        K[i][i] = -P_calc[i] + Gii * V[i] * V[i]
        L[i][i] = -Q_calc[i] + Bii * V[i] * V[i]

    # 将 H, N, K, L 拼接到大雅可比矩阵 J 中
    # 1. J11 (H): dP / dTheta
    for i, row_node in enumerate(theta_idx):
        for j, col_node in enumerate(theta_idx):
            J[i, j] = H[row_node, col_node]
            
    # 2. J12 (N): dP / dV (除以V的修正版，视具体推导而定，这里使用常规 V * dP/dV 形式)
    for i, row_node in enumerate(theta_idx):
        for j, col_node in enumerate(v_idx):
            J[i, len(theta_idx) + j] = N[row_node, col_node] / V[col_node]
            
    # 3. J21 (K): dQ / dTheta
    for i, row_node in enumerate(v_idx):
        for j, col_node in enumerate(theta_idx):
            J[len(theta_idx) + i, j] = K[row_node, col_node]
            
    # 4. J22 (L): dQ / dV
    for i, row_node in enumerate(v_idx):
        for j, col_node in enumerate(v_idx):
            J[len(theta_idx) + i, len(theta_idx) + j] = L[row_node, col_node] / V[col_node]

    return J, theta_idx, v_idx