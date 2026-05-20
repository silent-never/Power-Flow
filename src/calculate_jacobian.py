import numpy as np

def build_jacobian(V, theta, Y, bus_type, P_calc, Q_calc):
    """
    构建牛顿-拉夫逊法的雅可比矩阵。

    参数:
        V (ndarray): 节点电压幅值，形状 (n,)
        theta (ndarray): 节点相角（弧度），形状 (n,)
        Y (ndarray): 节点导纳矩阵，形状 (n, n)，复数
        bus_type (ndarray): 节点类型数组，形状 (n,)，编码：1=PQ, 2=PV, 3=平衡
        P_calc (ndarray): 当前有功注入计算值，形状 (n,)
        Q_calc (ndarray): 当前无功注入计算值，形状 (n,)

    返回:
        tuple: (J, theta_idx, v_idx)
            J (ndarray): 雅可比矩阵，形状 (m, m)
            theta_idx (list): 非平衡节点索引（需要求解 θ 的节点）
            v_idx (list): PQ 节点索引（需要求解 V 的节点）
    """
    n = len(V)

    # 确定未知量索引
    theta_idx = [i for i in range(n) if bus_type[i] != 3]   # 非平衡节点（PV 和 PQ）
    v_idx = [i for i in range(n) if bus_type[i] == 1]       # PQ 节点

    n_theta = len(theta_idx)
    n_v = len(v_idx)
    m = n_theta + n_v
    J = np.zeros((m, m))

    # 辅助数组：对角线元素
    Gii = Y.real.diagonal()
    Bii = Y.imag.diagonal()

    # 构建完整的 H, N, K, L 矩阵（n×n）
    H = np.zeros((n, n))
    N = np.zeros((n, n))
    K = np.zeros((n, n))
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            G = Y[i][j].real
            B = Y[i][j].imag
            delta = theta[i] - theta[j]
            #print(f"V[{i}]: {V[i]}, V[{j}]: {V[j]}, delta_{i}{j}: {delta}, G_{i}{j}: {G}, B_{i}{j}: {B}")
            if i != j:
                # 标准公式
                H[i][j] = -V[i] * V[j] * (G * np.sin(delta) - B * np.cos(delta))
                N[i][j] = -V[i] * V[j] * (G * np.cos(delta) + B * np.sin(delta))
                K[i][j] = -N[i][j]      # =  V[i]*V[j]*(G*cos + B*sin)
                L[i][j] = H[i][j]       # = -V[i]*V[j]*(G*sin - B*cos)
            else:
                # 对角元公式（使用传入的 P_calc, Q_calc）
                H[i][i] =  Q_calc[i] + Bii[i] * V[i] * V[i]
                N[i][i] = -P_calc[i] - Gii[i] * V[i] * V[i]
                K[i][i] = -P_calc[i] + Gii[i] * V[i] * V[i]
                L[i][i] = -Q_calc[i] + Bii[i] * V[i] * V[i]
            #print(f"i: {i}, j: {j}, H[{i}][{j}]: {H[i][j]}")
    # 填充 J
    row = 0
    # dP 行（对应所有非平衡节点）
    for i in theta_idx:
        col = 0
        for j in theta_idx:
            J[row, col] = H[i, j]
            #print(f"i:{i} j:{j} J[{row},{col}]:{J[row, col]}")
            col += 1
        for j in v_idx:
            J[row, col] = N[i, j]
            #print(f"i:{i} j:{j} J[{row},{col}]:{J[row, col]}")
            col += 1
        row += 1

    # dQ 行（只对应 PQ 节点）
    for i in v_idx:
        col = 0
        for j in theta_idx:
            J[row, col] = K[i, j]
            #print(f"i:{i} j:{j} J[{row},{col}]:{J[row, col]}")
            col += 1
        for j in v_idx:
            J[row, col] = L[i, j]
            #print(f"i:{i} j:{j} J[{row},{col}]:{J[row, col]}")
            col += 1
        row += 1

    return J, theta_idx, v_idx

def build_jacobian_PQ(V, theta, Y, bus_type, P_calc, Q_calc):
    """
    构建牛顿-拉夫逊法的雅可比矩阵。

    参数:
        V (ndarray): 节点电压幅值，形状 (n,)
        theta (ndarray): 节点相角（弧度），形状 (n,)
        Y (ndarray): 节点导纳矩阵，形状 (n, n)，复数
        bus_type (ndarray): 节点类型数组，形状 (n,)，编码：1=PQ, 2=PV, 3=平衡
        P_calc (ndarray): 当前有功注入计算值，形状 (n,)
        Q_calc (ndarray): 当前无功注入计算值，形状 (n,)

    返回:
        tuple: (J, theta_idx, v_idx)
            J (ndarray): 雅可比矩阵，形状 (m, m)
            theta_idx (list): 非平衡节点索引（需要求解 θ 的节点）
            v_idx (list): PQ 节点索引（需要求解 V 的节点）
    """
    n = len(V)

    # 确定未知量索引
    theta_idx = [i for i in range(n) if bus_type[i] != 3]   # 非平衡节点（PV 和 PQ）
    v_idx = [i for i in range(n) if bus_type[i] == 1]       # PQ 节点

    n_theta = len(theta_idx)
    n_v = len(v_idx)
    m = n_theta + n_v
    J = np.zeros((m, m))

    # 辅助数组：对角线元素
    Gii = Y.real.diagonal()
    Bii = Y.imag.diagonal()

    # 构建完整的 H, N, K, L 矩阵（n×n）
    H = np.zeros((n, n))
    N = np.zeros((n, n))
    K = np.zeros((n, n))
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            G = Y[i][j].real
            B = Y[i][j].imag
            # 标准公式
            H[i][j] = V[i] * V[j] * B
            L[i][j] = H[i][j]  # = -V[i]*V[j]*(G*sin - B*cos)

    # 填充 J
    row = 0
    # dP 行（对应所有非平衡节点）
    for i in theta_idx:
        col = 0
        for j in theta_idx:
            J[row, col] = H[i, j]
            #print(f"i:{i} j:{j} J[{row},{col}]:{J[row, col]}")
            col += 1
        for j in v_idx:
            J[row, col] = N[i, j]
            #print(f"i:{i} j:{j} J[{row},{col}]:{J[row, col]}")
            col += 1
        row += 1

    # dQ 行（只对应 PQ 节点）
    for i in v_idx:
        col = 0
        for j in theta_idx:
            J[row, col] = K[i, j]
            #print(f"i:{i} j:{j} J[{row},{col}]:{J[row, col]}")
            col += 1
        for j in v_idx:
            J[row, col] = L[i, j]
            #print(f"i:{i} j:{j} J[{row},{col}]:{J[row, col]}")
            col += 1
        row += 1

    return J, theta_idx, v_idx
