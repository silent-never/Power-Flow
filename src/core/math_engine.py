import numpy as np


def build_y_bus(buses, branches, idx_map):
    """
    构建节点导纳矩阵 Y_bus。

    支路类型约定：
        type = 1：普通输电线路
        type = 2：普通变压器，只考虑非标准变比
        type = 4：移相变压器，同时考虑变比和移相角

    变压器复变比放置在 from_bus 一侧：

        a = tap * exp(j * phase)

    对应支路导纳矩阵为：

        Yff = (y + j*b/2) / |a|^2
        Yft = -y / conj(a)
        Ytf = -y / a
        Ytt = y + j*b/2

    其中：
        y = 1 / (r + jx)
        tap 为非标准变比
        phase 为移相角，单位为度
    """
    n = len(buses)
    Y = np.zeros((n, n), dtype=complex)

    # 1. 节点并联导纳-
    for bus in buses:
        i = idx_map[bus["number"]]

        g_shunt = float(bus.get("g_shunt", 0.0))
        b_shunt = float(bus.get("b_shunt", 0.0))

        Y[i, i] += g_shunt + 1j * b_shunt

    # 2. 支路导纳
    for br in branches:
        from_bus = br["from_bus"]
        to_bus = br["to_bus"]

        i = idx_map[from_bus]
        j = idx_map[to_bus]

        r = float(br.get("r", 0.0))
        x = float(br.get("x", 0.0))
        b = float(br.get("b", 0.0))

        # 1=线路，2=普通变压器，4=移相变压器
        br_type = int(br.get("type", 1))

        # 串联阻抗与导纳
        z = r + 1j * x

        if abs(z) < 1e-12:
            raise ValueError(
                f"支路 {from_bus}->{to_bus} 的阻抗接近零："
                f"r={r}, x={x}"
            )

        y_series = 1.0 / z
        y_shunt_half = 1j * b / 2.0

        # 普通线路
        if br_type == 1:
            Y[i, i] += y_series + y_shunt_half
            Y[j, j] += y_series + y_shunt_half

            Y[i, j] -= y_series
            Y[j, i] -= y_series

        # 普通变压器或移相变压器
        elif br_type in (2, 4):
            tap = float(br.get("final_ratio", 1.0))

            if abs(tap) < 1e-12:
                print(
                    f"[警告] 变压器支路 {from_bus}->{to_bus} "
                    f"变比为 0，已自动改为 1.0"
                )
                tap = 1.0

            # 普通变压器不考虑移相角
            if br_type == 2:
                phase_deg = 0.0
            # 移相变压器读取移相角，单位为度
            else:
                phase_deg = float(br.get("final_phase", 0.0))
            phase_rad = np.radians(phase_deg)

            # 复变比放置在 from_bus 一侧
            complex_tap = tap * np.exp(1j * phase_rad)

            # 支路四个导纳元素
            Y[i, i] += (y_series + y_shunt_half) / (abs(complex_tap) ** 2)
            Y[i, j] += -y_series / np.conj(complex_tap)
            Y[j, i] += -y_series / complex_tap
            Y[j, j] += y_series + y_shunt_half

        else:
            raise ValueError(
                f"支路 {from_bus}->{to_bus} 存在未知类型："
                f"type={br_type}"
            )

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


def analyze_matrix(J):
    """
    分析矩阵数值特性。

    参数:
        J : ndarray
            雅可比矩阵或其他待分析矩阵

    返回:
        dict:
            condition_number : 条件数
            singular_values  : 奇异值
            min_singular_value : 最小奇异值
            eigenvalues : 特征值
    """

    result = {}

    # 条件数
    try:
        result["condition_number"] = np.linalg.cond(J)
    except np.linalg.LinAlgError:
        result["condition_number"] = np.inf


    # 奇异值分解
    try:
        singular_values = np.linalg.svd(
            J,
            compute_uv=False
        )

        result["singular_values"] = singular_values
        result["min_singular_value"] = np.min(
            singular_values
        )
    except np.linalg.LinAlgError:

        result["singular_values"] = None
        result["min_singular_value"] = 0.0


    # 特征值
    try:
        result["eigenvalues"] = np.linalg.eigvals(J)
    except np.linalg.LinAlgError:

        result["eigenvalues"] = None

    return result
