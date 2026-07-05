import numpy as np

def build_y_bus(buses, branches, base_mva=100.0):
    """
    构建节点导纳矩阵
    buses: 母线列表，每个元素为字典，包含 number, g_shunt, b_shunt 等
    branches: 支路列表，每个元素为字典，包含 from_bus, to_bus, r, x, b, type 等
    base_mva: 基准容量（MVA），用于将标幺值统一（本数据已经是标幺值，所以不需要转换）
    返回: Y (numpy 复数矩阵), bus_index_map (原始节点编号到矩阵索引的映射)
    """
    # 获取所有节点编号
    bus_numbers = [bus['number'] for bus in buses]
    n = len(bus_numbers)
    # 建立节点编号 -> 索引的映射（索引从0开始）
    idx_map = {num: i for i, num in enumerate(bus_numbers)}

    # 初始化 Y 矩阵为复数零矩阵
    Y = np.zeros((n, n), dtype=complex)

    # 1. 处理并联元件（对地导纳）
    for bus in buses:
        i = idx_map[bus['number']]
        # 并联电导 G_shunt 和电纳 B_shunt（B 电容为正，电抗为负）
        G_shunt = bus.get('g_shunt', 0.0)
        B_shunt = bus.get('b_shunt', 0.0)
        Y[i][i] += complex(G_shunt, B_shunt)

    # 2. 处理支路
    for br in branches:
        f = br['from_bus']
        t = br['to_bus']
        r = br['r']
        x = br['x']
        b = br['b']   # 总充电电纳（标幺值）
        br_type = br.get('type', 1)  # 1=线路, 2=变压器
        tap = br.get('final_ratio', 1.0)  # 变比（标幺值）
        # 计算支路导纳
        if r == 0.0 and x == 0.0:
            # 零阻抗支路，不应出现，若出现则跳过或报错
            continue
        y_series = 1.0 / complex(r, x)

        # 获取节点索引
        i = idx_map[f]
        j = idx_map[t]
        if br_type == 2 and tap != 1.0:
            if tap == 0:
                print(f"Warning: Transformer branch {f}->{t} has tap=0, set to 1")
                tap = 1.0
            # 变压器模型：变比放在 from_bus 侧
            # 并联充电电纳忽略（通常变压器 b 为0）
            Y[i][i] += y_series / (tap * tap)
            Y[j][j] += y_series
            Y[i][j] -= y_series / tap
            Y[j][i] -= y_series / tap
        else:
            # 普通线路
            Y[i][i] += y_series
            Y[j][j] += y_series
            Y[i][j] -= y_series
            Y[j][i] -= y_series
        # 处理充电电纳（平分）
        if b != 0.0:
            Y[i][i] += 1j * b / 2.0
            Y[j][j] += 1j * b / 2.0


    return Y, idx_map

def print_matrix(Y, precision, title="节点导纳矩阵"):
    """
    按数学格式打印复数矩阵（例如节点导纳矩阵），显示行号、列号及复数元素。

    参数:
        Y: numpy 复数矩阵 (n x n)
        precision: 显示的小数位数
        title: 打印标题
    """
    n = Y.shape[0]
    # 设置复数显示格式（虚部用 j 表示，符合电力系统习惯）
    # 定义每个单元格的固定宽度（根据精度调整）
    width = 2 * (precision + 5) + 4  # 例如 precision=4 => 宽度≈18

    print(f"\n{title}:")
    # 打印列号
    print("     ", end="")
    for j in range(n):
        print(f"  col{j+1:^{width-2}}", end="")
    print()

    for i in range(n):
        # 打印行号
        print(f"row{i+1:<3}", end=" ")
        for j in range(n):
            z = Y[i, j]
            # 处理接近零的值
            real = z.real if abs(z.real) > 1e-10 else 0.0
            imag = z.imag if abs(z.imag) > 1e-10 else 0.0

            if real == 0.0 and imag == 0.0:
                elem_str = "0"
            elif real == 0.0:
                elem_str = f"{imag:.{precision}f}j"
            elif imag == 0.0:
                elem_str = f"{real:.{precision}f}"
            else:
                sign = "+" if imag > 0 else "-"
                elem_str = f"{real:.{precision}f} {sign} j{abs(imag):.{precision}f}"
            # 右对齐输出，保持列宽一致
            print(f"{elem_str:>{width}}", end=" ")
        print()
    print()