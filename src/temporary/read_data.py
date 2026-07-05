import os

def safe_int(s, default=0):
    """安全地将字符串转换为整数，支持 '0.' 等带小数点的形式"""
    if s is None or s == '':
        return default
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return default

def special_type_deal(parts,type_idx):
    if len(parts) > type_idx-1:
        try:
            int(parts[type_idx-1])  # 尝试转换为整数
        except ValueError:
            # 如果转换失败，在索引5插入'0'（即在第五项后插入一项）
            parts.insert(type_idx-1, '1')
        return parts

def parse_dat_txt(file_path):
    """
    解析 DAT 格式的 TXT 文件（基于空格分割）
    返回 (buses, branches)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f if line.strip() != '']

    # 定义母线字段顺序（共 19 个字段）
    bus_fields = [
        'number', 'name', 'useless_num',
        'area', 'zone', 'type',
        'v_final', 'angle',
        'load_mw', 'load_mvar',
        'gen_mw', 'gen_mvar',
        'base_kv', 'v_desired',
        'max_q_v', 'min_q_v',
        'g_shunt', 'b_shunt',
        'remote_bus', 'extra'
    ]
        # 一般大多无用，仅number，type，base_kv，v_desired为计算初值，
        # load_mw，load_mvar，gen_mw，gen_mvar为计算用
        # max_q_v，max_q_v为计算限制，v_final，angle为已知计算结果，
        # type可能为空，1=PQ, 2=PV, 3=平衡


    # 定义支路字段顺序（简化版，共 21 个字段）
    branch_fields = [
        'from_bus', 'to_bus',
        'area', 'zone', 'type',
        'r', 'x', 'b',
        'rate1', 'rate2', 'rate3',
        'control_bus', 'side',
        'final_ratio', 'final_phase',
        'v_min', 'v_max', 'step_size',
        'phase_min', 'phase_max', 'extra'
    ]
        # 一般大多无用，仅from_bus,to_bus，type为计算初值，
        # r，x，b为计算用
        # rate1，rate2，rate13为计算限制，
        # type可能为空，1=正常, 2=有变压器

    buses = []
    branches = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]                         #遍历取出每行
        if line.startswith('BUS DATA FOLLOWS'):
            i += 1
            while i < n and not lines[i].startswith('-999'):

                parts = lines[i].split()
                # 调整type不占位问题
                parts = special_type_deal(parts,6)

                # 确保字段数量足够，不足用 None 填充
                if len(parts) < len(bus_fields):
                    parts += [None] * (len(bus_fields) - len(parts))

                bus = {}
                for idx, field in enumerate(bus_fields):
                    val = parts[idx] if idx < len(parts) else None
                    # print(idx, field, val)        #测试用
                    if field == 'number':
                        bus[field] = safe_int(val, 0)
                    elif field == 'name':
                        bus[field] = val.strip() if val else ''
                    elif field == 'type':
                        bus[field] = safe_int(val, 0)
                    elif field in ('useless_num', 'area', 'zone', 'remote_bus', 'extra'):
                        continue # 跳过不记录
                        # bus[field] = safe_int(val, 0)
                    else:
                        # 浮点字段
                        bus[field] = float(val) if val else 0.0
                buses.append(bus)
                i += 1

            i += 1  # 跳过 -999

        elif line.startswith('BRANCH DATA FOLLOWS'):
            i += 1
            while i < n and not lines[i].startswith('-999'):
                parts = lines[i].split()
                # 调整type不占位问题
                parts = special_type_deal(parts,5)
                if len(parts) < len(branch_fields):
                    parts += [None] * (len(branch_fields) - len(parts))
                branch = {}
                for idx, field in enumerate(branch_fields):
                    val = parts[idx] if idx < len(parts) else None
                    if field in ('from_bus', 'to_bus', 'type'):
                        branch[field] = safe_int(val, 0)
                    elif field in ('area', 'zone', 'rate1', 'rate2', 'rate3', 'control_bus', 'side','v_min', 'v_max', 'step_size', 'phase_min', 'phase_max', 'extra'):
                        continue # 跳过不记录
                    else:
                        branch[field] = float(val) if val else 0.0
                branches.append(branch)
                i += 1
            i += 1  # 跳过 -999

        #跳过剩余部分
        elif line.startswith('LOSS ZONES FOLLOWS') or line.startswith('INTERCHANGE DATA FOLLOWS') or line.startswith('TIE LINES FOLLOWS'):
            i += 1
            while i < n and not (lines[i].startswith('-99') or lines[i].startswith('-9')):
                i += 1
            i += 1
        elif line.startswith('END OF DATA'):
            break
        else:
            i += 1

    return buses, branches

# 主执行部分
file_path = r"D:\Data\Program\Python\Power_Flow\data\300ieee.txt"
test_number = 3 #测试sefe_int-->1

if __name__ == "__main__":
    if test_number == 1:
        print(safe_int("100"))
        print(safe_int("3.14"))
        print(safe_int("0."))
        print(safe_int(" "))
    if test_number == 2:
        line1 = '   1 bus_1   100   1  1    0.862 -4.778 160.     80.       0.      0.       100.    1.       0.     0.      0.      0.        0      '
        line2 = '   5 bus_5   100   1  1  3 1.05  0.     0.       0.        257.9427229.9402 100.    1.05   999900 -99990    0.      0.        0       '
        parts1 = line1.split()
        parts2 = line2.split()
        print(parts1)
        print(parts2)
        parts1 = special_type_deal(parts1, 6)
        parts2 = special_type_deal(parts2, 6)
        print(parts1)
        print(parts2)
        line3 = '   3    5  1  1   2  0.        0.03        0.     0    0     0      0    0  1.05   0      0      0       0      0      0         '
        line4 = '   1    2  1  1      0.04      0.25        0.5    0    0     0      0    0  0      0      0      0       0      0      0    '
        parts3 = line3.split()
        parts4 = line4.split()
        parts3 = special_type_deal(parts3, 5)
        parts4 = special_type_deal(parts4, 5)
        print(parts3)
        print(parts4)
    if test_number == 3:
        if test_number == 3:
            try:
                if not os.path.exists(file_path):
                    print(f"错误：文件不存在 - {file_path}")
                    dir_path = os.path.dirname(file_path)
                    if os.path.exists(dir_path):
                        print("目录中的文件：")
                        for f in os.listdir(dir_path):
                            print(f"  {f}")
                else:
                    buses, branches = parse_dat_txt(file_path)
                    print("=== 母线数据 ===")
                    print(f"共 {len(buses)} 条母线")
                    for i, bus in enumerate(buses[:5]):
                        print(f"\n母线 {i + 1}:")
                        for key, value in bus.items():
                            print(f"  {key}: {value}")
                    print("\n=== 支路数据 ===")
                    print(f"共 {len(branches)} 条支路")
                    for i, branch in enumerate(branches[:5]):  # 仅打印前5条支路示例
                        print(f"\n支路 {i + 1}:")
                        for key, value in branch.items():
                            print(f"  {key}: {value}")
            except Exception as e:
                print(f"解析过程中发生错误：{e}")
                import traceback
                traceback.print_exc()





