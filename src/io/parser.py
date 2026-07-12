# src/io/parser.py
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

def special_type_deal(parts, type_idx):
    if len(parts) > type_idx - 1:
        try:
            int(parts[type_idx - 1])  # 尝试转换为整数
        except ValueError:
            # 如果转换失败，插入 '1'
            parts.insert(type_idx - 1, '1')
        return parts
    return parts

def parse_dat_txt(file_path):
    """
    解析 IEEE DAT 格式的母线与支路数据。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误：文件不存在 - {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f if line.strip() != '']

    bus_fields = [
        'number',        # 母线编号（整数，唯一标识）
        'name',          # 母线名称（字符串，如 "BusA"）
        'useless_num',   # 无用占位编号（兼容旧格式）
        'area',          # 区域编号（整数，用于经济调度分区）
        'zone',          # 供电分区编号（整数，比 area 更细的子区域）
        'type',          # 母线类型（整数，参考标准：1=PQ节点, 2=PV节点, 3=平衡节点(参考), 4=孤立节点）
        'v_final',       # 最终电压幅值（标幺值 p.u.，潮流计算后的结果）
        'angle_final',   # 最终电压相角（度，潮流计算后的结果）
        'load_mw',       # 有功负荷（MW，该母线所接负荷的有功功率）
        'load_mvar',     # 无功负荷（Mvar，该母线所接负荷的无功功率）
        'gen_mw',        # 发电机有功出力（MW，该母线所接发电机的有功注入，正值为发出）
        'gen_mvar',      # 发电机无功出力（Mvar，该母线所接发电机的无功注入，正值为发出）
        'base_kv',       # 基准电压（kV，该母线的电压等级基准值）
        'v_desired',     # 期望电压（标幺值 p.u.，用于电压控制的目标值，作为初始设定值）
        'max_q_v',       # 无功出力上限（Mvar，该母线上发电机或无功补偿设备的最大无功注入量）
        'min_q_v',       # 无功出力下限（Mvar，最小无功注入量，通常为负值表示吸收无功）
        'g_shunt',       # 并联电导（标幺值 p.u.，母线对地并联支路的电导，通常为 0）
        'b_shunt',       # 并联电纳（标幺值 p.u.，母线对地并联支路的电纳，用于补偿电容/电抗器）
        'remote_bus',    # 远程控制母线编号（整数，用于该母线受远方母线电压控制时指定目标节点，0 表示无）
        'extra'          # 预留附加备用字段（标准填 0，用于自定义扩展）
    ]

    branch_fields = [
        'from_bus',      # 起始母线编号（整数）
        'to_bus',        # 终止母线编号（整数）
        'area',          # 区域编号（整数，用于经济调度分区）
        'zone',          # 供电分区编号（整数，比area更细的子区域）
        'type',          # 支路类型（1=输电线路, 2=固定变压器, 3=移相变压器）
        'r',             # 串联电阻（标幺值 p.u.）
        'x',             # 串联电抗（标幺值 p.u.）
        'b',             # 总充电电纳（标幺值 p.u.；线路为对地电容，变压器通常为0）
        'rate1',         # 长期连续额定容量（MVA，正常工况限额）
        'rate2',         # 短期过载额定容量（MVA，N-1后短期紧急限额）
        'rate3',         # 瞬时/紧急过载额定容量（MVA，极端瞬时限额，不用可填0）
        'control_bus',   # 受控母线编号（0表示不参与电压控制，仅固定变比）
        'side',          # 控制侧标识（1=控制作用在from侧, 2=控制作用在to侧）
        'final_ratio',   # 最终变比幅值（标幺值 p.u.；固定变比数值 或 OPF计算出的理想值）
        'final_phase',   # 最终移相角度（度；普通变压器/线路为0，移相器为计算出的角度）
        'v_min',         # 受控母线电压下限（标幺值 p.u.，如 0.95）
        'v_max',         # 受控母线电压上限（标幺值 p.u.，如 1.05）
        'step_size',     # 有载调压分接头离散步长（标幺值 p.u.；0表示连续可调无级调节）
        'phase_min',     # 移相角度可调下限（度；普通变压器填0，移相器如 -30°）
        'phase_max',     # 移相角度可调上限（度；普通变压器填0，移相器如 +30°）
        'extra'          # 预留附加备用字段（标准定义填0，可自定义扩展存储）
    ]

    buses = []
    branches = []
    parsing_bus = False
    parsing_branch = False

    for line in lines:
        if line.startswith('BUS DATA FOLLOWS'):
            parsing_bus = True
            parsing_branch = False
            continue
        elif line.startswith('BRANCH DATA FOLLOWS'):
            parsing_branch = True
            parsing_bus = False
            continue
        elif line.startswith('-999'):
            parsing_bus = False
            parsing_branch = False
            continue

        if parsing_bus:
            parts = line.split()
            parts = special_type_deal(parts, 6)
            if len(parts) < len(bus_fields):
                parts += [None] * (len(bus_fields) - len(parts))

            bus = {}
            for i, field in enumerate(bus_fields):
                val = parts[i] if i < len(parts) else None
                if field == 'name':
                    bus[field] = val.strip() if val else ''
                elif field in ['number', 'type', 'remote_bus', 'extra']:
                    bus[field] = safe_int(val, 0)
                else:
                    try:
                        bus[field] = float(val) if val else 0.0
                    except ValueError:
                        bus[field] = 0.0

            bus['v_final'] = bus.get('v_final', 1.0)
            bus['angle_final'] = bus.get('angle_final', bus.get('angle', 0.0))
            bus['load_mw'] = bus.get('load_mw', 0.0)
            bus['load_mvar'] = bus.get('load_mvar', 0.0)
            bus['gen_mw'] = bus.get('gen_mw', 0.0)
            bus['gen_mvar'] = bus.get('gen_mvar', 0.0)
            bus['g_shunt'] = bus.get('g_shunt', 0.0)
            bus['b_shunt'] = bus.get('b_shunt', 0.0)
            buses.append(bus)

        elif parsing_branch:
            parts = line.split()
            parts = special_type_deal(parts, 5)
            if len(parts) < len(branch_fields):
                parts += [None] * (len(branch_fields) - len(parts))

            branch = {}
            for i, field in enumerate(branch_fields):
                val = parts[i] if i < len(parts) else None
                if field in ['from_bus', 'to_bus', 'type', 'control_bus', 'side']:
                    branch[field] = safe_int(val, 0)
                else:
                    try:
                        branch[field] = float(val) if val else 0.0
                    except ValueError:
                        branch[field] = 0.0

            branch['from_bus'] = branch.get('from_bus', 0)
            branch['to_bus'] = branch.get('to_bus', 0)
            branch['r'] = branch.get('r', 0.0)
            branch['x'] = branch.get('x', 0.0)
            branch['b'] = branch.get('b', 0.0)
            branches.append(branch)

    return buses, branches