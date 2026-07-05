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
    这里按当前仓库中临时版本的字段定义来解析，以尽量复现旧版结果。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误：文件不存在 - {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f if line.strip() != '']

    bus_fields = [
        'number', 'name', 'useless_num', 'area', 'zone', 'type',
        'v_final', 'angle', 'load_mw', 'load_mvar',
        'gen_mw', 'gen_mvar', 'base_kv', 'v_desired',
        'max_q_v', 'min_q_v', 'g_shunt', 'b_shunt',
        'remote_bus', 'extra'
    ]

    branch_fields = [
        'from_bus', 'to_bus', 'area', 'zone', 'type',
        'r', 'x', 'b', 'rate1', 'rate2', 'rate3',
        'control_bus', 'side', 'final_ratio', 'final_phase',
        'v_min', 'v_max', 'step_size', 'phase_min', 'phase_max', 'extra'
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
            bus['angle'] = bus.get('angle', 0.0)
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