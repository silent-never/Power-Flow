# src/core/grid.py
import copy
import sys
import numpy as np
from src.core.math_engine import build_y_bus, calc_mismatch, build_jacobian

class PowerGrid:
    """
    电网模型实体类。
    封装了网络的拓扑结构、基准值、状态变量(V, theta)和给定量(P_spec, Q_spec)。
    """
    def __init__(self, buses, branches, base_mva=100.0):
        self.base_mva = base_mva
        self.buses = buses
        self.branches = branches
        self.n = len(buses)
        
        # 建立节点编号到内部索引(0~n-1)的映射
        self.idx_map = {bus['number']: i for i, bus in enumerate(buses)}
        
        # 节点类型数组: 1=PQ, 2=PV, 3=Slack
        self.bus_type = np.array([bus.get('type', 1) for bus in buses])
        
        # 1. 运行状态变量 (State variables)
        self.V = np.ones(self.n)
        self.theta = np.zeros(self.n)
        
        # 2. 节点给定量 (Specified values)
        self.P_spec = np.zeros(self.n)
        self.Q_spec = np.zeros(self.n)
        
        # 3. 网络拓扑矩阵
        self.Y_bus = build_y_bus(self.buses, self.branches, self.idx_map)
        
        # 4. 内部缓存变量（避免重复计算）
        self.P_calc = np.zeros(self.n)
        self.Q_calc = np.zeros(self.n)
        
        # 初始化运行数据
        self._init_state()

    def _init_state(self):
        """初始化节点电压初值和给定功率注入量"""
        for bus in self.buses:
            i = self.idx_map[bus['number']]
            
            # 设置初值
            self.V[i] = bus.get('v_final', 1.0)
            self.theta[i] = np.radians(bus.get('angle', 0.0))
            
            # 标幺化给定功率：P_spec = (P_gen - P_load) / S_base
            p_gen = bus.get('gen_mw', 0.0)
            p_load = bus.get('load_mw', 0.0)
            q_gen = bus.get('gen_mvar', 0.0)
            q_load = bus.get('load_mvar', 0.0)
            
            self.P_spec[i] = (p_gen - p_load) / self.base_mva
            self.Q_spec[i] = (q_gen - q_load) / self.base_mva

    def get_mismatch(self):
        """
        获取当前状态下的功率不平衡量
        返回: dP, dQ
        """
        dP, dQ, self.P_calc, self.Q_calc = calc_mismatch(
            self.V, self.theta, self.Y_bus, self.bus_type, self.P_spec, self.Q_spec
        )
        return dP, dQ

    def get_jacobian(self):
        """
        获取当前状态下的雅可比矩阵
        返回: J矩阵, 参与有功迭代的节点索引, 参与无功迭代的节点索引
        """
        J, theta_idx, v_idx = build_jacobian(
            self.V, self.theta, self.Y_bus, self.bus_type, self.P_calc, self.Q_calc
        )
        return J, theta_idx, v_idx

    def update_state(self, dV_over_V, dTheta):
        """
        更新系统的电压和相角状态。
        这里把 dV_over_V 视为相对增量 Delta V / V，因此采用 V *= (1 + dV_over_V) 的方式更新。
        """
        self.V *= (1.0 + dV_over_V)
        self.theta += dTheta

    def clone(self):
        """
        深拷贝当前电网状态。
        这在研究连续潮流(CPF)或者尝试不同迭代步长(防止病态发散)时极为重要！
        """
        return copy.deepcopy(self)