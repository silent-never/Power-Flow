# src/io/reporter.py
import numpy as np

class ResultReporter:
    """潮流计算结果打印与越限检测工具类"""
    
    def __init__(self, grid):
        self.grid = grid

    def print_node_results(self, info=None):
        """打印潮流计算终端结果表"""
        print("\n" + "="*55)
        print("                 潮流计算最终结果")
        print("-"*55)
        
        if info:
            print(f"迭代次数: {info['iterations']}")
            print(f"总耗时  : {info['time_elapsed']:.4f} 秒")
            print("-" * 55)

        print(f"{'节点':<8}{'电压 V(pu)':>12}{'相角 θ(deg)':>12}{'P计算(pu)':>12}{'Q计算(pu)':>12}")
        print("-" * 55)

        theta_deg = np.degrees(self.grid.theta)
        
        for i, bus in enumerate(self.grid.buses):
            bus_num = bus['number']
            print(f"{bus_num:<8}{self.grid.V[i]:>12.4f}{theta_deg[i]:>12.4f}"
                  f"{self.grid.P_calc[i]:>12.4f}{self.grid.Q_calc[i]:>12.4f}")
        print("="*55 + "\n")

    def check_voltage_thresholds(self, v_max=1.05, v_min=0.95):
        """检查系统中是否存在电压越限的节点"""
        print(">>> 正在执行电压越限检测...")
        violations = []
        
        for i, v in enumerate(self.grid.V):
            bus_num = self.grid.buses[i]['number']
            if v > v_max:
                violations.append(f"节点 {bus_num:<4} 电压偏高: {v:.4f} > {v_max}")
            elif v < v_min:
                violations.append(f"节点 {bus_num:<4} 电压偏低: {v:.4f} < {v_min}")
                
        if violations:
            print(f"[警告] 发现 {len(violations)} 处节点电压越限：")
            for v in violations:
                print(f"  - {v}")
        else:
            print("[正常] 全网节点电压均在安全范围内。")