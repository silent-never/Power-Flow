# src/io/reporter.py
import numpy as np

class ResultReporter:
    """潮流计算结果打印与越限检测工具类"""
    
    def __init__(self, grid):
        self.grid = grid

    def print_node_results(self, info=None):
        """打印潮流计算终端结果表"""
        print("\n" + "=" * 60)
        print("                 潮流计算最终结果")
        print("-" * 60)
        
        if info:
            print(f"迭代次数: {info['iterations']}")
            print(f"总耗时  : {info['time_elapsed']:.4f} 秒")
            print("-" * 60)

        # 表头宽度：节点左对齐8，其余右对齐12（含小数点及符号）
        header = (f"{'节点':<8}" f"{'电压 V.pu':<12}" f"{'相角 θ.deg':<12}"
                f"{'有功 P.pu':<12}" f"{'无功 Q.pu':<12}")
        print(header)
        print("-" * 60)

        theta_deg = np.degrees(self.grid.theta)
        
        for i, bus in enumerate(self.grid.buses):
            bus_num = bus['number']
            print(f"{bus_num:<6}{self.grid.V[i]:>12.4f}{theta_deg[i]:>14.4f}"
                  f"{self.grid.P_calc[i]:>14.4f}{self.grid.Q_calc[i]:>14.4f}")
        print("="*60 + "\n")

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