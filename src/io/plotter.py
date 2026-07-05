# src/io/plotter.py
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 1. 全局配置环境
def _setup_matplotlib_environment():
    # 强制将 font.family 设置为 sans-serif
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 获取可用字体
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    selected_font = next((f for f in preferred_fonts if f in available_fonts), 'DejaVu Sans')
    
    plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
    
    # --- 关键修改：针对 MathText (科学计数法) 的强制策略 ---
    plt.rcParams['axes.unicode_minus'] = False
    
    # 尝试切换为 'cm' 字体集，它通常对数学符号支持更好
    plt.rcParams['mathtext.fontset'] = 'cm' 
    
    # 彻底禁用 Matplotlib 自动加载某些诡异的默认配置
    plt.rcParams['mathtext.default'] = 'regular'

_setup_matplotlib_environment()

class GridPlotter:
    def __init__(self, save_dir="output/plots"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_voltage_profile(self, grid, save_name="voltage_profile.png"):
        """绘制系统节点电压幅值分布图"""
        bus_numbers = [bus['number'] for bus in grid.buses]
        
        plt.figure(figsize=(10, 5))
        plt.plot(bus_numbers, grid.V, marker='o', linestyle='-', color='#1f77b4', label="计算电压 (V)")
        plt.axhline(1.05, color='red', linestyle='--', alpha=0.7, label="上限 (1.05)")
        plt.axhline(0.95, color='green', linestyle='--', alpha=0.7, label="下限 (0.95)")
        
        plt.title("电力系统节点电压幅值分布", fontsize=14)
        plt.xlabel("节点编号", fontsize=12)
        plt.ylabel("电压幅值 (pu)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        file_path = os.path.join(self.save_dir, save_name)
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"[绘图] 电压分布图已保存至: {file_path}")
    
    def plot_convergence(self, info, algo_name="Newton-Raphson", save_name="convergence.png"):
        if 'max_error_history' not in info or not info['max_error_history']:
            return
        errors = info['max_error_history']
        iterations = range(1, len(errors) + 1)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(iterations, errors, marker='s', color='#ff7f0e', linewidth=2)
        ax.set_yscale('log')
        
        # 获取当前刻度位置，固定它，再自定义标签
        y_ticks = ax.get_yticks()
        ax.set_yticks(y_ticks)   # 新增固定
        y_labels = [f"{y:.0e}".replace('e', 'e').replace('-', '-') for y in y_ticks]
        ax.set_yticklabels(y_labels)
        
        ax.set_title(f"{algo_name} 算法收敛过程", fontsize=14)
        ax.set_xlabel("迭代次数", fontsize=12)
        ax.set_ylabel("最大不平衡量 (Max Error)", fontsize=12)
        ax.grid(True, which="both", linestyle='--', alpha=0.6)
        
        file_path = os.path.join(self.save_dir, save_name)
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"[绘图] 收敛特性图已保存至: {file_path}")