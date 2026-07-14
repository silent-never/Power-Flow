# src/solvers/nr_solver.py
import time

import numpy as np

from .base_solver import BaseSolver

class NewtonRaphsonSolver(BaseSolver):
    """
    牛顿-拉夫逊法求解器
    """
    def solve(self, grid):
        start_time = time.perf_counter()
        
        # 记录迭代信息（对于你画收敛特性曲线非常有用）
        info = {
            'iterations': 0,
            'max_error_history': [],
            'time_elapsed': 0.0,
            'condition_numbers': []  # 预留给病态潮流：记录雅可比矩阵条件数
        }

        if self.verbose:
            print("开始牛顿-拉夫逊迭代...")
        
        for it in range(self.max_iter):
            # 1. 获取当前状态下的功率不平衡量 (dP = P_spec - P_calc)
            dP, dQ = grid.get_mismatch()
            
            # 2. 检查收敛性
            max_dP = np.max(np.abs(dP))
            max_dQ = np.max(np.abs(dQ))
            max_error = max(max_dP, max_dQ)
            info['max_error_history'].append(max_error)
            
            if self.verbose:
                print(f"Iter {it:2d}: max|dP| = {max_dP:.6e}, max|dQ| = {max_dQ:.6e}")
            
            if max_error < self.tol:
                info['iterations'] = it
                info['time_elapsed'] = time.perf_counter() - start_time
                if self.verbose:
                    print(f"潮流计算在第 {it} 次迭代成功收敛！耗时: {info['time_elapsed']:.4f}秒")
                return True, info

            # 3. 获取你定义的雅可比矩阵
            J, theta_idx, v_idx = grid.get_jacobian()
            
            # 【病态潮流研究插入点】如果需要监控条件数，取消下行注释
            # info['condition_numbers'].append(np.linalg.cond(J))

            # 4. 构建右端项 dW
            dW = np.concatenate([dP[theta_idx], dQ[v_idx]])
            
            # 5. 求解线性方程组: J * dx = dW
            dx = -np.linalg.solve(J, dW)
            
            # 6. 解析修正量
            n_theta = len(theta_idx)
            dTheta_solve = dx[:n_theta]
            dV_over_V_solve = dx[n_theta:]
            
            dTheta = np.zeros(grid.n)
            dV_over_V = np.zeros(grid.n)
            dTheta[theta_idx] = dTheta_solve
            dV_over_V[v_idx] = dV_over_V_solve
            
            grid.update_state(dV_over_V, dTheta)
            
        # 达到最大迭代次数仍未收敛
        info['iterations'] = self.max_iter
        info['time_elapsed'] = time.perf_counter() - start_time
        if self.verbose:
            print(f"未能收敛！在达到最大迭代次数 {self.max_iter} 时发散。")
        return False, info
