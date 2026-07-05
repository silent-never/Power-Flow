# src/solvers/base_solver.py
from abc import ABC, abstractmethod
import time

class BaseSolver(ABC):
    """
    潮流求解器抽象基类
    所有具体的求解器算法都应继承此类，并实现 solve 方法。
    """
    def __init__(self, tol=1e-6, max_iter=20):
        self.tol = tol
        self.max_iter = max_iter
        
    @abstractmethod
    def solve(self, grid):
        """
        统一的求解接口
        :param grid: PowerGrid 实例（电网模型）
        :return: (success: bool, info: dict)
                 success 表示是否收敛
                 info 包含迭代次数、误差历史、耗时等统计信息
        """
        pass