"""非线性潮流求解器共用的内部工具。"""

from __future__ import annotations

import numpy as np


def active_mismatch(grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回当前状态的有效失配向量及其节点索引。"""
    dP, dQ = grid.get_mismatch()
    # 索引只由节点类型决定，无需为了获取索引提前重复构造雅可比矩阵。
    theta_idx = np.flatnonzero(grid.bus_type != 3)
    v_idx = np.flatnonzero(grid.bus_type == 1)
    mismatch = np.concatenate((dP[theta_idx], dQ[v_idx]))
    return mismatch, theta_idx, v_idx


def split_direction(
    grid,
    direction: np.ndarray,
    theta_idx: np.ndarray,
    v_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把紧凑的求解方向还原为全网相角和相对电压方向。"""
    theta_direction = np.zeros(grid.n, dtype=float)
    voltage_direction = np.zeros(grid.n, dtype=float)
    n_theta = len(theta_idx)
    theta_direction[theta_idx] = direction[:n_theta]
    voltage_direction[v_idx] = direction[n_theta:]
    return voltage_direction, theta_direction


def set_trial_state(
    grid,
    base_voltage: np.ndarray,
    base_angle: np.ndarray,
    voltage_direction: np.ndarray,
    angle_direction: np.ndarray,
    multiplier: float,
) -> bool:
    """直接设置试探状态；电压非正或含非有限数时返回失败。"""
    trial_voltage = base_voltage * (1.0 + multiplier * voltage_direction)
    trial_angle = base_angle + multiplier * angle_direction
    if (
        np.any(~np.isfinite(trial_voltage))
        or np.any(~np.isfinite(trial_angle))
        or np.any(trial_voltage <= 1e-8)
    ):
        return False
    grid.V[:] = trial_voltage
    grid.theta[:] = trial_angle
    return True


def mismatch_objective(grid, theta_idx: np.ndarray, v_idx: np.ndarray) -> float:
    """计算有效潮流失配平方和的一半。"""
    dP, dQ = grid.get_mismatch()
    mismatch = np.concatenate((dP[theta_idx], dQ[v_idx]))
    if np.any(~np.isfinite(mismatch)):
        return float("inf")
    return 0.5 * float(mismatch @ mismatch)
