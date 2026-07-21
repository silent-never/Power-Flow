"""Fast-decoupled (PQ) power-flow solver.

Implements the Stott–Alsac XB scheme, the most widely used variant of the
fast decoupled load flow (FDLF).  Compared to full Newton–Raphson, the method:

* neglects the off-diagonal coupling submatrices N and K,
* replaces the remaining H and L blocks with constant approximations B' and B'',
* factors B' and B'' once and reuses the factorisations throughout.

This reduces the repeated linear-solve cost from O(n³) to O(n²), although
the decoupled method usually needs more iterations than Newton–Raphson.

Reference
---------
Stott, B., & Alsac, O. (1974).  "Fast Decoupled Load Flow."
IEEE Transactions on Power Apparatus and Systems, PAS-93(3), 859–869.
"""

from __future__ import annotations

import time

import numpy as np

try:
    from scipy.linalg import lu_factor, lu_solve
except ImportError:  # PyCharm 的轻量环境可能只安装了 NumPy。
    lu_factor = None
    lu_solve = None

from .base_solver import BaseSolver


class FastDecoupledSolver(BaseSolver):
    """Fast-decoupled load-flow solver using the Stott–Alsac XB scheme.

    The two constant matrices are built once at the start:

    **B'** (P–θ)
        Constructed solely from branch series reactances.  Resistance,
        shunt charging, and off-nominal transformer taps are ignored::

            B'(i, j) = –1 / x_ij          (off-diagonal)
            B'(i, i) =  Σ 1 / x_ik        (diagonal)

    **B''** (Q–V)
        Taken as the negative imaginary part of the full Y_bus, restricted
        to PQ buses.  Unlike B', shunt elements are retained here so that
        voltage corrections remain accurate::

            B'' = –Im[Y_bus]

    Each full iteration consists of two half-steps:

    1.  **P–θ**:   Δθ = B'⁻¹ · (ΔP / V)
    2.  **Q–V**:   ΔV = B''⁻¹ · (ΔQ / V)

    The matrices are constant and are factorised once.  SciPy environments
    reuse LU factors; NumPy-only environments cache one inverse matrix and
    perform matrix–vector multiplication during the iteration.

    Parameters
    ----------
    tol : float
        Convergence tolerance on max |ΔP|, |ΔQ| (per‑unit).
    max_iter : int
        Maximum number of P–θ + Q–V full iterations.
    verbose : bool
        Print per-iteration mismatch values.
    """

    # ── internal helpers ────────────────────────────────────────────

    @staticmethod
    def _factor_constant_matrix(matrix: np.ndarray):
        """一次性分解常数矩阵，并返回可重复使用的求解数据。"""
        if lu_factor is not None:
            return "lu", lu_factor(matrix, check_finite=False)

        # 没有 SciPy 时缓存逆矩阵。与每轮 np.linalg.solve 相比，
        # 迭代阶段只需 O(n²) 的矩阵—向量乘法。
        return "inverse", np.linalg.inv(matrix)

    @staticmethod
    def _solve_factored(factor, right_hand_side: np.ndarray) -> np.ndarray:
        """使用已缓存的 LU 分解或逆矩阵求解新的右端项。"""
        factor_type, factor_data = factor
        if factor_type == "lu":
            return lu_solve(
                factor_data,
                right_hand_side,
                check_finite=False,
            )
        return factor_data @ right_hand_side

    @staticmethod
    def _build_b_prime(grid) -> tuple[np.ndarray, np.ndarray]:
        """Build B' from branch series reactances (no R, no shunt, no tap)."""
        n = grid.n
        b_full = np.zeros((n, n))

        for branch in grid.branches:
            i = grid.idx_map[int(branch["from_bus"])]
            j = grid.idx_map[int(branch["to_bus"])]
            x = float(branch.get("x", 0.0))
            if abs(x) < 1e-12:
                continue
            b_series = 1.0 / x
            b_full[i, i] += b_series
            b_full[j, j] += b_series
            b_full[i, j] -= b_series
            b_full[j, i] -= b_series

        non_slack = np.array(
            [i for i in range(n) if grid.bus_type[i] != 3], dtype=int
        )
        return b_full[np.ix_(non_slack, non_slack)], non_slack

    @staticmethod
    def _build_b_double_prime(grid) -> tuple[np.ndarray, np.ndarray]:
        """Build B'' = –Im[Y_bus] restricted to PQ buses."""
        n = grid.n
        pq_buses = np.array(
            [i for i in range(n) if grid.bus_type[i] == 1], dtype=int
        )
        imag_y = grid.Y_bus.imag
        return -imag_y[np.ix_(pq_buses, pq_buses)], pq_buses

    # ── main entry point ────────────────────────────────────────────

    def solve(self, grid):
        """Run the fast-decoupled power flow on *grid* (modified in place).

        Returns
        -------
        success : bool
        info : dict
            ``iterations``, ``max_error_history``, ``time_elapsed``.
        """
        start_time = time.perf_counter()
        info: dict = {
            "iterations": 0,
            "max_error_history": [],
            "time_elapsed": 0.0,
            "timing_breakdown": {
                "mismatch_time": 0.0,
                "matrix_build_time": 0.0,
                "factorization_time": 0.0,
                "linear_solve_time": 0.0,
                "state_update_time": 0.0,
            },
        }
        timing = info["timing_breakdown"]

        # ---- build constant matrices (once) -------------------------
        phase_start = time.perf_counter()
        B_prime, non_slack = self._build_b_prime(grid)
        B_double_prime, pq_buses = self._build_b_double_prime(grid)
        timing["matrix_build_time"] += time.perf_counter() - phase_start

        n_theta = len(non_slack)
        n_v = len(pq_buses)
        phase_start = time.perf_counter()
        B_prime_factor = (
            self._factor_constant_matrix(B_prime)
            if n_theta > 0
            else None
        )
        B_double_prime_factor = (
            self._factor_constant_matrix(B_double_prime)
            if n_v > 0
            else None
        )
        timing["factorization_time"] += time.perf_counter() - phase_start

        if self.verbose:
            print("开始快速解耦法 (XB scheme) 迭代...")

        # ---- iteration loop -----------------------------------------
        for iteration in range(self.max_iter):
            # ---------- P–θ half-step ----------
            phase_start = time.perf_counter()
            dP, dQ = grid.get_mismatch()
            timing["mismatch_time"] += time.perf_counter() - phase_start

            max_dP = float(np.max(np.abs(dP[non_slack]))) if n_theta else 0.0
            max_dQ = float(np.max(np.abs(dQ[pq_buses]))) if n_v else 0.0
            max_error = max(max_dP, max_dQ)
            info["max_error_history"].append(max_error)

            if self.verbose:
                print(
                    f"Iter {iteration:2d}: "
                    f"max|dP| = {max_dP:.6e}, "
                    f"max|dQ| = {max_dQ:.6e}"
                )

            if max_error < self.tol:
                info["iterations"] = iteration
                info["time_elapsed"] = time.perf_counter() - start_time
                if self.verbose:
                    print(
                        f"快速解耦法在第 {iteration} 次迭代收敛！"
                        f"耗时: {info['time_elapsed']:.4f} 秒"
                    )
                return True, info

            # Δθ = B'⁻¹ · (ΔP / V)    —  only for non-slack buses
            if n_theta > 0:
                rhs_p = dP[non_slack] / grid.V[non_slack]
                phase_start = time.perf_counter()
                d_theta_slice = self._solve_factored(
                    B_prime_factor,
                    rhs_p,
                )
                timing["linear_solve_time"] += (
                    time.perf_counter() - phase_start
                )

                # Optional stabiliser: half the correction when the
                # raw Newton step would overshoot on very flat starts.
                # max_angle_corr = np.max(np.abs(d_theta_slice))
                # if max_angle_corr > 0.5:            # 0.5 rad ≈ 29°
                #     d_theta_slice *= 0.5 / max_angle_corr

                d_theta = np.zeros(grid.n)
                d_theta[non_slack] = d_theta_slice
                phase_start = time.perf_counter()
                grid.theta += d_theta
                timing["state_update_time"] += (
                    time.perf_counter() - phase_start
                )

            # ---------- Q–V half-step ----------
            if n_v > 0:
                # recompute mismatches with updated angles
                phase_start = time.perf_counter()
                dP, dQ = grid.get_mismatch()
                timing["mismatch_time"] += time.perf_counter() - phase_start

                # ΔV = B''⁻¹ · (ΔQ / V)    —  only for PQ buses
                rhs_q = dQ[pq_buses] / grid.V[pq_buses]
                phase_start = time.perf_counter()
                d_v = self._solve_factored(
                    B_double_prime_factor,
                    rhs_q,
                )
                timing["linear_solve_time"] += (
                    time.perf_counter() - phase_start
                )
                phase_start = time.perf_counter()
                grid.V[pq_buses] += d_v
                timing["state_update_time"] += (
                    time.perf_counter() - phase_start
                )

        # ---- did not converge ---------------------------------------
        info["iterations"] = self.max_iter
        info["time_elapsed"] = time.perf_counter() - start_time
        if self.verbose:
            print(
                f"快速解耦法未收敛！"
                f"达到最大迭代次数 {self.max_iter}"
            )
        return False, info
