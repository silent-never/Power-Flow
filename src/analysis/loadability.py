"""Static load-multiplier scan used before implementing full CPF."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ..core.grid import PowerGrid
from ..solvers.nr_solver import NewtonRaphsonSolver


@dataclass(frozen=True)
class LoadabilityPoint:
    multiplier: float
    converged: bool
    iterations: int
    min_voltage: float
    min_voltage_bus: int | None
    max_error: float
    reason: str = ""


@dataclass
class LoadabilityResult:
    points: list[LoadabilityPoint]
    last_converged_multiplier: float | None
    first_failed_multiplier: float | None
    base_load_mw: float
    base_load_mvar: float

    @property
    def collapse_bracket(self) -> tuple[float, float] | None:
        if (
            self.last_converged_multiplier is None
            or self.first_failed_multiplier is None
        ):
            return None
        return self.last_converged_multiplier, self.first_failed_multiplier


def _scaled_grid(
    base_grid: PowerGrid,
    multiplier: float,
    initial_grid: PowerGrid | None,
) -> PowerGrid:
    buses = copy.deepcopy(base_grid.buses)
    for bus in buses:
        bus["load_mw"] = float(bus.get("load_mw", 0.0)) * multiplier
        bus["load_mvar"] = float(bus.get("load_mvar", 0.0)) * multiplier

    grid = PowerGrid(
        buses=buses,
        branches=copy.deepcopy(base_grid.branches),
        base_mva=base_grid.base_mva,
    )
    if initial_grid is not None:
        grid.V = initial_grid.V.copy()
        grid.theta = initial_grid.theta.copy()
    return grid


def _solve_point(
    base_grid: PowerGrid,
    multiplier: float,
    initial_grid: PowerGrid | None,
    tolerance: float,
    max_iterations: int,
) -> tuple[LoadabilityPoint, PowerGrid | None]:
    grid = _scaled_grid(base_grid, multiplier, initial_grid)
    solver = NewtonRaphsonSolver(
        tol=tolerance,
        max_iter=max_iterations,
        verbose=False,
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            converged, info = solver.solve(grid)
        finite_state = bool(np.all(np.isfinite(grid.V)) and np.all(np.isfinite(grid.theta)))
        converged = bool(converged and finite_state and np.all(grid.V > 0.0))
        errors = info.get("max_error_history", [])
        point = LoadabilityPoint(
            multiplier=multiplier,
            converged=converged,
            iterations=int(info.get("iterations", 0)),
            min_voltage=float(np.min(grid.V)) if finite_state else float("nan"),
            min_voltage_bus=(
                int(grid.buses[int(np.argmin(grid.V))]["number"])
                if finite_state
                else None
            ),
            max_error=float(errors[-1]) if errors else float("nan"),
            reason="" if converged else "Newton-Raphson did not converge",
        )
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        point = LoadabilityPoint(
            multiplier=multiplier,
            converged=False,
            iterations=0,
            min_voltage=float("nan"),
            min_voltage_bus=None,
            max_error=float("nan"),
            reason=f"{type(exc).__name__}: {exc}",
        )
        return point, None

    return point, grid if point.converged else None


def scan_loadability(
    base_grid: PowerGrid,
    start: float = 1.0,
    stop: float = 3.0,
    step: float = 0.1,
    refinement_tolerance: float = 1e-3,
    tolerance: float = 1e-6,
    max_iterations: int = 30,
) -> LoadabilityResult:
    """Scale every P/Q load together and bracket the first failed PF point.

    Each converged solution initializes the next multiplier.  Once a coarse
    step fails, bisection narrows the numerical convergence boundary.
    """
    if start <= 0 or stop <= start:
        raise ValueError("load scan requires 0 < start < stop")
    if step <= 0 or refinement_tolerance <= 0:
        raise ValueError("step and refinement_tolerance must be positive")

    points: list[LoadabilityPoint] = []
    last_factor: float | None = None
    failed_factor: float | None = None
    last_grid: PowerGrid | None = None

    count = int(np.floor((stop - start) / step + 1e-12))
    factors = [start + index * step for index in range(count + 1)]
    if factors[-1] < stop - 1e-12:
        factors.append(stop)

    for factor in factors:
        point, solved_grid = _solve_point(
            base_grid,
            factor,
            last_grid,
            tolerance,
            max_iterations,
        )
        points.append(point)
        if not point.converged:
            failed_factor = factor
            break
        last_factor = factor
        last_grid = solved_grid

    if last_factor is not None and failed_factor is not None:
        low = last_factor
        high = failed_factor
        while high - low > refinement_tolerance:
            middle = (low + high) / 2.0
            point, solved_grid = _solve_point(
                base_grid,
                middle,
                last_grid,
                tolerance,
                max_iterations,
            )
            points.append(point)
            if point.converged:
                low = middle
                last_factor = middle
                last_grid = solved_grid
            else:
                high = middle
                failed_factor = middle

    return LoadabilityResult(
        points=sorted(points, key=lambda item: item.multiplier),
        last_converged_multiplier=last_factor,
        first_failed_multiplier=failed_factor,
        base_load_mw=sum(float(bus.get("load_mw", 0.0)) for bus in base_grid.buses),
        base_load_mvar=sum(float(bus.get("load_mvar", 0.0)) for bus in base_grid.buses),
    )
