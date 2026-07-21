"""Power-system study workflows built on top of the core solvers."""

from .loadability import LoadabilityPoint, LoadabilityResult, scan_loadability
from .critical_stagnation import (
    CriticalStagnationPoint,
    CriticalStagnationResult,
    run_critical_stagnation_analysis,
)
from .robustness import (
    RobustnessSummary,
    RobustnessTrial,
    SolverRobustnessResult,
    run_solver_robustness,
)
from .scaling import (
    GridScaleMetrics,
    PowerLawFit,
    fit_power_law,
    measure_grid_scale,
)
from .solver_comparison import (
    SolverBenchmark,
    SolverComparisonResult,
    SolverTimingBreakdown,
    compare_power_flow_solvers,
)

__all__ = [
    "LoadabilityPoint",
    "LoadabilityResult",
    "scan_loadability",
    "CriticalStagnationPoint",
    "CriticalStagnationResult",
    "run_critical_stagnation_analysis",
    "RobustnessSummary",
    "RobustnessTrial",
    "SolverRobustnessResult",
    "run_solver_robustness",
    "GridScaleMetrics",
    "PowerLawFit",
    "fit_power_law",
    "measure_grid_scale",
    "SolverBenchmark",
    "SolverComparisonResult",
    "SolverTimingBreakdown",
    "compare_power_flow_solvers",
]
