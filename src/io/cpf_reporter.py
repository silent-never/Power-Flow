"""Continuation-power-flow reporting helpers."""

from .pf_reporter import ResultReporter


def report_cpf_results(cpf_data, info=None):
    """Print the grid state carried by a CPF result object.

    The dedicated PV-curve and stability-margin output can be added here once
    the CPF solver exposes those result fields.
    """
    grid = getattr(cpf_data, "grid", None)
    if grid is None:
        raise ValueError("cpf_data must expose a `grid` attribute")
    reporter = ResultReporter(grid)
    reporter.print_node_results(info)
    return reporter
