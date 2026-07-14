"""Continuation parameter management.
Empty scaffold file to be implemented.
"""

from dataclasses import dataclass

@dataclass
class CPFParameter:
    lambda_init: float = 0.0
    step_size: float = 0.01
    max_step: float = 0.1
    min_step: float = 1e-6
    max_iters: int = 20
