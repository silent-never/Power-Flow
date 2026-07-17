"""连续潮流的参数、预测、校正与步长控制组件。"""

from .corrector import CorrectorResult, apply_corrector
from .parameter import (
    ABSOLUTE_VP_ANGLE_PARAMETERIZATION,
    TANGENT_ANGLE_PARAMETERIZATION,
    NATURAL_PARAMETERIZATION,
    PSEUDO_ARCLENGTH_PARAMETERIZATION,
    CPFParameter,
    CPFParameterState,
)
from .predictor import CPFTangent, PredictorResult, compute_predictor, compute_tangent
from .step_control import StepControlResult, adjust_step

__all__ = [
    "ABSOLUTE_VP_ANGLE_PARAMETERIZATION",
    "TANGENT_ANGLE_PARAMETERIZATION",
    "NATURAL_PARAMETERIZATION",
    "PSEUDO_ARCLENGTH_PARAMETERIZATION",
    "CPFParameter",
    "CPFParameterState",
    "CPFTangent",
    "PredictorResult",
    "CorrectorResult",
    "StepControlResult",
    "compute_tangent",
    "compute_predictor",
    "apply_corrector",
    "adjust_step",
]
