"""Application configuration utilities.

The project configuration deliberately uses a small, flat YAML subset so it
can be read without adding a YAML dependency to this learning project.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class SolverConfig:
    data_file: str = "data/039ieee.txt"
    base_mva: float = 100.0
    algorithm: str = "nr"
    tolerance: float = 1e-6
    max_iterations: int = 20
    plot_dir: str = "output/plots"
    run_loadability_scan: bool = True
    load_multiplier_start: float = 1.0
    load_multiplier_stop: float = 3.0
    load_multiplier_step: float = 0.1
    load_refinement_tolerance: float = 1e-4
    load_scan_max_iterations: int = 50

    # 连续潮流配置
    run_cpf: bool = True
    cpf_parameterization: str = "pseudo_arclength"
    cpf_lambda_init: float = 0.0
    cpf_lambda_min: float = -0.99
    cpf_lambda_max: float = 2.0
    cpf_step_size: float = 0.02
    cpf_min_step: float = 1e-4
    cpf_max_step: float = 0.1
    cpf_step_increase_factor: float = 1.25
    cpf_step_decrease_factor: float = 0.5
    cpf_fast_convergence_iters: int = 4
    cpf_slow_convergence_iters: int = 15
    cpf_max_step_retries: int = 8
    cpf_corrector_tolerance: float = 1e-8
    cpf_max_corrector_iterations: int = 15
    cpf_max_steps: int = 500
    cpf_post_nose_steps: int = 20
    cpf_initial_direction: int = 1
    cpf_enforce_q_limits: bool = True
    cpf_tangent_angle_bus: int = 0
    cpf_tangent_angle_refinement: bool = True
    cpf_tangent_angle_refinement_cos_threshold: float = 0.35
    cpf_tangent_angle_refinement_min_step_ratio: float = 0.02
    cpf_tangent_angle_pseudo_fallback: bool = True
    cpf_absolute_vp_angle_bus: int = 0
    cpf_absolute_vp_angle_pseudo_fallback: bool = True
    cpf_monitor_bus: int = 0
    cpf_verbose: bool = True


def _parse_scalar(value: str) -> Any:
    """Parse a scalar value used by the project's flat YAML file."""
    value = value.strip()
    if not value:
        return ""
    if value[0:1] in {"'", '"'} and value[-1:] == value[0]:
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_config(path: str | Path = "config.yaml") -> SolverConfig:
    """Load the project's flat YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    values: dict[str, Any] = {}
    for line_number, raw_line in enumerate(
        config_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"配置文件第 {line_number} 行缺少 ':'")
        key, value = line.split(":", maxsplit=1)
        values[key.strip()] = _parse_scalar(value)

    return load_from_dict(values)


def load_from_dict(values: dict[str, Any]) -> SolverConfig:
    """Build and validate a solver configuration from a mapping."""
    config = SolverConfig(**values)
    if config.base_mva <= 0:
        raise ValueError("base_mva must be positive")
    if config.tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if config.load_multiplier_start <= 0:
        raise ValueError("load_multiplier_start must be positive")
    if config.load_multiplier_stop <= config.load_multiplier_start:
        raise ValueError("load_multiplier_stop must exceed load_multiplier_start")
    if config.load_multiplier_step <= 0 or config.load_refinement_tolerance <= 0:
        raise ValueError("load scan step and refinement tolerance must be positive")
    if config.load_scan_max_iterations <= 0:
        raise ValueError("load_scan_max_iterations must be positive")
    if config.cpf_corrector_tolerance <= 0:
        raise ValueError("cpf_corrector_tolerance must be positive")
    if config.cpf_max_corrector_iterations <= 0 or config.cpf_max_steps <= 0:
        raise ValueError("CPF iteration and step limits must be positive")
    if config.cpf_post_nose_steps <= 0:
        raise ValueError("cpf_post_nose_steps must be positive")
    if config.cpf_monitor_bus < 0:
        raise ValueError("cpf_monitor_bus cannot be negative")
    if config.cpf_tangent_angle_bus < 0:
        raise ValueError("cpf_tangent_angle_bus cannot be negative")
    if config.cpf_absolute_vp_angle_bus < 0:
        raise ValueError("cpf_absolute_vp_angle_bus cannot be negative")
    if not 0 < config.cpf_tangent_angle_refinement_cos_threshold <= 1:
        raise ValueError(
            "cpf_tangent_angle_refinement_cos_threshold must be in (0, 1]"
        )
    if not 0 < config.cpf_tangent_angle_refinement_min_step_ratio <= 1:
        raise ValueError(
            "cpf_tangent_angle_refinement_min_step_ratio must be in (0, 1]"
        )
    return config
