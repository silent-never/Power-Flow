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
    comparison_repeat_count: int = 5
    comparison_all_cases: bool = True
    comparison_detail_case: int = 118
    comparison_flat_start: bool = True
    comparison_fast_decoupled_max_iterations: int = 100
    comparison_run_tolerance_sweep: bool = True
    comparison_tolerance_values: str = "1e-4,1e-6,1e-8,1e-10"
    comparison_tolerance_repeat_count: int = 31
    comparison_run_robustness: bool = True
    robustness_random_trials: int = 20
    robustness_voltage_perturbations: str = "0.02,0.05,0.10"
    robustness_angle_perturbations_deg: str = "2,5,10"
    robustness_load_multipliers: str = "1.58,1.60,1.605,1.6074,1.61,1.615,1.63"
    robustness_load_refinement_tolerance: float = 1e-5
    robustness_resistance_multipliers: str = "1.0,2.0,3.0"
    robustness_random_seed: int = 2026
    comparison_run_scaling_analysis: bool = True
    comparison_scaling_min_nodes: int = 30
    comparison_run_critical_stagnation: bool = True
    critical_stagnation_cpf_nose_multiplier: float = 1.6074
    critical_stagnation_load_multipliers: str = (
        "1.600,1.605,1.6074,1.610,1.612,1.613,1.614,1.615,1.617,1.620"
    )
    critical_stagnation_max_iterations: int = 60
    critical_stagnation_enforce_q_limits: bool = True
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
    if config.comparison_repeat_count <= 1:
        raise ValueError("comparison_repeat_count must exceed 1")
    if not isinstance(config.comparison_all_cases, bool):
        raise ValueError("comparison_all_cases must be boolean")
    if config.comparison_detail_case <= 0:
        raise ValueError("comparison_detail_case must be positive")
    if not isinstance(config.comparison_flat_start, bool):
        raise ValueError("comparison_flat_start must be boolean")
    if config.comparison_fast_decoupled_max_iterations <= 0:
        raise ValueError(
            "comparison_fast_decoupled_max_iterations must be positive"
        )
    if not isinstance(config.comparison_run_tolerance_sweep, bool):
        raise ValueError("comparison_run_tolerance_sweep must be boolean")
    try:
        tolerance_values = [
            float(value.strip())
            for value in config.comparison_tolerance_values.replace("，", ",").split(",")
            if value.strip()
        ]
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            "comparison_tolerance_values must be comma-separated numbers"
        ) from exc
    if not tolerance_values or any(value <= 0.0 for value in tolerance_values):
        raise ValueError("comparison tolerance values must be positive")
    if config.comparison_tolerance_repeat_count <= 1:
        raise ValueError("comparison_tolerance_repeat_count must exceed 1")
    if not isinstance(config.comparison_run_robustness, bool):
        raise ValueError("comparison_run_robustness must be boolean")
    if config.robustness_random_trials <= 0:
        raise ValueError("robustness_random_trials must be positive")
    robustness_value_names = (
        "robustness_voltage_perturbations",
        "robustness_angle_perturbations_deg",
        "robustness_load_multipliers",
        "robustness_resistance_multipliers",
    )
    parsed_robustness_values: dict[str, list[float]] = {}
    for name in robustness_value_names:
        raw_value = getattr(config, name)
        try:
            parsed = [
                float(value.strip())
                for value in raw_value.replace("，", ",").split(",")
                if value.strip()
            ]
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"{name} must be comma-separated numbers") from exc
        if not parsed or any(value <= 0.0 for value in parsed):
            raise ValueError(f"{name} values must be positive")
        parsed_robustness_values[name] = parsed
    if len(parsed_robustness_values["robustness_voltage_perturbations"]) != len(
        parsed_robustness_values["robustness_angle_perturbations_deg"]
    ):
        raise ValueError("robustness voltage and angle levels must have equal length")
    if config.robustness_load_refinement_tolerance <= 0.0:
        raise ValueError("robustness_load_refinement_tolerance must be positive")
    if not isinstance(config.comparison_run_scaling_analysis, bool):
        raise ValueError("comparison_run_scaling_analysis must be boolean")
    if config.comparison_scaling_min_nodes <= 0:
        raise ValueError("comparison_scaling_min_nodes must be positive")
    if not isinstance(config.comparison_run_critical_stagnation, bool):
        raise ValueError("comparison_run_critical_stagnation must be boolean")
    if config.critical_stagnation_cpf_nose_multiplier <= 0.0:
        raise ValueError("critical_stagnation_cpf_nose_multiplier must be positive")
    try:
        stagnation_values = [
            float(value.strip())
            for value in config.critical_stagnation_load_multipliers.replace(
                "，", ","
            ).split(",")
            if value.strip()
        ]
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            "critical_stagnation_load_multipliers must be comma-separated numbers"
        ) from exc
    if not stagnation_values or any(value <= 0.0 for value in stagnation_values):
        raise ValueError("critical stagnation load multipliers must be positive")
    if config.critical_stagnation_max_iterations <= 0:
        raise ValueError("critical_stagnation_max_iterations must be positive")
    if not isinstance(config.critical_stagnation_enforce_q_limits, bool):
        raise ValueError("critical_stagnation_enforce_q_limits must be boolean")
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
