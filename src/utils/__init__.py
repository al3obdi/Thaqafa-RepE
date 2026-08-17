"""Shared helpers: probes, baselines, evaluation, plotting."""

from src.utils.baselines import (
    INSTRUCTION_TEMPLATES,
    ComparisonResult,
    ConditionResult,
    InstructionTemplate,
    compare_steering_vs_prompting,
    generate_prompt_baseline,
    generate_steering_condition,
    get_template,
)
from src.utils.evaluation import (
    DEFAULT_STRENGTHS,
    SteeringResult,
    compute_prompt_loss,
    evaluate_layer_sets,
    evaluate_steering,
    generate_steered,
    measure_steering_effect,
    summarize_layer_sets,
    summarize_sweep,
)
from src.utils.probes import (
    LinearProbe,
    ProbeResult,
    best_layer,
    chance_accuracy,
    probe_layer,
    summarize_probe_sweep,
    sweep_layers_with_probe,
)
from src.utils.visualization import plot_concept_similarity, plot_layer_scores, plot_steering_sweep

__all__ = [
    "DEFAULT_STRENGTHS",
    "INSTRUCTION_TEMPLATES",
    "ComparisonResult",
    "ConditionResult",
    "InstructionTemplate",
    "LinearProbe",
    "ProbeResult",
    "SteeringResult",
    "best_layer",
    "chance_accuracy",
    "compare_steering_vs_prompting",
    "compute_prompt_loss",
    "evaluate_layer_sets",
    "evaluate_steering",
    "generate_prompt_baseline",
    "generate_steered",
    "generate_steering_condition",
    "get_template",
    "measure_steering_effect",
    "plot_concept_similarity",
    "plot_layer_scores",
    "plot_steering_sweep",
    "probe_layer",
    "summarize_layer_sets",
    "summarize_probe_sweep",
    "summarize_sweep",
    "sweep_layers_with_probe",
]
