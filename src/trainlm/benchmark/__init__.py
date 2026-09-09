"""Versioned benchmark result and MFU contracts."""

from trainlm.benchmark.mfu import (
    CausalLMFlopBreakdown,
    calculate_causal_lm_flops,
    calculate_mfu,
)
from trainlm.benchmark.result import BenchmarkResult
from trainlm.benchmark.baseline import (
    BaselineEvaluation,
    BaselineWorkload,
    evaluate_baseline,
    load_baseline_workload,
)
from trainlm.benchmark.attention_stage import (
    AttentionStageEvaluation,
    evaluate_attention_stage,
)
from trainlm.benchmark.parity_closure import (
    ParityClosureEvaluation,
    ParityClosureEvidence,
    evaluate_parity_closure,
)
from trainlm.benchmark.numerical_alignment import (
    NumericalAlignmentReport,
    NumericalDifference,
    REQUIRED_NUMERICAL_PATHS,
    compare_numerical_alignment,
)

__all__ = [
    "BenchmarkResult",
    "CausalLMFlopBreakdown",
    "calculate_causal_lm_flops",
    "calculate_mfu",
    "BaselineEvaluation",
    "BaselineWorkload",
    "AttentionStageEvaluation",
    "ParityClosureEvaluation",
    "ParityClosureEvidence",
    "NumericalAlignmentReport",
    "NumericalDifference",
    "REQUIRED_NUMERICAL_PATHS",
    "evaluate_baseline",
    "evaluate_attention_stage",
    "evaluate_parity_closure",
    "compare_numerical_alignment",
    "load_baseline_workload",
]
