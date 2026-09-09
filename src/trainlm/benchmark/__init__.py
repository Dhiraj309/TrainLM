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
from trainlm.benchmark.repeated_parity import (
    RepeatedParityEvaluation,
    evaluate_repeated_parity,
)
from trainlm.benchmark.stability import (
    RealShardStabilityEvaluation,
    RealShardStabilityEvidence,
    evaluate_real_shard_stability,
)
from trainlm.benchmark.export_certification import (
    PlainHFExportEvaluation,
    PlainHFExportEvidence,
    evaluate_plain_hf_export,
)
from trainlm.benchmark.parity_report import ParityReport

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
    "RepeatedParityEvaluation",
    "RealShardStabilityEvaluation",
    "RealShardStabilityEvidence",
    "PlainHFExportEvaluation",
    "PlainHFExportEvidence",
    "ParityReport",
    "evaluate_baseline",
    "evaluate_attention_stage",
    "evaluate_parity_closure",
    "compare_numerical_alignment",
    "evaluate_repeated_parity",
    "evaluate_real_shard_stability",
    "evaluate_plain_hf_export",
    "load_baseline_workload",
]
