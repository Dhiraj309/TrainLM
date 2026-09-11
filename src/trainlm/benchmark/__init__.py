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
    load_parity_closure_evidence,
)
from trainlm.benchmark.numerical_alignment import (
    NumericalAlignmentReport,
    NumericalDifference,
    REQUIRED_NUMERICAL_PATHS,
    compare_numerical_alignment,
    load_numerical_alignment_report,
)
from trainlm.benchmark.repeated_parity import (
    RepeatedParityEvaluation,
    evaluate_repeated_parity,
    load_repeated_parity_evaluation,
)
from trainlm.benchmark.stability import (
    RealShardStabilityEvaluation,
    RealShardStabilityEvidence,
    evaluate_real_shard_stability,
    load_real_shard_stability_evaluation,
)
from trainlm.benchmark.export_certification import (
    PlainHFExportEvaluation,
    PlainHFExportEvidence,
    evaluate_plain_hf_export,
)
from trainlm.benchmark.parity_report import ParityReport
from trainlm.benchmark.cross_family_matrix import (
    CrossFamilyMatrixEvaluation,
    CrossFamilyRecord,
    SupportLevel,
    evaluate_cross_family_matrix,
)
from trainlm.benchmark.fsdp_scaling import (
    FSDPScalingEvaluation,
    FSDPScalingEvidence,
    FSDPScalingTarget,
    evaluate_fsdp_scaling,
)

__all__ = [
    "BenchmarkResult",
    "CausalLMFlopBreakdown",
    "CrossFamilyMatrixEvaluation",
    "CrossFamilyRecord",
    "FSDPScalingEvaluation",
    "FSDPScalingEvidence",
    "FSDPScalingTarget",
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
    "SupportLevel",
    "RealShardStabilityEvaluation",
    "RealShardStabilityEvidence",
    "PlainHFExportEvaluation",
    "PlainHFExportEvidence",
    "ParityReport",
    "evaluate_baseline",
    "evaluate_attention_stage",
    "evaluate_cross_family_matrix",
    "evaluate_fsdp_scaling",
    "evaluate_parity_closure",
    "load_parity_closure_evidence",
    "compare_numerical_alignment",
    "load_numerical_alignment_report",
    "evaluate_repeated_parity",
    "load_repeated_parity_evaluation",
    "evaluate_real_shard_stability",
    "load_real_shard_stability_evaluation",
    "evaluate_plain_hf_export",
    "load_baseline_workload",
]
