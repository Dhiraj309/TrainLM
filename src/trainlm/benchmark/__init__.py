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
    load_plain_hf_export_evaluation,
)
from trainlm.benchmark.parity_report import ParityReport, load_parity_report
from trainlm.benchmark.cross_family_matrix import (
    CrossFamilyMatrixEvaluation,
    CrossFamilyRecord,
    SupportLevel,
    evaluate_cross_family_matrix,
    load_cross_family_matrix_evaluation,
)
from trainlm.benchmark.fsdp_scaling import (
    FSDPScalingEvaluation,
    FSDPScalingEvidence,
    FSDPScalingTarget,
    evaluate_fsdp_scaling,
    load_fsdp_scaling_evaluation,
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
    "load_parity_report",
    "evaluate_baseline",
    "evaluate_attention_stage",
    "evaluate_cross_family_matrix",
    "load_cross_family_matrix_evaluation",
    "evaluate_fsdp_scaling",
    "load_fsdp_scaling_evaluation",
    "evaluate_parity_closure",
    "load_parity_closure_evidence",
    "compare_numerical_alignment",
    "load_numerical_alignment_report",
    "evaluate_repeated_parity",
    "load_repeated_parity_evaluation",
    "evaluate_real_shard_stability",
    "load_real_shard_stability_evaluation",
    "evaluate_plain_hf_export",
    "load_plain_hf_export_evaluation",
    "load_baseline_workload",
]
