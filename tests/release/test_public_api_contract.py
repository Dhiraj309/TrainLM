from pathlib import Path

import trainlm
from trainlm.release import (
    evaluate_public_api_compatibility,
    load_public_api_contract,
)


CONTRACT = Path(__file__).resolve().parents[2] / "support" / "public_api_v1.json"


def test_installed_public_surface_matches_versioned_contract():
    contract = load_public_api_contract(CONTRACT)

    result = evaluate_public_api_compatibility(
        contract,
        api_version=trainlm.PUBLIC_API_VERSION,
        public_symbols=tuple(trainlm.__all__),
        config_keys=("api_version", "model", "training_args", "args"),
        deprecated_config_keys=trainlm.DEPRECATED_CONFIG_KEYS,
    )

    assert result.compatible, result.reasons


def test_public_contract_detects_breaking_changes_without_version_bump():
    contract = load_public_api_contract(CONTRACT)

    result = evaluate_public_api_compatibility(
        contract,
        api_version="1",
        public_symbols=("TrainLMTrainer",),
        config_keys=("model",),
        deprecated_config_keys={},
    )

    assert not result.compatible
    assert len(result.reasons) == 3
