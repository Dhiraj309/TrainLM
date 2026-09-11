import pytest
import torch

from trainlm.optimization import ParameterLayoutMapping, StateDictLayoutConverter


def _mapping(dtype="float32"):
    return ParameterLayoutMapping(
        mapping_id="pack-qkv",
        canonical_keys=("q.weight", "k.weight", "v.weight"),
        transformed_key="qkv.weight",
        canonical_shapes=((4, 3), (2, 3), (2, 3)),
        axis=0,
        dtype=dtype,
    )


def test_pack_split_round_trip_preserves_values_shapes_and_dtype():
    canonical = {
        "q.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "k.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "v.weight": torch.arange(6, 12, dtype=torch.float32).reshape(2, 3),
        "other.weight": torch.ones(1),
    }
    converter = StateDictLayoutConverter((_mapping(),))

    transformed = converter.to_transformed(canonical)
    restored = converter.to_canonical(transformed)

    assert transformed["qkv.weight"].shape == (8, 3)
    assert "q.weight" not in transformed
    assert restored.keys() == canonical.keys()
    assert all(torch.equal(restored[key], value) for key, value in canonical.items())
    assert all(restored[key].dtype == value.dtype for key, value in canonical.items())


def test_manifest_round_trip_is_checkpoint_safe():
    converter = StateDictLayoutConverter(
        (_mapping(),), alias_groups=(("embed.weight", "lm_head.weight"),)
    )

    restored = StateDictLayoutConverter.from_manifest(converter.manifest())

    assert restored.mappings == converter.mappings
    assert restored.alias_groups == converter.alias_groups


def test_alias_groups_are_validated_and_restored_as_shared_objects():
    tied = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    converter = StateDictLayoutConverter(
        (), alias_groups=(("embed.weight", "lm_head.weight"),)
    )

    result = converter.to_canonical(
        {"embed.weight": tied.clone(), "lm_head.weight": tied.clone()}
    )

    assert result["embed.weight"] is result["lm_head.weight"]
    with pytest.raises(ValueError, match="disagree"):
        converter.to_transformed(
            {"embed.weight": tied, "lm_head.weight": tied + 1}
        )


def test_converter_rejects_overlapping_alias_groups():
    with pytest.raises(ValueError, match="multiple alias groups"):
        StateDictLayoutConverter(
            (),
            alias_groups=(
                ("embed.weight", "lm_head.weight"),
                ("lm_head.weight", "output.weight"),
            ),
        )


@pytest.mark.parametrize("alias_key", ("q.weight", "qkv.weight"))
def test_converter_rejects_aliases_for_mapped_layout_keys(alias_key):
    with pytest.raises(ValueError, match="cannot also belong to alias groups"):
        StateDictLayoutConverter(
            (_mapping(),),
            alias_groups=((alias_key, "shared.weight"),),
        )


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({"q.weight": torch.zeros(3, 3), "k.weight": torch.zeros(2, 3), "v.weight": torch.zeros(2, 3)}, "expected shape"),
        ({"q.weight": torch.zeros(4, 3, dtype=torch.float64), "k.weight": torch.zeros(2, 3, dtype=torch.float64), "v.weight": torch.zeros(2, 3, dtype=torch.float64)}, "expected dtype"),
    ],
)
def test_pack_rejects_shape_and_dtype_mismatches(state, message):
    with pytest.raises(ValueError, match=message):
        StateDictLayoutConverter((_mapping(),)).to_transformed(state)


def test_converter_rejects_missing_and_colliding_keys_without_mutating_input():
    canonical = {
        "q.weight": torch.zeros(4, 3),
        "k.weight": torch.zeros(2, 3),
        "v.weight": torch.zeros(2, 3),
        "qkv.weight": torch.zeros(8, 3),
    }
    before = dict(canonical)

    with pytest.raises(ValueError, match="already exists"):
        StateDictLayoutConverter((_mapping(),)).to_transformed(canonical)
    with pytest.raises(KeyError, match="v.weight"):
        StateDictLayoutConverter((_mapping(),)).to_transformed(
            {"q.weight": canonical["q.weight"], "k.weight": canonical["k.weight"]}
        )

    assert canonical == before
