"""Validated reversible mappings between canonical and optimized state dicts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True, slots=True)
class ParameterLayoutMapping:
    """One concatenated parameter layout and its canonical split geometry."""

    mapping_id: str
    canonical_keys: tuple[str, ...]
    transformed_key: str
    canonical_shapes: tuple[tuple[int, ...], ...]
    axis: int = 0
    dtype: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mapping_id, str) or not self.mapping_id.strip():
            raise ValueError("Layout mapping ID cannot be empty.")
        if len(self.canonical_keys) < 2 or any(
            not isinstance(key, str) or not key.strip() for key in self.canonical_keys
        ):
            raise ValueError("Layout mappings require at least two canonical keys.")
        if len(self.canonical_keys) != len(set(self.canonical_keys)):
            raise ValueError("Canonical layout keys must be unique.")
        if not isinstance(self.transformed_key, str) or not self.transformed_key.strip():
            raise ValueError("Transformed layout key cannot be empty.")
        if self.transformed_key in self.canonical_keys:
            raise ValueError("Transformed and canonical layout keys must differ.")
        if len(self.canonical_shapes) != len(self.canonical_keys):
            raise ValueError("Every canonical key requires one declared shape.")
        if not self.canonical_shapes or any(
            not shape
            or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in shape)
            for shape in self.canonical_shapes
        ):
            raise ValueError("Canonical shapes must contain positive dimensions.")
        ranks = {len(shape) for shape in self.canonical_shapes}
        if len(ranks) != 1:
            raise ValueError("Canonical tensors must have equal rank.")
        rank = ranks.pop()
        if isinstance(self.axis, bool) or not isinstance(self.axis, int) or not -rank <= self.axis < rank:
            raise ValueError("Layout mapping axis is outside the tensor rank.")
        axis = self.axis % rank
        reference = self.canonical_shapes[0]
        if any(
            any(size != reference[index] for index, size in enumerate(shape) if index != axis)
            for shape in self.canonical_shapes[1:]
        ):
            raise ValueError("Canonical shapes must match outside the concatenation axis.")
        object.__setattr__(self, "axis", axis)
        if self.dtype is not None and (
            not isinstance(self.dtype, str) or not self.dtype.strip()
        ):
            raise ValueError("Layout mapping dtype cannot be empty.")

    @property
    def transformed_shape(self) -> tuple[int, ...]:
        shape = list(self.canonical_shapes[0])
        shape[self.axis] = sum(item[self.axis] for item in self.canonical_shapes)
        return tuple(shape)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParameterLayoutMapping":
        if not isinstance(value, Mapping):
            raise TypeError("Layout mapping manifest entry must be a mapping.")
        data = dict(value)
        required = {
            "mapping_id",
            "canonical_keys",
            "transformed_key",
            "canonical_shapes",
        }
        allowed = required | {"axis", "dtype"}
        missing = sorted(required - data.keys())
        unknown = sorted(
            repr(key) for key in data if not isinstance(key, str) or key not in allowed
        )
        if missing:
            raise ValueError(
                "Layout mapping manifest is missing keys: " + ", ".join(missing)
            )
        if unknown:
            raise ValueError(
                "Layout mapping manifest has unknown keys: " + ", ".join(unknown)
            )
        if not isinstance(data["canonical_keys"], (list, tuple)):
            raise TypeError("canonical_keys must be a list or tuple.")
        if not isinstance(data["canonical_shapes"], (list, tuple)) or any(
            not isinstance(shape, (list, tuple))
            for shape in data["canonical_shapes"]
        ):
            raise TypeError("canonical_shapes must be a sequence of shapes.")
        data["canonical_keys"] = tuple(data["canonical_keys"])
        data["canonical_shapes"] = tuple(tuple(shape) for shape in data["canonical_shapes"])
        return cls(**data)


class StateDictLayoutConverter:
    """Pack for optimized resume and split for canonical Hugging Face export."""

    def __init__(
        self,
        mappings: tuple[ParameterLayoutMapping, ...],
        *,
        alias_groups: tuple[tuple[str, ...], ...] = (),
    ) -> None:
        if any(not isinstance(item, ParameterLayoutMapping) for item in mappings):
            raise TypeError("mappings must contain ParameterLayoutMapping values.")
        mapping_ids = [item.mapping_id for item in mappings]
        if len(mapping_ids) != len(set(mapping_ids)):
            raise ValueError("Layout mapping IDs must be unique.")
        canonical_keys = [key for item in mappings for key in item.canonical_keys]
        transformed_keys = [item.transformed_key for item in mappings]
        if len(canonical_keys) != len(set(canonical_keys)):
            raise ValueError("Canonical keys cannot participate in multiple mappings.")
        if len(transformed_keys) != len(set(transformed_keys)):
            raise ValueError("Transformed keys must be unique.")
        if set(canonical_keys) & set(transformed_keys):
            raise ValueError("Canonical and transformed key sets cannot overlap.")
        alias_keys: list[str] = []
        for group in alias_groups:
            if len(group) < 2 or len(group) != len(set(group)) or any(
                not isinstance(key, str) or not key.strip() for key in group
            ):
                raise ValueError("Alias groups require at least two unique keys.")
            alias_keys.extend(group)
        if len(alias_keys) != len(set(alias_keys)):
            raise ValueError("State-dict keys cannot participate in multiple alias groups.")
        mapped_keys = set(canonical_keys) | set(transformed_keys)
        conflicting_aliases = sorted(mapped_keys & set(alias_keys))
        if conflicting_aliases:
            raise ValueError(
                "Mapped layout keys cannot also belong to alias groups: "
                + ", ".join(conflicting_aliases)
            )
        self.mappings = mappings
        self.alias_groups = alias_groups

    def to_transformed(self, state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pack canonical tensors into optimized layouts for transformed resume."""

        result = self._copy_and_validate(state_dict)
        self._validate_alias_values(result)
        for mapping in self.mappings:
            tensors = [self._pop_tensor(result, key) for key in mapping.canonical_keys]
            self._validate_canonical(mapping, tensors)
            if mapping.transformed_key in result:
                raise ValueError(f"Transformed key already exists: {mapping.transformed_key}")
            result[mapping.transformed_key] = torch.cat(tensors, dim=mapping.axis)
        self._restore_aliases(result)
        return result

    def to_canonical(self, state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Split optimized tensors into plain Hugging Face parameter layouts."""

        result = self._copy_and_validate(state_dict)
        for mapping in reversed(self.mappings):
            tensor = self._pop_tensor(result, mapping.transformed_key)
            self._validate_tensor(mapping, tensor, mapping.transformed_shape)
            if any(key in result for key in mapping.canonical_keys):
                raise ValueError(f"Canonical keys already exist for mapping {mapping.mapping_id}.")
            sizes = tuple(shape[mapping.axis] for shape in mapping.canonical_shapes)
            pieces = torch.split(tensor, sizes, dim=mapping.axis)
            result.update(zip(mapping.canonical_keys, pieces, strict=True))
        self._restore_aliases(result)
        self._validate_alias_values(result)
        return result

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mappings": [mapping.to_dict() for mapping in self.mappings],
            "alias_groups": [list(group) for group in self.alias_groups],
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> "StateDictLayoutConverter":
        if not isinstance(value, Mapping):
            raise TypeError("State-dict layout manifest must be a mapping.")
        allowed = {"schema_version", "mappings", "alias_groups"}
        unknown = sorted(
            repr(key) for key in value if not isinstance(key, str) or key not in allowed
        )
        if unknown:
            raise ValueError(
                "State-dict layout manifest has unknown keys: " + ", ".join(unknown)
            )
        missing = sorted(allowed - value.keys())
        if missing:
            raise ValueError(
                "State-dict layout manifest is missing keys: " + ", ".join(missing)
            )
        schema_version = value["schema_version"]
        if isinstance(schema_version, bool) or schema_version != 1:
            raise ValueError("State-dict layout manifest supports schema_version=1 only.")
        mappings = value["mappings"]
        alias_groups = value["alias_groups"]
        if not isinstance(mappings, (list, tuple)):
            raise TypeError("State-dict layout mappings must be a list or tuple.")
        if not isinstance(alias_groups, (list, tuple)) or any(
            not isinstance(group, (list, tuple)) for group in alias_groups
        ):
            raise TypeError("State-dict alias_groups must be a sequence of groups.")
        return cls(
            tuple(ParameterLayoutMapping.from_dict(item) for item in mappings),
            alias_groups=tuple(tuple(group) for group in alias_groups),
        )

    @staticmethod
    def _copy_and_validate(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping.")
        result = dict(state_dict)
        if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in result.items()):
            raise TypeError("State-dict entries must map string keys to tensors.")
        return result

    @staticmethod
    def _pop_tensor(state_dict: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        try:
            return state_dict.pop(key)
        except KeyError as exc:
            raise KeyError(f"State dict is missing required layout key: {key}") from exc

    def _validate_canonical(self, mapping, tensors) -> None:
        dtypes = {tensor.dtype for tensor in tensors}
        devices = {tensor.device for tensor in tensors}
        if len(dtypes) != 1 or len(devices) != 1:
            raise ValueError(f"Mapping {mapping.mapping_id} tensors require one dtype and device.")
        for tensor, shape in zip(tensors, mapping.canonical_shapes, strict=True):
            self._validate_tensor(mapping, tensor, shape)

    @staticmethod
    def _validate_tensor(mapping, tensor, shape) -> None:
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"Mapping {mapping.mapping_id} expected shape {tuple(shape)}, "
                f"received {tuple(tensor.shape)}."
            )
        if mapping.dtype is not None and str(tensor.dtype).removeprefix("torch.") != mapping.dtype:
            raise ValueError(
                f"Mapping {mapping.mapping_id} expected dtype {mapping.dtype}, "
                f"received {tensor.dtype}."
            )

    def _validate_alias_values(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        for group in self.alias_groups:
            present = [key for key in group if key in state_dict]
            if present and len(present) != len(group):
                raise ValueError(f"Alias group is incomplete: {group!r}.")
            if present:
                reference = state_dict[group[0]]
                if any(
                    value.shape != reference.shape
                    or value.dtype != reference.dtype
                    or not torch.equal(value, reference)
                    for value in (state_dict[key] for key in group[1:])
                ):
                    raise ValueError(f"Alias group tensors disagree: {group!r}.")

    def _restore_aliases(self, state_dict: dict[str, torch.Tensor]) -> None:
        for group in self.alias_groups:
            present = [key for key in group if key in state_dict]
            if present:
                primary = state_dict[present[0]]
                for key in group:
                    if key in state_dict:
                        state_dict[key] = primary


__all__ = ["ParameterLayoutMapping", "StateDictLayoutConverter"]
