"""Planning boundary for canonical Hugging Face checkpoint export."""

from __future__ import annotations

from dataclasses import dataclass

from .export import HFExportManifest


@dataclass(frozen=True, slots=True)
class CanonicalHFExportPlan:
    """Validated export intent passed to a backend-specific writer."""

    export_id: str
    target_parameter_layout: str
    safe_serialization: bool
    required_roles: tuple[str, ...]
    forbidden_roles: tuple[str, ...]
    reversed_transform_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.export_id.strip():
            raise ValueError("export_id cannot be empty.")
        if self.target_parameter_layout != "huggingface":
            raise ValueError("Canonical exports require Hugging Face layout.")
        if self.safe_serialization is not True:
            raise ValueError("Canonical exports require safetensors.")
        if self.required_roles != ("config", "model_weights"):
            raise ValueError("Canonical export required roles are invalid.")
        if self.forbidden_roles:
            raise ValueError("Canonical exports cannot retain training-only roles.")


def plan_canonical_hf_export(
    manifest: HFExportManifest,
) -> CanonicalHFExportPlan:
    """Validate a committed manifest before invoking ``save_pretrained``."""

    if not isinstance(manifest, HFExportManifest):
        raise TypeError("manifest must be an HFExportManifest.")
    manifest.assert_loadable()
    roles = {artifact.role for artifact in manifest.artifacts}
    forbidden = tuple(
        sorted(roles & HFExportManifest._FORBIDDEN_ROLES)  # noqa: SLF001
    )
    required = tuple(
        role for role in ("config", "model_weights") if role in roles
    )
    return CanonicalHFExportPlan(
        export_id=manifest.export_id,
        target_parameter_layout=manifest.layout.target_parameter_layout,
        safe_serialization=manifest.safe_serialization,
        required_roles=required,
        forbidden_roles=forbidden,
        reversed_transform_ids=manifest.layout.reversed_transform_ids,
    )
