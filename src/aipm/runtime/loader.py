from pathlib import Path
from typing import List

from aipm.registry.store import Store
from aipm.registry.lockfile import load_lockfile
from aipm.manifest.loader import load_manifest
from aipm.manifest.models import Capability


def load_installed_capabilities(store: Store) -> List[Capability]:
    """
    Load all installed capabilities from the local store.

    Returns:
        List[Capability]
    """
    lockfile = load_lockfile(store)

    capabilities: List[Capability] = []

    for item in lockfile.capabilities:
        path = Path(item.path)

        manifest_path = path / "capability.yaml"

        if not manifest_path.exists():
            # fail hard — lockfile is invalid
            raise RuntimeError(
                f"Missing manifest for installed capability: {item.name}"
            )

        capability = load_manifest(manifest_path)

        capabilities.append(capability)

    return capabilities
