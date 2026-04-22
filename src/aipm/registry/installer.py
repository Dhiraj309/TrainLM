import shutil
from pathlib import Path

from aipm.manifest.loader import load_manifest
from aipm.manifest.errors import ManifestError

from .store import Store
from .lockfile import (
    load_lockfile,
    save_lockfile,
    add_capability,
    LockedCapability,
)


class InstallError(Exception):
    pass


def _copy_package(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)

    shutil.copytree(src, dest)


def install_from_path(source: str | Path, store: Store) -> LockedCapability:
    """
    Install a capability from a local directory.

    Steps:
    - validate manifest
    - copy package into store
    - update lockfile
    """
    source = Path(source).resolve()

    if not source.exists():
        raise InstallError(f"Source path does not exist: {source}")

    manifest_path = source / "capability.yaml"

    if not manifest_path.exists():
        raise InstallError(f"Missing capability.yaml in: {source}")

    # Step 1: validate manifest
    try:
        manifest = load_manifest(manifest_path)
    except ManifestError as e:
        raise InstallError(f"Manifest validation failed: {e}") from e

    # Step 2: copy package
    store.ensure()

    dest = store.package_path(manifest.name)

    _copy_package(source, dest)

    # Step 3: update lockfile
    lockfile = load_lockfile(store)

    locked = LockedCapability(
        name=manifest.name,
        version=manifest.version,
        source=str(source),
        path=str(dest),
    )

    lockfile = add_capability(lockfile, locked)

    save_lockfile(store, lockfile)

    return locked
