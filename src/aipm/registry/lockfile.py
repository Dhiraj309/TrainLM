import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from .store import Store


LOCKFILE_VERSION = "1"


@dataclass(frozen=True)
class LockedCapability:
    name: str
    version: str
    source: str
    path: str


@dataclass(frozen=True)
class Lockfile:
    version: str
    capabilities: List[LockedCapability]


def _serialize(lockfile: Lockfile) -> dict:
    return {
        "version": lockfile.version,
        "capabilities": [asdict(c) for c in lockfile.capabilities],
    }


def _deserialize(data: dict) -> Lockfile:
    capabilities = [
        LockedCapability(**item) for item in data.get("capabilities", [])
    ]

    return Lockfile(
        version=data.get("version", LOCKFILE_VERSION),
        capabilities=capabilities,
    )


def load_lockfile(store: Store) -> Lockfile:
    """
    Load lockfile from disk.
    If not present, return empty lockfile.
    """
    path = store.lockfile

    if not path.exists():
        return Lockfile(version=LOCKFILE_VERSION, capabilities=[])

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return _deserialize(data)


def save_lockfile(store: Store, lockfile: Lockfile) -> None:
    """
    Write lockfile to disk (deterministic format).
    """
    store.ensure()

    path = store.lockfile
    data = _serialize(lockfile)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def add_capability(
    lockfile: Lockfile,
    capability: LockedCapability
) -> Lockfile:
    """
    Add or replace a capability in the lockfile.
    """
    filtered = [
        c for c in lockfile.capabilities if c.name != capability.name
    ]

    filtered.append(capability)

    return Lockfile(
        version=lockfile.version,
        capabilities=sorted(filtered, key=lambda c: c.name),
    )
