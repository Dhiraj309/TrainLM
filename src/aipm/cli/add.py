from pathlib import Path

from aipm.registry.store import Store
from aipm.registry.installer import install_from_path, InstallError


def run_add(source: str) -> None:
    """
    Install a capability from a local path.
    """
    store = Store()

    try:
        result = install_from_path(Path(source), store)
    except InstallError as e:
        print(f"ERROR: {e}")
        return

    print(f"Installed {result.name}@{result.version}")
