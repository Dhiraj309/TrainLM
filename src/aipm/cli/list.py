from aipm.registry.store import Store
from aipm.registry.lockfile import load_lockfile


def run_list() -> None:
    """
    List installed capabilities.
    """
    store = Store()
    lockfile = load_lockfile(store)

    if not lockfile.capabilities:
        print("No capabilities installed.")
        return

    for cap in lockfile.capabilities:
        print(f"{cap.name}@{cap.version}")
