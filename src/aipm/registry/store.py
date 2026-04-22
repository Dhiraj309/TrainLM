from pathlib import Path


AIPM_DIR = ".aipm"
PACKAGES_DIR = "packages"
LOCKFILE_NAME = "lock.json"


class Store:
    """
    Manages local aipm storage paths.

    Does NOT perform installs or validation.
    Only handles filesystem structure.
    """

    def __init__(self, root: str | Path = "."):
        self.root = Path(root).resolve()
        self.base = self.root / AIPM_DIR
        self.packages = self.base / PACKAGES_DIR
        self.lockfile = self.base / LOCKFILE_NAME

    def ensure(self) -> None:
        """
        Ensure required directories exist.
        """
        self.base.mkdir(parents=True, exist_ok=True)
        self.packages.mkdir(parents=True, exist_ok=True)

    def package_path(self, name: str) -> Path:
        """
        Get path to a specific installed package.
        """
        return self.packages / name

    def has_package(self, name: str) -> bool:
        """
        Check if a package is already installed.
        """
        return self.package_path(name).exists()
