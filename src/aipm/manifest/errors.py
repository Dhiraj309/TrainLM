class ManifestError(Exception):
    """Base class for all manifest-related errors."""
    pass


class ManifestLoadError(ManifestError):
    """Raised when a manifest file cannot be read or parsed."""
    pass


class ManifestValidationError(ManifestError):
    """Raised when a manifest fails schema validation."""
    pass


class ManifestStructureError(ManifestError):
    """Raised when manifest structure is invalid after schema validation."""
    pass
