"""Small public greeting helper for TrainLM."""


def greet(name: str = "TrainLM") -> str:
    """Return a friendly greeting for ``name``."""

    if not isinstance(name, str):
        raise TypeError("name must be a string.")
    name = name.strip()
    if not name:
        raise ValueError("name must not be empty.")
    return f"Hello, {name}!"


__all__ = ["greet"]
