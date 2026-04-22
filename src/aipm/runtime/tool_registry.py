import importlib.util
from pathlib import Path
from typing import Callable, Dict

from aipm.manifest.models import Capability, ToolSpec


class ToolRegistry:
    """
    Maps tool names → callable functions.

    Tool names are namespaced:
        <capability>.<tool>
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register_capability(self, capability: Capability, base_path: Path) -> None:
        """
        Load tools from a capability and register them.
        """
        module = self._load_module(capability, base_path)

        for tool in capability.tools:
            func = self._resolve_tool_function(module, tool)

            namespaced = f"{capability.name}.{tool.name}"

            self._tools[namespaced] = func

    def get(self, name: str) -> Callable:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def list_tools(self) -> Dict[str, Callable]:
        return dict(self._tools)

    # --- internal ---

    def _load_module(self, capability: Capability, base_path: Path):
        """
        Dynamically load the tool module.
        """
        module_path = base_path / capability.entrypoint

        if not module_path.exists():
            raise RuntimeError(
                f"Entrypoint not found: {module_path}"
            )

        spec = importlib.util.spec_from_file_location(
            capability.name, module_path
        )

        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module: {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    def _resolve_tool_function(self, module, tool: ToolSpec) -> Callable:
        """
        Find function in module matching tool name.
        """
        if not hasattr(module, tool.name):
            raise RuntimeError(
                f"Tool function '{tool.name}' not found in module"
            )

        func = getattr(module, tool.name)

        if not callable(func):
            raise RuntimeError(
                f"Tool '{tool.name}' is not callable"
            )

        return func
