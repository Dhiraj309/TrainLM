import jsonschema
from typing import Any, Callable, Dict

from aipm.manifest.models import ToolSpec


class ExecutionError(Exception):
    """Raised when tool execution fails."""
    pass


class ToolExecutor:
    """
    Responsible for executing tools safely and consistently.

    Responsibilities:
    - Validate input against schema
    - Execute tool function
    - Capture errors
    - Return structured result

    Does NOT:
    - Resolve tools (handled by ToolRegistry)
    - Manage permissions (policy layer later)
    - Handle loops or retries (runtime layer)
    """

    def execute(
        self,
        tool: ToolSpec,
        func: Callable,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a tool with validated input.

        Returns:
            {
                "ok": bool,
                "result": Any | None,
                "error": str | None
            }
        """

        # Step 1: Validate input
        try:
            jsonschema.validate(instance=arguments, schema=tool.input_schema)
        except jsonschema.ValidationError as e:
            return {
                "ok": False,
                "result": None,
                "error": f"Invalid input: {e.message}",
            }

        # Step 2: Execute tool
        try:
            result = func(**arguments)
        except Exception as e:
            return {
                "ok": False,
                "result": None,
                "error": f"Execution error: {str(e)}",
            }

        # Step 3: Return structured result
        return {
            "ok": True,
            "result": result,
            "error": None,
        }
