from pathlib import Path
from typing import List, Dict, Any

from aipm.adapters.transformers import TransformersAdapter
from aipm.runtime.tool_registry import ToolRegistry
from aipm.execution.executor import ToolExecutor
from aipm.runtime.tool_call import parse_tool_call
from aipm.manifest.models import Capability, ToolSpec


class Session:
    """
    Runtime session that orchestrates:
    - model interaction
    - tool calling
    - execution loop

    Now includes:
    - argument error retry loop
    - stricter control flow
    """

    def __init__(
        self,
        adapter: TransformersAdapter,
        capabilities: List[Capability],
        base_path: Path,
        max_steps: int = 5,
    ):
        self.adapter = adapter
        self.max_steps = max_steps

        # Tool registry
        self.registry = ToolRegistry()

        for cap in capabilities:
            self.registry.register_capability(cap, base_path / cap.name)

        # Execution engine
        self.executor = ToolExecutor()

        # Cache tool specs
        self._tool_specs: Dict[str, ToolSpec] = {}
        for cap in capabilities:
            for tool in cap.tools:
                namespaced = f"{cap.name}.{tool.name}"
                self._tool_specs[namespaced] = tool

    def run(self, user_input: str) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_input},
        ]

        for step in range(self.max_steps):
            print(f"\n========== STEP {step} ==========")

            output = self.adapter.generate(messages)

            print(f"\n[MODEL OUTPUT]:\n{output}\n")

            tool_call = parse_tool_call(output)

            print(f"[PARSED TOOL CALL]: {tool_call}")

            stripped = output.strip()
            looks_like_json = stripped.startswith("{") and stripped.endswith("}")

            # --- CASE 1: No parsed tool call ---
            if tool_call is None:
                if looks_like_json:
                    print("[INFO] JSON detected but parsing failed → retry")

                    messages.append({
                        "role": "assistant",
                        "content": output,
                    })

                    messages.append({
                        "role": "user",
                        "content": self._invalid_json_error(),
                    })

                    continue

                print("[INFO] No JSON → final answer")
                return output

            tool_name = tool_call.tool

            # --- CASE 2: Invalid tool name ---
            if tool_name not in self._tool_specs:
                print(f"[ERROR] Invalid tool: {tool_name}")

                messages.append({
                    "role": "assistant",
                    "content": output,
                })

                messages.append({
                    "role": "user",
                    "content": self._tool_name_error(tool_name),
                })

                continue

            # --- CASE 3: Execute tool ---
            print(f"[EXECUTE] {tool_name}")

            func = self.registry.get(tool_name)
            tool_spec = self._tool_specs[tool_name]

            result = self.executor.execute(
                tool=tool_spec,
                func=func,
                arguments=tool_call.arguments,
            )

            print(f"[RESULT]: {result}")

            messages.append({
                "role": "assistant",
                "content": output,
            })

            # --- NEW: HANDLE TOOL FAILURE (retry loop) ---
            if not result["ok"]:
                print("[RETRY] Tool execution failed → asking model to fix arguments")

                messages.append({
                    "role": "user",
                    "content": self._tool_error_retry(tool_name, result["error"]),
                })

                continue

            # --- SUCCESS PATH ---
            messages.append({
                "role": "user",
                "content": self._format_tool_result(tool_name, result),
            })

        return "ERROR: Max steps exceeded"

    # --- helpers ---

    def _build_system_prompt(self) -> str:
        tool_blocks = []

        for name, spec in self._tool_specs.items():
            tool_blocks.append(
                f"""
Tool: {name}
Description: {spec.description}
Input JSON Schema:
{spec.input_schema}
""".strip()
            )

        tools_text = "\n\n".join(tool_blocks)

        return f"""
You are an AI assistant that can use tools.

IMPORTANT RULES:
- Use tools when needed
- Use ONLY listed tool names
- Return ONLY JSON for tool calls

AVAILABLE TOOLS:
{tools_text}

EXAMPLE:
{{
  "tool": "http_fetch.fetch",
  "arguments": {{
    "url": "https://example.com"
  }}
}}
""".strip()

    def _invalid_json_error(self) -> str:
        return """
Your previous response had invalid JSON.

Return ONLY valid JSON:

{
  "tool": "<tool_name>",
  "arguments": { ... }
}
""".strip()

    def _tool_name_error(self, invalid_name: str) -> str:
        valid_tools = "\n".join(self._tool_specs.keys())

        return f"""
Invalid tool: {invalid_name}

Use one of:
{valid_tools}

Return ONLY JSON.
""".strip()

    def _tool_error_retry(self, tool_name: str, error: str) -> str:
        return f"""
The tool '{tool_name}' failed with this error:

{error}

Fix your arguments and call the tool again.

Return ONLY valid JSON.
""".strip()

    def _format_tool_result(self, tool_name: str, result: Dict[str, Any]) -> str:
        return f"""
You have received data from the tool '{tool_name}'.

DATA:
{result['result']}

Your task:
- Summarize this content clearly for the user.
- Use your own words.
- Do NOT repeat the text verbatim.
- Keep it concise and useful.

Final answer:
""".strip()
