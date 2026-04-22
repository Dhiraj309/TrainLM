from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass(frozen=True)
class Permissions:
    network: bool = False
    filesystem: bool = False
    subprocess: bool = False


@dataclass(frozen=True)
class Capability:
    name: str
    version: str
    entrypoint: str

    tools: List[ToolSpec]

    description: Optional[str] = None
    permissions: Permissions = field(default_factory=Permissions)
    dependencies: List[str] = field(default_factory=list)
