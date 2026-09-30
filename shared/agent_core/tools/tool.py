from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    tool_id: str
    name: str
    description: str
    input_schema: dict
    output_schema: dict
    permissions: list[str]
    risk_level: str
    execute: Callable[..., Any]