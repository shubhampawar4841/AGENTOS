"""MCP tool errors and shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


class MCPError(Exception):
    """Raised when an MCP tool call or discovery fails."""


@dataclass(frozen=True)
class MCPTool:
    """Descriptor for a single MCP-style tool."""

    name: str
    description: str
    handler: Callable[..., Awaitable[Any]]
    input_schema: dict[str, Any] = field(default_factory=dict)


class MCPServer:
    """Base class for in-process MCP tool servers."""

    name: str = "base"

    def list_tools(self) -> list[MCPTool]:
        raise NotImplementedError

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        tools = {tool.name: tool for tool in self.list_tools()}
        tool = tools.get(tool_name)
        if tool is None:
            raise MCPError(f"Unknown tool '{tool_name}' on server '{self.name}'")
        try:
            return await tool.handler(**(arguments or {}))
        except MCPError:
            raise
        except Exception as exc:
            raise MCPError(f"Tool '{tool_name}' failed: {exc}") from exc
