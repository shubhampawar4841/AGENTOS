"""In-process MCP client used by the agent layer."""

from __future__ import annotations

import logging
from typing import Any

from app.mcp import MCPError, MCPServer, MCPTool
from app.mcp.servers.calendar import CalendarMCPServer
from app.mcp.servers.gmail import GmailMCPServer
from app.mcp.servers.youtube import YouTubeMCPServer

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Discovers and invokes tools exposed by MCP servers.

    Pattern:
        Agent → MCPClient → Gmail/Calendar/YouTube MCP servers → Google APIs
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServer] = {}

    def register_server(self, server: MCPServer) -> None:
        if server.name in self._servers:
            raise MCPError(f"MCP server '{server.name}' is already registered")
        self._servers[server.name] = server
        logger.info("Registered MCP server '%s'", server.name)

    def list_servers(self) -> list[str]:
        return sorted(self._servers)

    def list_tools(self) -> list[MCPTool]:
        tools: list[MCPTool] = []
        for server in self._servers.values():
            tools.extend(server.list_tools())
        return tools

    def get_tool(self, tool_name: str) -> MCPTool:
        for tool in self.list_tools():
            if tool.name == tool_name:
                return tool
        raise MCPError(f"Tool '{tool_name}' not found")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Resolve a tool by fully-qualified name and invoke it."""
        if "." in tool_name:
            server_name, _ = tool_name.split(".", 1)
            server = self._servers.get(server_name)
            if server is not None:
                logger.debug("Calling MCP tool '%s' via server '%s'", tool_name, server_name)
                return await server.call_tool(tool_name, arguments)

        for server in self._servers.values():
            names = {t.name for t in server.list_tools()}
            if tool_name in names:
                logger.debug("Calling MCP tool '%s' via server '%s'", tool_name, server.name)
                return await server.call_tool(tool_name, arguments)

        raise MCPError(f"Tool '{tool_name}' not found")


def create_default_mcp_client() -> MCPClient:
    """Build the default MCP client with built-in servers registered."""
    client = MCPClient()
    client.register_server(GmailMCPServer())
    client.register_server(CalendarMCPServer())
    client.register_server(YouTubeMCPServer())
    return client
