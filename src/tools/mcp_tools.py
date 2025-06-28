"""
MCP (Model Context Protocol) tools configuration and management.
"""

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
from typing import Dict, Any


class MCPToolsManager:
    """Manages MCP tools configuration and client initialization."""
    
    def __init__(self, workspace_path: str = "/home/anni/dev/python/redo"):
        """
        Initialize the MCP tools manager.
        
        Args:
            workspace_path: Path to the workspace for filesystem tools
        """
        self.workspace_path = workspace_path
        self._client = None
        self._tools = None
        self._tool_node = None
    
    @property
    def server_config(self) -> Dict[str, Dict[str, Any]]:
        """Get the MCP server configuration."""
        return {
            # "fetch": {
            #     "command": "uvx",
            #     "args": ["mcp-server-fetch"],
            #     "transport": "stdio",
            # },
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    self.workspace_path,
                ],
                "transport": "stdio",
            },
            "time": {
                "command": "uvx",
                "args": ["mcp-server-time", "--local-timezone=Europe/Berlin"],
                "transport": "stdio",
            },
            "playwright": {
                "command": "npx",
                "args": ["-y", "@executeautomation/playwright-mcp-server"],
                "transport": "stdio",
            }
        }
    
    async def initialize_client(self) -> MultiServerMCPClient:
        """Initialize and return the MCP client."""
        if self._client is None:
            self._client = MultiServerMCPClient(self.server_config)
        return self._client
    
    async def get_tools(self):
        """Get tools from the MCP client."""
        if self._tools is None:
            client = await self.initialize_client()
            self._tools = await client.get_tools()
        return self._tools
    
    async def get_tool_node(self) -> ToolNode:
        """Get the LangGraph ToolNode for the tools."""
        if self._tool_node is None:
            tools = await self.get_tools()
            self._tool_node = ToolNode(tools=tools)
        return self._tool_node
    
    def route_tools(self, state):
        """
        Route to tools node if the last message has tool calls.
        
        Args:
            state: The graph state containing messages
            
        Returns:
            "tools" if tool calls are present, "END" otherwise
        """
        from langgraph.graph import END
        
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END
