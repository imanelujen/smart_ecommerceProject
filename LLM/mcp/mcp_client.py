"""
LLM/mcp/mcp_client.py
---------------------------
MCP Client — the single gateway between the Streamlit app and all MCP servers.

Responsibilities (per MCP spec):
  1. Receive tool call requests from the Host (app)
  2. Validate permissions via the Audit server
  3. Check rate limits
  4. Route to the correct MCP server
  5. Log every call (allowed or denied)
  6. Return results or error messages to the Host
"""

import time
import logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from LLM.mcp.mcp_server_audit import get_audit
from LLM.mcp.mcp_server_data  import call_tool as data_call
from LLM.mcp.mcp_server_llm   import call_tool as llm_call

logger = logging.getLogger(__name__)

# Map each tool name → which server handles it
TOOL_ROUTING = {
    # Data server
    "get_top_products":      "data_server",
    "get_cluster_profile":   "data_server",
    "get_association_rules": "data_server",
    "get_market_stats":      "data_server",
    # LLM server
    "summarize_product":     "llm_server",
    "generate_trend_report": "llm_server",
    "build_client_profile":  "llm_server",
    "recommend_strategy":    "llm_server",
    "answer_question":       "llm_server",
    "generate_bi_report":    "llm_server",
}

SERVER_HANDLERS = {
    "data_server": data_call,
    "llm_server":  llm_call,
}


class MCPClient:
    """
    The MCP Client — the only entry point for the Host application.
    Every tool call goes through here.
    """

    def __init__(self):
        self.audit = get_audit()

    def call(self, tool_name: str, params: dict = None) -> dict:
        """
        Execute a tool call with full permission + rate-limit checking.

        Returns:
            {"result": ..., "allowed": True, "latency_ms": ...}
            or
            {"error": "...", "allowed": False, "reason": "..."}
        """
        params = params or {}
        server = TOOL_ROUTING.get(tool_name)

        if server is None:
            self.audit.log("unknown", tool_name, allowed=False, reason="Unknown tool")
            return {"error": f"Unknown tool: '{tool_name}'", "allowed": False}

        # Permission check
        allowed, reason = self.audit.check_permission(server, tool_name)
        if not allowed:
            self.audit.log(server, tool_name, allowed=False, reason=reason)
            logger.warning(f"[MCP] DENIED {server}.{tool_name}: {reason}")
            return {"error": reason, "allowed": False, "reason": reason}

        # Rate limit check
        rate_ok, rate_reason = self.audit.check_rate_limit(server)
        if not rate_ok:
            self.audit.log(server, tool_name, allowed=False, reason=rate_reason)
            logger.warning(f"[MCP] RATE-LIMITED {server}.{tool_name}")
            return {"error": rate_reason, "allowed": False, "reason": rate_reason}

        # Execute
        t0 = time.time()
        handler = SERVER_HANDLERS[server]
        response = handler(tool_name, params)
        latency  = (time.time() - t0) * 1000

        result_size = len(str(response.get("result", "")))
        has_error   = "error" in response

        self.audit.log(
            server, tool_name,
            allowed=True,
            reason="error" if has_error else "OK",
            latency_ms=latency,
            result_size=result_size,
        )

        if has_error:
            logger.error(f"[MCP] Tool error {tool_name}: {response['error']}")
        else:
            logger.info(f"[MCP] {tool_name} → {result_size} chars in {latency:.0f}ms")

        return {**response, "allowed": True, "latency_ms": round(latency, 1)}


# Singleton for use in Streamlit
_mcp_client = None

def get_mcp_client() -> MCPClient:
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client