"""
LLM/mcp/mcp_server_audit.py
---------------------------------
MCP Server: Audit — logs every tool call, enforces permissions and rate limits.

MCP principle: agents must declare their intentions and respect usage rules.
This server is the single source of truth for access control.

Permissions matrix:
  data_server:  get_top_products, get_cluster_profile, get_rules, get_stats
  llm_server:   summarize_product, generate_trend_report,
                build_client_profile, recommend_strategy, answer_question
  rate_limit:   max 20 LLM calls / minute per session
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LOG_PATH = Path("LLM/logs/audit.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Permission table ───────────────────────────────────────────────────
PERMISSIONS = {
    "data_server": {
        "get_top_products", "get_cluster_profile",
        "get_association_rules", "get_market_stats",
    },
    "llm_server": {
        "summarize_product",
        "generate_trend_report",
        "build_client_profile",
        "recommend_strategy",
        "answer_question",
        "generate_bi_report",
    },
}

RATE_LIMITS = {
    "llm_server": {"calls_per_minute": 20},
    "data_server": {"calls_per_minute": 60},
}

# In-memory counters (reset every minute)
_call_counts: dict = defaultdict(list)    # server → [timestamps]


class AuditServer:
    """
    Centralised audit and access control for all MCP tool calls.
    Logs every request with: timestamp, server, tool, allowed, latency.
    """

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = Path(log_path or LOG_PATH)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def check_permission(self, server: str, tool: str) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        allowed_tools = PERMISSIONS.get(server, set())
        if tool not in allowed_tools:
            return False, f"Tool '{tool}' not permitted for server '{server}'"
        return True, "OK"

    def check_rate_limit(self, server: str) -> tuple[bool, str]:
        """Returns (allowed, reason). Sliding 60-second window."""
        limit = RATE_LIMITS.get(server, {}).get("calls_per_minute", 100)
        now   = time.time()
        window_start = now - 60
        # Keep only timestamps in the last 60s
        _call_counts[server] = [t for t in _call_counts[server] if t > window_start]
        if len(_call_counts[server]) >= limit:
            return False, f"Rate limit exceeded: {limit} calls/min for '{server}'"
        _call_counts[server].append(now)
        return True, "OK"

    def log(self, server: str, tool: str, allowed: bool,
            reason: str = "OK", latency_ms: float = 0,
            result_size: int = 0):
        """Append a structured log entry to the audit JSONL file."""
        entry = {
            "ts":          datetime.utcnow().isoformat(),
            "server":      server,
            "tool":        tool,
            "allowed":     allowed,
            "reason":      reason,
            "latency_ms":  round(latency_ms, 1),
            "result_size": result_size,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Audit log write failed: {e}")
        logger.debug(f"[AUDIT] {entry}")
        return entry

    def get_recent_logs(self, n: int = 50) -> list:
        """Read the last n log entries."""
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text().strip().split("\n")
        return [json.loads(l) for l in lines[-n:] if l]

    def get_stats(self) -> dict:
        """Compute audit statistics: calls per server, blocked rate."""
        logs = self.get_recent_logs(n=1000)
        if not logs:
            return {}
        total      = len(logs)
        blocked    = sum(1 for l in logs if not l["allowed"])
        by_server  = defaultdict(int)
        for l in logs:
            by_server[l["server"]] += 1
        return {
            "total_calls":   total,
            "blocked_calls": blocked,
            "block_rate":    round(blocked / total, 3) if total else 0,
            "by_server":     dict(by_server),
            "avg_latency_ms":round(sum(l["latency_ms"] for l in logs) / total, 1),
        }


# Singleton instance used across the app
_audit = AuditServer()

def get_audit() -> AuditServer:
    return _audit