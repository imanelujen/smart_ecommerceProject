"""
module5/mcp/mcp_server_data.py
--------------------------------
MCP Server: Data — exposes product datasets as tools.

Follows the Model Context Protocol pattern:
  - Each tool is a declared function with a typed schema
  - The server only exposes what is strictly needed (principle of isolation)
  - All calls are logged to the audit server
"""

import json
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from module4.data_loader import load_scored, load_top_k, load_clusters, load_rules, load_shops


# ── Tool registry ──────────────────────────────────────────────────────
# Each tool declares: name, description, parameters, handler function.
# The MCP Client uses this registry to know what it can call.

TOOLS = {
    "get_top_products": {
        "description": "Returns the Top-K products sorted by composite score.",
        "parameters": {
            "k":        {"type": "integer", "default": 20, "description": "Number of products"},
            "category": {"type": "string",  "default": None, "description": "Optional category filter"},
        },
        "handler": lambda params: _get_top_products(**params),
    },
    "get_cluster_profile": {
        "description": "Returns average metrics per product cluster.",
        "parameters": {},
        "handler": lambda params: _get_cluster_profile(),
    },
    "get_association_rules": {
        "description": "Returns top association rules ranked by lift.",
        "parameters": {
            "min_lift": {"type": "number", "default": 1.0},
            "top_n":    {"type": "integer","default": 10},
        },
        "handler": lambda params: _get_rules(**params),
    },
    "get_market_stats": {
        "description": "Returns aggregate market statistics.",
        "parameters": {
            "group_by": {"type": "string", "default": "category",
                         "description": "Field to group by (category, platform, shop_country)"},
        },
        "handler": lambda params: _get_stats(**params),
    },
}


def call_tool(tool_name: str, params: dict) -> dict:
    """
    Dispatch a tool call. Returns {result, error}.
    This is the single entry point — audit logging happens in mcp_client.py.
    """
    if tool_name not in TOOLS:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        result = TOOLS[tool_name]["handler"](params)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


# ── Tool implementations ───────────────────────────────────────────────

def _get_top_products(k: int = 20, category: str = None) -> list:
    df = load_top_k()
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    cols = [c for c in ["title","category","price","rating","review_count",
                         "discount_pct","score","shop_name","platform"] if c in df.columns]
    return df.head(k)[cols].to_dict(orient="records")


def _get_cluster_profile() -> list:
    df = load_clusters()
    if "cluster_label" not in df.columns:
        return []
    cols = [c for c in ["price","rating","review_count","discount_pct","score"] if c in df.columns]
    profile = df.groupby("cluster_label")[cols].mean().round(3)
    profile["count"] = df.groupby("cluster_label").size()
    return profile.reset_index().to_dict(orient="records")


def _get_rules(min_lift: float = 1.0, top_n: int = 10) -> list:
    df = load_rules()
    if df.empty:
        return []
    return (df[df["lift"] >= min_lift]
              .head(top_n)
              [["antecedents","consequents","support","confidence","lift"]]
              .to_dict(orient="records"))


def _get_stats(group_by: str = "category") -> list:
    df = load_scored()
    if group_by not in df.columns:
        group_by = "category"
    cols = [c for c in ["price","rating","review_count","score"] if c in df.columns]
    stats = df.groupby(group_by)[cols].mean().round(3)
    stats["count"] = df.groupby(group_by).size()
    return stats.reset_index().to_dict(orient="records")