# pyre-ignore-all-errors
"""
Single source of truth for sidebar page navigation.
Import NAV_ITEMS wherever navigation labels or routing are defined.
"""

from typing import Tuple

NAV_ITEMS: Tuple[str, ...] = (
    "Overview",
    "Top-K Products",
    "Clusters",
    "Pricing Analysis",
    "Association Rules",
    "Shops & Geography",
    "LLM Insights",
    "Assistant BI",
)
