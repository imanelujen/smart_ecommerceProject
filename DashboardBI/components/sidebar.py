# pyre-ignore-all-errors
"""
DashboardBI/components/sidebar.py
──────────────────────────────────
Professional sidebar with brand header, navigation, filters, and quick actions.
"""

import streamlit as st  # type: ignore
import pandas as pd  # type: ignore

from DashboardBI.nav_config import NAV_ITEMS


def render_sidebar(df_all: pd.DataFrame) -> tuple:
    """
    Render the full sidebar and return (page, filters_dict).
    filters_dict has keys: sel_cat, sel_plat, price_range
    """
    with st.sidebar:
        # ── Brand Header ──────────────────────────────────────────────
        st.markdown("""
<div class="sidebar-brand">
    <div class="brand-icon">🛒</div>
    <div class="brand-text">
        <strong>Smart eCommerce</strong>
        <span class="version-badge">v2.0 · Intelligence</span>
    </div>
</div>""", unsafe_allow_html=True)
        st.divider()

        # ── Navigation ────────────────────────────────────────────────
        st.markdown('<div class="nav-section-label">Navigation</div>',
                    unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            NAV_ITEMS,
            label_visibility="collapsed",
            key="bi_main_nav",
        )
        st.divider()

        # ── Filters ───────────────────────────────────────────────────
        st.markdown('<div class="nav-section-label">Filters</div>',
                    unsafe_allow_html=True)

        cats = ["All"] + sorted(df_all["category"].dropna().unique().tolist())
        platforms = ["All"] + sorted(df_all["platform"].dropna().unique().tolist())

        sel_cat = st.selectbox("Category", cats, label_visibility="collapsed")
        sel_plat = st.selectbox("Platform", platforms, label_visibility="collapsed")

        price_min = float(df_all["price"].min())
        price_max = float(df_all["price"].max())
        price_range = st.slider(
            "Price range ($)", price_min, price_max, (price_min, price_max),
        )
        st.divider()

        # ── Dataset Stats ─────────────────────────────────────────────
        n_shops = df_all["shop_name"].nunique() if "shop_name" in df_all.columns else 0
        n_cats = df_all["category"].nunique() if "category" in df_all.columns else 0
        st.markdown(f"""
<div class="sidebar-stats">
    <div class="stat-row"><span>Products</span><span class="stat-value">{len(df_all):,}</span></div>
    <div class="stat-row"><span>Shops</span><span class="stat-value">{n_shops}</span></div>
    <div class="stat-row"><span>Categories</span><span class="stat-value">{n_cats}</span></div>
</div>""", unsafe_allow_html=True)

    return page, {
        "sel_cat": sel_cat,
        "sel_plat": sel_plat,
        "price_range": price_range,
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filter selections to a DataFrame."""
    mask = (
        (df["price"] >= filters["price_range"][0])
        & (df["price"] <= filters["price_range"][1])
    )
    if filters["sel_cat"] != "All":
        mask &= df["category"] == filters["sel_cat"]
    if filters["sel_plat"] != "All":
        mask &= df["platform"] == filters["sel_plat"]
    return df[mask]
