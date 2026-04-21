# pyre-ignore-all-errors
"""
DashboardBI/components/layout.py
─────────────────────────────────
Reusable layout helpers: page header, section dividers, alerts, insight cards, footer.
"""

import streamlit as st  # type: ignore
from datetime import datetime


def page_header(title: str, subtitle: str = "", product_count: int = 0):
    """Render a professional page header with optional product count badge."""
    sub_html = f'<span class="page-subtitle">· {subtitle}</span>' if subtitle else ""
    badge_html = ""
    if product_count > 0:
        badge_html = f'''
        <div class="header-badge">
            <span class="dot"></span> {product_count:,} products
        </div>'''
    st.markdown(f"""
<div class="page-header">
    <div class="page-header-left">
        <h1 class="page-title">{title}</h1>
        {sub_html}
    </div>
    <div class="page-header-right">
        {badge_html}
    </div>
</div>""", unsafe_allow_html=True)


def section(label: str):
    """Render an uppercase section divider."""
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)


def alert(msg: str, kind: str = "info"):
    """Render a colored alert banner (info/success/warning/danger)."""
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "danger": "🔴"}
    st.markdown(
        f'<div class="alert-banner alert-{kind}">{icons.get(kind, "ℹ️")} {msg}</div>',
        unsafe_allow_html=True,
    )


def insight_box(title: str, body: str, accent_color: str = ""):
    """Render a styled insight card."""
    style = f' style="border-left: 4px solid {accent_color};"' if accent_color else ""
    title_style = f' style="color:{accent_color}"' if accent_color else ""
    st.markdown(f"""
<div class="insight-box"{style}>
    <h4{title_style}>{title}</h4>
    <p>{body}</p>
</div>""", unsafe_allow_html=True)


def footer():
    """Render a dashboard footer with last-updated timestamp."""
    now = datetime.now().strftime("%b %d, %Y at %H:%M")
    st.markdown(f"""
<div class="dashboard-footer">
    Smart eCommerce Intelligence · Last updated {now} · v2.0
</div>""", unsafe_allow_html=True)
