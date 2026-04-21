# pyre-ignore-all-errors
"""
DashboardBI/components/kpi_card.py
───────────────────────────────────
Reusable KPI card component with accent color, delta indicator, and hover effects.
"""


def kpi_card(value, label, delta=None, delta_dir="neutral", accent_color="#3b82f6"):
    """Return HTML string for a single KPI card."""
    dir_class = {"up": "delta-up", "down": "delta-down", "neutral": "delta-neutral"}[delta_dir]
    arrow = {"up": "▲", "down": "▼", "neutral": ""}[delta_dir]
    delta_html = f'<div class="kpi-delta {dir_class}">{arrow} {delta}</div>' if delta else ""
    return f"""
<div class="kpi-card">
    <div class="kpi-accent" style="background:{accent_color};"></div>
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>
    {delta_html}
</div>"""


def kpi_row(cards_html: list) -> str:
    """Wrap a list of kpi_card HTML strings in a responsive grid."""
    return f'<div class="kpi-grid">{"".join(cards_html)}</div>'
