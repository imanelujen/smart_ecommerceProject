# pyre-ignore-all-errors
"""
DashboardBI/components/charts.py
─────────────────────────────────
Plotly theme configuration and chart builder helpers.
"""

import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore

# ── Color palettes ────────────────────────────────────────────────────
PALETTE = {
    "blue":    "#3b82f6",
    "teal":    "#14b8a6",
    "amber":   "#f59e0b",
    "purple":  "#8b5cf6",
    "rose":    "#f43f5e",
    "slate":   "#64748b",
    "green":   "#10b981",
    "indigo":  "#6366f1",
}

CLUSTER_COLORS = {
    "Premium":            "#8b5cf6",
    "Top rated":          "#14b8a6",
    "Discount / Promo":   "#f59e0b",
    "Niche / peu connu":  "#94a3b8",
    "Mainstream":         "#3b82f6",
    "Anomalie (DBSCAN)":  "#f43f5e",
}

PLATFORM_COLORS = {
    "shopify":     "#3b82f6",
    "woocommerce": "#8b5cf6",
    "other":       "#64748b",
}

# ── Plotly theme ──────────────────────────────────────────────────────
pio.templates["smart_ecommerce"] = go.layout.Template(
    layout={
        "font": {"family": "Inter, sans-serif", "size": 12, "color": "#334155"},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 0, "r": 0, "t": 10, "b": 0},
        "xaxis": {"gridcolor": "#f1f5f9", "gridwidth": 1},
        "yaxis": {"gridcolor": "#f1f5f9", "gridwidth": 1},
    }
)
pio.templates.default = "plotly_white+smart_ecommerce"


# ── Chart shortcuts ───────────────────────────────────────────────────

def styled_bar(df, x, y, orientation="h", color=None, color_scale=None,
               text=None, height=340, show_legend=False, **kwargs):
    """Common horizontal bar chart with consistent styling."""
    if color_scale is None:
        color_scale = [[0, "#eff6ff"], [1, "#3b82f6"]]
    fig = px.bar(df, x=x, y=y, orientation=orientation,
                 color=color or x, color_continuous_scale=color_scale,
                 text=text or x, **kwargs)
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(
        height=height, showlegend=show_legend, coloraxis_showscale=False,
        xaxis={"visible": False} if orientation == "h" else {"gridcolor": "#f8fafc"},
        yaxis={"title": "", "tickfont": {"size": 11}},
    )
    return fig


def styled_scatter(df, x, y, color=None, size=None, color_scale=None,
                   hover_data=None, height=360, **kwargs):
    """Common scatter chart."""
    if color_scale is None:
        color_scale = [[0, "#eff6ff"], [0.5, "#3b82f6"], [1, "#1e3a8a"]]
    fig = px.scatter(df, x=x, y=y, color=color, size=size,
                     color_continuous_scale=color_scale,
                     hover_data=hover_data, size_max=36, **kwargs)
    fig.update_layout(
        height=height,
        xaxis={"gridcolor": "#f8fafc"},
        yaxis={"gridcolor": "#f1f5f9"},
    )
    return fig


def styled_pie(labels, values, colors=None, height=320, hole=0.55):
    """Common donut chart."""
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=hole,
        marker_colors=colors,
        textinfo="percent+label", textfont_size=11,
    ))
    fig.update_layout(height=height, showlegend=False)
    return fig
