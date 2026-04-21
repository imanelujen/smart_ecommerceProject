# pyre-ignore-all-errors
"""
DashboardBI/app.py
------------------
Smart eCommerce Intelligence — Professional BI Dashboard v2.0

Single-file architecture for maximum Streamlit compatibility.

Run:
    streamlit run DashboardBI/app.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from DashboardBI.data_loader import (
    load_scored, load_top_k, load_clusters, load_pca, load_rules, load_shops
)
from DashboardBI.nav_config import NAV_ITEMS

# ════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart eCommerce Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — CSS
# ════════════════════════════════════════════════════════════════════════
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# THEME PALETTE
# ════════════════════════════════════════════════════════════════════════
PALETTE = {
    "blue":   "#3b82f6", "teal":   "#14b8a6", "amber":  "#f59e0b",
    "purple": "#8b5cf6", "rose":   "#f43f5e", "slate":  "#64748b",
    "green":  "#10b981", "indigo": "#6366f1",
}
CLUSTER_COLORS = {
    "Premium": "#8b5cf6", "Top rated": "#14b8a6",
    "Discount / Promo": "#f59e0b", "Niche / peu connu": "#94a3b8",
    "Mainstream": "#3b82f6", "Anomalie (DBSCAN)": "#f43f5e",
}
PLATFORM_COLORS = {"shopify": "#3b82f6", "woocommerce": "#8b5cf6", "other": "#64748b"}

pio.templates["smart_ecommerce"] = go.layout.Template(layout={
    "font": {"family": "Inter, sans-serif", "size": 12, "color": "#334155"},
    "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": {"l": 0, "r": 0, "t": 10, "b": 0},
})
pio.templates.default = "plotly_white+smart_ecommerce"

# ════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════
def page_header(title, subtitle="", count=0):
    sub = f'<span class="page-subtitle">{subtitle}</span>' if subtitle else ""
    badge = f'<div class="header-badge"><span class="dot"></span> {count:,} products</div>' if count else ""
    st.markdown(f"""
<div class="page-header">
  <div class="page-header-left"><h1 class="page-title">{title}</h1>{sub}</div>
  <div class="page-header-right">{badge}</div>
</div>""", unsafe_allow_html=True)

def section(label):
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)

def kpi_card(value, label, delta=None, delta_dir="neutral", accent="#3b82f6"):
    dc = {"up":"delta-up","down":"delta-down","neutral":"delta-neutral"}[delta_dir]
    ar = {"up":"▲","down":"▼","neutral":""}[delta_dir]
    dh = f'<div class="kpi-delta {dc}">{ar} {delta}</div>' if delta else ""
    return f"""<div class="kpi-card"><div class="kpi-accent" style="background:{accent};"></div>
<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{dh}</div>"""

def kpi_row(cards):
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

def alert(msg, kind="info"):
    icons = {"info":"ℹ","success":"✓","warning":"⚠","danger":"✕"}
    st.markdown(f'<div class="alert-banner alert-{kind}"><b>{icons.get(kind,"")}</b> {msg}</div>',
                unsafe_allow_html=True)

def insight_box(title, body, accent=""):
    s = f' style="border-left:4px solid {accent};"' if accent else ""
    ts = f' style="color:{accent}"' if accent else ""
    st.markdown(f'<div class="insight-box"{s}><h4{ts}>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True)

def footer():
    st.markdown(f'<div class="dashboard-footer">Smart CARE</div>',
                unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div class="sidebar-brand">
  <div class="brand-icon">S</div>
  <div class="brand-text">
    <strong>Smart CARE</strong>
    <span class="version-badge">AI</span>
  </div>
</div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="nav-section-label">NAVIGATION</div>', unsafe_allow_html=True)
    page = st.radio(
        "Nav",
        NAV_ITEMS,
        label_visibility="collapsed",
        key="bi_main_nav",
    )
    st.divider()

    st.markdown('<div class="nav-section-label">FILTERS</div>', unsafe_allow_html=True)
    df_all = load_scored()
    cats = ["All"] + sorted(df_all["category"].dropna().unique().tolist())
    plats = ["All"] + sorted(df_all["platform"].dropna().unique().tolist())
    sel_cat = st.selectbox("Category", cats, label_visibility="collapsed")
    sel_plat = st.selectbox("Platform", plats, label_visibility="collapsed")
    price_range = st.slider("Price ($)",
                            float(df_all["price"].min()), float(df_all["price"].max()),
                            (float(df_all["price"].min()), float(df_all["price"].max())))
    st.divider()

    n_shops = df_all["shop_name"].nunique() if "shop_name" in df_all.columns else 0
    n_cats = df_all["category"].nunique()
    st.markdown(f"""
<div class="sidebar-stats">
  <div class="stat-row"><span>Products</span><span class="stat-value">{len(df_all):,}</span></div>
  <div class="stat-row"><span>Shops</span><span class="stat-value">{n_shops}</span></div>
  <div class="stat-row"><span>Categories</span><span class="stat-value">{n_cats}</span></div>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# FILTER APPLICATION
# ════════════════════════════════════════════════════════════════════════
def apply_filters(df):
    m = (df["price"] >= price_range[0]) & (df["price"] <= price_range[1])
    if sel_cat != "All":  m &= df["category"] == sel_cat
    if sel_plat != "All": m &= df["platform"] == sel_plat
    return df[m]

df = apply_filters(df_all)

# ════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════
if page == "Overview":
    page_header("Overview", "Executive summary", count=len(df))

    top_pct = int(df["is_top_product"].mean()*100) if "is_top_product" in df.columns else 20
    avg_score = df["score"].mean() if "score" in df.columns else 0
    med_price = df["price"].median()
    avg_rating = df["rating"].mean() if "rating" in df.columns else 0
    on_sale = int(df["discount_pct"].gt(0).mean()*100) if "discount_pct" in df.columns else 0

    kpi_row([
        kpi_card(f"{len(df):,}", "Total Products", accent=PALETTE["blue"]),
        kpi_card(f"{avg_score:.2f}", "Avg Score", delta="vs baseline", delta_dir="up", accent=PALETTE["green"]),
        kpi_card(f"${med_price:.0f}", "Median Price", accent=PALETTE["amber"]),
        kpi_card(f"{avg_rating:.1f}", "Avg Rating", accent=PALETTE["purple"]),
        kpi_card(f"{top_pct}%", "Top-K Share", accent=PALETTE["rose"]),
    ])

    section("Product Distribution")
    c1, c2 = st.columns([3, 2])
    with c1:
        cat_df = df["category"].value_counts().reset_index()
        cat_df.columns = ["category","count"]
        fig = px.bar(cat_df, x="count", y="category", orientation="h",
                     color="count", color_continuous_scale=[[0,"#eff6ff"],[1,"#3b82f6"]], text="count")
        fig.update_traces(texttemplate="%{text}", textposition="outside", marker_line_width=0)
        fig.update_layout(height=320, showlegend=False, coloraxis_showscale=False,
                          xaxis={"showgrid":False,"visible":False}, yaxis={"title":""})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        plat_df = df["platform"].value_counts().reset_index()
        plat_df.columns = ["platform","count"]
        fig2 = go.Figure(go.Pie(labels=plat_df["platform"], values=plat_df["count"], hole=0.55,
                                marker_colors=[PLATFORM_COLORS.get(p,"#64748b") for p in plat_df["platform"]],
                                textinfo="percent+label", textfont_size=11))
        fig2.update_layout(height=320, showlegend=False,
                           annotations=[{"text":f"<b>{len(df):,}</b><br>products","x":0.5,"y":0.5,
                                         "font_size":14,"showarrow":False,"font_color":"#0f172a"}])
        st.plotly_chart(fig2, use_container_width=True)

    section("Score Distribution by Platform")
    fig3 = px.histogram(df, x="score", nbins=40, color="platform", barmode="overlay",
                        opacity=0.72, color_discrete_map=PLATFORM_COLORS)
    fig3.update_layout(height=200, bargap=0.02,
                       legend={"orientation":"h","y":1.15,"x":0,"font_size":11},
                       xaxis={"title":"Score","gridcolor":"#f1f5f9"},
                       yaxis={"title":"Products","gridcolor":"#f1f5f9"})
    st.plotly_chart(fig3, use_container_width=True)

    section("Key Insights")
    c1,c2,c3 = st.columns(3)
    with c1:
        insight_box("Top Category",
                    f"<strong>{cat_df.iloc[0]['category']}</strong> leads with "
                    f"{cat_df.iloc[0]['count']:,} products.", accent=PALETTE["blue"])
    with c2:
        insight_box("Score Health",
                    f"Average score is <strong>{avg_score:.2f}</strong>. "
                    f"{top_pct}% qualify as top-K products.", accent=PALETTE["green"])
    with c3:
        insight_box("Pricing Signal",
                    f"Median price <strong>${med_price:.0f}</strong> with "
                    f"{on_sale}% currently discounted.", accent=PALETTE["amber"])
    

    low_stock = df["stock_quantity"].lt(5).sum() if "stock_quantity" in df.columns else 0
    high_disc = df["discount_pct"].gt(40).sum() if "discount_pct" in df.columns else 0
    if low_stock > 0:
        alert(f"<strong>{low_stock} products</strong> are critically low on stock (< 5 units).", "warning")
    if high_disc > 0:
        alert(f"<strong>{high_disc} products</strong> have discounts > 40% — review pricing strategy.", "info")

    footer()


# ════════════════════════════════════════════════════════════════════════
# PAGE 2 — TOP-K PRODUCTS
# ════════════════════════════════════════════════════════════════════════
elif page == "Top-K Products":
    page_header("Top-K Products", "Highest-scoring products by composite score")

    df_top = load_top_k()
    df_top = apply_filters(df_top) if not df_top.empty else df_top

    cc1, cc2, _ = st.columns([2,2,4])
    with cc1:
        k_sel = st.slider("Show top N", 10, min(100, max(10,len(df_top))), 20, step=5)
    with cc2:
        sort_by = st.selectbox("Sort by", ["score","rating","price","review_count"])
    df_show = df_top.sort_values(sort_by, ascending=False).head(k_sel).reset_index(drop=True)
    df_show.index += 1

    kpi_row([
        kpi_card(f"{len(df_top):,}", "Eligible Products", accent=PALETTE["blue"]),
        kpi_card(f"{df_show['score'].max():.3f}", "Top Score", accent=PALETTE["green"]),
        kpi_card(f"${df_show['price'].mean():.0f}", "Avg Price (top)", accent=PALETTE["amber"]),
        kpi_card(f"{df_show['rating'].mean():.1f}", "Avg Rating (top)", accent=PALETTE["purple"]),
        kpi_card(f"{int(df_show['discount_pct'].gt(0).mean()*100)}%", "On Sale", accent=PALETTE["rose"]),
    ])

    section("Price vs Rating")
    fig = px.scatter(df_show, x="price", y="rating",
                     size="review_count" if "review_count" in df_show.columns else None,
                     color="score", color_continuous_scale=[[0,"#eff6ff"],[0.5,"#3b82f6"],[1,"#1e3a8a"]],
                     hover_data=[c for c in ["title","platform","category"] if c in df_show.columns],
                     size_max=36)
    fig.add_hline(y=df_show["rating"].median(), line_dash="dot", line_color="#cbd5e1", line_width=1)
    fig.add_vline(x=df_show["price"].median(), line_dash="dot", line_color="#cbd5e1", line_width=1)
    fig.update_layout(height=340, xaxis={"title":"Price ($)","gridcolor":"#f1f5f9"},
                      yaxis={"title":"Rating","gridcolor":"#f1f5f9"},
                      coloraxis_colorbar={"title":"Score","len":0.7,"thickness":10})
    st.plotly_chart(fig, use_container_width=True)

    section("Score Ranking — Top 10")
    t10 = df_show.head(10)
    fb = px.bar(t10, x="score", y="title" if "title" in t10.columns else t10.index.astype(str),
                orientation="h", color="score", color_continuous_scale=[[0,"#eff6ff"],[1,"#1e3a8a"]], text="score")
    fb.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
    fb.update_layout(height=320, showlegend=False, coloraxis_showscale=False,
                     xaxis={"visible":False}, yaxis={"title":""})
    st.plotly_chart(fb, use_container_width=True)

    section("Ranked Product Table")
    dcols = [c for c in ["score_rank","title","category","price","price_promo",
                         "discount_pct","rating","review_count","availability",
                         "shop_name","platform","score"] if c in df_show.columns]
    styled = df_show[dcols].style.background_gradient(
        subset=["score"] if "score" in dcols else [], cmap="Blues"
    ).format({"price":"${:.2f}",
              "price_promo": lambda v: f"${v:.2f}" if pd.notna(v) else "—",
              "discount_pct": lambda v: f"{v:.0f}%" if pd.notna(v) and v>0 else "—",
              "rating":"{:.1f}","score":"{:.3f}"})
    st.dataframe(styled, use_container_width=True, height=400)
    st.download_button("Export Top-K CSV", df_show.to_csv(index=False).encode(),
                       "top_k_products.csv", "text/csv")
    footer()


# ════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLUSTERS
# ════════════════════════════════════════════════════════════════════════
elif page == "Clusters":
    page_header("Product Clusters", "Segmentation via KMeans / DBSCAN + PCA")

    df_pca = load_pca()
    df_clust = apply_filters(load_clusters())

    if "cluster_label" in df_clust.columns:
        sizes = df_clust["cluster_label"].value_counts()
        kpi_row([
            kpi_card(f"{df_clust['cluster_label'].nunique()}", "Clusters Found", accent=PALETTE["blue"]),
            kpi_card(f"{sizes.max():,}", f"Largest: {sizes.idxmax()[:14]}", accent=PALETTE["purple"]),
            kpi_card(f"{sizes.min():,}", "Smallest Cluster", accent=PALETTE["slate"]),
            kpi_card(f"{len(df_clust):,}", "Products Clustered", accent=PALETTE["teal"]),
            kpi_card(f"{df_clust.loc[df_clust['cluster_label']=='Anomalie (DBSCAN)'].shape[0]}",
                     "Anomalies", accent=PALETTE["rose"]),
        ])

    ca, cb = st.columns([3,1])
    with ca:
        section("PCA 2D — Product Space")
        fig = px.scatter(df_pca, x="pc1", y="pc2", color="cluster_label",
                         color_discrete_map=CLUSTER_COLORS,
                         hover_data=[c for c in ["title","score"] if c in df_pca.columns],
                         opacity=0.7, size_max=7)
        fig.update_traces(marker={"size":5,"line":{"width":0}})
        fig.update_layout(height=400, legend={"title":"Segment","orientation":"v","x":1.01,"font":{"size":11}},
                          xaxis={"title":"PC 1","gridcolor":"#f8fafc"},
                          yaxis={"title":"PC 2","gridcolor":"#f8fafc"})
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        section("Segment Sizes")
        if "cluster_label" in df_clust.columns:
            sz = df_clust["cluster_label"].value_counts().reset_index()
            sz.columns = ["cluster","count"]
            f2 = go.Figure(go.Bar(x=sz["count"], y=sz["cluster"], orientation="h",
                                  marker_color=[CLUSTER_COLORS.get(c,"#94a3b8") for c in sz["cluster"]],
                                  text=sz["count"], textposition="outside"))
            f2.update_layout(height=400, showlegend=False, xaxis={"visible":False},
                             yaxis={"title":"","tickfont":{"size":10}})
            st.plotly_chart(f2, use_container_width=True)

    section("Segment Profiles — Average Metrics")
    if "cluster_label" in df_clust.columns:
        pcols = [c for c in ["price","rating","review_count","discount_pct","score"] if c in df_clust.columns]
        profile = df_clust.groupby("cluster_label")[pcols].mean().round(2)
        st.dataframe(profile.style.background_gradient(
            cmap="Blues", subset=["score"] if "score" in profile.columns else []
        ).format("{:.2f}"), use_container_width=True)

    section("Segment Interpretation")
    cols = st.columns(3)
    guides = [
        ("Premium", PALETTE["purple"], "High price + high rating. Focus on brand storytelling and upselling."),
        ("Top rated", PALETTE["teal"], "Popular with customers. Prioritize inventory and marketing."),
        ("Discount / Promo", PALETTE["amber"], "Aggressive pricing. Evaluate margin impact."),
        ("Mainstream", PALETTE["blue"], "Average on all metrics. Differentiate through content or bundling."),
        ("Niche / peu connu", PALETTE["slate"], "Low visibility. Test boosted listings or targeted ads."),
        ("Anomalie (DBSCAN)", PALETTE["rose"], "Outliers. Review for data quality or unique opportunities."),
    ]
    for i,(n,c,t) in enumerate(guides):
        with cols[i%3]:
            insight_box(n, t, accent=c)
    footer()


# ════════════════════════════════════════════════════════════════════════
# PAGE 4 — PRICING ANALYSIS
# ════════════════════════════════════════════════════════════════════════
elif page == "Pricing Analysis":
    page_header("Pricing Analysis", "Price bands, discounts, and premium detection")

    on_sale = df["discount_pct"].gt(0).sum() if "discount_pct" in df.columns else 0
    avg_disc = df.loc[df["discount_pct"]>0,"discount_pct"].mean() if "discount_pct" in df.columns else 0
    max_disc = df["discount_pct"].max() if "discount_pct" in df.columns else 0

    kpi_row([
        kpi_card(f"${df['price'].min():.0f} – ${df['price'].max():.0f}", "Price Range", accent=PALETTE["blue"]),
        kpi_card(f"${df['price'].mean():.0f}", "Mean Price", accent=PALETTE["teal"]),
        kpi_card(f"{on_sale:,}", "Products on Sale", accent=PALETTE["amber"]),
        kpi_card(f"{avg_disc:.1f}%", "Avg Discount", accent=PALETTE["rose"]),
        kpi_card(f"{max_disc:.0f}%", "Max Discount", accent=PALETTE["purple"]),
    ])

    ca, cb = st.columns(2)
    with ca:
        section("Price Distribution by Category")
        fig = px.box(df, x="category", y="price", color="category",
                     color_discrete_sequence=list(PALETTE.values()), points="outliers")
        fig.update_layout(height=340, showlegend=False,
                          xaxis={"title":"","tickangle":-30}, yaxis={"title":"Price ($)","gridcolor":"#f1f5f9"})
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        section("Discount Distribution")
        if "discount_pct" in df.columns:
            disc = df[df["discount_pct"]>0]["discount_pct"]
            f2 = px.histogram(disc, nbins=20, opacity=0.85, color_discrete_sequence=[PALETTE["amber"]])
            f2.update_traces(marker_line_width=0)
            f2.update_layout(height=340, bargap=0.04, xaxis={"title":"Discount %"}, yaxis={"title":"# Products"})
            st.plotly_chart(f2, use_container_width=True)

    section("Price vs Score — Premium Detection Quadrant")
    f3 = px.scatter(df, x="price", y="score", color="category",
                    size="review_count" if "review_count" in df.columns else None,
                    opacity=0.6, size_max=20, color_discrete_sequence=list(PALETTE.values()),
                    hover_data=["title"] if "title" in df.columns else [])
    px_med = df["price"].median()
    sc_med = df["score"].median() if "score" in df.columns else 0
    f3.add_hline(y=sc_med, line_dash="dash", line_color="#94a3b8", line_width=1,
                 annotation_text="Median score", annotation_position="bottom right", annotation_font_size=10)
    f3.add_vline(x=px_med, line_dash="dash", line_color="#94a3b8", line_width=1,
                 annotation_text="Median price", annotation_position="top right", annotation_font_size=10)
    f3.update_layout(height=320, xaxis={"title":"Price ($)","gridcolor":"#f8fafc"},
                     yaxis={"title":"Score","gridcolor":"#f1f5f9"},
                     legend={"orientation":"h","y":1.12,"font_size":10})
    st.plotly_chart(f3, use_container_width=True)

    q1,q2,q3,q4 = st.columns(4)
    for col,(n,d,c) in zip([q1,q2,q3,q4],[
        ("Premium Value","High price · high score",PALETTE["green"]),
        ("Overpriced","High price · low score",PALETTE["rose"]),
        ("Hidden Gems","Low price · high score",PALETTE["blue"]),
        ("Weak Products","Low price · low score",PALETTE["slate"]),
    ]):
        with col:
            st.markdown(f"""<div style="border-left:4px solid {c}; padding:0.5rem 0.75rem;
            background:#f8fafc; border-radius:0 8px 8px 0; margin-bottom:0.5rem;">
            <div style="font-size:0.85rem; font-weight:600; color:#1e293b;">{n}</div>
            <div style="font-size:0.75rem; color:#64748b;">{d}</div></div>""", unsafe_allow_html=True)
    footer()


# ════════════════════════════════════════════════════════════════════════
# PAGE 5 — ASSOCIATION RULES
# ════════════════════════════════════════════════════════════════════════
elif page == "Association Rules":
    page_header("Association Rules", "Co-purchase patterns — lift, confidence, support")

    df_rules = load_rules()
    if df_rules.empty:
        alert("No rules found yet. Run the ML pipeline first.", "warning")
    else:
        c1,_ = st.columns([2,4])
        with c1:
            min_lift = st.slider("Min lift", 1.0, float(df_rules["lift"].max()), 1.0, step=0.1)
        df_r = df_rules[df_rules["lift"]>=min_lift].head(30)

        kpi_row([
            kpi_card(f"{len(df_rules):,}", "Rules Found", accent=PALETTE["blue"]),
            kpi_card(f"{df_rules['lift'].max():.2f}", "Max Lift", accent=PALETTE["rose"]),
            kpi_card(f"{df_rules['confidence'].max():.2f}", "Max Confidence", accent=PALETTE["green"]),
            kpi_card(f"{df_rules['support'].max():.3f}", "Max Support", accent=PALETTE["amber"]),
            kpi_card(f"{len(df_r)}", "Shown (filtered)", accent=PALETTE["slate"]),
        ])

        section("Support vs Confidence — bubble size = lift")
        fig = px.scatter(df_r, x="support", y="confidence", size="lift", color="lift",
                         color_continuous_scale=[[0,"#fef3c7"],[0.5,"#f59e0b"],[1,"#92400e"]],
                         hover_data=["antecedents","consequents","lift"], size_max=40)
        fig.update_layout(height=360, xaxis={"title":"Support","gridcolor":"#f8fafc"},
                          yaxis={"title":"Confidence","gridcolor":"#f1f5f9"},
                          coloraxis_colorbar={"title":"Lift","thickness":10,"len":0.7})
        st.plotly_chart(fig, use_container_width=True)

        section("Top Rules by Lift")
        st.dataframe(
            df_r[["antecedents","consequents","support","confidence","lift"]]
                .style.background_gradient(subset=["lift"], cmap="YlOrRd")
                .format({"support":"{:.3f}","confidence":"{:.3f}","lift":"{:.2f}"}),
            use_container_width=True, height=340)

        best = df_r.sort_values("lift", ascending=False).iloc[0]
        insight_box("Strongest Rule",
                    f"<strong>{best['antecedents']}</strong> → <strong>{best['consequents']}</strong> "
                    f"with lift <strong>{best['lift']:.2f}</strong> and confidence "
                    f"<strong>{best['confidence']:.0%}</strong>. High-confidence cross-sell opportunity.",
                    accent=PALETTE["amber"])
    footer()


# ════════════════════════════════════════════════════════════════════════
# PAGE 6 — SHOPS & GEOGRAPHY
# ════════════════════════════════════════════════════════════════════════
elif page == "Shops & Geography":
    page_header("Shops & Geography", "Competitive landscape by shop and country")

    df_shops = load_shops()
    kpi_row([
        kpi_card(f"{len(df_shops):,}", "Total Shops", accent=PALETTE["blue"]),
        kpi_card(f"{df_shops['avg_score'].max():.3f}", "Best Shop Score", accent=PALETTE["green"]),
        kpi_card(f"{df_shops['avg_score'].mean():.3f}", "Avg Shop Score", accent=PALETTE["teal"]),
        kpi_card(f"{df_shops['product_count'].max():,}" if "product_count" in df_shops.columns else "—",
                 "Largest Shop", accent=PALETTE["amber"]),
        kpi_card(f"{df['shop_country'].nunique() if 'shop_country' in df.columns else '—'}",
                 "Countries", accent=PALETTE["purple"]),
    ])

    ca,cb = st.columns(2)
    with ca:
        section("Top 10 Shops by Avg Score")
        fig = px.bar(df_shops.head(10), x="avg_score", y="shop_name", orientation="h",
                     color="avg_score", color_continuous_scale=[[0,"#e0f2fe"],[1,"#0369a1"]], text="avg_score")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
        fig.update_layout(height=340, showlegend=False, coloraxis_showscale=False,
                          xaxis={"visible":False}, yaxis={"title":"","tickfont":{"size":11}})
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        section("Products by Country")
        if "shop_country" in df.columns:
            cdf = df["shop_country"].value_counts().reset_index()
            cdf.columns = ["country","count"]
            f2 = go.Figure(go.Pie(labels=cdf["country"], values=cdf["count"], hole=0.5,
                                  marker_colors=px.colors.qualitative.Set2,
                                  textinfo="percent+label", textfont_size=10))
            f2.update_layout(height=340, showlegend=False)
            st.plotly_chart(f2, use_container_width=True)

    section("Avg Score by Country")
    if "shop_country" in df.columns:
        cs = df.groupby("shop_country")["score"].mean().sort_values(ascending=False).reset_index()
        f3 = px.bar(cs, x="shop_country", y="score", color="score",
                    color_continuous_scale=[[0,"#eff6ff"],[1,"#1e40af"]], text="score")
        f3.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
        f3.update_layout(height=260, coloraxis_showscale=False,
                         xaxis={"title":""}, yaxis={"title":"Avg Score","gridcolor":"#f1f5f9"})
        st.plotly_chart(f3, use_container_width=True)

    section("Shop Detail Table")
    scols = [c for c in ["shop_name","shop_country","avg_score","product_count","avg_rating"]
             if c in df_shops.columns]
    st.dataframe(df_shops[scols].style.background_gradient(subset=["avg_score"], cmap="Blues")
                 .format({"avg_score":"{:.3f}","avg_rating":"{:.2f}"}),
                 use_container_width=True, height=320)
    footer()


# ════════════════════════════════════════════════════════════════════════
# PAGE 7 — LLM INSIGHTS
# ════════════════════════════════════════════════════════════════════════
elif page == "LLM Insights":
    from LLM.mcp.mcp_client import get_mcp_client

    page_header("LLM Insights", "AI-generated analysis and strategic recommendations")

    REPORT_KIND_MAP = {
        "Top-K product summary": "topk_summary",
        "Market trend analysis": "market_trend",
        "Competitive pricing report": "pricing",
        "Cross-sell opportunities": "cross_sell",
        "Segment strategy brief": "segment",
    }

    cl, cr = st.columns([1, 2])
    with cl:
        section("Generate Report")
        report_type = st.selectbox("Report type", list(REPORT_KIND_MAP.keys()))
        tone = st.selectbox(
            "Tone",
            ["Executive summary", "Detailed analysis", "Bullet points"],
        )
        n_products = st.slider("Products to analyse", 5, 50, 10, 5)
        with st.expander("LLM & MCP configuration", expanded=False):
            st.caption(
                "Reports run through the **MCP client** (audit + rate limits) and call the "
                "**LLM server** tools. Set `LLM_PROVIDER` and API keys in `.env` or the environment "
                "(see `LLM/env.example`). Use `mock` for demos without API keys."
            )
        if st.button("Generate with LLM", use_container_width=True):
            st.session_state["llm_trigger"] = True
            st.session_state["llm_report_type"] = report_type
            st.session_state["llm_tone"] = tone
            st.session_state["llm_n"] = n_products

    with cr:
        section("AI Recommendations")
        if st.session_state.get("llm_trigger"):
            report_type = st.session_state.get("llm_report_type", report_type)
            tone = st.session_state.get("llm_tone", tone)
            n_products = int(st.session_state.get("llm_n", n_products))
            mcp = get_mcp_client()
            with st.spinner("Generating insights via MCP + LLM…"):
                try:
                    resp = mcp.call(
                        "generate_bi_report",
                        {
                            "report_kind": REPORT_KIND_MAP.get(
                                report_type, "market_trend"
                            ),
                            "tone": tone,
                            "n_products": n_products,
                            "category": None
                            if sel_cat == "All"
                            else sel_cat,
                            "platform": None
                            if sel_plat == "All"
                            else sel_plat,
                            "price_min": float(price_range[0]),
                            "price_max": float(price_range[1]),
                        },
                    )
                    if not resp.get("allowed", False):
                        st.error(resp.get("error") or resp.get("reason") or "Request denied.")
                    elif resp.get("error"):
                        st.error(f"LLM tool error: {resp['error']}")
                    else:
                        st.session_state["llm_output"] = resp.get("result") or ""
                        st.session_state["llm_latency_ms"] = resp.get("latency_ms")
                    st.session_state["llm_trigger"] = False
                except Exception as e:
                    st.error(f"LLM / MCP call failed: {e}")
                    st.session_state["llm_trigger"] = False

        if st.session_state.get("llm_output"):
            lat = st.session_state.get("llm_latency_ms")
            if lat is not None:
                st.caption(f"MCP round-trip: **{lat} ms** (audited)")
            st.markdown(st.session_state["llm_output"])
            rt = st.session_state.get("llm_report_type", "report")
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in rt.replace(" ", "_"))
            st.download_button(
                "Export Report",
                st.session_state["llm_output"].encode(),
                f"llm_{safe}.md",
                "text/markdown",
            )
        else:
            st.markdown(
                """<div style="text-align:center; padding:3rem 2rem; color:#94a3b8;">
                <div style="font-size:2rem; margin-bottom:0.75rem;">AI</div>
                Configure report options and click <strong>Generate</strong> to get AI-powered insights.
            </div>""",
                unsafe_allow_html=True,
            )

    section("Automated Insight Cards")
    c1,c2,c3 = st.columns(3)
    with c1:
        insight_box("Top Scoring Category",
                    f"<strong>{df['category'].value_counts().index[0]}</strong> dominates in "
                    f"product count. Focus ML scoring on this segment for highest ROI.",
                    accent=PALETTE["blue"])
    with c2:
        insight_box("Pricing Opportunity",
                    f"Products priced ${df['price'].quantile(0.25):.0f}–${df['price'].median():.0f} "
                    f"cluster in 'hidden gems' quadrant — premium value at accessible price.",
                    accent=PALETTE["amber"])
    with c3:
        insight_box("Engagement Signal",
                    f"Products with rating >= {df['rating'].quantile(0.75):.1f} represent the top "
                    f"25th percentile. Use as a quality filter for the Top-K shortlist.",
                    accent=PALETTE["green"])
    footer()


# ════════════════════════════════════════════════════════════════════════
# PAGE 8 — ASSISTANT BI
# ════════════════════════════════════════════════════════════════════════
elif page == "Assistant BI":
    try:
        from LLM.chatbot_page import render_chatbot
        render_chatbot()
    except Exception as e:
        page_header("Assistant BI", "Interactive chatbot")
        alert(f"Chatbot module could not load: {e}", "warning")
        footer()