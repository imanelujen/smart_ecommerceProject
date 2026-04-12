"""
module4/app.py
--------------
Smart eCommerce — Streamlit BI Dashboard

Pages:
  1. Overview        — KPI cards + platform & category distributions
  2. Top-K Products  — ranked table + price/rating scatter
  3. Clusters        — PCA 2D scatter + cluster profile table
  4. Pricing         — price band analysis + discount distribution
  5. Association Rules — top rules ranked by lift
  6. Shops & Geo     — shop leaderboard + country breakdown

Run:
    streamlit run module4/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from module4.data_loader import (
    load_scored, load_top_k, load_clusters, load_pca, load_rules, load_shops
)

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart eCommerce Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .kpi-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #4e79a7;
        margin-bottom: 0.5rem;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .kpi-label { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }
    .kpi-delta { font-size: 0.8rem; color: #2ecc71; font-weight: 600; }
    .section-header {
        font-size: 1.15rem; font-weight: 600;
        color: #1a1a2e; margin: 1.5rem 0 0.8rem;
        border-bottom: 2px solid #e9ecef; padding-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

CLUSTER_COLORS = {
    "Premium":            "#7F77DD",
    "Top rated":          "#1D9E75",
    "Discount / Promo":   "#EF9F27",
    "Niche / peu connu":  "#888780",
    "Mainstream":         "#378ADD",
    "Anomalie (DBSCAN)":  "#E24B4A",
}

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛒 Smart eCommerce")
    st.caption("Business Intelligence Dashboard")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Overview", "Top-K Products", "Clusters", "Pricing Analysis",
         "Association Rules", "Shops & Geography"],
        label_visibility="collapsed",
    )
    st.divider()

    # Global filters
    df_all = load_scored()
    cats       = ["All"] + sorted(df_all["category"].dropna().unique().tolist())
    platforms  = ["All"] + sorted(df_all["platform"].dropna().unique().tolist())

    sel_cat  = st.selectbox("Category filter", cats)
    sel_plat = st.selectbox("Platform filter", platforms)
    price_range = st.slider(
        "Price range ($)",
        float(df_all["price"].min()),
        float(df_all["price"].max()),
        (float(df_all["price"].min()), float(df_all["price"].max())),
    )
    st.divider()
    st.caption(f"Dataset: **{len(df_all):,}** products")

# ── Apply filters ──────────────────────────────────────────────────────
def apply_filters(df):
    mask = (df["price"] >= price_range[0]) & (df["price"] <= price_range[1])
    if sel_cat  != "All": mask &= df["category"] == sel_cat
    if sel_plat != "All": mask &= df["platform"] == sel_plat
    return df[mask]

df = apply_filters(df_all)

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Overview")
    st.caption(f"Showing {len(df):,} products after filters")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)

    def kpi(col, value, label, delta=None, color="#4e79a7"):
        with col:
            delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color:{color}">
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
                {delta_html}
            </div>""", unsafe_allow_html=True)

    kpi(col1, f"{len(df):,}",   "Total products",       color="#4e79a7")
    kpi(col2, f"{df['score'].mean():.2f}", "Avg score",  color="#1D9E75")
    kpi(col3, f"${df['price'].median():.0f}", "Median price", color="#EF9F27")
    kpi(col4, f"{df['rating'].mean():.1f} ★", "Avg rating", color="#7F77DD")
    top_pct = int(df["is_top_product"].mean() * 100) if "is_top_product" in df.columns else 20
    kpi(col5, f"{top_pct}%", "Top products",            color="#E24B4A")

    st.markdown('<div class="section-header">Products by category</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig = px.bar(cat_counts, x="count", y="category", orientation="h",
                     color="count", color_continuous_scale="Blues",
                     template="plotly_white")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0,r=0,t=10,b=0), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        plat_counts = df["platform"].value_counts().reset_index()
        plat_counts.columns = ["platform","count"]
        fig2 = px.pie(plat_counts, names="platform", values="count",
                      color_discrete_sequence=["#4e79a7","#f28e2b","#59a14f"],
                      template="plotly_white")
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        fig2.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=320, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Score distribution</div>', unsafe_allow_html=True)
    fig3 = px.histogram(df, x="score", nbins=40, color="platform",
                        barmode="overlay", opacity=0.75,
                        color_discrete_map={"shopify":"#4e79a7","woocommerce":"#f28e2b"},
                        template="plotly_white")
    fig3.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=220,
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — Top-K Products
# ══════════════════════════════════════════════════════════════════════
elif page == "Top-K Products":
    st.title("Top-K Products")

    df_top = load_top_k()
    df_top = apply_filters(df_top) if not df_top.empty else df_top

    k_sel = st.slider("Show top N", 10, min(100, len(df_top)), 20, step=5)
    df_show = df_top.head(k_sel)

    # Scatter: price vs rating, sized by review_count
    st.markdown('<div class="section-header">Price vs rating (size = review count)</div>',
                unsafe_allow_html=True)
    fig = px.scatter(
        df_show,
        x="price", y="rating",
        size="review_count" if "review_count" in df_show.columns else None,
        color="score",
        color_continuous_scale="viridis",
        hover_data=["title","platform","category","discount_pct"],
        template="plotly_white",
        size_max=40,
    )
    fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=380,
                      coloraxis_colorbar=dict(title="Score"))
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown('<div class="section-header">Ranked product table</div>',
                unsafe_allow_html=True)
    display_cols = [c for c in ["score_rank","title","category","price","price_promo",
                                 "discount_pct","rating","review_count","availability",
                                 "shop_name","platform","score"] if c in df_show.columns]

    styled = df_show[display_cols].style.background_gradient(
        subset=["score"] if "score" in display_cols else [],
        cmap="YlGn"
    ).format({
        "price":       "${:.2f}",
        "price_promo": lambda v: f"${v:.2f}" if pd.notna(v) else "—",
        "discount_pct":lambda v: f"{v:.0f}%" if pd.notna(v) and v > 0 else "—",
        "rating":      "{:.1f}",
        "score":       "{:.3f}",
    })
    st.dataframe(styled, use_container_width=True, height=420)

    # Download button
    csv_bytes = df_show.to_csv(index=False).encode()
    st.download_button("Download Top-K CSV", csv_bytes, "top_k_products.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — Clusters
# ══════════════════════════════════════════════════════════════════════
elif page == "Clusters":
    st.title("Product Clusters")

    df_pca   = load_pca()
    df_clust = apply_filters(load_clusters())

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown('<div class="section-header">PCA 2D — product space</div>',
                    unsafe_allow_html=True)
        fig = px.scatter(
            df_pca, x="pc1", y="pc2",
            color="cluster_label",
            color_discrete_map=CLUSTER_COLORS,
            hover_data=["title","score"] if "score" in df_pca.columns else ["title"],
            opacity=0.75,
            template="plotly_white",
            size_max=8,
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=420,
                          legend=dict(title="Cluster", orientation="h",
                                      yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Cluster sizes</div>',
                    unsafe_allow_html=True)
        if "cluster_label" in df_clust.columns:
            sizes = df_clust["cluster_label"].value_counts().reset_index()
            sizes.columns = ["cluster","count"]
            fig2 = px.bar(sizes, x="count", y="cluster", orientation="h",
                          color="cluster", color_discrete_map=CLUSTER_COLORS,
                          template="plotly_white")
            fig2.update_layout(showlegend=False, margin=dict(l=0,r=0,t=10,b=0),
                               height=280)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Cluster profiles — avg metrics</div>',
                unsafe_allow_html=True)
    if "cluster_label" in df_clust.columns:
        profile_cols = [c for c in ["price","rating","review_count","discount_pct","score"]
                        if c in df_clust.columns]
        profile = df_clust.groupby("cluster_label")[profile_cols].mean().round(2)
        st.dataframe(profile.style.background_gradient(cmap="YlGn"),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — Pricing Analysis
# ══════════════════════════════════════════════════════════════════════
elif page == "Pricing Analysis":
    st.title("Pricing Analysis")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Price distribution by category</div>',
                    unsafe_allow_html=True)
        fig = px.box(df, x="category", y="price",
                     color="category", template="plotly_white",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=10,b=0),
                          height=350, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Discount distribution</div>',
                    unsafe_allow_html=True)
        disc = df[df["discount_pct"] > 0]["discount_pct"]
        fig2 = px.histogram(disc, nbins=20, color_discrete_sequence=["#EF9F27"],
                            template="plotly_white")
        fig2.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=350,
                           xaxis_title="Discount %", yaxis_title="Products")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Price vs score — premium detection</div>',
                unsafe_allow_html=True)
    fig3 = px.scatter(df, x="price", y="score",
                      color="category", size="review_count" if "review_count" in df else None,
                      opacity=0.6, template="plotly_white",
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      hover_data=["title"] if "title" in df else [])
    fig3.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=320)
    st.plotly_chart(fig3, use_container_width=True)

    # On-sale vs full price KPIs
    on_sale = df["discount_pct"].gt(0).sum()
    avg_discount = df[df["discount_pct"] > 0]["discount_pct"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Products on sale",  f"{on_sale:,}")
    c2.metric("Avg discount",      f"{avg_discount:.1f}%" if not np.isnan(avg_discount) else "—")
    c3.metric("Max discount",      f"{df['discount_pct'].max():.0f}%")


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — Association Rules
# ══════════════════════════════════════════════════════════════════════
elif page == "Association Rules":
    st.title("Association Rules")
    st.caption("Patterns discovered with Apriori algorithm")

    df_rules = load_rules()

    if df_rules.empty:
        st.info("No rules found yet. Run the ML pipeline first.")
    else:
        min_lift = st.slider("Min lift", 1.0,
                             float(df_rules["lift"].max()), 1.0, step=0.1)
        df_r = df_rules[df_rules["lift"] >= min_lift].head(30)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rules found",      len(df_rules))
        c2.metric("Max lift",         f"{df_rules['lift'].max():.2f}")
        c3.metric("Max confidence",   f"{df_rules['confidence'].max():.2f}")

        st.markdown('<div class="section-header">Top rules by lift</div>',
                    unsafe_allow_html=True)
        fig = px.scatter(
            df_r, x="support", y="confidence",
            size="lift", color="lift",
            color_continuous_scale="Reds",
            hover_data=["antecedents","consequents","lift"],
            template="plotly_white",
            size_max=40,
        )
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=360)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Rules table</div>',
                    unsafe_allow_html=True)
        st.dataframe(
            df_r[["antecedents","consequents","support","confidence","lift"]]
              .style.background_gradient(subset=["lift"], cmap="YlOrRd")
              .format({"support":"{:.3f}","confidence":"{:.3f}","lift":"{:.2f}"}),
            use_container_width=True, height=360,
        )


# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — Shops & Geography
# ══════════════════════════════════════════════════════════════════════
elif page == "Shops & Geography":
    st.title("Shops & Geography")

    df_shops = load_shops()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Shop leaderboard</div>',
                    unsafe_allow_html=True)
        fig = px.bar(
            df_shops.head(10), x="avg_score", y="shop_name",
            orientation="h", color="avg_score",
            color_continuous_scale="Teal",
            text="avg_score", template="plotly_white",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0,r=0,t=10,b=0), height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Products by country</div>',
                    unsafe_allow_html=True)
        if "shop_country" in df.columns:
            country_counts = df["shop_country"].value_counts().reset_index()
            country_counts.columns = ["country","count"]
            fig2 = px.pie(country_counts, names="country", values="count",
                          color_discrete_sequence=px.colors.qualitative.Set2,
                          template="plotly_white")
            fig2.update_traces(textposition="inside", textinfo="percent+label")
            fig2.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=350,
                               showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Average score by country</div>',
                unsafe_allow_html=True)
    if "shop_country" in df.columns:
        country_score = (df.groupby("shop_country")["score"].mean()
                           .sort_values(ascending=False).reset_index())
        fig3 = px.bar(country_score, x="shop_country", y="score",
                      color="score", color_continuous_scale="Blues",
                      template="plotly_white")
        fig3.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280,
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Shop detail table</div>',
                unsafe_allow_html=True)
    show_cols = [c for c in ["shop_name","shop_country","avg_score",
                              "product_count","avg_rating"] if c in df_shops.columns]
    st.dataframe(
        df_shops[show_cols].style
          .background_gradient(subset=["avg_score"], cmap="YlGn")
          .format({"avg_score":"{:.3f}", "avg_rating":"{:.2f}"}),
        use_container_width=True, height=340,
    )