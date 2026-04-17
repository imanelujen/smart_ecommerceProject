"""
DashboardBI/data_loader.py
----------------------
Loads all TopKselection output files into DataFrames.
Falls back to synthetic demo data if outputs are not available yet.
"""

import pandas as pd
import numpy as np
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent.parent / "TopKselection" / "output"


def _synth(n=300) -> pd.DataFrame:
    """Generate realistic demo data when real outputs don't exist yet."""
    np.random.seed(42)
    cats     = ["Electronics","Sport","Home","Fashion","Books","Beauty","Toys"]
    shops    = ["TechWorld","FitShop","BrightHome","StyleHub","BookNest"]
    countries= ["USA","UK","FR","DE","CA"]
    platforms= ["shopify","woocommerce"]
    colors   = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc948","#b07aa1"]

    prices      = np.random.lognormal(3.5, 0.8, n).clip(2, 500)
    ratings     = np.random.uniform(2.5, 5.0, n)
    reviews     = np.random.randint(0, 3000, n)
    discounts   = np.random.choice([0,0,0,10,20,30,50], n).astype(float)
    stocks      = np.random.randint(0, 200, n).astype(float)
    category    = np.random.choice(cats, n)
    platform    = np.random.choice(platforms, n)
    shop        = np.random.choice(shops, n)
    country     = np.random.choice(countries, n)

    scores = (
        0.35 * (ratings - 2.5) / 2.5 +
        0.25 * np.log1p(reviews) / np.log1p(3000) +
        0.15 * (1 - (prices - 2) / 498) +
        0.10 * discounts / 50 +
        0.10 * np.ones(n) +
        0.05 * stocks / 200
    ).clip(0, 1)

    df = pd.DataFrame({
        "product_id":    range(1, n+1),
        "title":         [f"Product {i}" for i in range(1, n+1)],
        "category":      category,
        "platform":      platform,
        "price":         prices.round(2),
        "price_promo":   np.where(discounts > 0, (prices * (1 - discounts/100)).round(2), np.nan),
        "price_old":     np.where(discounts > 0, prices.round(2), np.nan),
        "discount_pct":  discounts,
        "rating":        ratings.round(1),
        "review_count":  reviews,
        "availability":  np.random.choice([True, False], n, p=[0.85, 0.15]),
        "stock_quantity":stocks,
        "shop_name":     shop,
        "shop_country":  country,
        "score":         scores.round(4),
        "is_top_product":np.where(scores >= np.quantile(scores, 0.80), 1, 0),
        "cluster_label": np.random.choice(["Premium","Top rated","Discount / Promo","Niche / peu connu","Mainstream"], n),
    })
    df["score_rank"] = df["score"].rank(ascending=False, method="min").astype(int)
    return df


def load_scored() -> pd.DataFrame:
    p = OUTPUT_DIR / "products_scored.csv"
    return pd.read_csv(p) if p.exists() else _synth()


def load_top_k() -> pd.DataFrame:
    p = OUTPUT_DIR / "top_k_products.csv"
    if p.exists():
        return pd.read_csv(p)
    df = _synth()
    return df.nlargest(50, "score").reset_index(drop=True)


def load_clusters() -> pd.DataFrame:
    p = OUTPUT_DIR / "products_clustered.csv"
    return pd.read_csv(p) if p.exists() else _synth()


def load_pca() -> pd.DataFrame:
    p = OUTPUT_DIR / "pca_2d.csv"
    if p.exists():
        return pd.read_csv(p)
    df = _synth()
    np.random.seed(1)
    return pd.DataFrame({
        "pc1": np.random.randn(len(df)),
        "pc2": np.random.randn(len(df)),
        "cluster": df["cluster_label"],
        "cluster_label": df["cluster_label"],
        "title": df["title"],
        "price": df["price"],
        "score": df["score"],
    })


def load_rules() -> pd.DataFrame:
    p = OUTPUT_DIR / "association_rules.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame({
        "antecedents": ["Electronics","Sport","Books","Home","Fashion"],
        "consequents": ["Home","Beauty","Fashion","Electronics","Sport"],
        "support":     [0.18, 0.14, 0.11, 0.09, 0.08],
        "confidence":  [0.72, 0.65, 0.58, 0.50, 0.44],
        "lift":        [3.8,  3.1,  2.7,  2.2,  1.9],
    })


def load_shops() -> pd.DataFrame:
    p = OUTPUT_DIR / "shop_leaderboard.csv"
    if p.exists():
        return pd.read_csv(p)
    df = _synth()
    return (df.groupby(["shop_name","shop_country"])
              .agg(avg_score=("score","mean"),
                   product_count=("title","count"),
                   avg_rating=("rating","mean"))
              .sort_values("avg_score", ascending=False)
              .reset_index())