"""
TopKselection/scoring.py
------------------
Step 2 — Composite scoring and Top-K selection.

Score formula (weighted sum, all inputs normalised 0→1):
  score = w_rating * rating_norm
        + w_reviews * log_reviews_norm
        + w_price_competitiveness * (1 - price_norm)   ← lower price = better
        + w_discount * discount_norm
        + w_availability * availability
        + w_stock * stock_norm

Weights are configurable; defaults match the project spec priorities.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "rating":       0.35,
    "reviews":      0.25,
    "price_comp":   0.15,   # inversed: cheap = good
    "discount":     0.10,
    "availability": 0.10,
    "stock":        0.05,
}


def _minmax(series: pd.Series) -> pd.Series:
    rng = series.max() - series.min()
    return (series - series.min()) / rng if rng > 0 else series * 0


def compute_scores(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """
    Add a 'score' column (0.0–1.0) to df.
    Also adds 'score_rank' and 'is_top_product' (label for supervised learning).
    """
    w = weights or DEFAULT_WEIGHTS

    df = df.copy()

    rating_norm      = _minmax(df["rating"].fillna(0))
    reviews_norm     = _minmax(np.log1p(df["review_count"].fillna(0)))
    price_norm       = _minmax(df["price"].clip(lower=0))
    discount_norm    = _minmax(df["discount_pct"].fillna(0))
    availability_col = df["availability"].map({True:1, False:0, 1:1, 0:0}).fillna(1)
    stock_norm       = _minmax(df["stock_quantity"].fillna(df["stock_quantity"].median()))

    df["score"] = (
        w["rating"]       * rating_norm
      + w["reviews"]      * reviews_norm
      + w["price_comp"]   * (1 - price_norm)
      + w["discount"]     * discount_norm
      + w["availability"] * availability_col
      + w["stock"]        * stock_norm
    ).round(4)

    df["score_rank"]     = df["score"].rank(ascending=False, method="min").astype(int)

    # Top-K label: top 20% by score = success label for supervised models
    threshold = df["score"].quantile(0.80)
    df["is_top_product"] = (df["score"] >= threshold).astype(int)

    logger.info(
        f"Scoring done. Top products: {df['is_top_product'].sum()} / {len(df)} "
        f"(threshold score={threshold:.3f})"
    )
    return df


def select_top_k(df: pd.DataFrame, k: int = 50) -> pd.DataFrame:
    """
    Return the Top-K products sorted by score descending.
    Also includes per-shop top products for geographic analysis.
    """
    top_k = df.nlargest(k, "score").reset_index(drop=True)
    top_k.index = top_k.index + 1   # rank starts at 1
    top_k.index.name = "rank"
    return top_k


def shop_leaderboard(df: pd.DataFrame, top_n_shops: int = 10) -> pd.DataFrame:
    """
    Rank shops by their average product score.
    Useful for geographic / competitive analysis.
    """
    return (
        df.groupby(["shop_name", "shop_country"])
        .agg(
            avg_score    = ("score",        "mean"),
            product_count= ("title",        "count"),
            avg_rating   = ("rating",       "mean"),
            top_product  = ("title",        lambda x: x.iloc[df.loc[x.index, "score"].argmax()]),
        )
        .sort_values("avg_score", ascending=False)
        .head(top_n_shops)
        .reset_index()
    )


def run(df_clean: pd.DataFrame, k: int = 50, output_dir: str = "TopKselection/output") -> dict:
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df_scored = compute_scores(df_clean)
    top_k     = select_top_k(df_scored, k=k)
    shops     = shop_leaderboard(df_scored)

    df_scored.to_csv(f"{output_dir}/products_scored.csv", index=False)
    top_k.to_csv(    f"{output_dir}/top_k_products.csv")
    shops.to_csv(    f"{output_dir}/shop_leaderboard.csv", index=False)

    logger.info(f"Top-{k} saved to {output_dir}/top_k_products.csv")
    return {"df_scored": df_scored, "top_k": top_k, "shop_leaderboard": shops}