"""
module3/components/scoring_component.py
----------------------------------------
Kubeflow Pipeline component — Composite Scoring + Top-K Selection.
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Output


@dsl.component(
    base_image="smart-ecommerce/ml:latest",
    packages_to_install=[],
)
def scoring_component(
    input_clean:    Input[Dataset],
    k:              int,
    output_scored:  Output[Dataset],
    output_topk:    Output[Dataset],
    output_shops:   Output[Dataset],
):
    """
    Computes composite scores, selects Top-K, builds shop leaderboard.
    """
    import sys
    sys.path.insert(0, "/app")
    import pandas as pd
    from TopKselection.scoring import compute_scores, select_top_k, shop_leaderboard

    df       = pd.read_csv(input_clean.path)
    df       = compute_scores(df)
    top_k    = select_top_k(df, k=k)
    shops    = shop_leaderboard(df)

    df.to_csv(output_scored.path, index=False)
    top_k.to_csv(output_topk.path)
    shops.to_csv(output_shops.path, index=False)

    print(f"[scoring] Top-{k} selected. Score range: {df['score'].min():.3f}–{df['score'].max():.3f}")