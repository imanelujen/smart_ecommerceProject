"""
TopKselection/pipeline.py
-------------------
Main entry point for Module 2.
Chains all steps: preprocess → score → supervised → cluster → association rules.

Usage:
    python TopKselection/pipeline.py --csv Scrapingdata/products_history.csv --k 50

Outputs in TopKselection/output/:
    df_clean.csv               — cleaned dataset
    products_scored.csv        — with score column
    top_k_products.csv         — Top-K ranked products
    shop_leaderboard.csv       — best shops by avg score
    supervised_metrics.json    — RF + XGBoost evaluation
    clustering_metrics.json    — KMeans / DBSCAN / hierarchical silhouettes
    elbow_curve.json           — inertia + silhouette per K
    association_rules.csv      — all rules (support, confidence, lift)
    top_rules.json             — top 20 rules
    pca_2d.csv                 — 2D coordinates for visualization
    rf_model.pkl               — saved Random Forest
    xgb_model.pkl              — saved XGBoost (if available)
    kmeans_model.pkl           — saved KMeans
    scaler.pkl                 — saved MinMaxScaler

    python TopKselection/pipeline.py --csv Scraping/data/products_history.csv --k 50

"""

import argparse
import logging
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("Module2")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from TopKselection import preprocessing, scoring, supervised, clustering, association_rules


def run_pipeline(csv_path: str, k: int = 50, output_dir: str = "TopKselection/output"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("MODULE 2 — ML Analysis & Top-K Selection")
    logger.info("=" * 60)

    # Step 1 — Preprocess
    logger.info("STEP 1: Preprocessing")
    artefacts = preprocessing.run(csv_path, output_dir)
    df_clean  = artefacts["df"]

    # Step 2 — Scoring + Top-K
    logger.info("STEP 2: Scoring & Top-K")
    scoring_out = scoring.run(df_clean, k=k, output_dir=output_dir)
    df_scored   = scoring_out["df_scored"]

    # Update artefacts with scored labels
    import numpy as np
    artefacts["df"] = df_scored
    # Re-build X and y with the scored labels
    X, scaler, feat_cols = preprocessing.build_feature_matrix(df_scored, artefacts["scaler"])
    artefacts["X"]       = X
    artefacts["y_train"] = df_scored.loc[artefacts["idx_train"], "is_top_product"].values
    artefacts["y_test"]  = df_scored.loc[artefacts["idx_test"],  "is_top_product"].values

    # Step 3a — Supervised
    logger.info("STEP 3a: Supervised learning (RF + XGBoost)")
    sup_out = supervised.run(artefacts, output_dir)

    # Step 3b — Clustering
    logger.info("STEP 3b: Clustering (KMeans + DBSCAN + Hierarchical)")
    clust_out = clustering.run(artefacts, output_dir)

    # Step 3c — Association Rules
    logger.info("STEP 3c: Association Rules")
    assoc_out = association_rules.run(df_scored, output_dir)

    # Final summary
    summary = {
        "dataset": {
            "total_products": len(df_scored),
            "top_k": k,
            "features_used": len(feat_cols),
        },
        "supervised": {m["model"]: {"f1": m.get("f1"), "accuracy": m.get("accuracy")}
                       for m in sup_out["metrics"]},
        "clustering": clust_out["metrics"],
        "association_rules": {
            "total_rules": len(assoc_out["rules"]),
            "top_lift": float(assoc_out["rules"]["lift"].max())
                        if not assoc_out["rules"].empty else 0,
        },
    }
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("MODULE 2 COMPLETE")
    logger.info(f"  Products analysed : {summary['dataset']['total_products']}")
    logger.info(f"  Top-{k} saved      : {output_dir}/top_k_products.csv")
    logger.info(f"  Summary            : {output_dir}/summary.json")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 2 — ML Pipeline")
    parser.add_argument("--csv",    required=True, help="Path to products_history.csv")
    parser.add_argument("--k",      type=int, default=50, help="Number of top products")
    parser.add_argument("--output", default="TopKselection/output", help="Output directory")
    args = parser.parse_args()

    summary = run_pipeline(args.csv, k=args.k, output_dir=args.output)
    print(json.dumps(summary, indent=2))