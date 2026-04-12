"""
KubeflowPipelines/components/training_component.py
-----------------------------------------
Kubeflow Pipeline component — Model Training (RF + XGBoost + KMeans + Assoc. Rules).
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Output, Model, Metrics


@dsl.component(
    base_image="smart-ecommerce/ml:latest",
    packages_to_install=[],
)
def training_component(
    input_scored:   Input[Dataset],
    feat_cols_path: Input[Dataset],
    output_rf:      Output[Model],
    output_kmeans:  Output[Model],
    output_scaler:  Output[Model],
    output_metrics: Output[Metrics],
    output_clusters:Output[Dataset],
    output_rules:   Output[Dataset],
    output_pca:     Output[Dataset],
):
    """
    Trains all models and writes artefacts + metrics.
    """
    import sys, json, tempfile
    sys.path.insert(0, "/app")
    import numpy as np
    import pandas as pd
    import joblib
    from TopKselection.preprocessing import engineer_features, build_feature_matrix, make_train_test_split
    from TopKselection.scoring       import compute_scores
    from TopKselection.supervised    import train_random_forest
    from TopKselection.clustering    import find_optimal_k, run_kmeans, label_clusters, run_pca, run_dbscan
    from TopKselection.association_rules import build_baskets_from_category, run_apriori_manual

    df = pd.read_csv(input_scored.path)
    with open(feat_cols_path.path) as f:
        feat_cols = json.load(f)

    # Re-build feature matrix
    df = engineer_features(df)
    X, scaler, feat_cols = build_feature_matrix(df)
    y = df["is_top_product"].values

    # Train/test split
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = make_train_test_split(df, X, y)

    # Random Forest
    with tempfile.TemporaryDirectory() as tmp:
        rf, rf_metrics = train_random_forest(X_tr, y_tr, X_te, y_te, feat_cols, tmp)

    # Clustering
    elbow   = find_optimal_k(X, k_range=(2, min(8, len(df)//10 or 2)))
    best_k  = elbow["best_k"]
    km, km_labels, km_sil = run_kmeans(X, best_k)
    df_clust, _ = label_clusters(df, km_labels)
    _, X_2d, explained = run_pca(X)
    pca_df = pd.DataFrame({"pc1": X_2d[:,0], "pc2": X_2d[:,1],
                            "cluster": km_labels, "title": df["title"],
                            "score": df.get("score", 0)})

    # Association rules
    baskets = build_baskets_from_category(df)
    rules   = run_apriori_manual(baskets, min_support=0.05, min_confidence=0.1)

    # Persist models
    joblib.dump(rf,    output_rf.path)
    joblib.dump(km,    output_kmeans.path)
    joblib.dump(scaler,output_scaler.path)

    # Persist datasets
    df_clust.to_csv(output_clusters.path, index=False)
    rules.to_csv(   output_rules.path,    index=False)
    pca_df.to_csv(  output_pca.path,      index=False)

    # Log metrics to Kubeflow UI
    output_metrics.log_metric("rf_f1",         rf_metrics["f1"])
    output_metrics.log_metric("rf_accuracy",   rf_metrics["accuracy"])
    output_metrics.log_metric("rf_precision",  rf_metrics["precision"])
    output_metrics.log_metric("rf_recall",     rf_metrics["recall"])
    output_metrics.log_metric("km_silhouette", round(km_sil, 4))
    output_metrics.log_metric("n_rules",       len(rules))
    output_metrics.log_metric("pca_var_pc1",   round(explained[0], 4))

    print(f"[training] RF F1={rf_metrics['f1']:.3f}  KMeans sil={km_sil:.3f}  rules={len(rules)}")