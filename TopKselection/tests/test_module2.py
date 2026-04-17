"""Tests for TopKselection — ML pipeline"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import tempfile

def make_synthetic_df(n=200):
    np.random.seed(42)
    cats = ["Electronics", "Sport", "Home", "Fashion", "Books"]
    return pd.DataFrame({
        "title":          [f"Product {i}" for i in range(n)],
        "price":          np.random.lognormal(3.5, 0.8, n).clip(1, 500),
        "rating":         np.random.uniform(2.5, 5.0, n),
        "review_count":   np.random.randint(0, 2000, n),
        "discount_pct":   np.random.choice([0, 10, 20, 30, 50], n).astype(float),
        "stock_quantity": np.random.randint(0, 200, n).astype(float),
        "availability":   np.random.choice([True, False], n, p=[0.85, 0.15]),
        "variant_count":  np.random.randint(1, 6, n),
        "category":       np.random.choice(cats, n),
        "platform":       np.random.choice(["shopify", "woocommerce"], n),
        "shop_country":   np.random.choice(["USA", "UK", "FR", "DE"], n),
        "shop_name":      np.random.choice(["ShopA", "ShopB", "ShopC"], n),
        "related_products": ["Product A, Product B" if i%3==0 else "" for i in range(n)],
        "is_top_product": 0,
    })


def test_preprocessing():
    from TopKselection.preprocessing import load_and_clean, engineer_features, build_feature_matrix
    df = make_synthetic_df()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name, index=False)
        path = f.name
    df_clean = load_and_clean(path)
    assert len(df_clean) > 0 and df_clean["price"].min() > 0
    df_eng = engineer_features(df_clean)
    assert all(c in df_eng.columns for c in ["log_price","popularity_score","is_budget"])
    X, scaler, feat_cols = build_feature_matrix(df_eng)
    assert X.shape[0] == len(df_eng)
    assert X.min() >= 0.0 and X.max() <= 1.0 + 1e-6
    print(f"  preprocessing: {X.shape[1]} features, {len(df_eng)} rows — OK")


def test_scoring():
    from TopKselection.scoring import compute_scores, select_top_k, shop_leaderboard
    df = make_synthetic_df()
    df_scored = compute_scores(df)
    assert df_scored["score"].between(0, 1).all()
    assert df_scored["is_top_product"].sum() > 0
    top_k = select_top_k(df_scored, k=20)
    assert len(top_k) == 20
    assert top_k["score"].is_monotonic_decreasing
    shops = shop_leaderboard(df_scored)
    assert "avg_score" in shops.columns
    print(f"  scoring: top-K={len(top_k)}, shops={len(shops)} — OK")


def test_supervised():
    from TopKselection.preprocessing import engineer_features, build_feature_matrix, make_train_test_split
    from TopKselection.scoring import compute_scores
    from TopKselection.supervised import train_random_forest
    df = make_synthetic_df()
    df = engineer_features(df)
    df = compute_scores(df)
    X, scaler, feat_cols = build_feature_matrix(df)
    y = df["is_top_product"].values
    X_tr, X_te, y_tr, y_te, _, _ = make_train_test_split(df, X, y)
    with tempfile.TemporaryDirectory() as tmp:
        rf, metrics = train_random_forest(X_tr, y_tr, X_te, y_te, feat_cols, tmp)
    assert 0 <= metrics["f1"] <= 1
    assert "confusion_matrix" in metrics
    assert len(metrics["top_features"]) > 0
    print(f"  supervised RF: F1={metrics['f1']:.3f}, acc={metrics['accuracy']:.3f} — OK")


def test_clustering():
    from TopKselection.preprocessing import engineer_features, build_feature_matrix
    from TopKselection.scoring import compute_scores
    from TopKselection.clustering import find_optimal_k, run_kmeans, run_dbscan, label_clusters, run_pca
    df = make_synthetic_df(150)
    df = engineer_features(df)
    df = compute_scores(df)
    X, _, _ = build_feature_matrix(df)
    elbow = find_optimal_k(X, k_range=(2, 5))
    assert 2 <= elbow["best_k"] <= 5
    km, labels, sil = run_kmeans(X, elbow["best_k"])
    assert len(labels) == len(df) and sil > -1.0
    _, db_labels, _, n_noise = run_dbscan(X)
    assert n_noise >= 0
    df_clust, profile = label_clusters(df, labels)
    assert "cluster_label" in df_clust.columns
    pca, X_2d, explained = run_pca(X)
    assert X_2d.shape == (len(df), 2)
    print(f"  clustering: K={elbow['best_k']}, silhouette={sil:.3f}, PCA={[round(v,2) for v in explained]} — OK")


def test_association_rules():
    from TopKselection.association_rules import build_baskets_from_category, run_apriori_manual
    df = make_synthetic_df(100)
    cat_baskets = build_baskets_from_category(df)
    assert len(cat_baskets) >= 1
    rules = run_apriori_manual(cat_baskets, min_support=0.05, min_confidence=0.1)
    assert isinstance(rules, pd.DataFrame)
    if not rules.empty:
        assert all(c in rules.columns for c in ["support","confidence","lift"])
        assert (rules["support"] >= 0.05).all()
    print(f"  association rules: {len(rules)} rules — OK")


if __name__ == "__main__":
    print("Running TopKselection  tests...")
    test_preprocessing()
    test_scoring()
    test_supervised()
    test_clustering()
    test_association_rules()
    print("\nAll TopKselection  tests passed.")