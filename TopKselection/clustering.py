"""
TopKselection/clustering.py
---------------------
Step 3b — Unsupervised learning: KMeans, DBSCAN, hierarchical clustering.

Task: segment products into meaningful groups (premium, discount, popular…).

Evaluation (as required by project spec):
  - Silhouette score
  - Inertia curve (elbow method for KMeans optimal K)
  - Cluster interpretation labels
  - PCA 2D projection for visual validation
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib

logger = logging.getLogger(__name__)


def find_optimal_k(X: np.ndarray, k_range=(2, 10)) -> dict:
    """
    Elbow method: compute inertia and silhouette for each K.
    Returns dict with inertia list and best K by silhouette.
    """
    inertias    = []
    silhouettes = []
    ks          = list(range(k_range[0], k_range[1] + 1))

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(round(km.inertia_, 2))
        sil = silhouette_score(X, labels)
        silhouettes.append(round(sil, 4))
        logger.info(f"K={k}  inertia={km.inertia_:.1f}  silhouette={sil:.4f}")

    best_k = ks[np.argmax(silhouettes)]
    logger.info(f"Best K by silhouette: {best_k}")
    return {"ks": ks, "inertias": inertias, "silhouettes": silhouettes, "best_k": best_k}


def run_kmeans(X: np.ndarray, k: int) -> tuple:
    """Fit KMeans with the best K."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil    = silhouette_score(X, labels)
    logger.info(f"KMeans k={k}  silhouette={sil:.4f}")
    return km, labels, sil


def run_dbscan(X: np.ndarray, eps: float = 0.3, min_samples: int = 5) -> tuple:
    """
    DBSCAN — detects outlier products (noise label = -1).
    Good for anomaly detection: products with unusual price/rating combos.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()

    if n_clusters >= 2:
        mask = labels != -1
        sil  = silhouette_score(X[mask], labels[mask]) if mask.sum() > 1 else 0.0
    else:
        sil = 0.0

    logger.info(
        f"DBSCAN: {n_clusters} clusters, {n_noise} outliers, silhouette={sil:.4f}"
    )
    return db, labels, sil, n_noise


def run_hierarchical(X: np.ndarray, k: int) -> tuple:
    """Agglomerative (hierarchical) clustering — Ward linkage."""
    hc     = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = hc.fit_predict(X)
    sil    = silhouette_score(X, labels)
    logger.info(f"Hierarchical k={k}  silhouette={sil:.4f}")
    return hc, labels, sil


def label_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Assign business-meaningful names to clusters based on their centroid
    characteristics (avg price, avg rating, avg review_count).
    """
    df = df.copy()
    df["cluster"] = labels

    # Compute cluster profiles
    profile = df[df["cluster"] != -1].groupby("cluster").agg(
        avg_price   = ("price",        "mean"),
        avg_rating  = ("rating",       "mean"),
        avg_reviews = ("review_count", "mean"),
        avg_discount= ("discount_pct", "mean"),
        count       = ("title",        "count"),
    )

    # Simple rule-based naming
    def _name(row):
        if row["avg_price"] > df["price"].quantile(0.75):
            return "Premium"
        if row["avg_discount"] > 20:
            return "Discount / Promo"
        if row["avg_rating"] >= 4.2 and row["avg_reviews"] >= 200:
            return "Top rated"
        if row["avg_reviews"] < 50:
            return "Niche / peu connu"
        return "Mainstream"

    profile["cluster_label"] = profile.apply(_name, axis=1)
    logger.info(f"Cluster profiles:\n{profile.to_string()}")

    # Map labels back
    label_map = profile["cluster_label"].to_dict()
    label_map[-1] = "Anomalie (DBSCAN)"
    df["cluster_label"] = df["cluster"].map(label_map)
    return df, profile


def run_pca(X: np.ndarray, n_components: int = 2) -> tuple:
    """Reduce to 2D for visualization."""
    pca = PCA(n_components=n_components, random_state=42)
    X_2d = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.tolist()
    logger.info(f"PCA explained variance: {[round(v,3) for v in explained]}")
    return pca, X_2d, explained


def run(artefacts: dict, output_dir: str = "TopKselection/output") -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    X   = artefacts["X"]
    df  = artefacts["df"]

    # ── Find optimal K ─────────────────────────────────────────────────
    elbow = find_optimal_k(X, k_range=(2, min(10, len(df) // 5)))
    best_k = elbow["best_k"]

    with open(f"{output_dir}/elbow_curve.json", "w") as f:
        json.dump(elbow, f, indent=2)

    # ── KMeans ────────────────────────────────────────────────────────
    km, km_labels, km_sil = run_kmeans(X, best_k)
    joblib.dump(km, f"{output_dir}/kmeans_model.pkl")

    # ── Hierarchical ──────────────────────────────────────────────────
    hc, hc_labels, hc_sil = run_hierarchical(X, best_k)

    # ── DBSCAN ────────────────────────────────────────────────────────
    db, db_labels, db_sil, n_noise = run_dbscan(X)

    # ── Label and interpret ───────────────────────────────────────────
    df_clustered, cluster_profile = label_clusters(df, km_labels)
    df_clustered["dbscan_label"]  = db_labels
    df_clustered["hc_label"]      = hc_labels

    df_clustered.to_csv(f"{output_dir}/products_clustered.csv", index=False)
    cluster_profile.to_csv(f"{output_dir}/cluster_profiles.csv")

    # ── PCA 2D ────────────────────────────────────────────────────────
    pca, X_2d, explained = run_pca(X)
    joblib.dump(pca, f"{output_dir}/pca_model.pkl")
    pca_df = pd.DataFrame({
        "pc1": X_2d[:, 0], "pc2": X_2d[:, 1],
        "cluster": km_labels, "cluster_label": df_clustered["cluster_label"],
        "title": df["title"], "price": df["price"], "score": df.get("score", 0),
    })
    pca_df.to_csv(f"{output_dir}/pca_2d.csv", index=False)

    metrics = {
        "kmeans":      {"k": best_k, "silhouette": km_sil},
        "hierarchical":{"k": best_k, "silhouette": hc_sil},
        "dbscan":      {"n_clusters": int((db_labels != -1).max()) + 1 if len(db_labels) > 0 else 0,
                        "n_outliers": int(n_noise), "silhouette": db_sil},
        "pca_explained": explained,
    }
    with open(f"{output_dir}/clustering_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Clustering complete.")
    return {
        "df_clustered": df_clustered,
        "cluster_profile": cluster_profile,
        "pca_df": pca_df,
        "metrics": metrics,
        "km_labels": km_labels,
        "X_2d": X_2d,
    }