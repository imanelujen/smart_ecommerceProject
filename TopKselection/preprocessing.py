"""
TopKselection/preprocessing.py
------------------------
Step 1 of the ML pipeline.
Loads products_history.csv, cleans and prepares features for all algorithms.

Operations:
  - Imputation of missing values
  - Categorical encoding (Label + One-Hot)
  - Min-Max normalisation of numeric features
  - Train / test split (80/20, stratified on category)
  - Exports: X_train, X_test, y_train, y_test, df_clean
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import logging

logger = logging.getLogger(__name__)

# Features used by ML models
NUMERIC_FEATURES = [
    "price", "rating", "review_count", "discount_pct",
    "stock_quantity", "variant_count",
]
CATEGORICAL_FEATURES = ["category", "platform", "shop_country"]
TARGET_COL = "is_top_product"   # computed during scoring step


def infer_category(title: str, brand: str) -> tuple:
    """Task 6: Impute missing category/subcategory based on title keywords."""
    title = str(title).lower()
    
    # Simple keyword mapping
    skincare_keywords = ["serum", "cream", "lotion", "cleanser", "moisturizer", "toner", "oil", "mask", "peel"]
    makeup_keywords = ["lipstick", "palette", "foundation", "concealer", "mascara", "liner", "blush", "gloss"]
    hair_keywords = ["shampoo", "conditioner", "hair", "scalp"]
    kit_keywords = ["kit", "set", "bundle", "collection"]

    category = "Other"
    subcategory = "General"

    if any(k in title for k in skincare_keywords):
        category = "Skincare"
        if "serum" in title: subcategory = "Serums"
        elif "mask" in title: subcategory = "Masks"
        elif "cream" in title or "moisturizer" in title: subcategory = "Moisturizers"
    elif any(k in title for k in makeup_keywords):
        category = "Makeup"
        if "lipstick" in title or "gloss" in title: subcategory = "Lips"
        elif "palette" in title: subcategory = "Eyes"
    elif any(k in title for k in hair_keywords):
        category = "Haircare"
    elif any(k in title for k in kit_keywords):
        category = "Kits & Sets"

    return category, subcategory


def generate_synthetic_metrics(row: pd.Series) -> pd.Series:
    """Task 4: Generate plausible ratings and review counts based on price/brand."""
    import random
    
    # Logic: Higher price or recognized brands tend to have more/higher ratings in beauty
    price = row.get("price", 0)
    brand = str(row.get("brand", "")).lower()
    
    # Base ranges
    if price > 100:
        base_rating = 4.2
        count_range = (50, 500)
    elif price > 30:
        base_rating = 4.0
        count_range = (10, 200)
    else:
        base_rating = 3.8
        count_range = (0, 50)

    # Brand prestige multiplier
    prestige_brands = ["glossier", "banish", "drunk elephant", "chanel", "dior"]
    if any(pb in brand for pb in prestige_brands):
        base_rating += 0.3
        count_range = (count_range[0] * 2, count_range[1] * 2)

    rating = base_rating + random.uniform(-0.5, 0.3)
    rating = min(5.0, max(1.0, rating))
    count = random.randint(*count_range)

    return pd.Series([round(rating, 1), count], index=["rating", "review_count"])


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load CSV and fix basic quality issues."""
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows × {len(df.columns)} cols from {csv_path}")

    # ── Task 2: Fix Product IDs (Remove scientific notation) ──────────
    if "product_id" in df.columns:
        # Convert to float first (if it was read as scientific notation) then to int string
        df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce")
        df["product_id"] = df["product_id"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "Unknown")

    # ── Task 6: Missing categorical data ──────────────────────────────
    for idx, row in df.iterrows():
        if pd.isna(row.get("category")) or str(row.get("category")) == "Unknown" or str(row.get("category")) == "NaN":
            cat, subcat = infer_category(row["title"], row.get("brand", ""))
            df.at[idx, "category"] = cat
            df.at[idx, "subcategory"] = subcat
        elif pd.isna(row.get("subcategory")) or str(row.get("subcategory")) == "Unknown":
            _, subcat = infer_category(row["title"], row.get("brand", ""))
            df.at[idx, "subcategory"] = subcat

    # ── Task 4: Synthetic ratings if missing ──────────────────────────
    # If 90%+ is default/missing, we generate
    if df["rating"].fillna(3.0).value_counts(normalize=True).get(3.0, 0) > 0.9:
        logger.info("Generating synthetic ratings to improve analysis quality (Task 4)")
        metrics = df.apply(generate_synthetic_metrics, axis=1)
        df["rating"] = metrics["rating"]
        df["review_count"] = metrics["review_count"]

    # ── Numeric imputation fallback ─────────────────────────────────────
    df["rating"] = df["rating"].fillna(df["rating"].median()).fillna(3.5)
    df["review_count"]  = df["review_count"].fillna(0).clip(lower=0)
    df["discount_pct"]  = df["discount_pct"].fillna(0).clip(0, 100)

    median_stock = df["stock_quantity"].median()
    df["stock_quantity"] = df["stock_quantity"].fillna(median_stock if pd.notna(median_stock) else 0)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price"] = df["price"].replace([np.inf, -np.inf], np.nan)
    df["price"] = df["price"].fillna(df["price"].median())
    df["variant_count"] = df["variant_count"].fillna(1)

    # ── Boolean coercion ──────────────────────────────────────────────
    df["availability"]  = df["availability"].map(
        {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
    ).fillna(1).astype(int)

    # ── Categorical fill ──────────────────────────────────────────────
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").str.strip()

    # ── Drop rows with no title or zero price ─────────────────────────
    before = len(df)
    df = df[(df["title"].notna()) & (df["price"] > 0)]
    logger.info(f"Dropped {before - len(df)} invalid rows → {len(df)} remain")

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived ML features."""
    # Log-transform skewed distributions
    df["log_price"]        = np.log1p(df["price"])
    #df["log_review_count"] = np.log1p(df["review_count"])
    df["review_count"] = df["review_count"].fillna(0)

    # Normalised popularity proxy
    eps = 1e-6
    df["popularity_score"] = (
        df["rating"].fillna(0) * np.log1p(df["review_count"])
    )

    # Price competitiveness flag
    price_median = df["price"].median()
    df["is_budget"]  = (df["price"] <= price_median * 0.5).astype(int)
    df["is_premium"] = (df["price"] >= price_median * 2.0).astype(int)

    # Encode categorical features
    le = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

    return df


def build_feature_matrix(df: pd.DataFrame, scaler=None):
    """
    Build X (feature matrix) ready for sklearn.

    Returns:
        X        — normalised NumPy array
        scaler   — fitted MinMaxScaler (reuse for inference)
        feat_cols — list of column names used
    """
    feat_cols = (
        ["log_price", "log_review_count", "rating",
         "discount_pct", "stock_quantity", "variant_count",
         "availability", "is_budget", "is_premium", "popularity_score"]
        + [f"{c}_enc" for c in CATEGORICAL_FEATURES if f"{c}_enc" in df.columns]
    )
    feat_cols = [c for c in feat_cols if c in df.columns]

    X_raw = df[feat_cols].fillna(0).values
    if scaler is None:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X_raw)
    else:
        X = scaler.transform(X_raw)

    return X, scaler, feat_cols


def make_train_test_split(df: pd.DataFrame, X: np.ndarray, y: np.ndarray):
    """Stratified 80/20 split. Falls back to random if too few classes."""
    try:
        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            X, y, df.index,
            test_size=0.20, random_state=42, stratify=y
        )
    except ValueError:
        logger.warning("Stratified split failed — using random split")
        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            X, y, df.index, test_size=0.20, random_state=42
        )
    logger.info(f"Train: {len(X_tr)}  Test: {len(X_te)}")
    return X_tr, X_te, y_tr, y_te, idx_tr, idx_te


def run(csv_path: str, output_dir: str = "TopKselection/output") -> dict:
    """Full preprocessing pipeline. Returns dict of artefacts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = load_and_clean(csv_path)
    df = engineer_features(df)

    # Target is set by scoring.py — placeholder here
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = 0

    X, scaler, feat_cols = build_feature_matrix(df)
    y = df[TARGET_COL].values

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = make_train_test_split(df, X, y)

    # Save artefacts
    df.to_csv(f"{output_dir}/df_clean.csv", index=False)
    joblib.dump(scaler,    f"{output_dir}/scaler.pkl")
    joblib.dump(feat_cols, f"{output_dir}/feat_cols.pkl")
    np.save(f"{output_dir}/X_train.npy", X_tr)
    np.save(f"{output_dir}/X_test.npy",  X_te)
    np.save(f"{output_dir}/y_train.npy", y_tr)
    np.save(f"{output_dir}/y_test.npy",  y_te)
    logger.info(f"Artefacts saved to {output_dir}/")

    return {
        "df": df, "X": X, "X_train": X_tr, "X_test": X_te,
        "y_train": y_tr, "y_test": y_te,
        "scaler": scaler, "feat_cols": feat_cols,
        "idx_train": idx_tr, "idx_test": idx_te,
    }