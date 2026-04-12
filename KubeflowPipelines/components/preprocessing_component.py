"""
module3/components/preprocessing_component.py
----------------------------------------------
Kubeflow Pipeline component — Preprocessing (wraps TopKselection/preprocessing.py).
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Output, Model
import json


@dsl.component(
    base_image="smart-ecommerce/ml:latest",
    packages_to_install=[],
)
def preprocessing_component(
    input_csv:    Input[Dataset],
    output_clean: Output[Dataset],
    output_scaler:Output[Model],
    feat_cols_out: Output[Dataset],
):
    """
    Cleans, imputes, engineers features, normalises.
    Outputs: cleaned CSV, fitted scaler (pkl), feature column list (json).
    """
    import sys, json
    sys.path.insert(0, "/app")
    from TopKselection.preprocessing import load_and_clean, engineer_features, build_feature_matrix
    import joblib, pandas as pd
    df      = load_and_clean(input_csv.path)
    df      = engineer_features(df)
    X, scaler, feat_cols = build_feature_matrix(df)

    df.to_csv(output_clean.path, index=False)
    joblib.dump(scaler, output_scaler.path)
    with open(feat_cols_out.path, "w") as f:
        json.dump(feat_cols, f)

    print(f"[preprocessing] {len(df)} rows × {len(feat_cols)} features")