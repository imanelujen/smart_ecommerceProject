"""
KubeflowPipelines/components/export_component.py
----------------------------------------
Kubeflow Pipeline component — Assembles final outputs for Module 4 (BI Dashboard).
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Output


@dsl.component(
    base_image="smart-ecommerce/ml:latest",
    packages_to_install=[],
)
def export_component(
    input_topk:    Input[Dataset],
    input_shops:   Input[Dataset],
    input_clusters:Input[Dataset],
    input_rules:   Input[Dataset],
    input_pca:     Input[Dataset],
    output_bundle: Output[Dataset],
):
    """
    Bundles all artefacts into a single directory that Module 4 consumes.
    In production this would write to MinIO / S3 / GCS.
    """
    import json, shutil, os
    from pathlib import Path
    import pandas as pd

    bundle_dir = Path("/tmp/export_bundle")
    bundle_dir.mkdir(exist_ok=True)

    # Copy all datasets
    files = {
        "top_k_products.csv":   input_topk.path,
        "shop_leaderboard.csv": input_shops.path,
        "products_clustered.csv": input_clusters.path,
        "association_rules.csv":  input_rules.path,
        "pca_2d.csv":           input_pca.path,
    }
    manifest = {}
    for name, src in files.items():
        dst = bundle_dir / name
        shutil.copy(src, dst)
        rows = len(pd.read_csv(dst))
        manifest[name] = {"rows": rows}
        print(f"[export] {name}: {rows} rows")

    # Write manifest
    manifest["generated_at"] = __import__("datetime").datetime.utcnow().isoformat()
    with open(bundle_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Zip bundle for artifact store
    shutil.make_archive(output_bundle.path.replace(".zip",""), "zip", bundle_dir)
    if not output_bundle.path.endswith(".zip"):
        shutil.copy(output_bundle.path + ".zip", output_bundle.path)

    print(f"[export] Bundle ready → {output_bundle.path}")