"""
KubeflowPipelines/pipeline/smart_ecommerce_pipeline.py
---------------------------------------------
Main Kubeflow Pipeline definition.

To compile to YAML:
    python smart_ecommerce_pipeline.py

To submit to a running Kubeflow cluster:
    python smart_ecommerce_pipeline.py --submit \
        --host http://localhost:8080 \
        --urls https://store1.myshopify.com https://store2.com

To run locally (no Kubernetes needed):
    python smart_ecommerce_pipeline.py --local

python pipeline/smart_ecommerce_pipeline.py --local --urls https://gymshark.com

"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Import components ──────────────────────────────────────────────────
from KubeflowPipelines.components.scraping_component     import scraping_component
from KubeflowPipelines.components.preprocessing_component import preprocessing_component
from KubeflowPipelines.components.scoring_component      import scoring_component
from KubeflowPipelines.components.training_component     import training_component
from KubeflowPipelines.components.export_component       import export_component

from kfp import dsl, compiler
import kfp


# ── Pipeline definition ───────────────────────────────────────────────

@dsl.pipeline(
    name="smart-ecommerce-pipeline",
    description="End-to-end ML pipeline: scraping → preprocessing → scoring → training → export",
)
def smart_ecommerce_pipeline(
    urls: list = ["https://hydrogen-demo-store.myshopify.com"],
    shop_country: str = "USA",
    top_k: int = 50,
):
    """
    DAG of the Smart eCommerce ML pipeline.

    Steps run sequentially; training branches into RF+clustering in parallel.
    All artefacts are passed between steps via Kubeflow's artifact store.
    """

    # ── Step 1: Scraping ─────────────────────────────────────────────
    scrape_task = scraping_component(
        urls=urls,
        shop_country=shop_country,
    )
    scrape_task.set_display_name("Web Scraping (A2A Agents)")
    scrape_task.set_cpu_request("500m").set_memory_request("512Mi")

    # ── Step 2: Preprocessing ─────────────────────────────────────────
    preprocess_task = preprocessing_component(
        input_csv=scrape_task.outputs["output_csv"],
    )
    preprocess_task.set_display_name("Preprocessing & Feature Engineering")
    preprocess_task.set_cpu_request("500m").set_memory_request("1Gi")
    preprocess_task.after(scrape_task)

    # ── Step 3: Scoring + Top-K ───────────────────────────────────────
    scoring_task = scoring_component(
        input_clean=preprocess_task.outputs["output_clean"],
        k=top_k,
    )
    scoring_task.set_display_name(f"Composite Scoring → Top-{top_k}")
    scoring_task.set_cpu_request("500m").set_memory_request("512Mi")
    scoring_task.after(preprocess_task)

    # ── Step 4: Training (RF + clustering + assoc. rules) ────────────
    training_task = training_component(
        input_scored=scoring_task.outputs["output_scored"],
        feat_cols_path=preprocess_task.outputs["feat_cols_out"],
    )
    training_task.set_display_name("Model Training & Evaluation")
    training_task.set_cpu_request("2").set_memory_request("2Gi")
    training_task.after(scoring_task)

    # ── Step 5: Export bundle ─────────────────────────────────────────
    export_task = export_component(
        input_topk=scoring_task.outputs["output_topk"],
        input_shops=scoring_task.outputs["output_shops"],
        input_clusters=training_task.outputs["output_clusters"],
        input_rules=training_task.outputs["output_rules"],
        input_pca=training_task.outputs["output_pca"],
    )
    export_task.set_display_name("Export Artefacts for BI Dashboard")
    export_task.set_cpu_request("200m").set_memory_request("256Mi")
    export_task.after(training_task)


# ── CLI ───────────────────────────────────────────────────────────────

def compile_pipeline(output_path: str = "KubeflowPipelines/pipeline/smart_ecommerce_pipeline.yaml"):
    """Compile the pipeline to a YAML spec file."""
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path=output_path,
    )
    print(f"Pipeline compiled → {output_path}")


def submit_pipeline(host: str, urls: list, shop_country: str, top_k: int):
    """Submit pipeline run to a live Kubeflow instance."""
    client = kfp.Client(host=host)
    run = client.create_run_from_pipeline_func(
        smart_ecommerce_pipeline,
        arguments={
            "urls":         urls,
            "shop_country": shop_country,
            "top_k":        top_k,
        },
        run_name="smart-ecommerce-run",
        experiment_name="Smart eCommerce",
    )
    print(f"Pipeline submitted. Run ID: {run.run_id}")
    print(f"Track at: {host}/#/runs/details/{run.run_id}")


def run_local(urls: list, shop_country: str, top_k: int):
    """
    Local execution — runs each component as a plain Python function.
    Useful for development and CI testing without a Kubernetes cluster.
    """
    import tempfile, os, shutil
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("=" * 60)
    print("LOCAL PIPELINE RUN (no Kubernetes)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Step 1: Scraping
        print("\nStep 1: Scraping...")
        from Scraping.orchestrator import Orchestrator
        orch = Orchestrator(output_dir=str(tmp / "scraping"))
        df_raw = orch.run(urls=urls)
        csv_path = tmp / "scraping" / "products_history.csv"
        print(f"  {len(df_raw)} products scraped")

        # Step 2: Preprocessing
        print("\nStep 2: Preprocessing...")
        from TopKselection.preprocessing import load_and_clean, engineer_features, build_feature_matrix
        df = load_and_clean(str(csv_path))
        df = engineer_features(df)
        X, scaler, feat_cols = build_feature_matrix(df)
        print(f"  {len(df)} rows × {len(feat_cols)} features")

        # Step 3: Scoring
        print("\nStep 3: Scoring...")
        from TopKselection.scoring import compute_scores, select_top_k, shop_leaderboard
        df = compute_scores(df)
        top = select_top_k(df, k=top_k)
        shops = shop_leaderboard(df)
        print(f"  Top-{top_k} selected. Best score: {df['score'].max():.3f}")

        # Step 4: Training
        print("\nStep 4: Training...")
        from TopKselection.pipeline import run_pipeline
        out = str(tmp / "ml_output")
        run_pipeline(str(csv_path), k=top_k, output_dir=out)

        # Step 5: Export
        print("\nStep 5: Exporting...")
        bundle = tmp / "bundle"
        bundle.mkdir()
        for fname in ["top_k_products.csv","shop_leaderboard.csv",
                      "products_clustered.csv","association_rules.csv","pca_2d.csv"]:
            src = Path(out) / fname
            if src.exists():
                shutil.copy(src, bundle / fname)
                print(f"  {fname}")

        print(f"\nAll outputs in: {out}/")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart eCommerce Kubeflow Pipeline")
    parser.add_argument("--submit", action="store_true", help="Submit to Kubeflow cluster")
    parser.add_argument("--local",  action="store_true", help="Run locally (no K8s)")
    parser.add_argument("--compile",action="store_true", help="Compile to YAML only")
    parser.add_argument("--host",   default="http://localhost:8080", help="Kubeflow host")
    parser.add_argument("--urls",   nargs="+", default=["https://hydrogen-demo-store.myshopify.com"])
    parser.add_argument("--country",default="USA")
    parser.add_argument("--k",      type=int, default=50)
    parser.add_argument("--output", default="KubeflowPipelines/pipeline/smart_ecommerce_pipeline.yaml")
    args = parser.parse_args()

    if args.local:
        run_local(args.urls, args.country, args.k)
    elif args.submit:
        submit_pipeline(args.host, args.urls, args.country, args.k)
    else:
        compile_pipeline(args.output)