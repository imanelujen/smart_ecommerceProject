"""
KubeflowPipelines/components/scraping_component.py
-----------------------------------------
Kubeflow Pipeline component — Web Scraping (wraps Module 1 orchestrator).

Each kfp component is a self-contained Python function that Kubeflow
runs inside its own Docker container. Typed Input/Output objects are
the interface between steps — Kubeflow handles moving data between them.
"""

from kfp import dsl
from kfp.dsl import Dataset, Output


@dsl.component(
    base_image="smart-ecommerce/scraping:latest",
    packages_to_install=[],   # all deps baked into base_image
)
def scraping_component(
    urls: list,
    shop_country: str,
    output_csv: Output[Dataset],
):
    """
    Runs the A2A orchestrator against the given URLs.
    Output: products_history.csv written to output_csv.path.
    """
    import sys
    sys.path.insert(0, "/app")
    from orchestrator import Orchestrator
    import shutil

    out_dir = "/tmp/scraping_out"
    orch    = Orchestrator(output_dir=out_dir)
    df      = orch.run(urls=urls)

    if df.empty:
        raise RuntimeError("Scraping returned 0 products — check URLs/credentials")

    shutil.copy(f"{out_dir}/products_history.csv", output_csv.path)
    print(f"[scraping] {len(df)} products saved to {output_csv.path}")