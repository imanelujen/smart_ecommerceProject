"""
module5/enrichment_pipeline.py
--------------------------------
Batch enrichment — runs all 4 LLM chains on the scraped dataset
and saves enriched outputs for the BI dashboard and report.

Usage:
    python module5/enrichment_pipeline.py --input module2/output/products_scored.csv
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("Enrichment")

sys.path.insert(0, str(Path(__file__).parent.parent))

from module5.llm_client import LLMClient
from module5.chains     import (batch_summarize, generate_trend_report,
                                  build_client_profile, recommend_strategy)
from module4.data_loader import load_scored, load_top_k, load_clusters, load_rules


def run(csv_path: str = None, output_dir: str = "module5/output",
        max_summaries: int = 50) -> dict:

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    client = LLMClient()

    logger.info("=" * 55)
    logger.info("MODULE 5 — LLM Enrichment Pipeline")
    logger.info(f"  Provider : {client.provider}")
    logger.info("=" * 55)

    import pandas as pd
    df = pd.read_csv(csv_path) if csv_path else load_scored()
    top_k   = load_top_k()
    clusters= load_clusters()
    rules   = load_rules()

    results = {}

    # ── Task 1: Batch product summaries ───────────────────────────────
    logger.info(f"Task 1: Summarising up to {max_summaries} products...")
    df_enriched = batch_summarize(client, df, max_products=max_summaries)
    df_enriched.to_json(f"{output_dir}/enriched_products.jsonl",
                        orient="records", lines=True, force_ascii=False)
    results["summaries_count"] = int(df_enriched["llm_summary"].ne("").sum())
    logger.info(f"  {results['summaries_count']} summaries generated")

    # ── Task 2: Weekly trend report ───────────────────────────────────
    logger.info("Task 2: Generating trend report...")
    trend = generate_trend_report(client, df, top_k)
    Path(f"{output_dir}/trend_report.md").write_text(
        f"# Rapport de tendances — {datetime.utcnow().strftime('%Y-%m-%d')}\n\n{trend}",
        encoding="utf-8"
    )
    results["trend_report"] = trend[:200] + "..."
    logger.info("  Trend report saved")

    # ── Task 3: Customer profile ──────────────────────────────────────
    logger.info("Task 3: Building customer profile...")
    profile = build_client_profile(client, top_k)
    Path(f"{output_dir}/client_profile.md").write_text(
        f"# Profil client cible\n\n{profile}", encoding="utf-8"
    )
    results["client_profile"] = profile[:200] + "..."
    logger.info("  Client profile saved")

    # ── Task 4: Marketing strategy ────────────────────────────────────
    logger.info("Task 4: Generating marketing strategy...")
    strategy = recommend_strategy(client, clusters, rules)
    Path(f"{output_dir}/marketing_strategy.md").write_text(
        f"# Stratégie marketing recommandée\n\n{strategy}", encoding="utf-8"
    )
    results["strategy"] = strategy[:200] + "..."
    logger.info("  Marketing strategy saved")

    # ── Summary ───────────────────────────────────────────────────────
    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "llm_provider": client.provider,
        **results,
    }
    with open(f"{output_dir}/enrichment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 55)
    logger.info("MODULE 5 COMPLETE")
    logger.info(f"  Outputs → {output_dir}/")
    logger.info("=" * 55)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 5 — LLM Enrichment")
    parser.add_argument("--input",    default=None,            help="Path to products_scored.csv")
    parser.add_argument("--output",   default="module5/output")
    parser.add_argument("--max",      type=int, default=50,    help="Max products to summarise")
    args = parser.parse_args()
    summary = run(args.input, args.output, args.max)
    print(json.dumps(summary, indent=2, ensure_ascii=False))