"""
orchestrator.py  (v2)
---------------------
Updated orchestrator — now produces a dataset with all 20 required columns.
Output is ready for Module 2 ML analysis (Top-K selection).

python orchestrator.py --urls https://your-shopify-store.myshopify.com
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

from agents.base_agent import Product
from agents.shopify_agent import ShopifyAgent
from agents.woocommerce_agent import WooCommerceAgent
from agents.generic_agent import GenericHTMLAgent

logger = logging.getLogger("Orchestrator")

# Final column order for the CSV (matches spec Section 10-11)
COLUMN_ORDER = [
    "product_id", "title", "category", "subcategory", "brand",
    "price", "price_promo", "price_old", "discount_pct", "currency",
    "rating", "review_count", "category_rank",
    "availability", "stock_quantity", "delivery_days",
    "variant_count", "colors", "sizes",
    "shop_name", "shop_country", "shop_product_count",
    "related_products",
    "published_at", "scraped_at",
    "tags", "description", "customer_reviews",
    "url", "image_url", "platform",
]


class Orchestrator:
    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute des colonnes de métadonnées, par exemple price_band et scraped_at."""
        if "price" in df.columns:
            df["price_band"] = pd.cut(
                df["price"],
                bins=[0, 10, 30, 100, 300, float("inf")],
                labels=["<$10", "$10-30", "$30-100", "$100-300", ">$300"],
            )
        if "scraped_at" not in df.columns:
            from datetime import datetime, timezone
            df["scraped_at"] = datetime.now(timezone.utc).isoformat()
        return df
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.agents = [
            ShopifyAgent(),
            WooCommerceAgent(),
            GenericHTMLAgent(),
        ]
        logger.info(f"Agents ready: {[a.name for a in self.agents]}")

    def run(self, urls: List[str]) -> pd.DataFrame:
        all_products: List[Product] = []
        for url in urls:
            agent = self._detect_agent(url)
            logger.info(f"{url}  →  [{agent.name}]")
            try:
                products = agent.run(url)
                # Stamp scraped_at
                ts = datetime.now(timezone.utc).isoformat()
                for p in products:
                    p.scraped_at = ts
                all_products.extend(products)
            except Exception as e:
                logger.error(f"Agent {agent.name} failed on {url}: {e}")

        if not all_products:
            logger.warning("No products scraped.")
            return pd.DataFrame()

        df = self._to_dataframe(all_products)
        df = self._deduplicate(df)
        df = self._add_derived_columns(df)
        df = self._reorder_columns(df)
        self._save(df)
        logger.info(f"Done — {len(df)} unique products, {len(df.columns)} columns")
        return df

    # ── Agent routing ─────────────────────────────────────────────────

    def _detect_agent(self, url: str):
        for agent in self.agents:
            try:
                if agent.detect(url):
                    return agent
            except Exception:
                pass
        return self.agents[-1]

    # ── Data processing ───────────────────────────────────────────────

    def _to_dataframe(self, products: List[Product]) -> pd.DataFrame:
        return pd.DataFrame([p.to_dict() for p in products])

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["title", "price", "platform"], keep="first")
        logger.info(f"Dedup: {before} → {len(df)} rows")
        return df.reset_index(drop=True)

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ML-ready helper columns."""
        # Score placeholder (Module 2 fills this)
        df["score"] = 0.0

        # Price band — useful for clustering
        df["price_band"] = pd.cut(
            df["price"],
            bins=[0, 10, 30, 100, 300, float("inf")],
            labels=["<$10", "$10-30", "$30-100", "$100-300", ">$300"],
        )

        # On-sale flag
        df["is_on_sale"] = (
            df["price_promo"].notna() & (df["price_promo"] < df["price"])
        )

        # Popular flag: rating >= 4 AND review_count >= 100
        df["is_popular"] = (
            (df["rating"].fillna(0) >= 4.0) & (df["review_count"] >= 100)
        )

        # Low-stock flag
        df["is_low_stock"] = (
            df["stock_quantity"].notna() & (df["stock_quantity"] < 10)
        )

        return df

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        ordered = [c for c in COLUMN_ORDER if c in df.columns]
        extras  = [c for c in df.columns if c not in ordered]
        return df[ordered + extras]

    # ── Persistence ───────────────────────────────────────────────────

    def _save(self, df: pd.DataFrame):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Full CSV (for ML pipeline — Module 2)
        csv_path = self.output_dir / f"products_{ts}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")

        # JSONL (for LLM enrichment — Module 5)
        jsonl_path = self.output_dir / f"products_{ts}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), default=str) + "\n")
        logger.info(f"Saved: {jsonl_path}")

        # Master history (append all scraped data)
        import os
        history_path = self.output_dir / "products_history.csv"
        if history_path.exists():
            try:
                history_df = pd.read_csv(history_path)
                combined_df = pd.concat([history_df, df], ignore_index=True)
                # Deduplicate history to avoid exact duplicates if scraping rerun
                combined_df = combined_df.drop_duplicates(subset=["title", "price", "platform", "scraped_at"], keep="last")
                combined_df.to_csv(history_path, index=False)
                logger.info(f"Updated history: {len(combined_df)} rows in {history_path}")
                df = combined_df # Use combined for summary
            except Exception as e:
                logger.error(f"Failed to append to history: {e}")
        else:
            df.to_csv(history_path, index=False)
            logger.info(f"Created new history file: {history_path}")

        # Print summary stats
        self._print_summary(df)

    def _print_summary(self, df: pd.DataFrame):
        print(f"\n{'='*60}")
        print(f"Dataset: {len(df)} produits × {len(df.columns)} variables")
        print(f"Plateformes : {df['platform'].value_counts().to_dict()}")
        if "category" in df.columns:
            top_cats = df["category"].value_counts().head(5)
            print(f"Top catégories :\n{top_cats.to_string()}")
        if "price" in df.columns:
            print(f"Prix — min:{df['price'].min():.2f}  "
                  f"mean:{df['price'].mean():.2f}  "
                  f"max:{df['price'].max():.2f}")
        if "rating" in df.columns:
            rated = df["rating"].dropna()
            if len(rated):
                print(f"Rating moyen : {rated.mean():.2f}  (sur {len(rated)} produits notés)")
        print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Smart eCommerce A2A scraper v2")
    parser.add_argument("--urls", nargs="+", required=True)
    parser.add_argument("--output", default="data")
    args = parser.parse_args()

    orch = Orchestrator(output_dir=args.output)
    df   = orch.run(urls=args.urls)
    if not df.empty:
        available_cols = [c for c in ["title", "price", "rating", "review_count", "platform"] if c in df.columns]
        if available_cols:
            print(df[available_cols].head(10))