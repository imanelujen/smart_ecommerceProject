"""
TopKselection/association_rules.py
-----------------------------
Step 3c — Association rules (Apriori / FP-Growth).

Discovers product co-occurrence patterns like:
  {coque iphone} → {chargeur}   (support=0.12, confidence=0.78, lift=3.4)

Input: either
  - related_products column (scraped "customers also bought")
  - category co-purchases (basket = all products in same category/shop)

Evaluation (as required by spec):
  - support, confidence, lift for each rule
  - top-N rules ranked by lift
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from itertools import combinations
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
    from mlxtend.preprocessing import TransactionEncoder
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False
    logger.warning("mlxtend not installed — using manual Apriori fallback")


def build_baskets_from_related(df: pd.DataFrame) -> list:
    """
    Build transaction baskets from 'related_products' column.
    Each basket = [product_title] + [related_product_titles].
    """
    baskets = []
    for _, row in df.iterrows():
        basket = [str(row["title"])]
        if pd.notna(row.get("related_products")) and row["related_products"]:
            related = [r.strip() for r in str(row["related_products"]).split(",") if r.strip()]
            basket.extend(related)
        if len(basket) >= 2:
            baskets.append(basket)
    return baskets


def build_baskets_from_category(df: pd.DataFrame) -> list:
    """
    Fallback: treat each category as a basket of product titles.
    Useful when related_products is sparse.
    """
    baskets = []
    for (platform, cat), group in df.groupby(["platform", "category"]):
        MAX_ITEMS_PER_BASKET = 20  # limit to avoid huge baskets from broad categories for Killed problem !!
        titles = group["title"].dropna().tolist()[:MAX_ITEMS_PER_BASKET]
        if len(titles) >= 2:
            baskets.append(titles)

    return baskets


def run_apriori_mlxtend(baskets: list, min_support=0.05, min_confidence=0.3) -> pd.DataFrame:
    """Run Apriori via mlxtend."""
    te     = TransactionEncoder()
    te_arr = te.fit(baskets).transform(baskets)
    df_bool = pd.DataFrame(te_arr, columns=te.columns_)

    #freq_items = apriori(df_bool, min_support=min_support, use_colnames=True)
    freq_items = fpgrowth(df_bool, min_support=min_support, use_colnames=True)
    if freq_items.empty:
        return pd.DataFrame()

    rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    return rules.sort_values("lift", ascending=False)


def run_apriori_manual(baskets: list, min_support=0.05, min_confidence=0.3) -> pd.DataFrame:
    """
    Pure-Python Apriori — runs when mlxtend is not installed.
    Returns a DataFrame with antecedents, consequents, support, confidence, lift.
    """
    n = len(baskets)
    # Item counts
    item_counts = defaultdict(int)
    for basket in baskets:
        for item in set(basket):
            item_counts[item] += 1

    # Frequent 1-itemsets
    freq_1 = {frozenset([k]): v / n for k, v in item_counts.items()
               if v / n >= min_support}

    # Frequent 2-itemsets
    pair_counts = defaultdict(int)
    for basket in baskets:
        items = list(set(basket))
        for a, b in combinations(items, 2):
            key = frozenset([a, b])
            pair_counts[key] += 1

    freq_2 = {k: v / n for k, v in pair_counts.items() if v / n >= min_support}

    # Generate rules from 2-itemsets
    rows = []
    for pair, sup in freq_2.items():
        items_list = list(pair)
        for i in range(2):
            ant  = frozenset([items_list[i]])
            cons = frozenset([items_list[1 - i]])
            ant_sup  = freq_1.get(ant, 0)
            if ant_sup == 0:
                continue
            confidence = sup / ant_sup
            if confidence < min_confidence:
                continue
            cons_sup = freq_1.get(cons, 0)
            lift     = confidence / cons_sup if cons_sup > 0 else 0
            rows.append({
                "antecedents": ", ".join(ant),
                "consequents": ", ".join(cons),
                "support":     round(sup, 4),
                "confidence":  round(confidence, 4),
                "lift":        round(lift, 4),
            })

    df_rules = pd.DataFrame(rows)

    # 🔥 sécurité absolue
    if df_rules.empty:
        logger.warning("No rules generated (manual Apriori).")
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    if "lift" not in df_rules.columns:
        logger.warning("Column 'lift' missing — returning empty DataFrame.")
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    return df_rules.sort_values("lift", ascending=False)


def run(df: pd.DataFrame, output_dir: str = "TopKselection/output",
        min_support: float = 0.05, min_confidence: float = 0.3) -> dict:

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Try related_products first, fall back to category baskets
    baskets = build_baskets_from_related(df)
    MAX_BASKETS = 50
    if len(baskets) < MAX_BASKETS:
        #logger.info("Sparse related_products — using category baskets")
        #baskets = build_baskets_from_category(df)
        logger.warning(f"Reducing baskets from {len(baskets)} to {MAX_BASKETS}")
        baskets = baskets[:MAX_BASKETS]

    logger.info(f"Built {len(baskets)} baskets for association rule mining")

    if HAS_MLXTEND:
        rules = run_apriori_mlxtend(baskets, min_support, min_confidence)
    else:
        rules = run_apriori_manual(baskets, min_support, min_confidence)

    if rules.empty:
        logger.warning("No rules found — try lowering min_support or min_confidence")
        return {"rules": rules, "top_rules": []}

    # Save
    rules.to_csv(f"{output_dir}/association_rules.csv", index=False)

    # Top 20 rules by lift as JSON (for the report)
    top_rules = rules.head(20)[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].to_dict(orient="records")
    with open(f"{output_dir}/top_rules.json", "w", encoding="utf-8") as f:
        json.dump(top_rules, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Found {len(rules)} rules. "
        f"Top lift={rules['lift'].max():.2f}  "
        f"Top confidence={rules['confidence'].max():.2f}"
    )
    return {"rules": rules, "top_rules": top_rules}