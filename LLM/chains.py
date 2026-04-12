"""
module5/chains.py
-----------------
LangChain-style prompt chains for the 4 LLM tasks required by the spec:

  1. summarize_product    — résumé de description produit
  2. generate_trend_report — rapport hebdomadaire des tendances
  3. build_client_profile  — profil client basé sur les Top-K
  4. recommend_strategy    — stratégie marketing recommandée

Each function accepts a LLMClient + data, returns a text result.
Chain of Thought prompting is used throughout (spec requirement).
"""

import json
import logging
import pandas as pd
from typing import Optional
from module5.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ── System prompt shared by all chains ────────────────────────────────
BASE_SYSTEM = """Tu es un expert en e-commerce et data mining.
Tu analyses des données produits scrappées de plateformes Shopify et WooCommerce.
Tes réponses sont concises, structurées, et orientées décision business.
Utilise toujours un raisonnement étape par étape (Chain of Thought) avant ta conclusion."""


# ══════════════════════════════════════════════════════════════════════
# Chain 1 — Product summarization
# ══════════════════════════════════════════════════════════════════════

SUMMARIZE_PROMPT = """Voici la description brute d'un produit e-commerce :

Titre       : {title}
Catégorie   : {category}
Prix        : {price} {currency}
Note        : {rating}/5 ({review_count} avis)
Description : {description}

Étape 1 — Identifie les 3 caractéristiques clés du produit.
Étape 2 — Évalue son positionnement prix/qualité en une phrase.
Étape 3 — Rédige un résumé marketing en 2-3 phrases maximum, adapté à une fiche produit."""


def summarize_product(client: LLMClient, product: dict) -> str:
    """Generate a clean 2-3 sentence product summary."""
    prompt = SUMMARIZE_PROMPT.format(
        title=product.get("title", ""),
        category=product.get("category", ""),
        price=product.get("price", ""),
        currency=product.get("currency", "USD"),
        rating=product.get("rating", "N/A"),
        review_count=product.get("review_count", 0),
        description=str(product.get("description", ""))[:400],
    )
    result = client.chat(prompt, system=BASE_SYSTEM, max_tokens=300)
    logger.info(f"Summarized: {product.get('title','?')}")
    return result


def batch_summarize(client: LLMClient, df: pd.DataFrame,
                    max_products: int = 50) -> pd.DataFrame:
    """Summarize up to max_products rows and add 'llm_summary' column."""
    df = df.copy()
    summaries = []
    for _, row in df.head(max_products).iterrows():
        try:
            summary = summarize_product(client, row.to_dict())
        except Exception as e:
            logger.warning(f"Summary failed for '{row.get('title','')}': {e}")
            summary = ""
        summaries.append(summary)

    # Pad with empty strings for rows beyond max_products
    summaries += [""] * (len(df) - len(summaries))
    df["llm_summary"] = summaries
    return df


# ══════════════════════════════════════════════════════════════════════
# Chain 2 — Weekly trend report
# ══════════════════════════════════════════════════════════════════════

TREND_REPORT_PROMPT = """Tu es un analyste marché e-commerce.
Voici les données agrégées des produits scrapés cette semaine :

{stats_json}

Top 5 produits (score composite) :
{top5_text}

Top catégories par nombre de produits :
{cat_counts}

Étape 1 — Identifie 3 tendances majeures dans ces données.
Étape 2 — Signale les anomalies ou opportunités remarquables.
Étape 3 — Rédige un rapport hebdomadaire structuré avec :
  - Résumé exécutif (2 phrases)
  - Tendances clés (liste bullet)
  - Produits émergents à surveiller
  - Recommandation décisionnelle finale"""


def generate_trend_report(client: LLMClient, df: pd.DataFrame,
                          top_k: Optional[pd.DataFrame] = None) -> str:
    """Generate a business-oriented weekly market trend report."""
    stats = {
        "total_products":   int(len(df)),
        "avg_score":        round(float(df["score"].mean()), 3) if "score" in df else 0,
        "avg_price":        round(float(df["price"].mean()), 2),
        "avg_rating":       round(float(df["rating"].mean()), 2) if "rating" in df else 0,
        "pct_on_sale":      round(float((df["discount_pct"].fillna(0) > 0).mean() * 100), 1),
        "platforms":        df["platform"].value_counts().to_dict() if "platform" in df else {},
    }

    if top_k is None:
        top5 = df.nlargest(5, "score") if "score" in df else df.head(5)
    else:
        top5 = top_k.head(5)

    top5_text = "\n".join(
        f"  {i+1}. {row.get('title','?')} — score={row.get('score',0):.3f}, "
        f"prix={row.get('price',0):.2f}$, rating={row.get('rating','N/A')}"
        for i, (_, row) in enumerate(top5.iterrows())
    )
    cat_counts = df["category"].value_counts().head(5).to_string() if "category" in df else "N/A"

    prompt = TREND_REPORT_PROMPT.format(
        stats_json=json.dumps(stats, ensure_ascii=False, indent=2),
        top5_text=top5_text,
        cat_counts=cat_counts,
    )
    result = client.chat(prompt, system=BASE_SYSTEM, max_tokens=700)
    logger.info("Trend report generated")
    return result


# ══════════════════════════════════════════════════════════════════════
# Chain 3 — Customer profile
# ══════════════════════════════════════════════════════════════════════

CLIENT_PROFILE_PROMPT = """Sur la base des produits les plus populaires suivants, crée un profil client cible.

Produits consultés (Top-K) :
{products_text}

Catégories dominantes : {top_categories}
Gamme de prix : {price_min}$ – {price_max}$
Note moyenne : {avg_rating}/5

Étape 1 — Déduis les préférences et comportements d'achat typiques.
Étape 2 — Identifie le segment démographique le plus probable.
Étape 3 — Rédige un profil client détaillé incluant :
  - Persona (âge, style de vie, valeurs)
  - Motivations d'achat
  - Sensibilité prix / qualité
  - Canaux de communication préférés
  - Message marketing recommandé"""


def build_client_profile(client: LLMClient, top_k: pd.DataFrame) -> str:
    """Build a target customer persona from the Top-K products."""
    products_text = "\n".join(
        f"  - {row.get('title','?')} ({row.get('category','?')}, "
        f"{row.get('price',0):.0f}$, ★{row.get('rating','?')})"
        for _, row in top_k.head(15).iterrows()
    )
    top_categories = (
        top_k["category"].value_counts().head(3).index.tolist()
        if "category" in top_k.columns else []
    )
    prompt = CLIENT_PROFILE_PROMPT.format(
        products_text=products_text,
        top_categories=", ".join(str(c) for c in top_categories),
        price_min=round(float(top_k["price"].min()), 2) if "price" in top_k else 0,
        price_max=round(float(top_k["price"].max()), 2) if "price" in top_k else 0,
        avg_rating=round(float(top_k["rating"].mean()), 1) if "rating" in top_k else 0,
    )
    result = client.chat(prompt, system=BASE_SYSTEM, max_tokens=600)
    logger.info("Client profile generated")
    return result


# ══════════════════════════════════════════════════════════════════════
# Chain 4 — Marketing strategy recommendation
# ══════════════════════════════════════════════════════════════════════

STRATEGY_PROMPT = """Tu es un consultant en stratégie marketing e-commerce.
Voici l'analyse du marché scrapé :

Segments identifiés (clustering) :
{cluster_summary}

Règles d'association découvertes (top 5) :
{rules_text}

Top produits par catégorie :
{top_by_cat}

Étape 1 — Identifie les opportunités de cross-selling et up-selling.
Étape 2 — Analyse la compétitivité prix par segment.
Étape 3 — Propose un plan marketing actionnable avec :
  - 3 actions immédiates (quick wins)
  - 2 initiatives moyen terme
  - 1 recommandation stratégique long terme
  - KPIs à surveiller"""


def recommend_strategy(client: LLMClient, df: pd.DataFrame,
                       rules_df: Optional[pd.DataFrame] = None) -> str:
    """Generate actionable marketing strategy recommendations."""
    # Cluster summary
    if "cluster_label" in df.columns:
        cluster_summary = df.groupby("cluster_label").agg(
            count=("title","count"),
            avg_price=("price","mean"),
            avg_score=("score","mean") if "score" in df else ("price","count"),
        ).round(2).to_string()
    else:
        cluster_summary = "Clustering non disponible"

    # Rules text
    if rules_df is not None and not rules_df.empty:
        rules_text = "\n".join(
            f"  {{{r['antecedents']}}} → {{{r['consequents']}}} "
            f"(lift={r['lift']:.2f}, conf={r['confidence']:.2f})"
            for _, r in rules_df.head(5).iterrows()
        )
    else:
        rules_text = "Règles d'association non disponibles"

    # Top product per category
    if "category" in df.columns and "score" in df.columns:
        idx = df.groupby("category")["score"].idxmax()
        top_by_cat = (
            df.loc[idx, ["category","title","price","score"]]
              .head(6).to_string(index=False)
        )
    else:
        top_by_cat = "N/A"

    prompt = STRATEGY_PROMPT.format(
        cluster_summary=cluster_summary,
        rules_text=rules_text,
        top_by_cat=top_by_cat,
    )
    result = client.chat(prompt, system=BASE_SYSTEM, max_tokens=800)
    logger.info("Marketing strategy generated")
    return result