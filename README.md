# Smart eCommerce Intelligence

**| DM & SID | 2025/2026**

Système intelligent et automatisé pour l'analyse de produits e-commerce via scraping, Machine Learning, pipelines MLOps, visualisation BI, et enrichissement LLM.

---
# 2. Lancer
python LLM/enrichment_pipeline.py \
  --input TopKselection/output/products_scored.csv \
  --output LLM/output \
  --max 50

## Structure du projet

```
smart_ecommerce/
│
├── agents/                         # Scraping  — Agents A2A
│   ├── base_agent.py               #   Classe abstraite + dataclass Product (31 champs)
│   ├── shopify_agent.py            #   Agent Shopify (API /products.json + HTML fallback)
│   ├── woocommerce_agent.py        #   Agent WooCommerce (REST API + Selenium fallback)
│   └── generic_agent.py            #   Agent générique (BeautifulSoup heuristique)
├── orchestrator.py                 #   Orchestrateur A2A (détection, routing, dédup, export)
│
├── TopKselection/                  # TopKselection — Analyse ML & Top-K
│   ├── preprocessing.py            #   Nettoyage, feature engineering, normalisation, split
│   ├── scoring.py                  #   Score composite pondéré + sélection Top-K
│   ├── supervised.py               #   Random Forest + XGBoost (CV, F1, matrice confusion)
│   ├── clustering.py               #   KMeans + DBSCAN + Hiérarchique + PCA 2D
│   ├── association_rules.py        #   Apriori / FP-Growth (support, confidence, lift)
│   └── pipeline.py                 #   Point d'entrée unique : prétraitement→scoring→ML→règles
│
├── KubeflowPipelines/                        # KubeflowPipelines — Kubeflow Pipelines & CI/CD
│   ├── components/                 #   5 composants kfp (@dsl.component)
│   ├── pipeline/                   #   DAG du pipeline + CLI (--local / --submit / --compile)
│   ├── docker/                     #   Dockerfile.scraping + Dockerfile.ml + docker-compose
│   └── .github/workflows/          #   CI/CD GitHub Actions (test → build → deploy)
│
├── DashboardBI/                    # DashboardBI — Dashboard BI (Streamlit)
│   ├── app.py                      #   6 pages : Overview, Top-K, Clusters, Prix, Règles, Shops
│   └── data_loader.py              #   Chargement des outputs ML + données synthétiques demo
│
├── LLM/                            #  LLM  — MCP
│   ├── llm_client.py               #   Client LLM unifié (Anthropic / OpenAI / Mock)
│   ├── chains.py                   #   4 chaînes CoT : résumé, rapport, profil client, stratégie
│   ├── chatbot_page.py             #   Interface conversationnelle Streamlit
│   ├── enrichment_pipeline.py      #   Pipeline d'enrichissement batch
│   └── mcp/                        #   Architecture MCP responsable
│       ├── mcp_client.py           #     Client MCP (gateway unique, permission + rate-limit)
│       ├── mcp_server_data.py      #     Serveur Data (4 outils lecture seule)
│       ├── mcp_server_llm.py       #     Serveur LLM (5 outils d'enrichissement)
│       └── mcp_server_audit.py     #     Serveur Audit (logs, permissions, rate limiting)
│
└── tests/                          # Tests unitaires Scraping & TopKselection
```

---

## Démarrage rapide

### 1. Installation

```bash
git clone https://github.com/imanelujen/smart_ecommerceProject.git
cd smart_ecommerceProject

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # Remplir les clés API
```

### 2. Scraping (Scraping)

```bash
python Scraping/orchestrator.py --urls https://store.myshopify.com https://store2.com
# → Scraping/data/products_history.csv
```

### 3. Analyse ML (TopKselection)

```bash
python TopKselection/pipeline.py --csv Scraping/data/products_history.csv --k 50
# → TopKselection/output/top_k_products.csv + models/
```

### 4. Dashboard BI (DashboardBI)

```bash
pip install streamlit plotly
streamlit run DashboardBI/app.py
# → http://localhost:8501
```

### 5. Pipeline Kubeflow local (KubeflowPipelines)

```bash
pip install kfp
python KubeflowPipelines/pipeline/smart_ecommerce_pipeline.py --local --urls https://store.myshopify.com
```

### 6. Enrichissement LLM (LLM)

```bash
# Avec Claude (Anthropic)
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python LLM/enrichment_pipeline.py --input TopKselection/output/products_scored.csv

# Sans clé API (mode mock pour tests)
python LLM/enrichment_pipeline.py
```

---

## Variables d'environnement (.env)

| Variable | Description |
|----------|-------------|
| `WC_CONSUMER_KEY` | Clé API WooCommerce |
| `WC_CONSUMER_SECRET` | Secret API WooCommerce |
| `SHOPIFY_STOREFRONT_TOKEN` | Token Shopify (optionnel) |
| `LLM_PROVIDER` | `anthropic` / `openai` / `mock` |
| `ANTHROPIC_API_KEY` | Clé API Anthropic Claude |
| `OPENAI_API_KEY` | Clé API OpenAI GPT |

---

## Dataset produit (31 colonnes)

| Groupe | Colonnes |
|--------|----------|
| Descriptives | `product_id`, `title`, `category`, `subcategory`, `brand`, `tags`, `url`, `platform` |
| Prix | `price`, `price_promo`, `price_old`, `discount_pct`, `currency` |
| Popularité | `rating`, `review_count`, `category_rank` |
| Stock | `availability`, `stock_quantity`, `delivery_days` |
| Variantes | `variant_count`, `colors`, `sizes` |
| Vendeur | `shop_name`, `shop_country`, `shop_product_count` |
| Marketing | `related_products` |
| Temporel | `published_at`, `scraped_at` |
| Textuelles | `description`, `customer_reviews` |

---

## Livrables produits

| Livrable | Fichier |
|----------|---------|
| Code agents A2A | `agents/` + `orchestrator.py` |
| Pipeline Kubeflow (YAML) | `KubeflowPipelines/pipeline/smart_ecommerce_pipeline.yaml` |
| Tableau Top-K + Dashboard BI | `TopKselection/output/top_k_products.csv` + `DashboardBI/app.py` |
| LLM enrichissement | `LLM/enrichment_pipeline.py` |
| Rapport technique | `rapport_technique.docx` |