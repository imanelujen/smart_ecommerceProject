# Module 1 — Web Scraping with A2A Agents

## Project structure

```
smart_ecommerce/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Abstract base + Product dataclass
│   ├── shopify_agent.py       # Shopify: /products.json API + HTML fallback
│   ├── woocommerce_agent.py   # WooCommerce: REST API + Selenium fallback
│   └── generic_agent.py       # Generic: BeautifulSoup heuristic fallback
├── tests/
│   └── test_agents.py         # Full pytest suite
├── data/                      # Auto-created — scraped output lands here
├── orchestrator.py            # A2A coordinator (main entry point)
├── requirements.txt
├── .env.example               # Copy to .env and add credentials
└── README.md
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env → add WC_CONSUMER_KEY / WC_CONSUMER_SECRET if needed

# 4. (Optional) Install Playwright browsers for JS-heavy sites
playwright install chromium
```

## Usage

### Run the orchestrator from the command line

```bash
# Scrape a Shopify store
python orchestrator.py --urls https://hydrogen-demo-store.myshopify.com

# Scrape multiple stores at once
python orchestrator.py \
  --urls https://store1.myshopify.com https://store2.myshopify.com

# Specify output directory
python orchestrator.py --urls https://mystore.com --output my_data/
```

### Use the API in Python

```python
from orchestrator import Orchestrator

orch = Orchestrator(output_dir="data")
df = orch.run(urls=[
    "https://hydrogen-demo-store.myshopify.com",
    "https://demo.woothemes.com",
])

print(df[["title", "price", "platform", "availability"]].head())
```

### Use a single agent directly

```python
from agents import ShopifyAgent

agent = ShopifyAgent()
products = agent.run("https://hydrogen-demo-store.myshopify.com")

for p in products[:5]:
    print(f"{p.title:40s}  ${p.price:.2f}  ({p.platform})")
```

## Output files

After each run, `data/` contains:

| File | Description |
|------|-------------|
| `products_YYYYMMDD_HHMMSS.csv` | Timestamped CSV — one product per row |
| `products_YYYYMMDD_HHMMSS.jsonl` | JSONL for LLM enrichment (Module 5) |
| `products_latest.csv` | Always points to the most recent run |

### CSV columns

| Column | Type | Notes |
|--------|------|-------|
| `title` | str | Product name |
| `price` | float | Numeric, in `currency` |
| `currency` | str | ISO code (e.g. USD) |
| `url` | str | Product page URL |
| `platform` | str | shopify / woocommerce / generic |
| `availability` | bool | In-stock status |
| `rating` | float | 0–5 or null |
| `review_count` | int | Number of reviews |
| `description` | str | Cleaned text (HTML stripped) |
| `category` | str | Product category |
| `vendor` | str | Brand / vendor name |
| `image_url` | str | Main product image |
| `scraped_at` | datetime | UTC timestamp |
| `score` | float | Placeholder — filled by Module 2 |

## Run the tests

```bash
pytest tests/ -v
```

## How the A2A pattern works

```
Orchestrator
  ├─ detect(url) → tries ShopifyAgent first, then WooCommerce, then Generic
  ├─ agent.scrape(url) → platform-specific data fetch (API or HTML)
  ├─ agent.clean(raw) → normalise to Product dataclass
  └─ deduplicate + save → data/products_*.csv
```

Each agent is autonomous and interchangeable. To add a new platform:

1. Create `agents/myplatform_agent.py` extending `BaseAgent`
2. Implement `detect()`, `scrape()`, `clean()`
3. Register it in `Orchestrator.__init__` before `GenericHTMLAgent`

## Module handoff → Module 2 (ML Analysis)

The `products_latest.csv` file is the direct input to Module 2.
The `score` column (currently 0.0) will be populated by the
Top-K selection algorithm.