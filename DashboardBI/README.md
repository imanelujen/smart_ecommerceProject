# DashboardBI — BI Dashboard (Streamlit)

## Run locally

```bash
pip install streamlit plotly pandas numpy
streamlit run DashboardBI/app.py
# → http://localhost:8501
```

## Pages

| Page | Content |
|------|---------|
| Overview | KPI cards, category bar chart, platform pie, score histogram |
| Top-K Products | Price vs rating scatter, ranked table, CSV download |
| Clusters | PCA 2D scatter by cluster, cluster profile table |
| Pricing Analysis | Price boxplots, discount histogram, price vs score scatter |
| Association Rules | Support/confidence/lift scatter, rules table |
| Shops & Geography | Shop leaderboard, country breakdown, avg score by country |

## Filters (sidebar)

- Category, Platform, Price range — applied to all pages globally.

## Data source

Reads from `TopKselection/output/` automatically.
Falls back to **synthetic demo data** if outputs are not yet generated —
so the dashboard runs out of the box with no ML pipeline needed.