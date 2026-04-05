# Estimating the Incremental Impact of Retail Promotions Using Quasi-Experimental Methods

**Data:** [Rossmann Store Sales](https://www.kaggle.com/datasets/shahpranshu27/rossman-store-sales) (daily store-level panel, 2013–2015).

**Question:** Do promotions move incremental sales, or are they mostly aligned with periods that would have been strong anyway?

---

## TL;DR

- **What I did:** Built a **store × day** panel, merged store metadata, and estimated **how log daily sales move on promotion days** while stripping out slow-moving store differences and common calendar shocks.
- **What I found (Rossmann):** In the main specification, promotion days are associated with roughly **0.31 log points** higher sales (on the order of **one-third** higher volume in levels under a log link), after controlling for store fixed effects, **ISO week** fixed effects, day-of-week, and holiday indicators—with **standard errors clustered by store** (within-R² about **0.43** on the estimation sample).
- **Caveat:** In this dataset the **promo flag is national**—every store shares the same value on a given calendar day—so you **cannot** add a full **calendar-date** fixed effect and still estimate a standalone promo coefficient. Identification leans on **variation of promo within store-week** (promo on vs off days in the same week, for the same store) plus **heterogeneity** across stores. That rules out a naive “date FE + promo” story and is called out explicitly below.

---

## Why this is not just correlation (and what still worries me)

**What the fixed-effects design buys you**

- **Store fixed effects** difference out time-invariant store traits (format, baseline traffic, region).
- **ISO week (`%G-%V`) fixed effects** absorb shared weekly seasonality and macro shocks that hit all stores in the same week.
- **Day-of-week** and **holiday** controls address predictable calendar structure.

Together, the estimand is closer to “within the same store, comparing promo and non-promo days in the same ISO week,” rather than a raw pre/post on the calendar.

**Identification wrinkle (important)**

Exploratory checks show **promo is identical across stores on each date**. That makes **date fixed effects** and **promo** collinear in a model with both. The main specification therefore uses **week** fixed effects—not **day** fixed effects—so promo remains identified off **within-week** switches. That is a substantive modeling choice, not a detail to hide.

**Remaining threats**

- **Selection inside the week:** If HQ schedules promos on the highest-residual days *within* a week, our week FE do not remove that.
- **Demand controls:** `Customers` is downstream of traffic and promo; a robustness spec adds `log(Customers)` but it should be read as a **sensitivity check**, not a causal “closing the backdoor” claim.

---

## Methods (what is implemented)

| Piece | Role |
|--------|------|
| **Two-way FE (store + ISO week)** | Primary “panel with rich FE” specification; SE clustered at store |
| **Diff-in-diff style interactions** | `Promo × high_competition` and `Promo × Promo2` (store-level moderators); mains for time-invariant moderators are absorbed by store FE—only interactions are estimable |
| **Event-style leads/lags** | National calendar leads/lags of the promo indicator (±3 days in the shipped code) with the same FE stack. Coefficients are **not** the same object as the single `Promo` term in the main equation (neighboring days partition overlapping variation); treat as a **diagnostic**, not a second point estimate of the same parameter. |

---

## How to run

```bash
cd promotion-incrementality-causal-inference
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
python src/run_analysis.py
```

Optional: open `notebooks/promotion_incrementality_analysis.ipynb` for a **step-by-step** walkthrough (EDA, identification, FE, event study, heterogeneity). Run all cells after `python scripts/download_data.py`.

---

## Outputs (for reviewers)

| Path | Contents |
|------|-----------|
| `reports/figures/fig_promo_calendar.png` | How often promo is “on” over time (national series) |
| `reports/figures/fig_event_study.png` | Leads/lags of the promo calendar (±3 days) |
| `reports/figures/fig_heterogeneity.png` | Interaction coefficients (competition + Promo2) |
| `reports/tables/main_twfe.csv` | Main promo coefficient |
| `reports/tables/event_study_leads_lags.csv` | Event-study coefficients |
| `reports/tables/heterogeneity_interactions.csv` | Interaction terms |
| `reports/tables/robustness_customers.csv` | Promo with `log(Customers)` control |

---

## Repo layout

```
data/raw/          # train.csv, store.csv (downloaded, not required in git)
notebooks/         # promotion_incrementality_analysis.ipynb (narrative analysis)
reports/figures/   # Plots for README / portfolio
reports/tables/    # CSV results
scripts/           # download_data.py
src/run_analysis.py
```

---

## Tools

Python 3.9+; **linearmodels** (`PanelOLS`) for fixed effects; **pandas** for panel prep; **matplotlib** / **seaborn** for plots.

---

## License

Dataset use is subject to Kaggle’s terms. Analysis code in this repository: use freely with attribution.
