#!/usr/bin/env python3
"""
Main estimation pipeline: fixed-effects panel, event-study leads/lags, and
heterogeneity (DiD-style interactions).

Run from project root:
  .venv/bin/python src/run_analysis.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"


def load_and_prepare() -> pd.DataFrame:
    train_path = DATA_RAW / "train.csv"
    store_path = DATA_RAW / "store.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path}. Run: python scripts/download_data.py"
        )

    train = pd.read_csv(train_path, low_memory=False, parse_dates=["Date"])
    train["StateHoliday"] = train["StateHoliday"].astype(str).str.strip()
    store = pd.read_csv(store_path)

    train = train.loc[train["Open"] == 1].copy()
    train = train.loc[train["Sales"] > 0].copy()

    store["CompetitionDistance"] = pd.to_numeric(
        store["CompetitionDistance"], errors="coerce"
    )
    med_cd = store["CompetitionDistance"].median()
    store["CompetitionDistance"] = store["CompetitionDistance"].fillna(med_cd)
    store["high_comp"] = (
        store["CompetitionDistance"] >= store["CompetitionDistance"].median()
    ).astype(int)
    store["Promo2"] = pd.to_numeric(store["Promo2"], errors="coerce").fillna(0).astype(int)

    df = train.merge(
        store[["Store", "high_comp", "Promo2", "StoreType", "Assortment"]],
        on="Store",
        how="left",
        validate="many_to_one",
    )

    df["log_sales"] = np.log(df["Sales"].astype(float))
    df["year_week"] = df["Date"].dt.strftime("%G-%V")
    df["dow"] = df["Date"].dt.dayofweek
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


def add_promo_calendar_leads_lags(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """National promo schedule: same Promo for all stores on a given date."""
    cal = df[["Date", "Promo"]].drop_duplicates("Date").sort_values("Date")
    cal = cal.set_index("Date")["Promo"]

    out = df.copy()
    for i in range(1, k + 1):
        fwd = cal.shift(-i)
        back = cal.shift(i)
        out[f"promo_lead{i}"] = out["Date"].map(fwd)
        out[f"promo_lag{i}"] = out["Date"].map(back)
    out[[c for c in out.columns if c.startswith("promo_lead")]] = out[
        [c for c in out.columns if c.startswith("promo_lead")]
    ].fillna(0)
    out[[c for c in out.columns if c.startswith("promo_lag")]] = out[
        [c for c in out.columns if c.startswith("promo_lag")]
    ].fillna(0)
    return out


def fit_panel(
    df: pd.DataFrame,
    formula: str,
) -> PanelOLS:
    panel = df.set_index(["Store", "Date"])
    other = pd.DataFrame({"year_week": panel["year_week"]}, index=panel.index)
    return PanelOLS.from_formula(
        formula,
        data=panel,
        other_effects=other,
    )


def figure_event_study(coefs: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = coefs["k"].values
    ax.axhline(0, color="0.4", lw=1)
    ax.axvline(-0.5, color="0.75", ls="--", lw=1)
    ax.errorbar(
        x,
        coefs["coef"].values,
        yerr=1.96 * coefs["se"].values,
        fmt="o",
        color="#1f77b4",
        capsize=3,
    )
    ax.set_xlabel(
        "k (lead k: association with promo k days ahead; k=0 is today)"
    )
    ax.set_ylabel("Coefficient (log sales)")
    ax.set_title(
        "Leads/lags of the national promo calendar (±3 days; same FE as main spec)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_calendar(df: pd.DataFrame, path: Path) -> None:
    daily = df.groupby("Date", as_index=False)["Promo"].mean()
    daily["ma7"] = daily["Promo"].rolling(7, min_periods=1).mean()
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily["Date"], daily["ma7"], color="#2ca02c", lw=1.2)
    ax.set_ylabel("Promo (7-day mean)")
    ax.set_xlabel("Date")
    ax.set_title("National promo calendar (all stores share the same daily promo flag)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_heterogeneity(rows: list[dict], path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["name"] for r in rows]
    coefs = [r["coef"] for r in rows]
    ses = [r["se"] for r in rows]
    y = np.arange(len(names))
    ax.axvline(0, color="0.5", lw=1)
    ax.errorbar(coefs, y, xerr=[1.96 * s for s in ses], fmt="o", capsize=4, color="#d62728")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Coefficient (log sales)")
    ax.set_title("Heterogeneity & DiD-style interactions (clustered SE at store)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare()
    by_date = df.groupby("Date")["Promo"].nunique()
    assert (by_date <= 1).all(), "Unexpected: Promo varies across stores on the same date."
    figure_calendar(df, FIG_DIR / "fig_promo_calendar.png")

    mod = fit_panel(
        df,
        "log_sales ~ Promo + C(dow) + C(StateHoliday) + SchoolHoliday + EntityEffects",
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    rows_main = [
        {
            "model": "main_twfe",
            "coef": float(res.params["Promo"]),
            "se": float(res.std_errors["Promo"]),
            "pvalue": float(res.pvalues["Promo"]),
            "n": int(res.nobs),
            "r2_within": float(res.rsquared_within),
            "r2_overall": float(res.rsquared),
        }
    ]
    pd.DataFrame(rows_main).to_csv(TABLE_DIR / "main_twfe.csv", index=False)

    with_cont = fit_panel(
        df,
        "log_sales ~ Promo + log(Customers) + C(dow) + C(StateHoliday) + SchoolHoliday + EntityEffects",
    ).fit(cov_type="clustered", cluster_entity=True)
    df_customers = pd.DataFrame(
        [
            {
                "model": "with_log_customers",
                "coef_promo": float(with_cont.params["Promo"]),
                "se_promo": float(with_cont.std_errors["Promo"]),
                "note": "Customers may absorb part of the promo channel; interpret cautiously.",
            }
        ]
    )
    df_customers.to_csv(TABLE_DIR / "robustness_customers.csv", index=False)

    ev_k = 3
    df_ev = add_promo_calendar_leads_lags(df, k=ev_k)
    lead_cols = [f"promo_lead{i}" for i in range(1, ev_k + 1)]
    lag_cols = [f"promo_lag{i}" for i in range(1, ev_k + 1)]
    ev_formula = (
        "log_sales ~ "
        + " + ".join(lead_cols + ["Promo"] + lag_cols)
        + " + C(dow) + C(StateHoliday) + SchoolHoliday + EntityEffects"
    )
    ev_res = fit_panel(df_ev, ev_formula).fit(
        cov_type="clustered", cluster_entity=True
    )

    ev_rows = []
    for i in range(1, ev_k + 1):
        c = f"promo_lead{i}"
        ev_rows.append(
            {"k": -i, "coef": float(ev_res.params[c]), "se": float(ev_res.std_errors[c])}
        )
    ev_rows.append(
        {
            "k": 0,
            "coef": float(ev_res.params["Promo"]),
            "se": float(ev_res.std_errors["Promo"]),
        }
    )
    for i in range(1, ev_k + 1):
        c = f"promo_lag{i}"
        ev_rows.append(
            {"k": i, "coef": float(ev_res.params[c]), "se": float(ev_res.std_errors[c])}
        )
    df_ev_coef = pd.DataFrame(ev_rows).sort_values("k")
    df_ev_coef.to_csv(TABLE_DIR / "event_study_leads_lags.csv", index=False)
    figure_event_study(df_ev_coef, FIG_DIR / "fig_event_study.png")

    # Only interactions (not high_comp / Promo2 mains): those are absorbed by store FE.
    het_formula = (
        "log_sales ~ Promo + Promo:high_comp + Promo:Promo2 + C(dow) + C(StateHoliday) + SchoolHoliday + EntityEffects"
    )
    het = fit_panel(df, het_formula).fit(cov_type="clustered", cluster_entity=True)

    het_out = []
    for name in ["Promo", "Promo:high_comp", "Promo:Promo2"]:
        het_out.append(
            {
                "term": name,
                "coef": float(het.params[name]),
                "se": float(het.std_errors[name]),
                "pvalue": float(het.pvalues[name]),
            }
        )
    pd.DataFrame(het_out).to_csv(TABLE_DIR / "heterogeneity_interactions.csv", index=False)

    hr = [
        {
            "name": "Promo (baseline; low-competition ref)",
            "coef": float(het.params["Promo"]),
            "se": float(het.std_errors["Promo"]),
        },
        {
            "name": "Additional promo effect in high-competition markets",
            "coef": float(het.params["Promo:high_comp"]),
            "se": float(het.std_errors["Promo:high_comp"]),
        },
        {
            "name": "Additional effect for Promo2 (recurring promo) stores",
            "coef": float(het.params["Promo:Promo2"]),
            "se": float(het.std_errors["Promo:Promo2"]),
        },
    ]
    figure_heterogeneity(hr, FIG_DIR / "fig_heterogeneity.png")

    print("Main TWFE (store + ISO-week FE, clustered by store):")
    print(res.summary.tables[1])
    print("\nSaved figures to", FIG_DIR)
    print("Saved tables to", TABLE_DIR)


if __name__ == "__main__":
    main()
