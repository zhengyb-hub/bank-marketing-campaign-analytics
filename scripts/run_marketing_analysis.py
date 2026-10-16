from __future__ import annotations

import os
import sqlite3
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = PROJECT_ROOT / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from build_dashboard import build_dashboard


RAW_CSV = PROJECT_ROOT / "data" / "raw" / "bank-additional" / "bank-additional" / "bank-additional-full.csv"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
REPORTS_DIR = PROJECT_ROOT / "reports"
PORTFOLIO_DIR = PROJECT_ROOT / "portfolio"
SQL_DIR = PROJECT_ROOT / "sql"

CONTACT_COST = 3.0
CONVERSION_VALUE = 650.0
PRE_OUTREACH_FEATURES = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "was_previously_contacted", "days_since_previous_contact", "previous", "poutcome", "emp_var_rate", "cons_price_idx",
    "cons_conf_idx", "euribor3m", "nr_employed",
]


def money(value: float) -> str:
    return f"EUR {value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def wrap(text: str, width: int = 88) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    rows = []
    columns = list(frame.columns)
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def load_and_clean() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_CSV}")
    data = pd.read_csv(RAW_CSV, sep=";")
    data.columns = [column.replace(".", "_") for column in data.columns]
    data.insert(0, "contact_id", np.arange(1, len(data) + 1))
    data["converted"] = (data["y"] == "yes").astype(int)
    data["was_previously_contacted"] = (data["pdays"] != 999).astype(int)
    data["days_since_previous_contact"] = data["pdays"].where(data["pdays"] != 999, 0)
    data["age_band"] = pd.cut(
        data["age"],
        bins=[0, 24, 34, 44, 54, 64, 120],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        right=True,
    ).astype(str)
    data["campaign_bucket"] = pd.cut(
        data["campaign"],
        bins=[0, 1, 2, 3, 5, 99],
        labels=["1 touch", "2 touches", "3 touches", "4-5 touches", "6+ touches"],
        right=True,
    ).astype(str)
    data["duration_minutes"] = data["duration"] / 60
    data["month_order"] = data["month"].map(
        {"mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    )
    return data


def write_sqlite(data: pd.DataFrame) -> None:
    with sqlite3.connect(DATA_DIR / "bank_marketing.sqlite") as connection:
        data.to_sql("campaign_contacts", connection, if_exists="replace", index=False)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_contacts_job ON campaign_contacts(job)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_contacts_month ON campaign_contacts(month)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_contacts_converted ON campaign_contacts(converted)")


def summarize_segment(data: pd.DataFrame, field: str) -> pd.DataFrame:
    summary = (
        data.groupby(field, dropna=False)
        .agg(
            contacts=("contact_id", "count"),
            conversions=("converted", "sum"),
            avg_touches=("campaign", "mean"),
            avg_duration_min=("duration_minutes", "mean"),
        )
        .reset_index()
    )
    summary["conversion_rate"] = summary["conversions"] / summary["contacts"]
    summary["modeled_conversion_value"] = summary["conversions"] * CONVERSION_VALUE
    summary["modeled_historical_touch_cost"] = data.groupby(field, dropna=False)["campaign"].sum().values * CONTACT_COST
    summary["modeled_net_value"] = summary["modeled_conversion_value"] - summary["modeled_historical_touch_cost"]
    return summary.sort_values(["conversion_rate", "contacts"], ascending=[False, False])


def chronological_split(data: pd.DataFrame, train_share: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve the UCI full file's documented date ordering without shuffling."""
    if not 0 < train_share < 1:
        raise ValueError("train_share must lie between zero and one")
    split = int(len(data) * train_share)
    if split < 1 or split >= len(data):
        raise ValueError("not enough rows for both train and holdout")
    return data.iloc[:split], data.iloc[split:]


def build_propensity_model(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame]:
    # Only fields available before a new outreach decision. Current-campaign
    # touch count, call duration, channel and scheduled day/month are excluded.
    features = PRE_OUTREACH_FEATURES
    target = "converted"
    categorical = [column for column in features if not pd.api.types.is_numeric_dtype(data[column])]
    numeric = [column for column in features if column not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=25), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    # UCI documents bank-additional-full.csv as ordered by contact date.
    # Keep the last quarter untouched to test forward-in-time generalization.
    train, test = chronological_split(data)
    model.fit(train[features], train[target])
    test_scores = model.predict_proba(test[features])[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(test[target], test_scores),
        "average_precision": average_precision_score(test[target], test_scores),
        "brier_score": brier_score_loss(test[target], test_scores),
        "holdout_base_rate": float(test[target].mean()),
        "train_rows": len(train),
        "holdout_rows": len(test),
    }

    scored = test.copy()
    scored["propensity_score"] = test_scores
    scored["propensity_decile"] = np.ceil(
        scored["propensity_score"].rank(method="first", ascending=False) / len(scored) * 10
    ).astype(int)

    deciles = (
        scored.groupby("propensity_decile")
        .agg(
            contacts=("contact_id", "count"),
            conversions=("converted", "sum"),
            avg_score=("propensity_score", "mean"),
            avg_touches=("campaign", "mean"),
        )
        .reset_index()
        .sort_values("propensity_decile")
    )
    deciles["conversion_rate"] = deciles["conversions"] / deciles["contacts"]
    deciles["modeled_contact_cost"] = deciles["contacts"] * CONTACT_COST
    deciles["modeled_conversion_value"] = deciles["conversions"] * CONVERSION_VALUE
    deciles["modeled_net_value"] = deciles["modeled_conversion_value"] - deciles["modeled_contact_cost"]
    deciles["value_to_cost_ratio"] = deciles["modeled_conversion_value"] / deciles["modeled_contact_cost"]
    deciles["cumulative_contacts"] = deciles["contacts"].cumsum()
    deciles["cumulative_conversions"] = deciles["conversions"].cumsum()
    deciles["cumulative_contact_share"] = deciles["cumulative_contacts"] / len(scored)
    deciles["cumulative_conversion_capture"] = deciles["cumulative_conversions"] / scored[target].sum()
    deciles["cumulative_lift"] = deciles["cumulative_conversion_capture"] / deciles["cumulative_contact_share"]

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefs = model.named_steps["model"].coef_[0]
    drivers = (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        .assign(abs_coefficient=lambda frame: frame["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .head(25)
    )
    drivers["direction"] = np.where(drivers["coefficient"] > 0, "positive", "negative")
    return scored, deciles, metrics, drivers


def write_analysis_outputs(data: pd.DataFrame) -> dict[str, pd.DataFrame | float]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_sqlite(data)
    data.to_csv(DATA_DIR / "bank_marketing_clean.csv", index=False)

    overview = pd.DataFrame(
        [
            {
                "contacts": len(data),
                "conversions": int(data["converted"].sum()),
                "conversion_rate": data["converted"].mean(),
                "avg_touches": data["campaign"].mean(),
                "contact_cost_assumption": CONTACT_COST,
                "conversion_value_assumption": CONVERSION_VALUE,
                "modeled_conversion_value": data["converted"].sum() * CONVERSION_VALUE,
                "modeled_historical_touch_cost": data["campaign"].sum() * CONTACT_COST,
            }
        ]
    )
    overview["modeled_net_value"] = overview["modeled_conversion_value"] - overview["modeled_historical_touch_cost"]

    by_job = summarize_segment(data, "job")
    by_education = summarize_segment(data, "education")
    by_contact = summarize_segment(data, "contact")
    by_month = summarize_segment(data, "month").merge(
        data[["month", "month_order"]].drop_duplicates(), on="month", how="left"
    ).sort_values("month_order")
    by_campaign = summarize_segment(data, "campaign_bucket")
    by_poutcome = summarize_segment(data, "poutcome")
    scored, deciles, model_metrics, drivers = build_propensity_model(data)

    overview.to_csv(DATA_DIR / "overview_metrics.csv", index=False)
    by_job.to_csv(DATA_DIR / "conversion_by_job.csv", index=False)
    by_education.to_csv(DATA_DIR / "conversion_by_education.csv", index=False)
    by_contact.to_csv(DATA_DIR / "conversion_by_contact_method.csv", index=False)
    by_month.to_csv(DATA_DIR / "conversion_by_month.csv", index=False)
    by_campaign.to_csv(DATA_DIR / "conversion_by_campaign_frequency.csv", index=False)
    by_poutcome.to_csv(DATA_DIR / "conversion_by_previous_outcome.csv", index=False)
    deciles.to_csv(DATA_DIR / "propensity_decile_holdout.csv", index=False)
    pd.DataFrame([model_metrics]).to_csv(DATA_DIR / "model_metrics.csv", index=False)
    drivers.to_csv(DATA_DIR / "model_drivers.csv", index=False)
    scored[["contact_id", "converted", "propensity_score", "propensity_decile"]].to_csv(
        DATA_DIR / "propensity_scores_sample.csv", index=False
    )

    return {
        "overview": overview,
        "by_job": by_job,
        "by_education": by_education,
        "by_contact": by_contact,
        "by_month": by_month,
        "by_campaign": by_campaign,
        "by_poutcome": by_poutcome,
        "deciles": deciles,
        "drivers": drivers,
        "model_metrics": model_metrics,
    }


def add_card(ax: plt.Axes, label: str, value: str, note: str) -> None:
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f7f9fc", edgecolor="#d9e0ec"))
    ax.text(0.06, 0.72, label.upper(), transform=ax.transAxes, fontsize=8.5, color="#5f6b7a", weight="bold")
    ax.text(0.06, 0.40, value, transform=ax.transAxes, fontsize=17, color="#172033", weight="bold")
    ax.text(0.06, 0.16, note, transform=ax.transAxes, fontsize=8, color="#667085")


def create_dashboard_pdf(outputs: dict[str, pd.DataFrame | float]) -> None:
    overview = outputs["overview"].iloc[0]
    by_month = outputs["by_month"]
    by_job = outputs["by_job"].head(8).sort_values("conversion_rate")
    by_campaign = outputs["by_campaign"]
    deciles = outputs["deciles"]

    fig = plt.figure(figsize=(16, 10.5), facecolor="white")
    fig.suptitle("Bank Marketing Campaign Dashboard Snapshot", fontsize=22, weight="bold", x=0.04, ha="left")
    fig.text(0.04, 0.94, "Historical response, pre-outreach modeling, and cost scenarios.", fontsize=11, color="#4f5b6b")
    grid = fig.add_gridspec(4, 4, left=0.04, right=0.98, top=0.89, bottom=0.07, wspace=0.35, hspace=0.6)

    add_card(fig.add_subplot(grid[0, 0]), "Records", f"{int(overview['contacts']):,}", "Campaign records")
    add_card(fig.add_subplot(grid[0, 1]), "Conversions", f"{int(overview['conversions']):,}", "Term deposit subscriptions")
    add_card(fig.add_subplot(grid[0, 2]), "Conversion rate", pct(float(overview["conversion_rate"])), "Overall response")
    add_card(fig.add_subplot(grid[0, 3]), "Model AUC", f"{float(outputs['model_metrics']['roc_auc']):.2f}", "Chronological holdout")

    ax1 = fig.add_subplot(grid[1:3, :2])
    ax1.bar(by_month["month"], by_month["conversion_rate"], color="#1f77b4")
    ax1.set_title("Response by Month of Year (Pooled)", loc="left", fontsize=13, weight="bold")
    ax1.set_ylabel("Conversion rate")
    ax1.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = fig.add_subplot(grid[1:3, 2:])
    ax2.barh(by_job["job"], by_job["conversion_rate"], color="#2ca02c")
    ax2.set_title("Top Job Segments by Conversion Rate", loc="left", fontsize=13, weight="bold")
    ax2.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax2.grid(axis="x", alpha=0.25)

    ax3 = fig.add_subplot(grid[3, :2])
    ax3.bar(by_campaign["campaign_bucket"], by_campaign["conversion_rate"], color="#ff7f0e")
    ax3.set_title("Conversion by Contact Frequency", loc="left", fontsize=13, weight="bold")
    ax3.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax3.tick_params(axis="x", labelrotation=15)
    ax3.grid(axis="y", alpha=0.25)

    ax4 = fig.add_subplot(grid[3, 2:])
    ax4.plot(deciles["propensity_decile"], deciles["cumulative_lift"], color="#d62728", marker="o", linewidth=2.4)
    ax4.set_title("Cumulative Lift on Future Holdout", loc="left", fontsize=13, weight="bold")
    ax4.set_xlabel("Decile, 1 = highest propensity")
    ax4.set_ylabel("Lift vs holdout base rate")
    ax4.grid(alpha=0.25)

    fig.savefig(PORTFOLIO_DIR / "marketing_dashboard_snapshot.pdf", bbox_inches="tight")
    plt.close(fig)


def create_one_pager(outputs: dict[str, pd.DataFrame | float]) -> None:
    overview = outputs["overview"].iloc[0]
    by_job = outputs["by_job"].head(1).iloc[0]
    by_contact = outputs["by_contact"].sort_values("conversion_rate", ascending=False).iloc[0]
    deciles = outputs["deciles"]
    top_decile = deciles.iloc[0]

    with PdfPages(PORTFOLIO_DIR / "marketing_project_one_pager.pdf") as pdf:
        fig = plt.figure(figsize=(8.5, 11), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.07, 0.95, "Bank Marketing Campaign Analytics", fontsize=22, weight="bold", color="#172033")
        ax.text(0.07, 0.925, "Marketing portfolio project | Campaign KPI, segmentation, model validation", fontsize=9.5, color="#536070")

        y = 0.88
        sections = [
            (
                "Business Question",
                "Can historical campaign data support reliable, cost-aware prioritization of future outreach?",
            ),
            (
                "Dataset and Scope",
                f"UCI Bank Marketing dataset with {int(overview['contacts']):,} campaign records and {int(overview['conversions']):,} subscriptions. Records are not verified unique clients.",
            ),
            (
                "Method",
                "Built campaign KPIs and a pre-outreach propensity model. Evaluated ranking on the latest 25% of date-ordered records; excluded current-campaign touches and call duration.",
            ),
            (
                "Key Findings",
                f"Historical conversion rate: {pct(float(overview['conversion_rate']))}. Future holdout AUC: {float(outputs['model_metrics']['roc_auc']):.2f}. Top holdout propensity decile: {pct(float(top_decile['conversion_rate']))} conversion versus {pct(float(outputs['model_metrics']['holdout_base_rate']))} holdout baseline.",
            ),
            (
                "Recommendation",
                "Do not deploy this score: chronological AUC is near chance and response by decile is inconsistent. Rebuild with newer data, validate stability, then test a revised rule against a randomized control.",
            ),
        ]
        for title, body in sections:
            ax.text(0.07, y, title, fontsize=13, weight="bold", color="#172033")
            y -= 0.033
            lines = wrap(body, 88)
            ax.text(0.07, y, lines, fontsize=9.6, color="#303846", va="top", linespacing=1.5)
            y -= 0.024 * (lines.count("\n") + 1) + 0.043

        ax.plot([0.07, 0.93], [0.09, 0.09], color="#d9e0ec", linewidth=1, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_autoscale_on(False)
        ax.text(0.07, 0.066, "Historical UCI sample | Scenario values are assumptions, not campaign results.", fontsize=8.2, color="#667085")
        pdf.savefig(fig)
        plt.close(fig)


def write_reports(outputs: dict[str, pd.DataFrame | float]) -> None:
    overview = outputs["overview"].iloc[0]
    by_job = outputs["by_job"].head(5)
    deciles = outputs["deciles"]
    drivers = outputs["drivers"].head(10)

    summary = f"""# Executive Summary

## Business Question

Can historical campaign data support reliable, cost-aware prioritization of future outreach?

## Dataset and grain

UCI Bank Marketing dataset with {int(overview['contacts']):,} campaign records and {int(overview['conversions']):,} subscriptions. There is no persistent client identifier, so records are not asserted to be unique clients. The full dataset is date-ordered from May 2008 to November 2010; the latest 25% is reserved as an untouched chronological holdout.

## Key Findings

- Overall conversion rate is {pct(float(overview['conversion_rate']))}.
- Mean recorded current-campaign touches per record is {float(overview['avg_touches']):.2f}.
- A hypothetical EUR {CONTACT_COST:.0f} per touch and EUR {CONVERSION_VALUE:.0f} per subscription imply {money(float(overview['modeled_net_value']))} modeled net value across historical records; these are not observed costs, revenue, profit, or incremental ROI.
- The top **holdout-only** propensity decile has {int(deciles.iloc[0]['contacts']):,} records and a {pct(float(deciles.iloc[0]['conversion_rate']))} observed subscription rate, versus {pct(float(outputs['model_metrics']['holdout_base_rate']))} across the holdout. Its cumulative lift is {float(deciles.iloc[0]['cumulative_lift']):.2f}x.
- Chronological holdout ROC AUC is {float(outputs['model_metrics']['roc_auc']):.2f}; average precision is {float(outputs['model_metrics']['average_precision']):.2f}; Brier score is {float(outputs['model_metrics']['brier_score']):.3f}.

## Recommended Actions

1. Do not deploy this score as an automated outreach rule: chronological holdout AUC is only {float(outputs['model_metrics']['roc_auc']):.2f} and decile response is not monotonic. Improve and revalidate the model on newer campaigns before any controlled pilot.
2. Test a contact-frequency cap rather than inferring a causal benefit from observational frequency groups.
3. Treat contact-method and demographic differences as descriptive, not evidence that switching channel or excluding a group will improve outcomes. Review consent, fairness, and applicable marketing rules before any targeting use.
4. If a revised rule proves stable, compare it with a randomized business-as-usual control for incremental subscriptions and contact cost. This historical dataset has no reliable client ID for deduplication and no actual cost or value fields.

## Top Converting Job Segments

{dataframe_to_markdown(by_job[['job', 'contacts', 'conversion_rate']])}

## Top Model Drivers

{dataframe_to_markdown(drivers[['feature', 'coefficient', 'direction']])}

Coefficients show model associations after preprocessing, not causal campaign drivers or a fairness approval.
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "executive_summary.md").write_text(summary, encoding="utf-8")

    resume = f"""# Resume Project Description

## Recommended Resume Bullet

Built a marketing analytics project on {int(overview['contacts']):,} historical campaign records, using SQL and Python to analyze response by segment and contact pattern; evaluated a leakage-aware propensity model on a chronological holdout, surfaced its generalization limits, and designed an assumption-based outreach scenario.

## Short Version

Bank Marketing Campaign Analytics: analyzed {int(overview['contacts']):,} campaign records with SQL, Python, and Plotly; identified an out-of-time model performance gap and modeled outreach trade-offs under explicit assumptions.

## Interview Talking Points

- Translated raw campaign records into response KPIs and clearly separated observed conversion from hypothetical cost/value scenarios.
- Excluded current-campaign touch count, call duration, and last-contact channel/time from the pre-outreach model.
- Compared broad outreach with holdout-only propensity-decile selection, documented non-monotonic response and weak out-of-time AUC, and avoided a deployment claim.
- Delivered dashboard, SQL bundle, one-page PDF, and resume-ready project description.
"""
    (REPORTS_DIR / "resume_project_description.md").write_text(resume, encoding="utf-8")
    (PORTFOLIO_DIR / "resume_project_description.md").write_text(resume, encoding="utf-8")


def write_sql_files() -> None:
    SQL_DIR.mkdir(parents=True, exist_ok=True)
    sql_files = {
        "01_campaign_kpis.sql": """-- Historical response and hypothetical unit-economics inputs; not realized ROI.
SELECT
    COUNT(*) AS contacts,
    SUM(converted) AS conversions,
    ROUND(AVG(converted), 4) AS conversion_rate,
    ROUND(AVG(campaign), 2) AS avg_touches,
    ROUND(SUM(converted) * 650.0, 2) AS modeled_conversion_value,
    ROUND(SUM(campaign) * 3.0, 2) AS modeled_historical_touch_cost,
    ROUND(SUM(converted) * 650.0 - SUM(campaign) * 3.0, 2) AS modeled_net_value
FROM campaign_contacts;
""",
        "02_segment_conversion.sql": """-- Historical conversion by record segment; no causal or ROI inference.
SELECT
    job,
    COUNT(*) AS contacts,
    SUM(converted) AS conversions,
    ROUND(AVG(converted), 4) AS conversion_rate,
    ROUND(AVG(campaign), 2) AS avg_touches
FROM campaign_contacts
GROUP BY job
ORDER BY conversion_rate DESC, contacts DESC;
""",
        "03_channel_performance.sql": """-- Observed contact-method association; EUR 3 per touch is assumed.
SELECT
    contact,
    COUNT(*) AS contacts,
    SUM(converted) AS conversions,
    ROUND(AVG(converted), 4) AS conversion_rate,
    ROUND(SUM(campaign) * 3.0 / NULLIF(SUM(converted), 0), 2) AS cost_per_conversion
FROM campaign_contacts
GROUP BY contact
ORDER BY conversion_rate DESC;
""",
        "04_contact_frequency.sql": """-- Response rate by campaign contact frequency.
SELECT
    campaign_bucket,
    COUNT(*) AS contacts,
    SUM(converted) AS conversions,
    ROUND(AVG(converted), 4) AS conversion_rate,
    ROUND(AVG(campaign), 2) AS avg_touches
FROM campaign_contacts
GROUP BY campaign_bucket
ORDER BY MIN(campaign);
""",
        "05_previous_campaign_outcome.sql": """-- Prior campaign outcome as a retargeting signal.
SELECT
    poutcome,
    was_previously_contacted,
    COUNT(*) AS contacts,
    SUM(converted) AS conversions,
    ROUND(AVG(converted), 4) AS conversion_rate
FROM campaign_contacts
GROUP BY poutcome, was_previously_contacted
ORDER BY conversion_rate DESC;
""",
    }
    for name, content in sql_files.items():
        (SQL_DIR / name).write_text(content, encoding="utf-8")
    bundle = ["-- Bank Marketing Campaign Analytics SQL Bundle", ""]
    for sql_path in sorted(SQL_DIR.glob("*.sql")):
        bundle.extend(["", "-- " + "=" * 78, f"-- {sql_path.name}", "-- " + "=" * 78, "", sql_path.read_text(encoding="utf-8")])
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    (PORTFOLIO_DIR / "sql_code_bundle.sql").write_text("\n".join(bundle).rstrip() + "\n", encoding="utf-8")


def write_readmes(outputs: dict[str, pd.DataFrame | float]) -> None:
    overview = outputs["overview"].iloc[0]
    readme = f"""# Bank Marketing Campaign Analytics

Marketing-focused portfolio project for Campaign Analyst, Marketing Analyst, and Growth Marketing Analyst roles.

## Finished Work

- Dashboard HTML: [dashboard/marketing_dashboard.html](dashboard/marketing_dashboard.html)
- Dashboard PDF snapshot: [portfolio/marketing_dashboard_snapshot.pdf](portfolio/marketing_dashboard_snapshot.pdf)
- One-page project PDF: [portfolio/marketing_project_one_pager.pdf](portfolio/marketing_project_one_pager.pdf)
- SQL code bundle: [portfolio/sql_code_bundle.sql](portfolio/sql_code_bundle.sql)
- Resume project description: [portfolio/resume_project_description.md](portfolio/resume_project_description.md)
- Executive summary: [reports/executive_summary.md](reports/executive_summary.md)

## Key Results

- Campaign contacts analyzed: {int(overview['contacts']):,}
- Conversions: {int(overview['conversions']):,}
- Overall conversion rate: {pct(float(overview['conversion_rate']))}
- Chronological holdout AUC: {float(outputs['model_metrics']['roc_auc']):.2f}
- Holdout-only top-decile conversion: {pct(float(outputs['deciles'].iloc[0]['conversion_rate']))} versus {pct(float(outputs['model_metrics']['holdout_base_rate']))} holdout baseline
- Hypothetical scenario inputs: EUR {CONTACT_COST:.0f} per planned outreach and EUR {CONVERSION_VALUE:.0f} value per subscription. No realized or incremental ROI is claimed.

## Methods

- Campaign KPI analysis
- Segment conversion analysis
- Contact method and contact frequency analysis
- Experimental propensity modeling and chronological validation
- Assumption-based outreach cost/value scenario modeling

## Reproduce

Download the official UCI Bank Marketing archive, extract `bank-additional-full.csv` to `data/raw/bank-additional/bank-additional/`, install `requirements.txt`, and run `python scripts/run_marketing_analysis.py`. The dataset's documented row order is chronological, so the newest 25% is held out for evaluation. Raw data is excluded from Git.

## Important limitations

- Records lack persistent client IDs; repeat-client deduplication cannot be verified.
- Segment, contact-method and contact-frequency differences are observational, not causal effects.
- Assumed EUR amounts are sensitivity inputs, not recorded costs, deposit revenue or realized profit.
- Model probabilities and ranking must be revalidated on recent data before operational use; demographic targeting also needs consent, fairness and compliance review.

## Data Source

UCI Bank Marketing dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing
"""
    (PROJECT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    portfolio_readme = """# Marketing Portfolio Deliverables

Files for recruiter and hiring-manager review:

- `marketing_dashboard_snapshot.pdf`: dashboard PDF screenshot.
- `marketing_project_one_pager.pdf`: one-page project summary PDF.
- `sql_code_bundle.sql`: SQL code file.
- `resume_project_description.md`: resume-ready project description.
"""
    (PORTFOLIO_DIR / "README.md").write_text(portfolio_readme, encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    data = load_and_clean()
    outputs = write_analysis_outputs(data)
    build_dashboard()
    create_dashboard_pdf(outputs)
    create_one_pager(outputs)
    write_sql_files()
    write_reports(outputs)
    write_readmes(outputs)
    print("Marketing analytics project generated.")
    print(f"Dashboard: {DASHBOARD_DIR / 'marketing_dashboard.html'}")
    print(f"Portfolio: {PORTFOLIO_DIR}")


if __name__ == "__main__":
    main()
