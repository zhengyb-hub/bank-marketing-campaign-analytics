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
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_CSV = PROJECT_ROOT / "data" / "raw" / "bank-additional" / "bank-additional" / "bank-additional-full.csv"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
REPORTS_DIR = PROJECT_ROOT / "reports"
PORTFOLIO_DIR = PROJECT_ROOT / "portfolio"
SQL_DIR = PROJECT_ROOT / "sql"

CONTACT_COST = 3.0
CONVERSION_VALUE = 650.0


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
    summary["estimated_revenue"] = summary["conversions"] * CONVERSION_VALUE
    summary["estimated_cost"] = data.groupby(field, dropna=False)["campaign"].sum().values * CONTACT_COST
    summary["estimated_profit"] = summary["estimated_revenue"] - summary["estimated_cost"]
    summary["roi"] = np.where(summary["estimated_cost"] > 0, summary["estimated_profit"] / summary["estimated_cost"], np.nan)
    return summary.sort_values(["conversion_rate", "contacts"], ascending=[False, False])


def build_propensity_model(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    # Duration is excluded because it is only known after a call and would leak outcome information.
    features = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "emp_var_rate",
        "cons_price_idx",
        "cons_conf_idx",
        "euribor3m",
        "nr_employed",
    ]
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
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    train, test = train_test_split(data, test_size=0.25, random_state=42, stratify=data[target])
    model.fit(train[features], train[target])
    test_scores = model.predict_proba(test[features])[:, 1]
    auc = roc_auc_score(test[target], test_scores)

    scored = data.copy()
    scored["propensity_score"] = model.predict_proba(data[features])[:, 1]
    scored["propensity_decile"] = pd.qcut(scored["propensity_score"], 10, labels=False, duplicates="drop") + 1
    scored["propensity_decile"] = 11 - scored["propensity_decile"]

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
    deciles["estimated_cost"] = deciles["contacts"] * CONTACT_COST
    deciles["estimated_revenue"] = deciles["conversions"] * CONVERSION_VALUE
    deciles["estimated_profit"] = deciles["estimated_revenue"] - deciles["estimated_cost"]
    deciles["roi"] = deciles["estimated_profit"] / deciles["estimated_cost"]

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefs = model.named_steps["model"].coef_[0]
    drivers = (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        .assign(abs_coefficient=lambda frame: frame["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .head(25)
    )
    drivers["direction"] = np.where(drivers["coefficient"] > 0, "positive", "negative")
    return scored, deciles, auc, drivers


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
                "estimated_revenue": data["converted"].sum() * CONVERSION_VALUE,
                "estimated_cost": data["campaign"].sum() * CONTACT_COST,
            }
        ]
    )
    overview["estimated_profit"] = overview["estimated_revenue"] - overview["estimated_cost"]
    overview["roi"] = overview["estimated_profit"] / overview["estimated_cost"]

    by_job = summarize_segment(data, "job")
    by_education = summarize_segment(data, "education")
    by_contact = summarize_segment(data, "contact")
    by_month = summarize_segment(data, "month").merge(
        data[["month", "month_order"]].drop_duplicates(), on="month", how="left"
    ).sort_values("month_order")
    by_campaign = summarize_segment(data, "campaign_bucket")
    by_poutcome = summarize_segment(data, "poutcome")
    scored, deciles, auc, drivers = build_propensity_model(data)

    overview.to_csv(DATA_DIR / "overview_metrics.csv", index=False)
    by_job.to_csv(DATA_DIR / "conversion_by_job.csv", index=False)
    by_education.to_csv(DATA_DIR / "conversion_by_education.csv", index=False)
    by_contact.to_csv(DATA_DIR / "conversion_by_contact_method.csv", index=False)
    by_month.to_csv(DATA_DIR / "conversion_by_month.csv", index=False)
    by_campaign.to_csv(DATA_DIR / "conversion_by_campaign_frequency.csv", index=False)
    by_poutcome.to_csv(DATA_DIR / "conversion_by_previous_outcome.csv", index=False)
    deciles.to_csv(DATA_DIR / "propensity_decile_roi.csv", index=False)
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
        "auc": auc,
    }


def create_dashboard(outputs: dict[str, pd.DataFrame | float]) -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    overview = outputs["overview"].iloc[0]
    by_month = outputs["by_month"]
    by_job = outputs["by_job"].head(10).sort_values("conversion_rate")
    by_contact = outputs["by_contact"]
    by_campaign = outputs["by_campaign"]
    deciles = outputs["deciles"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Conversion Rate by Month",
            "Top Job Segments by Conversion Rate",
            "Contact Method Performance",
            "Propensity Decile ROI",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "scatter"}]],
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )
    fig.add_bar(x=by_month["month"], y=by_month["conversion_rate"], marker_color="#1f77b4", row=1, col=1)
    fig.add_bar(x=by_job["conversion_rate"], y=by_job["job"], orientation="h", marker_color="#2ca02c", row=1, col=2)
    fig.add_bar(x=by_contact["contact"], y=by_contact["conversion_rate"], marker_color="#ff7f0e", row=2, col=1)
    fig.add_scatter(
        x=deciles["propensity_decile"],
        y=deciles["roi"],
        mode="lines+markers",
        line=dict(color="#d62728", width=3),
        row=2,
        col=2,
    )
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_xaxes(tickformat=".0%", row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".1f", row=2, col=2)
    fig.update_layout(height=850, showlegend=False, title_text="Bank Marketing Campaign Analytics Dashboard")

    cards = f"""
    <section class="cards">
      <div><span>Contacts</span><strong>{int(overview['contacts']):,}</strong></div>
      <div><span>Conversions</span><strong>{int(overview['conversions']):,}</strong></div>
      <div><span>Conversion Rate</span><strong>{pct(float(overview['conversion_rate']))}</strong></div>
      <div><span>Estimated ROI</span><strong>{float(overview['roi']):.1f}x</strong></div>
    </section>
    """
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bank Marketing Campaign Analytics</title>
  <style>
    body{{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0;background:#f6f7fb;color:#172033;}}
    header{{background:#172033;color:white;padding:28px 40px;}}
    h1{{margin:0;font-size:28px;}}
    header p{{color:#d8deea;margin:8px 0 0;}}
    main{{max-width:1180px;margin:0 auto;padding:24px;}}
    .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px;}}
    .cards div{{background:white;border:1px solid #e1e5ee;border-radius:8px;padding:16px;}}
    .cards span{{display:block;font-size:12px;text-transform:uppercase;color:#69758a;font-weight:700;}}
    .cards strong{{display:block;font-size:26px;margin-top:8px;}}
    .panel{{background:white;border:1px solid #e1e5ee;border-radius:8px;padding:10px;}}
    @media(max-width:760px){{.cards{{grid-template-columns:1fr 1fr;}}}}
  </style>
</head>
<body>
  <header><h1>Bank Marketing Campaign Analytics</h1><p>Direct marketing conversion, audience segmentation, propensity scoring, and ROI targeting.</p></header>
  <main>{cards}<section class="panel">{fig.to_html(full_html=False, include_plotlyjs=True)}</section></main>
</body>
</html>
"""
    (DASHBOARD_DIR / "marketing_dashboard.html").write_text(html, encoding="utf-8")


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
    fig.text(0.04, 0.94, "Conversion, segmentation, propensity scoring, and ROI targeting.", fontsize=11, color="#4f5b6b")
    grid = fig.add_gridspec(4, 4, left=0.04, right=0.98, top=0.89, bottom=0.07, wspace=0.35, hspace=0.6)

    add_card(fig.add_subplot(grid[0, 0]), "Contacts", f"{int(overview['contacts']):,}", "Campaign records")
    add_card(fig.add_subplot(grid[0, 1]), "Conversions", f"{int(overview['conversions']):,}", "Term deposit subscriptions")
    add_card(fig.add_subplot(grid[0, 2]), "Conversion rate", pct(float(overview["conversion_rate"])), "Overall response")
    add_card(fig.add_subplot(grid[0, 3]), "Model AUC", f"{float(outputs['auc']):.2f}", "Holdout propensity model")

    ax1 = fig.add_subplot(grid[1:3, :2])
    ax1.bar(by_month["month"], by_month["conversion_rate"], color="#1f77b4")
    ax1.set_title("Conversion Rate by Month", loc="left", fontsize=13, weight="bold")
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
    ax4.plot(deciles["propensity_decile"], deciles["roi"], color="#d62728", marker="o", linewidth=2.4)
    ax4.set_title("ROI by Propensity Decile", loc="left", fontsize=13, weight="bold")
    ax4.set_xlabel("Decile, 1 = highest propensity")
    ax4.set_ylabel("ROI multiple")
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
        ax.text(0.07, 0.925, "Marketing portfolio project | Campaign KPI, segmentation, propensity scoring, ROI", fontsize=9.5, color="#536070")

        y = 0.87
        sections = [
            (
                "Business Question",
                "How should a marketing team prioritize direct marketing outreach to improve conversion efficiency and campaign ROI?",
            ),
            (
                "Dataset and Scope",
                f"UCI Bank Marketing dataset with {int(overview['contacts']):,} campaign contacts and {int(overview['conversions']):,} conversions. Outcome is whether the client subscribed to a term deposit.",
            ),
            (
                "Method",
                "Built campaign KPIs, segment conversion tables, contact-method analysis, contact-frequency diagnostics, and a logistic-regression propensity model excluding call duration to avoid leakage.",
            ),
            (
                "Key Findings",
                f"Overall conversion rate is {pct(float(overview['conversion_rate']))}. The highest-converting job segment is {by_job['job']} at {pct(float(by_job['conversion_rate']))}. The strongest contact method is {by_contact['contact']} at {pct(float(by_contact['conversion_rate']))}. Top propensity decile converts at {pct(float(top_decile['conversion_rate']))}.",
            ),
            (
                "Recommendation",
                "Prioritize high-propensity deciles for outbound campaigns, suppress over-contacted low-response audiences, and shift more budget toward segments and contact methods with stronger conversion efficiency.",
            ),
        ]
        for title, body in sections:
            ax.text(0.07, y, title, fontsize=13, weight="bold", color="#172033")
            y -= 0.035
            ax.text(0.07, y, wrap(body, 88), fontsize=9.6, color="#303846", va="top")
            y -= 0.13

        ax.add_patch(plt.Rectangle((0.07, 0.12), 0.86, 0.12, facecolor="#f7f9fc", edgecolor="#d9e0ec"))
        ax.text(0.09, 0.205, "Portfolio Deliverables", fontsize=11, weight="bold", color="#172033")
        ax.text(0.09, 0.175, "Dashboard PDF, SQL code bundle, one-page summary PDF, and resume project description.", fontsize=8.8, color="#303846")
        ax.text(0.09, 0.15, "Project: Bank Marketing Campaign Analytics", fontsize=8.8, color="#303846")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def write_reports(outputs: dict[str, pd.DataFrame | float]) -> None:
    overview = outputs["overview"].iloc[0]
    by_job = outputs["by_job"].head(5)
    deciles = outputs["deciles"]
    drivers = outputs["drivers"].head(10)

    summary = f"""# Executive Summary

## Business Question

How should a marketing team prioritize direct marketing outreach to improve conversion efficiency and campaign ROI?

## Dataset

UCI Bank Marketing dataset with {int(overview['contacts']):,} campaign contacts and {int(overview['conversions']):,} conversions. The target is whether a client subscribed to a term deposit.

## Key Findings

- Overall conversion rate is {pct(float(overview['conversion_rate']))}.
- Average contacts per client in the campaign is {float(overview['avg_touches']):.2f}.
- Estimated campaign ROI under the assumptions EUR {CONTACT_COST:.0f} per contact attempt and EUR {CONVERSION_VALUE:.0f} per conversion is {float(overview['roi']):.1f}x.
- The top propensity decile converts at {pct(float(deciles.iloc[0]['conversion_rate']))}.
- Model AUC on a holdout set is {float(outputs['auc']):.2f}.

## Recommended Actions

1. Prioritize top propensity deciles for outbound campaigns.
2. Suppress low-propensity, repeatedly contacted audiences to reduce wasted touches.
3. Shift budget toward stronger segments such as high-converting job groups and contact methods.
4. Use propensity score as a targeting layer, then monitor conversion rate, cost per conversion, and incremental lift.

## Top Converting Job Segments

{dataframe_to_markdown(by_job[['job', 'contacts', 'conversion_rate', 'roi']])}

## Top Model Drivers

{dataframe_to_markdown(drivers[['feature', 'coefficient', 'direction']])}
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "executive_summary.md").write_text(summary, encoding="utf-8")

    resume = f"""# Resume Project Description

## Recommended Resume Bullet

Built a marketing campaign analytics project on {int(overview['contacts']):,} direct marketing contacts, using SQL and Python to analyze conversion rates, segment performance, contact frequency, channel effectiveness, propensity scoring, and campaign ROI; identified high-propensity audiences and targeting rules to improve marketing efficiency.

## Short Version

Bank Marketing Campaign Analytics: analyzed {int(overview['contacts']):,} campaign contacts with SQL, Python, and Plotly to diagnose conversion drivers, build a propensity model, and recommend ROI-focused audience targeting.

## Interview Talking Points

- Translated raw campaign records into marketing KPIs: conversion rate, cost per conversion, ROI, and segment response.
- Excluded call duration from the targeting model because it is unavailable before outreach and would leak outcome information.
- Compared broad outreach with propensity-decile targeting to show how targeting can improve campaign efficiency.
- Delivered dashboard, SQL bundle, one-page PDF, and resume-ready project description.
"""
    (REPORTS_DIR / "resume_project_description.md").write_text(resume, encoding="utf-8")
    (PORTFOLIO_DIR / "resume_project_description.md").write_text(resume, encoding="utf-8")


def write_sql_files() -> None:
    SQL_DIR.mkdir(parents=True, exist_ok=True)
    sql_files = {
        "01_campaign_kpis.sql": """-- Campaign-level KPIs.
SELECT
    COUNT(*) AS contacts,
    SUM(converted) AS conversions,
    ROUND(AVG(converted), 4) AS conversion_rate,
    ROUND(AVG(campaign), 2) AS avg_touches,
    ROUND(SUM(converted) * 650.0, 2) AS estimated_revenue,
    ROUND(SUM(campaign) * 3.0, 2) AS estimated_cost,
    ROUND((SUM(converted) * 650.0 - SUM(campaign) * 3.0) / (SUM(campaign) * 3.0), 2) AS roi
FROM campaign_contacts;
""",
        "02_segment_conversion.sql": """-- Conversion rate and ROI by customer segment.
SELECT
    job,
    COUNT(*) AS contacts,
    SUM(converted) AS conversions,
    ROUND(AVG(converted), 4) AS conversion_rate,
    ROUND(AVG(campaign), 2) AS avg_touches,
    ROUND((SUM(converted) * 650.0 - SUM(campaign) * 3.0) / (SUM(campaign) * 3.0), 2) AS roi
FROM campaign_contacts
GROUP BY job
ORDER BY conversion_rate DESC, contacts DESC;
""",
        "03_channel_performance.sql": """-- Contact method performance.
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
- Dashboard PDF screenshot: [portfolio/marketing_dashboard_snapshot.pdf](portfolio/marketing_dashboard_snapshot.pdf)
- One-page project PDF: [portfolio/marketing_project_one_pager.pdf](portfolio/marketing_project_one_pager.pdf)
- SQL code bundle: [portfolio/sql_code_bundle.sql](portfolio/sql_code_bundle.sql)
- Resume project description: [portfolio/resume_project_description.md](portfolio/resume_project_description.md)
- Executive summary: [reports/executive_summary.md](reports/executive_summary.md)

## Key Results

- Campaign contacts analyzed: {int(overview['contacts']):,}
- Conversions: {int(overview['conversions']):,}
- Overall conversion rate: {pct(float(overview['conversion_rate']))}
- Propensity model AUC: {float(outputs['auc']):.2f}
- Estimated ROI assumption: EUR {CONTACT_COST:.0f} per contact attempt and EUR {CONVERSION_VALUE:.0f} value per conversion

## Methods

- Campaign KPI analysis
- Segment conversion analysis
- Contact method and contact frequency analysis
- Propensity scoring for targeting
- ROI and cost-per-conversion modeling

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
    create_dashboard(outputs)
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
