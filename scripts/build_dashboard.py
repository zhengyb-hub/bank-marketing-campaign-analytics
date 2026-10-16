"""Build a self-contained decision dashboard from the reviewed aggregate outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def rows(filename: str) -> list[dict]:
    return json.loads(pd.read_csv(PROCESSED / filename).to_json(orient="records"))


def build_dashboard() -> Path:
    payload = {
        "overview": rows("overview_metrics.csv")[0],
        "model": rows("model_metrics.csv")[0],
        "deciles": rows("propensity_decile_holdout.csv"),
        "jobs": rows("conversion_by_job.csv"),
        "methods": rows("conversion_by_contact_method.csv"),
        "touches": rows("conversion_by_campaign_frequency.csv"),
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
    html = html.replace("/* PLOTLY_BUNDLE */", get_plotlyjs()).replace("/* DASHBOARD_DATA */", data)
    for path in (ROOT / "dashboard" / "marketing_dashboard.html", ROOT / "docs" / "index.html"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    return ROOT / "dashboard" / "marketing_dashboard.html"


if __name__ == "__main__":
    print(build_dashboard())
