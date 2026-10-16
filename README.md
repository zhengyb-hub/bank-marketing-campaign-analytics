# Bank Marketing Campaign Analytics

Marketing-focused portfolio project for Campaign Analyst, Marketing Analyst, and Growth Marketing Analyst roles.

## Finished Work

- Dashboard HTML: [dashboard/marketing_dashboard.html](dashboard/marketing_dashboard.html)
- Dashboard PDF snapshot: [portfolio/marketing_dashboard_snapshot.pdf](portfolio/marketing_dashboard_snapshot.pdf)
- One-page project PDF: [portfolio/marketing_project_one_pager.pdf](portfolio/marketing_project_one_pager.pdf)
- SQL code bundle: [portfolio/sql_code_bundle.sql](portfolio/sql_code_bundle.sql)
- Resume project description: [portfolio/resume_project_description.md](portfolio/resume_project_description.md)
- Executive summary: [reports/executive_summary.md](reports/executive_summary.md)

## Key Results

- Campaign contacts analyzed: 41,188
- Conversions: 4,640
- Overall conversion rate: 11.3%
- Chronological holdout AUC: 0.53
- Holdout-only top-decile conversion: 57.0% versus 25.8% holdout baseline
- Hypothetical scenario inputs: EUR 3 per planned outreach and EUR 650 value per subscription. No realized or incremental ROI is claimed.

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
