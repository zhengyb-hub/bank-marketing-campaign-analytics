# Executive Summary

## Business Question

Can historical campaign data support reliable, cost-aware prioritization of future outreach?

## Dataset and grain

UCI Bank Marketing dataset with 41,188 campaign records and 4,640 subscriptions. There is no persistent client identifier, so records are not asserted to be unique clients. The full dataset is date-ordered from May 2008 to November 2010; the latest 25% is reserved as an untouched chronological holdout.

## Key Findings

- Overall conversion rate is 11.3%.
- Mean recorded current-campaign touches per record is 2.57.
- A hypothetical EUR 3 per touch and EUR 650 per subscription imply EUR 2,698,738 modeled net value across historical records; these are not observed costs, revenue, profit, or incremental ROI.
- The top **holdout-only** propensity decile has 1,029 records and a 57.0% observed subscription rate, versus 25.8% across the holdout. Its cumulative lift is 2.21x.
- Chronological holdout ROC AUC is 0.53; average precision is 0.37; Brier score is 0.199.

## Recommended Actions

1. Do not deploy this score as an automated outreach rule: chronological holdout AUC is only 0.53 and decile response is not monotonic. Improve and revalidate the model on newer campaigns before any controlled pilot.
2. Test a contact-frequency cap rather than inferring a causal benefit from observational frequency groups.
3. Treat contact-method and demographic differences as descriptive, not evidence that switching channel or excluding a group will improve outcomes. Review consent, fairness, and applicable marketing rules before any targeting use.
4. If a revised rule proves stable, compare it with a randomized business-as-usual control for incremental subscriptions and contact cost. This historical dataset has no reliable client ID for deduplication and no actual cost or value fields.

## Top Converting Job Segments

| job | contacts | conversion_rate |
| --- | --- | --- |
| student | 875 | 0.3143 |
| retired | 1720 | 0.2523 |
| unemployed | 1014 | 0.1420 |
| admin. | 10422 | 0.1297 |
| management | 2924 | 0.1122 |

## Top Model Drivers

| feature | coefficient | direction |
| --- | --- | --- |
| cat__poutcome_failure | -0.8565 | negative |
| cat__default_unknown | -0.6177 | negative |
| num__cons_conf_idx | -0.4883 | negative |
| num__nr_employed | -0.4353 | negative |
| cat__job_student | 0.4298 | positive |
| cat__job_retired | 0.4257 | positive |
| cat__marital_divorced | -0.4112 | negative |
| cat__housing_unknown | -0.3972 | negative |
| cat__loan_unknown | -0.3972 | negative |
| cat__default_no | -0.3855 | negative |

Coefficients show model associations after preprocessing, not causal campaign drivers or a fairness approval.
