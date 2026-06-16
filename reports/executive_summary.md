# Executive Summary

## Business Question

How should a marketing team prioritize direct marketing outreach to improve conversion efficiency and campaign ROI?

## Dataset

UCI Bank Marketing dataset with 41,188 campaign contacts and 4,640 conversions. The target is whether a client subscribed to a term deposit.

## Key Findings

- Overall conversion rate is 11.3%.
- Average contacts per client in the campaign is 2.57.
- Estimated campaign ROI under the assumptions EUR 3 per contact attempt and EUR 650 per conversion is 8.5x.
- The top propensity decile converts at 49.2%.
- Model AUC on a holdout set is 0.80.

## Recommended Actions

1. Prioritize top propensity deciles for outbound campaigns.
2. Suppress low-propensity, repeatedly contacted audiences to reduce wasted touches.
3. Shift budget toward stronger segments such as high-converting job groups and contact methods.
4. Use propensity score as a targeting layer, then monitor conversion rate, cost per conversion, and incremental lift.

## Top Converting Job Segments

| job | contacts | conversion_rate | roi |
| --- | --- | --- | --- |
| student | 875 | 0.3143 | 31.3647 |
| retired | 1720 | 0.2523 | 21.0736 |
| unemployed | 1014 | 0.1420 | 11.0000 |
| admin. | 10422 | 0.1297 | 9.7137 |
| management | 2924 | 0.1122 | 8.8158 |

## Top Model Drivers

| feature | coefficient | direction |
| --- | --- | --- |
| num__emp_var_rate | -2.1859 | negative |
| cat__month_mar | 1.2789 | positive |
| num__cons_price_idx | 1.0357 | positive |
| cat__month_jun | -0.7661 | negative |
| cat__month_may | -0.6471 | negative |
| cat__month_nov | -0.5822 | negative |
| num__euribor3m | 0.5420 | positive |
| cat__month_aug | 0.4514 | positive |
| cat__month_dec | 0.4240 | positive |
| cat__job_retired | 0.3557 | positive |
