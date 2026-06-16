-- Campaign-level KPIs.
SELECT
    COUNT(*) AS contacts,
    SUM(converted) AS conversions,
    ROUND(AVG(converted), 4) AS conversion_rate,
    ROUND(AVG(campaign), 2) AS avg_touches,
    ROUND(SUM(converted) * 650.0, 2) AS estimated_revenue,
    ROUND(SUM(campaign) * 3.0, 2) AS estimated_cost,
    ROUND((SUM(converted) * 650.0 - SUM(campaign) * 3.0) / (SUM(campaign) * 3.0), 2) AS roi
FROM campaign_contacts;
