-- Historical response and hypothetical unit-economics inputs; not realized ROI.
SELECT
    COUNT(*) AS contacts,
    SUM(converted) AS conversions,
    ROUND(AVG(converted), 4) AS conversion_rate,
    ROUND(AVG(campaign), 2) AS avg_touches,
    ROUND(SUM(converted) * 650.0, 2) AS modeled_conversion_value,
    ROUND(SUM(campaign) * 3.0, 2) AS modeled_historical_touch_cost,
    ROUND(SUM(converted) * 650.0 - SUM(campaign) * 3.0, 2) AS modeled_net_value
FROM campaign_contacts;
