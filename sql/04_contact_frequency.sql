-- Response rate by campaign contact frequency.
SELECT
    campaign_bucket,
    COUNT(*) AS contacts,
    SUM(converted) AS conversions,
    ROUND(AVG(converted), 4) AS conversion_rate,
    ROUND(AVG(campaign), 2) AS avg_touches
FROM campaign_contacts
GROUP BY campaign_bucket
ORDER BY MIN(campaign);
