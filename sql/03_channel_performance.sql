-- Contact method performance.
SELECT
    contact,
    COUNT(*) AS contacts,
    SUM(converted) AS conversions,
    ROUND(AVG(converted), 4) AS conversion_rate,
    ROUND(SUM(campaign) * 3.0 / NULLIF(SUM(converted), 0), 2) AS cost_per_conversion
FROM campaign_contacts
GROUP BY contact
ORDER BY conversion_rate DESC;
