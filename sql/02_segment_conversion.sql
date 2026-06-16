-- Conversion rate and ROI by customer segment.
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
