-- Historical conversion by record segment; no causal or ROI inference.
SELECT
    job,
    COUNT(*) AS contacts,
    SUM(converted) AS conversions,
    ROUND(AVG(converted), 4) AS conversion_rate,
    ROUND(AVG(campaign), 2) AS avg_touches
FROM campaign_contacts
GROUP BY job
ORDER BY conversion_rate DESC, contacts DESC;
