-- Prior campaign outcome as a retargeting signal.
SELECT
    poutcome,
    was_previously_contacted,
    COUNT(*) AS contacts,
    SUM(converted) AS conversions,
    ROUND(AVG(converted), 4) AS conversion_rate
FROM campaign_contacts
GROUP BY poutcome, was_previously_contacted
ORDER BY conversion_rate DESC;
