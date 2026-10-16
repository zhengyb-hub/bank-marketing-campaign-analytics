-- Bank Marketing Campaign Analytics SQL Bundle


-- ==============================================================================
-- 01_campaign_kpis.sql
-- ==============================================================================

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


-- ==============================================================================
-- 02_segment_conversion.sql
-- ==============================================================================

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


-- ==============================================================================
-- 03_channel_performance.sql
-- ==============================================================================

-- Observed contact-method association; EUR 3 per touch is assumed.
SELECT
    contact,
    COUNT(*) AS contacts,
    SUM(converted) AS conversions,
    ROUND(AVG(converted), 4) AS conversion_rate,
    ROUND(SUM(campaign) * 3.0 / NULLIF(SUM(converted), 0), 2) AS cost_per_conversion
FROM campaign_contacts
GROUP BY contact
ORDER BY conversion_rate DESC;


-- ==============================================================================
-- 04_contact_frequency.sql
-- ==============================================================================

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


-- ==============================================================================
-- 05_previous_campaign_outcome.sql
-- ==============================================================================

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
