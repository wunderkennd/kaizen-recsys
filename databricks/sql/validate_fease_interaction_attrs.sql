-- ==========================================================================
-- Validation queries for fease_interaction_attrs and fease_item_attrs.
-- Run after the backfill (and after two MERGEs for query 3); record the
-- results on the GitHub issue. Same parameters as the create / merge scripts.
-- ==========================================================================

-- 1. Row counts and date range (compare with the agg table's query 1).
SELECT
    COUNT(*)                     AS total_rows,
    COUNT(DISTINCT `user_id`)    AS distinct_users,
    COUNT(DISTINCT `item_id`)    AS distinct_items,
    MIN(`view_date`)             AS earliest_date,
    MAX(`view_date`)             AS latest_date
FROM `${catalog}`.`${schema}`.`${attrs_table}`;

-- 2. Key-set parity with the agg table over the last 30 complete days: both
--    counts must be 0 (a key in one table but not the other means the two
--    filters drifted).
SELECT
    SUM(CASE WHEN a.`user_id` IS NULL THEN 1 ELSE 0 END) AS agg_keys_missing_attrs,
    SUM(CASE WHEN g.`user_id` IS NULL THEN 1 ELSE 0 END) AS attrs_keys_missing_agg
FROM (
    SELECT `user_id`, `item_id`, `view_date` FROM `${catalog}`.`${schema}`.`${table}`
    WHERE `view_date` >= date_sub(current_date(), 30) AND `view_date` < current_date()
) g
FULL OUTER JOIN (
    SELECT `user_id`, `item_id`, `view_date` FROM `${catalog}`.`${schema}`.`${attrs_table}`
    WHERE `view_date` >= date_sub(current_date(), 30) AND `view_date` < current_date()
) a
  ON g.`user_id` = a.`user_id` AND g.`item_id` = a.`item_id` AND g.`view_date` = a.`view_date`;

-- 3. MERGE idempotency: run the merge twice for the same window; the second
--    run must report 0 inserted, 0 updated, 0 deleted.
SELECT
    `timestamp`,
    operationMetrics.numTargetRowsInserted AS inserted,
    operationMetrics.numTargetRowsUpdated  AS updated,
    operationMetrics.numTargetRowsDeleted  AS deleted
FROM (DESCRIBE HISTORY `${catalog}`.`${schema}`.`${attrs_table}`)
WHERE operation = 'MERGE'
ORDER BY `timestamp` DESC
LIMIT 2;

-- 4. Distribution of max_depth_ratio (NULL share = views without a known
--    duration) and of episodes per day.
SELECT
    AVG(CASE WHEN `max_depth_ratio` IS NULL THEN 1.0 ELSE 0.0 END) AS null_depth_share,
    PERCENTILE(`max_depth_ratio`, array(0.1, 0.25, 0.5, 0.75, 0.9)) AS depth_quantiles,
    AVG(CASE WHEN `max_depth_ratio` >= ${completed_ratio} THEN 1.0 ELSE 0.0 END) AS completed_share,
    PERCENTILE(`episodes_watched`, array(0.5, 0.9, 0.99)) AS episodes_quantiles,
    AVG(CASE WHEN `dub_views` + `sub_views` = 0 THEN 1.0 ELSE 0.0 END) AS unknown_audio_share
FROM `${catalog}`.`${schema}`.`${attrs_table}`
WHERE `view_date` >= date_sub(current_date(), 30);

-- 5. Normalized platform classes over the last 30 days.
SELECT `platform_top`, COUNT(*) AS rows
FROM `${catalog}`.`${schema}`.`${attrs_table}`
WHERE `view_date` >= date_sub(current_date(), 30)
GROUP BY `platform_top`
ORDER BY rows DESC;

-- 6. Raw platform values before normalization, with the class each maps to
--    (check the four-class mapping in kzn_recsys/cr_config.py against this).
SELECT `${platform_col}` AS raw_platform, ${platform_expr} AS normalized, COUNT(*) AS views
FROM ${source_table}
WHERE `${subsidiary_col}` = '${subsidiary}'
  AND `${seconds_col}` >= ${min_watch_seconds}
  AND `${date_col}` >= date_sub(current_date(), 30)
  AND `${date_col}` <  current_date()
GROUP BY `${platform_col}`
ORDER BY views DESC
LIMIT 100;

-- 7. Item attributes: premiere source mix and coverage of the agg table's series.
SELECT
    `premiere_source`,
    COUNT(*)                                   AS series,
    AVG(`episode_count`)                       AS avg_episode_count,
    AVG(`avg_episode_seconds`)                 AS avg_episode_seconds,
    MIN(`first_air_year`)                      AS min_air_year,
    MAX(`first_air_year`)                      AS max_air_year
FROM `${catalog}`.`${schema}`.`${item_attrs_table}`
GROUP BY `premiere_source`;
