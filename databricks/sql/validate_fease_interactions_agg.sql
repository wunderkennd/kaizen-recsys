-- ==========================================================================
-- Issue #102: validation queries for the series-grain table.
-- Run after the backfill (and after a MERGE for query 5); record the
-- results on the issue. Same parameters as the create / merge scripts.
-- ==========================================================================

-- 1. Row counts and date range.
SELECT
    COUNT(*)                     AS total_rows,
    COUNT(DISTINCT `user_id`)    AS distinct_users,
    COUNT(DISTINCT `item_id`)    AS distinct_items,
    MIN(`view_date`)             AS earliest_date,
    MAX(`view_date`)             AS latest_date,
    COUNT(DISTINCT `view_date`)  AS distinct_dates
FROM `${catalog}`.`${schema}`.`${table}`;

-- 2. Grain check: the busiest item_ids must be series ids, not episode ids.
SELECT `item_id`, COUNT(DISTINCT `user_id`) AS user_count
FROM `${catalog}`.`${schema}`.`${table}`
GROUP BY `item_id`
ORDER BY user_count DESC
LIMIT 10;

-- 3. Rows per day over the last 30 complete days (watch for gaps / spikes).
SELECT `view_date`, COUNT(*) AS rows
FROM `${catalog}`.`${schema}`.`${table}`
WHERE `view_date` >= date_sub(current_date(), 30)
GROUP BY `view_date`
ORDER BY `view_date` DESC;

-- 4. `value` is a sum of ln(seconds + 1) terms, so it is >= ln(min_watch_seconds + 1)
--    per view; a single 30 s view is ~3.43.
SELECT
    MIN(`value`)              AS min_value,
    AVG(`value`)              AS avg_value,
    PERCENTILE(`value`, 0.5)  AS median_value,
    MAX(`value`)              AS max_value,
    AVG(`num_views`)          AS avg_views_per_row
FROM `${catalog}`.`${schema}`.`${table}`;

-- 5. MERGE idempotency: run the merge twice for the same window; the second
--    run must report 0 inserted, 0 updated, 0 deleted.
SELECT
    `timestamp`,
    operationMetrics.numTargetRowsInserted AS inserted,
    operationMetrics.numTargetRowsUpdated  AS updated,
    operationMetrics.numTargetRowsDeleted  AS deleted
FROM (DESCRIBE HISTORY `${catalog}`.`${schema}`.`${table}`)
WHERE operation = 'MERGE'
ORDER BY `timestamp` DESC
LIMIT 2;

-- 6. Last 30 days: raw event rows vs aggregated rows, and that the set of
--    (user, item) pairs is identical.
SELECT
    'raw' AS source,
    COUNT(*) AS rows,
    COUNT(DISTINCT `${user_col}`, `${item_col}`) AS unique_pairs
FROM `${source_table}`
WHERE `${subsidiary_col}` = '${subsidiary}'
  AND NULLIF(TRIM(`${user_col}`), '') IS NOT NULL
  AND `${item_col}` IS NOT NULL
  AND `${seconds_col}` >= ${min_watch_seconds}
  AND `${date_col}` >= date_sub(current_date(), 30)
  AND `${date_col}` <  current_date()
UNION ALL
SELECT
    'agg' AS source,
    COUNT(*) AS rows,
    COUNT(DISTINCT `user_id`, `item_id`) AS unique_pairs
FROM `${catalog}`.`${schema}`.`${table}`
WHERE `view_date` >= date_sub(current_date(), 30)
  AND `view_date` <  current_date();
