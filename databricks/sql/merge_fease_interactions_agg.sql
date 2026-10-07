-- ==========================================================================
-- Issue #102: daily incremental refresh of the series-grain table.
--
-- Recomputes the last ${lookback_days} complete days from the gold table
-- and MERGEs them in. The window is *recomputed*, not added to, so a
-- re-run for the same window is a no-op: matched rows are rewritten with
-- the same values, new rows are inserted, and rows that no longer exist in
-- the source for that window (restated or removed views) are deleted.
-- `lookback_days` must cover the gold table's late-arrival window.
--
-- Parameters: same as create_fease_interactions_agg.sql plus
-- ${lookback_days}.
-- ==========================================================================

MERGE INTO `${catalog}`.`${schema}`.`${table}` AS target
USING (
    SELECT
        `${user_col}`                              AS `user_id`,
        `${item_col}`                              AS `item_id`,
        `${date_col}`                              AS `view_date`,
        SUM(LN(`${seconds_col}` + 1))              AS `value`,
        COUNT(*)                                   AS `num_views`
    FROM `${source_table}`
    WHERE `${subsidiary_col}` = '${subsidiary}'
      AND NULLIF(TRIM(`${user_col}`), '') IS NOT NULL
      AND `${item_col}` IS NOT NULL
      AND `${seconds_col}` >= ${min_watch_seconds}
      AND `${date_col}` >= date_sub(current_date(), ${lookback_days})
      AND `${date_col}` <  current_date()
    GROUP BY `${user_col}`, `${item_col}`, `${date_col}`
) AS source
ON  target.`user_id`   = source.`user_id`
AND target.`item_id`   = source.`item_id`
AND target.`view_date` = source.`view_date`
WHEN MATCHED AND (target.`value` <> source.`value` OR target.`num_views` <> source.`num_views`)
    THEN UPDATE SET
        target.`value`     = source.`value`,
        target.`num_views` = source.`num_views`
WHEN NOT MATCHED
    THEN INSERT (`user_id`, `item_id`, `view_date`, `value`, `num_views`)
    VALUES (source.`user_id`, source.`item_id`, source.`view_date`, source.`value`, source.`num_views`)
WHEN NOT MATCHED BY SOURCE
    AND target.`view_date` >= date_sub(current_date(), ${lookback_days})
    AND target.`view_date` <  current_date()
    THEN DELETE;
