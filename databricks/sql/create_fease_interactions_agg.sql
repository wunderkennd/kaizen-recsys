-- ==========================================================================
-- Issue #102: create and backfill the series-grain interactions table.
--
-- One row per (profile, series, view_date). Items are series
-- (${item_col}), not episode-level media ids. `days_ago` is NOT stored:
-- consumers derive it at read time (datediff(current_date(), view_date))
-- so temporal decay stays a tunable hyperparameter and the table is not
-- coupled to a point in time.
--
-- Parameters (substituted by databricks/src/00_refresh_interactions_agg.py
-- or by notebook widgets of the same names):
--   ${catalog}.${schema}.${table}  target table
--   ${source_table}                gold viewership table
--   ${user_col} ${item_col} ${date_col} ${seconds_col}
--   ${subsidiary_col} ${subsidiary} ${min_watch_seconds}
--
-- Run once. Re-running replaces the whole table (INSERT OVERWRITE).
-- ==========================================================================

CREATE TABLE IF NOT EXISTS `${catalog}`.`${schema}`.`${table}` (
    `user_id`    STRING  NOT NULL COMMENT 'Profile id (${user_col})',
    `item_id`    STRING  NOT NULL COMMENT 'Series id (${item_col})',
    `view_date`  DATE    NOT NULL COMMENT 'Date the aggregated views fell on',
    `value`      DOUBLE  NOT NULL COMMENT 'sum(ln(${seconds_col} + 1)) over that day''s qualifying views',
    `num_views`  BIGINT  NOT NULL COMMENT 'Count of qualifying view rows (diagnostics)'
)
USING DELTA
PARTITIONED BY (`view_date`)
COMMENT 'Pre-aggregated interactions at (profile, series, date) grain for FEASE / BERT4Rec / SASRec training (issue #102). days_ago is derived at read time.'
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact'   = 'true'
);

-- Backfill from the full history. Today is excluded because it is still
-- being written; the daily MERGE picks it up once it is complete.
INSERT OVERWRITE `${catalog}`.`${schema}`.`${table}`
SELECT
    `${user_col}`                                  AS `user_id`,
    `${item_col}`                                  AS `item_id`,
    `${date_col}`                                  AS `view_date`,
    SUM(LN(`${seconds_col}` + 1))                  AS `value`,
    COUNT(*)                                       AS `num_views`
FROM `${source_table}`
WHERE `${subsidiary_col}` = '${subsidiary}'
  AND NULLIF(TRIM(`${user_col}`), '') IS NOT NULL
  AND `${item_col}` IS NOT NULL
  AND `${seconds_col}` >= ${min_watch_seconds}
  AND `${date_col}` < current_date()
GROUP BY `${user_col}`, `${item_col}`, `${date_col}`;

-- Co-locate each user's rows inside every date partition. This helps the
-- per-user reads (histories, splits) more than the training groupBy, which
-- scans everything regardless.
OPTIMIZE `${catalog}`.`${schema}`.`${table}` ZORDER BY (`user_id`);
