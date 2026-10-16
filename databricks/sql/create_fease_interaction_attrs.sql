-- ==========================================================================
-- Per-interaction attributes: create and backfill the sibling table of
-- fease_interactions_agg (issue #102).
--
-- One row per (profile, series, view_date), for exactly the keys the agg
-- table holds: the WHERE clause below is the agg table's, verbatim, so the
-- two key sets match for any window (validate_fease_interaction_attrs.sql,
-- query 2). The agg table, its MERGE and its columns are untouched.
--
-- Nothing thresholded or time-relative is stored: depth / binge / dub-share
-- buckets and the discovery lag are derived at read time by
-- kzn_recsys.spark.interaction_attrs.
--
-- Parameters (substituted by databricks/src/00_refresh_interactions_agg.py,
-- target=attrs):
--   ${catalog}.${schema}.${attrs_table}  target table
--   ${source_table} ${user_col} ${item_col} ${date_col} ${ts_col}
--   ${seconds_col} ${subsidiary_col} ${subsidiary} ${min_watch_seconds}
--                                        as in create_fease_interactions_agg.sql
--   ${media_id_expr}      per-episode id (from media_id_col; NULL if blank)
--   ${duration_expr}      episode length in seconds (from media_duration_col
--                         x duration scale; NULL if blank)
--   ${platform_expr}      CASE normalizing platform_col to tv/mobile/web/other
--                         (generated from kzn_recsys.cr_config)
--   ${platform_rank_expr} tie-break rank over the normalized class
--   ${audio_expr}         audio locale (from audio_col; NULL if blank)
--   ${sub_audio_locale}   locale counted as "sub"; any other non-null is "dub"
--   ${completed_ratio}    seconds / duration at or above this = completed
--
-- ${source_table} / ${catalog_table} are spliced unquoted on purpose: a
-- backticked `a.b.c` is one identifier, not a three-part name. The notebook
-- restricts both to [A-Za-z0-9_.]+.
--
-- Run once. Re-running replaces the whole table (INSERT OVERWRITE).
-- ==========================================================================

CREATE TABLE IF NOT EXISTS `${catalog}`.`${schema}`.`${attrs_table}` (
    `user_id`             STRING    NOT NULL COMMENT 'Profile id (${user_col})',
    `item_id`             STRING    NOT NULL COMMENT 'Series id (${item_col})',
    `view_date`           DATE      NOT NULL COMMENT 'Date the aggregated views fell on',
    `seconds_watched`     DOUBLE    NOT NULL COMMENT 'sum(${seconds_col}) over that day''s qualifying views (raw, not log)',
    `episodes_watched`    BIGINT    NOT NULL COMMENT 'Distinct episode ids watched that day (0 if ids unknown)',
    `completed_episodes`  BIGINT             COMMENT 'Qualifying views with seconds / duration >= ${completed_ratio}; NULL if no duration known that day',
    `max_depth_ratio`     DOUBLE             COMMENT 'max(least(seconds / duration, 1)); NULL if no duration known that day',
    `platform_top`        STRING    NOT NULL COMMENT 'Most frequent normalized platform (tv/mobile/web/other), ties by fixed order',
    `dub_views`           BIGINT    NOT NULL COMMENT 'Qualifying views with a non-null audio locale other than ${sub_audio_locale}',
    `sub_views`           BIGINT    NOT NULL COMMENT 'Qualifying views with audio locale ${sub_audio_locale}',
    `first_view_ts`       TIMESTAMP          COMMENT 'Earliest ${ts_col} that day'
)
USING DELTA
PARTITIONED BY (`view_date`)
COMMENT 'Per-interaction attributes keyed like fease_interactions_agg. Buckets and discovery lag are derived at read time.'
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact'   = 'true'
);

INSERT OVERWRITE `${catalog}`.`${schema}`.`${attrs_table}`
WITH views AS (
    SELECT
        `${user_col}`                                 AS `user_id`,
        `${item_col}`                                 AS `item_id`,
        `${date_col}`                                 AS `view_date`,
        `${ts_col}`                                   AS `view_ts`,
        CAST(`${seconds_col}` AS DOUBLE)              AS `seconds`,
        ${media_id_expr}                              AS `media_id`,
        ${duration_expr}                              AS `duration`,
        ${platform_expr}                              AS `platform`,
        ${audio_expr}                                 AS `audio`
    FROM ${source_table}
    WHERE `${subsidiary_col}` = '${subsidiary}'
      AND NULLIF(TRIM(`${user_col}`), '') IS NOT NULL
      AND `${item_col}` IS NOT NULL
      AND `${seconds_col}` >= ${min_watch_seconds}
      AND `${date_col}` < current_date()
),
per_day AS (
    SELECT
        `user_id`, `item_id`, `view_date`,
        SUM(`seconds`)                                                       AS `seconds_watched`,
        COUNT(DISTINCT `media_id`)                                           AS `episodes_watched`,
        CASE WHEN MAX(`duration`) > 0 THEN
            SUM(CASE WHEN `duration` > 0 AND `seconds` / `duration` >= ${completed_ratio} THEN 1 ELSE 0 END)
        END                                                                  AS `completed_episodes`,
        MAX(CASE WHEN `duration` > 0 THEN LEAST(`seconds` / `duration`, 1.0) END) AS `max_depth_ratio`,
        SUM(CASE WHEN `audio` IS NOT NULL AND `audio` <> '${sub_audio_locale}' THEN 1 ELSE 0 END) AS `dub_views`,
        SUM(CASE WHEN `audio` = '${sub_audio_locale}' THEN 1 ELSE 0 END)     AS `sub_views`,
        MIN(`view_ts`)                                                       AS `first_view_ts`
    FROM views
    GROUP BY `user_id`, `item_id`, `view_date`
),
platform_top AS (
    SELECT `user_id`, `item_id`, `view_date`, `platform` AS `platform_top`
    FROM (
        SELECT
            `user_id`, `item_id`, `view_date`, `platform`,
            ROW_NUMBER() OVER (
                PARTITION BY `user_id`, `item_id`, `view_date`
                ORDER BY COUNT(*) DESC, ${platform_rank_expr}
            ) AS `rn`
        FROM views
        GROUP BY `user_id`, `item_id`, `view_date`, `platform`
    )
    WHERE `rn` = 1
)
SELECT
    d.`user_id`, d.`item_id`, d.`view_date`,
    d.`seconds_watched`, d.`episodes_watched`, d.`completed_episodes`, d.`max_depth_ratio`,
    p.`platform_top`, d.`dub_views`, d.`sub_views`, d.`first_view_ts`
FROM per_day d
JOIN platform_top p
  ON d.`user_id` = p.`user_id` AND d.`item_id` = p.`item_id` AND d.`view_date` = p.`view_date`;

OPTIMIZE `${catalog}`.`${schema}`.`${attrs_table}` ZORDER BY (`user_id`);
