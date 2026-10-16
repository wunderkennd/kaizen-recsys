-- ==========================================================================
-- Per-interaction attributes: daily incremental refresh.
--
-- Same shape as merge_fease_interactions_agg.sql: the last ${lookback_days}
-- complete days are *recomputed* from the gold table (same filters and
-- expressions as create_fease_interaction_attrs.sql) and MERGEd in. Changed
-- rows are updated (null-safe comparison of every non-key column), new rows
-- inserted, and rows that vanished from the source window deleted, so a
-- re-run for the same window reports 0 / 0 / 0.
--
-- Parameters: same as create_fease_interaction_attrs.sql plus
-- ${lookback_days}.
-- ==========================================================================

CREATE OR REPLACE TEMPORARY VIEW `fease_interaction_attrs_source` AS
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
      AND `${date_col}` >= date_sub(current_date(), ${lookback_days})
      AND `${date_col}` <  current_date()
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

MERGE INTO `${catalog}`.`${schema}`.`${attrs_table}` AS target
USING `fease_interaction_attrs_source` AS source
ON  target.`user_id`   = source.`user_id`
AND target.`item_id`   = source.`item_id`
AND target.`view_date` = source.`view_date`
WHEN MATCHED AND (
        NOT (target.`seconds_watched` <=> source.`seconds_watched`)
     OR NOT (target.`episodes_watched` <=> source.`episodes_watched`)
     OR NOT (target.`completed_episodes` <=> source.`completed_episodes`)
     OR NOT (target.`max_depth_ratio` <=> source.`max_depth_ratio`)
     OR NOT (target.`platform_top` <=> source.`platform_top`)
     OR NOT (target.`dub_views` <=> source.`dub_views`)
     OR NOT (target.`sub_views` <=> source.`sub_views`)
     OR NOT (target.`first_view_ts` <=> source.`first_view_ts`)
    )
    THEN UPDATE SET
        target.`seconds_watched` = source.`seconds_watched`,
        target.`episodes_watched` = source.`episodes_watched`,
        target.`completed_episodes` = source.`completed_episodes`,
        target.`max_depth_ratio` = source.`max_depth_ratio`,
        target.`platform_top` = source.`platform_top`,
        target.`dub_views` = source.`dub_views`,
        target.`sub_views` = source.`sub_views`,
        target.`first_view_ts` = source.`first_view_ts`
WHEN NOT MATCHED
    THEN INSERT (`user_id`, `item_id`, `view_date`, `seconds_watched`, `episodes_watched`, `completed_episodes`, `max_depth_ratio`, `platform_top`, `dub_views`, `sub_views`, `first_view_ts`)
    VALUES (source.`user_id`, source.`item_id`, source.`view_date`, source.`seconds_watched`, source.`episodes_watched`, source.`completed_episodes`, source.`max_depth_ratio`, source.`platform_top`, source.`dub_views`, source.`sub_views`, source.`first_view_ts`)
WHEN NOT MATCHED BY SOURCE
    AND target.`view_date` >= date_sub(current_date(), ${lookback_days})
    AND target.`view_date` <  current_date()
    THEN DELETE;
