-- ==========================================================================
-- Per-series attributes: one row per series (item_id). Small (thousands of
-- rows), so every refresh rebuilds it in full (INSERT OVERWRITE); there is
-- no MERGE.
--
-- premiere_date is the series' earliest catalog availability date when the
-- catalog has one ('catalog'), else the earliest view_date in the agg table
-- ('first_view'), which is a lower bound on how late the series was really
-- discovered and is flagged so a consumer can drop it. Series that exist only
-- in the catalog or only in the agg table are both kept.
--
-- Parameters (substituted by databricks/src/00_refresh_interactions_agg.py,
-- target=item_attrs):
--   ${catalog}.${schema}.${item_attrs_table}  target table
--   ${catalog}.${schema}.${table}             the agg table (first-view fallback)
--   ${catalog_table}                          series / media catalog
--   ${catalog_series_col} ${catalog_media_id_col}
--   ${catalog_duration_expr}   media length in seconds (NULL if column blank)
--   ${catalog_available_expr}  first CR availability (NULL if column blank)
--   ${catalog_air_year_expr}   original air year (NULL if column blank)
--
-- The source and catalog table names are spliced unquoted on purpose: a
-- backticked `a.b.c` is one identifier, not a three-part name. The notebook
-- restricts both to [A-Za-z0-9_.]+.
-- ==========================================================================

CREATE TABLE IF NOT EXISTS `${catalog}`.`${schema}`.`${item_attrs_table}` (
    `item_id`              STRING  NOT NULL COMMENT 'Series id',
    `premiere_date`        DATE             COMMENT 'Earliest CR availability (see premiere_source)',
    `premiere_source`      STRING           COMMENT 'catalog | first_view',
    `episode_count`        BIGINT           COMMENT 'Distinct media ids in the catalog for the series',
    `avg_episode_seconds`  DOUBLE           COMMENT 'Mean media duration in seconds',
    `first_air_year`       INT              COMMENT 'Earliest original air year (era bucket)'
)
USING DELTA
COMMENT 'Per-series attributes for discovery lag and era / length buckets. Rebuilt in full on each refresh.';

INSERT OVERWRITE `${catalog}`.`${schema}`.`${item_attrs_table}`
WITH catalog_series AS (
    SELECT
        CAST(`${catalog_series_col}` AS STRING)            AS `item_id`,
        MIN(${catalog_available_expr})                     AS `catalog_premiere`,
        COUNT(DISTINCT `${catalog_media_id_col}`)          AS `episode_count`,
        AVG(${catalog_duration_expr})                      AS `avg_episode_seconds`,
        MIN(${catalog_air_year_expr})                      AS `first_air_year`
    FROM ${catalog_table}
    WHERE `${catalog_series_col}` IS NOT NULL
    GROUP BY CAST(`${catalog_series_col}` AS STRING)
),
first_view AS (
    SELECT `item_id`, MIN(`view_date`) AS `first_view_date`
    FROM `${catalog}`.`${schema}`.`${table}`
    GROUP BY `item_id`
)
SELECT
    COALESCE(c.`item_id`, f.`item_id`)                     AS `item_id`,
    COALESCE(c.`catalog_premiere`, f.`first_view_date`)    AS `premiere_date`,
    CASE
        WHEN c.`catalog_premiere` IS NOT NULL THEN 'catalog'
        WHEN f.`first_view_date` IS NOT NULL THEN 'first_view'
    END                                                    AS `premiere_source`,
    c.`episode_count`,
    c.`avg_episode_seconds`,
    CAST(c.`first_air_year` AS INT)                        AS `first_air_year`
FROM catalog_series c
FULL OUTER JOIN first_view f
  ON c.`item_id` = f.`item_id`;
