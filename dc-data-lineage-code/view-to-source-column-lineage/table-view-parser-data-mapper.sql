create  or replace table VIEW_TO_SOURCE_COLUMN_MAPPING AS
WITH lineage AS (
  SELECT *
  FROM CPS_DB.CPS_DSCI_BR.VIEW_TO_SOURCE_COLUMN_LINEAGE
),

/* 1) Split on ';' to get one source ref per row */
exploded AS (
  SELECT
    l.VIEW_NAME,
    l.VIEW_COLUMN,
    l.SOURCE_COLUMN,
    TRIM(part.value) AS src_part
  FROM lineage l,
       LATERAL SPLIT_TO_TABLE(l.SOURCE_TABLE, ';') part
),

/* 2) Remove trailing alias "AS X" if present */
no_alias AS (
  SELECT
    VIEW_NAME,
    VIEW_COLUMN,
    SOURCE_COLUMN,
    REGEXP_REPLACE(src_part, '\s+AS\s+\w+\s*$', '') AS src_qual
  FROM exploded
),

/* 3) Parse schema/table from the qualified name */
parsed AS (
  SELECT
    VIEW_NAME,
    VIEW_COLUMN,
    SOURCE_COLUMN,
    src_qual,
    /* table = last identifier */
    REGEXP_SUBSTR(src_qual, '([^.]+)$') AS SOURCE_TABLE,
    /* schema if present in last two identifiers */
    REGEXP_SUBSTR(src_qual, '([^.]+)\.([^.]+)$', 1, 1, 'e', 1) AS SRC_SCHEMA
  FROM no_alias
)

/* 4) Join to BASE_TABLE on schema + table + column (case-insensitive) */
SELECT 
  p.VIEW_NAME,
  p.VIEW_COLUMN,
  p.SOURCE_TABLE,
  p.SOURCE_COLUMN,
  b.primary_key AS table_key,         -- one per (schema, table, column)
  v.primary_key AS view_key,
  b.SCHEMA_NAME AS BASE_SCHEMA,
  -- b.TABLE_NAME  AS BASE_TABLE,
  p.src_qual  AS SOURCE_TABLE_FULL
FROM parsed p
LEFT JOIN CPS_DSCI_BR.BASE_TABLE b
  ON  UPPER(b.TABLE_NAME)  = UPPER(p.SOURCE_TABLE)
 -- AND (p.SRC_SCHEMA IS NULL OR UPPER(b.SCHEMA_NAME) = UPPER(p.SRC_SCHEMA))
  AND UPPER(b.COLUMN_NAMES) = UPPER(p.SOURCE_COLUMN)        -- <<< added
LEFT JOIN view_table v
  ON UPPER(p.VIEW_NAME)   = UPPER(v.VIEW_NAME)
 AND UPPER(p.VIEW_COLUMN) = UPPER(v.COLUMN_NAMES);

 select * from VIEW_TO_SOURCE_COLUMN_MAPPING;
 
select * from (select * from VIEW_TO_SOURCE_COLUMN_LINEAGE limit 100) as  p
LEFT JOIN CPS_DSCI_BR.BASE_TABLE b
  ON  UPPER(b.TABLE_NAME)  = UPPER(p.SOURCE_TABLE)
    -- AND (p.SRC_SCHEMA IS NULL OR UPPER(b.SCHEMA_NAME) = UPPER(p.SRC_SCHEMA))
  AND UPPER(b.COLUMN_NAMES) = UPPER(p.SOURCE_COLUMN);