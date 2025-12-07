CREATE OR REPLACE VIEW CPS_DSCI_BR.SP_TABLE_COLUMN_MAPPING_V AS
SELECT 
    ROW_NUMBER() OVER (ORDER BY sp.SP_NAME, sp.TABLE_NAME, sp.COLUMN_NAME) AS ID,
    sp.SP_NAME,
    sp.SP_SCHEMA,
    sp.TABLE_NAME,
    sp.COLUMN_NAME,
    sp.RELATIONSHIP_TYPES,
    bn.base_primary_id AS BASE_TABLE_ID,
    bt.PRIMARY_KEY AS BASE_COLUMN_ID

FROM CPS_DSCI_BR.SP_TABLE_COLUMN_MAPPING sp
LEFT JOIN BASE_NAMES bn ON (
    -- Match exact table name
    UPPER(sp.TABLE_NAME) = UPPER(bn.table_name)
    OR
    -- Match table name without schema (last part after dot)
    UPPER(SPLIT_PART(sp.TABLE_NAME, '.', -1)) = UPPER(bn.table_name)
    OR
    -- Match base table name without schema
    UPPER(sp.TABLE_NAME) = UPPER(SPLIT_PART(bn.table_name, '.', -1))
)
LEFT JOIN CPS_DSCI_BR.BASE_TABLE bt ON (
    bn.base_primary_id = bt.BASE_PRIMARY_ID
    AND UPPER(sp.COLUMN_NAME) = UPPER(bt.COLUMN_NAMES)
)
WHERE 
    -- Only show records that have at least one match
    (bn.base_primary_id IS NOT NULL OR bt.PRIMARY_KEY IS NOT NULL)
ORDER BY sp.SP_NAME, sp.TABLE_NAME, sp.COLUMN_NAME;