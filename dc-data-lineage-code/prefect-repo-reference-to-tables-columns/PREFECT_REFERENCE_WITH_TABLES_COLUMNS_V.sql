CREATE OR REPLACE VIEW CPS_DSCI_BR.PREFECT_REFERENCE_WITH_TABLES_COLUMNS_V AS
SELECT 
    pref.repo_name,
    pref.function_name,
    pref.table_name,
    pref.column_name,
    pref.file_name,
    
    -- Add base_key and view_key through joins
    bt.PRIMARY_KEY AS base_key,
    vt.PRIMARY_KEY AS view_key
    
FROM CPS_DSCI_BR.PREFECT_REFERENCE_WITH_TABLES_COLUMNS pref

-- Left join with BASE_TABLE to get base_key
LEFT JOIN CPS_DSCI_BR.BASE_TABLE bt 
    ON UPPER(TRIM(pref.table_name)) = UPPER(TRIM(bt.TABLE_NAME))
    AND (pref.column_name IS NULL 
         OR TRIM(pref.column_name) = '' 
         OR UPPER(TRIM(pref.column_name)) = UPPER(TRIM(bt.COLUMN_NAMES)))

-- Left join with VIEW_TABLE to get view_key  
LEFT JOIN CPS_DSCI_BR.VIEW_TABLE vt 
    ON UPPER(TRIM(pref.table_name)) = UPPER(TRIM(vt.VIEW_NAME))
    AND (pref.column_name IS NULL 
         OR TRIM(pref.column_name) = '' 
         OR UPPER(TRIM(pref.column_name)) = UPPER(TRIM(vt.COLUMN_NAMES)));