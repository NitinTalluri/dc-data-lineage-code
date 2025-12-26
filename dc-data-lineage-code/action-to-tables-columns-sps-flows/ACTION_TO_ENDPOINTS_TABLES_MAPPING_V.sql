create or replace view CPS_DB.CPS_DSCI_BR.ACTION_TO_ENDPOINTS_TABLES_MAPPING_V(
	ID,
	FRONTEND_FILE,
	FRONTEND_FUNCTION,
	HTTP_METHOD,
	FRONTEND_URL,
	BACKEND_FILE,
	BACKEND_FUNCTION,
	BACKEND_ROUTE,
	TABLE_NAME,
	COLUMN_NAME,
	BASE_COLUMN_PRIMARY_ID,
	BASE_TABLE_PRIMARY_ID,
	SCHEMA_NAME
) as
WITH parsed_table_columns AS (
    SELECT 
        frontend_file,
        frontend_function,
        http_method,
        frontend_url,
        backend_file,
        backend_function,
        backend_route,
        SPLIT_PART(TRIM(table_detail.VALUE), ':', 1) as TABLE_NAME,
        TRIM(column_detail.VALUE) as COLUMN_NAME
    FROM CPS_DSCI_BR.ACTION_TO_ENDPOINTS_TABLES_MAPPING,
    LATERAL FLATTEN(INPUT => SPLIT(Table_Column_Details, ';')) as table_detail,
    LATERAL FLATTEN(INPUT => SPLIT(REPLACE(REPLACE(SPLIT_PART(TRIM(table_detail.VALUE), ':', 2), '[', ''), ']', ''), ',')) as column_detail
    WHERE Table_Column_Details IS NOT NULL
    AND TRIM(table_detail.VALUE) LIKE '%:%'
)
SELECT 
    ROW_NUMBER() OVER (ORDER BY p.frontend_file, p.frontend_function, p.TABLE_NAME, p.COLUMN_NAME) AS ID,
    p.frontend_file,
    p.frontend_function,
    p.http_method,
    p.frontend_url,
    p.backend_file,
    p.backend_function,
    p.backend_route,
    p.TABLE_NAME,
    p.COLUMN_NAME,
    bt.PRIMARY_KEY as BASE_COLUMN_PRIMARY_ID,
    bn.base_primary_id as BASE_TABLE_PRIMARY_ID,
    bt.SCHEMA_NAME
FROM parsed_table_columns p
LEFT JOIN BASE_NAMES bn 
    ON UPPER(p.TABLE_NAME) = UPPER(bn.table_name)
LEFT JOIN CPS_DSCI_BR.BASE_TABLE bt 
    ON UPPER(p.TABLE_NAME) = UPPER(bt.TABLE_NAME)
    AND UPPER(p.COLUMN_NAME) = UPPER(bt.COLUMN_NAMES)
ORDER BY p.frontend_file, p.frontend_function, p.TABLE_NAME, p.COLUMN_NAME;