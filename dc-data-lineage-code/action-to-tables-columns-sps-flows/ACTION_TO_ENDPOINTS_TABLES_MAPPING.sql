
-- uses base_names
create or replace table ACTION_TO_ENDPOINTS_TABLES_MAPPING_V as
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

-- uses base_table only 

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
    -- p.COLUMN_NAME,
    bt.PRIMARY_KEY as BASE_TABLE_PRIMARY_ID,
    bt.BASE_PRIMARY_ID,
    -- bt.TYPE as COLUMN_TYPE,
    -- bt.RELATIONSHIP_KIND,
    -- bt.PRIMARY_KEY_REFERENCE,
    bt.SCHEMA_NAME
    
FROM parsed_table_columns p
LEFT JOIN CPS_DSCI_BR.BASE_TABLE bt 
    ON UPPER(p.TABLE_NAME) = UPPER(bt.TABLE_NAME)
    AND UPPER(p.COLUMN_NAME) = UPPER(bt.COLUMN_NAMES)
ORDER BY p.frontend_file, p.frontend_function, p.TABLE_NAME, p.COLUMN_NAME;

-- workbook

-- inital action_to_tables_without_columns
SELECT
    ROW_NUMBER() OVER (ORDER BY t.FRONTEND_FILE, t.FRONTEND_FUNCTION, TRIM(s.VALUE)) AS ID,
    t.FRONTEND_FILE,
    t.FRONTEND_FUNCTION,
    t.HTTP_METHOD,
    t.FRONTEND_URL,
    t.BACKEND_FILE,
    t.BACKEND_FUNCTION,
    t.BACKEND_ROUTE,
    TRIM(s.VALUE)::VARCHAR AS TABLE_NAME
FROM
    ACTION_TO_ENDPOINTS_TABLES_MAPPING AS t,
    LATERAL FLATTEN(INPUT => SPLIT(t.TABLES, ',')) AS s;


WITH flattened_data AS (
    SELECT
        t.FRONTEND_FILE,
        t.FRONTEND_FUNCTION,
        t.HTTP_METHOD,
        t.FRONTEND_URL,
        t.BACKEND_FILE,
        t.BACKEND_FUNCTION,
        t.BACKEND_ROUTE,
        TRIM(s.VALUE)::VARCHAR AS TABLE_NAME
    FROM
        ACTION_TO_ENDPOINTS_TABLES_MAPPING AS t,
        LATERAL FLATTEN(INPUT => SPLIT(t.TABLES, ',')) AS s
)
SELECT
    ROW_NUMBER() OVER (ORDER BY f.FRONTEND_FILE, f.FRONTEND_FUNCTION, f.TABLE_NAME) AS ID,
    f.FRONTEND_FILE,
    f.FRONTEND_FUNCTION,
    f.HTTP_METHOD,
    f.FRONTEND_URL,
    f.BACKEND_FILE,
    f.BACKEND_FUNCTION,
    f.BACKEND_ROUTE,
    f.TABLE_NAME,
    b.BASE_PRIMARY_ID AS TABLE_KEY
FROM flattened_data f
LEFT JOIN BASE_NAMES b ON UPPER(f.TABLE_NAME) = UPPER(b.TABLE_NAME) ;     


--with_columns_action_to_tables_columns create table query


CREATE OR REPLACE TABLE CPS_DSCI_BR.ACTION_TO_ENDPOINTS_TABLES_MAPPING (
    frontend_file VARCHAR(100),
    frontend_function VARCHAR(100),
    http_method VARCHAR(100),
    frontend_url VARCHAR(500),
    backend_file VARCHAR(100),
    backend_function VARCHAR(100),
    backend_route VARCHAR(500),
    tables VARCHAR(5000),
    Response_Fields VARCHAR(5000),
    Table_Column_Details VARCHAR(5000),
    Response_Model VARCHAR(500),
    Nested_Fields VARCHAR(5000)
);





