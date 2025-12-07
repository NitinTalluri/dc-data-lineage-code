CREATE OR REPLACE VIEW CPS_DSCI_BR.SP_MAPPING_WITH_ENDPOINTS_AND_FLOWS_V AS
SELECT 
    ROW_NUMBER() OVER (ORDER BY sp.SP_NAME, sp.TABLE_NAME, sp.COLUMN_NAME) AS ID,
    sp.SP_NAME,
    sp.SP_SCHEMA,
    sp.TABLE_NAME,
    sp.COLUMN_NAME,
    sp.RELATIONSHIP_TYPES,
    sp.BASE_TABLE_ID,
    sp.BASE_COLUMN_ID,
    CASE 
        WHEN flows.sp_name IS NOT NULL AND endpoints.sp_found IS NOT NULL THEN 'flow_triggered,api_triggered'
        WHEN flows.sp_name IS NOT NULL THEN 'flow_triggered'
        WHEN endpoints.sp_found IS NOT NULL THEN 'api_triggered'
        ELSE ''
    END AS TRIGGER_TYPE,
    CASE 
        WHEN flows.sp_name IS NOT NULL THEN flows.repo_names
        WHEN endpoints.sp_found IS NOT NULL THEN endpoints.backend_routes
        ELSE NULL
    END AS SOURCE_REFERENCE
FROM CPS_DSCI_BR.SP_TABLE_COLUMN_MAPPING_V sp
LEFT JOIN (
    SELECT 
        sp_name,
        LISTAGG(DISTINCT repo_name, ',') WITHIN GROUP (ORDER BY repo_name) AS repo_names
    FROM sp_reference_to_flows 
    GROUP BY sp_name
) flows ON UPPER(sp.SP_NAME) = UPPER(flows.sp_name)
LEFT JOIN (
    WITH sp_extraction AS (
        SELECT 
            aet.BACKEND_ROUTE,
            aet.STORED_PROCEDURES,
            -- Extract individual SPs from comma-separated values
            TRIM(sp_individual.value) AS raw_sp_name
        FROM ACTION_TO_ENDPOINTS_TABLES_MAPPING aet,
        LATERAL FLATTEN(INPUT => SPLIT(aet.STORED_PROCEDURES, ',')) sp_individual
        WHERE aet.STORED_PROCEDURES IS NOT NULL 
        AND TRIM(sp_individual.value) != ''
        AND TRIM(sp_individual.value) NOT IN ('IDENTIFIER', 'logged_user', '', 'NULL', 'null')
    )
    SELECT 
        UPPER(
            CASE 
                -- Remove V2ProcedureNames. prefix
                WHEN raw_sp_name LIKE 'V2ProcedureNames.%' 
                THEN SUBSTR(raw_sp_name, 18)
                -- Handle other prefixes if needed
                ELSE raw_sp_name
            END
        ) AS sp_found,
        LISTAGG(DISTINCT BACKEND_ROUTE, ',') WITHIN GROUP (ORDER BY BACKEND_ROUTE) AS backend_routes
    FROM sp_extraction
    WHERE raw_sp_name IS NOT NULL
    AND raw_sp_name != ''
    GROUP BY UPPER(
        CASE 
            WHEN raw_sp_name LIKE 'V2ProcedureNames.%' 
            THEN SUBSTR(raw_sp_name, 18)
            ELSE raw_sp_name
        END
    )
) endpoints ON UPPER(sp.SP_NAME) = endpoints.sp_found
ORDER BY sp.SP_NAME, sp.TABLE_NAME, sp.COLUMN_NAME;