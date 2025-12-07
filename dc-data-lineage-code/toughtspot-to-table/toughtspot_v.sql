
create or replace view THOUGHTSPOT_TABLE_V as
SELECT 
    lm.table_name,
    lm.liveboard_name,
    lm.guid,
    lm.schema_name,
    lm.type,
    bn.base_primary_id,
    vn.view_primary_id,
    bn.table_name as full_base_table_name,
    
    vn.view_name
FROM THOUGHTSPOT_TABLE lm
LEFT JOIN BASE_NAMES bn ON (
    lm.table_name = SPLIT_PART(bn.table_name, '.', -1)  -- Extract last part after dot
    OR lm.table_name = bn.table_name  -- Direct match
    OR CONCAT(lm.schema_name, '.', lm.table_name) = bn.table_name  -- schema.table match
) AND lm.type = 'BASE TABLE'
LEFT JOIN VIEW_NAMES vn ON (
    lm.table_name = vn.view_name
    AND lm.type = 'VIEW'
);