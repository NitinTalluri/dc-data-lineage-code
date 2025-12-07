#!/usr/bin/env python3
"""
Enhanced script to connect to Snowflake and retrieve table metadata for all tables in main_tables_full.txt.
Handles tables across multiple databases and schemas.
Uses the same connection pattern as get_stored_procedures.py.
"""

import pandas as pd
from sqlalchemy import create_engine, text
from common import sec
import os

def get_correct_schema(env: str) -> str:
    """Get the correct schema based on environment."""
    if env == 'prod':
        return 'CPS_DSCI_API'
    else:
        return 'CPS_DSCI_BR'

def check_env(env: str) -> str:
    """Check and return the correct connection name for environment."""
    if env == "dev":
        cn = "dev_cps_dsci_etl_svc"
    elif env == "stage":
        cn = "stg_cps_dsci_etl_svc"
    elif env == "prod":
        cn = "prd_cps_dsci_etl_svc"
    else:
        cn = env
    return cn

def create_sf_connection_engine(sf_env: str):
    """
    Create Snowflake connection engine using existing strategy.
    
    Args:
        sf_env: Snowflake environment
        
    Returns:
        SQLAlchemy engine
    """
    try:
        cn = check_env(sf_env)
        correct_schema = get_correct_schema(sf_env)
        engine = create_engine(
            sec.get_sf_pw(cn, 'CPS_DSCI_ETL_EXT2_WH', correct_schema)
        )
        return engine
        
    except Exception as e:
        print(f"Failed to create Snowflake connection: {e}")
        raise

def parse_table_name(full_table_name: str):
    """Parse full table name into database, schema, and table components."""
    parts = full_table_name.split('.')
    
    if len(parts) == 3:
        # Format: DATABASE.SCHEMA.TABLE
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        # Format: SCHEMA.TABLE (assume current database)
        return None, parts[0], parts[1]
    elif len(parts) == 1:
        # Format: TABLE (assume current database and schema)
        return None, None, parts[0]
    else:
        # Invalid format
        return None, None, full_table_name

def load_table_names(filename: str = 'main_tables_full.txt'):
    """Load table names from text file."""
    try:
        with open(filename, 'r') as f:
            tables = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(tables)} table names from {filename}")
        return tables
    except Exception as e:
        print(f"Error loading table names from {filename}: {e}")
        raise

def get_table_type_cross_database(engine, table_name: str, schema_name: str = None, database_name: str = None):
    """Get table type from information_schema.tables, handling cross-database queries."""
    try:
        # Build the query based on available information
        if database_name and schema_name:
            # Query specific database and schema
            sql = f"""
            SELECT table_name, table_type, table_schema, table_catalog
            FROM {database_name}.information_schema.tables 
            WHERE UPPER(table_name) = UPPER('{table_name}') 
            AND UPPER(table_schema) = UPPER('{schema_name}')
            """
        elif schema_name:
            # Query specific schema in current database
            sql = f"""
            SELECT table_name, table_type, table_schema, table_catalog
            FROM information_schema.tables 
            WHERE UPPER(table_name) = UPPER('{table_name}') 
            AND UPPER(table_schema) = UPPER('{schema_name}')
            """
        else:
            # Search across all schemas in current database
            sql = f"""
            SELECT table_name, table_type, table_schema, table_catalog
            FROM information_schema.tables 
            WHERE UPPER(table_name) = UPPER('{table_name}')
            """
        
        with engine.connect() as connection:
            result = pd.read_sql(text(sql), connection)
        
        if not result.empty:
            # Handle different possible column name cases
            table_type_col = None
            schema_col = None
            catalog_col = None
            
            for col in result.columns:
                if col.upper() in ['TABLE_TYPE', 'TABLETYPE']:
                    table_type_col = col
                if col.upper() in ['TABLE_SCHEMA', 'TABLESCHEMA']:
                    schema_col = col
                if col.upper() in ['TABLE_CATALOG', 'TABLECATALOG']:
                    catalog_col = col
            
            if table_type_col:
                schema_value = result.iloc[0][schema_col] if schema_col else schema_name or 'UNKNOWN'
                catalog_value = result.iloc[0][catalog_col] if catalog_col else database_name or 'UNKNOWN'
                return result.iloc[0][table_type_col], schema_value, catalog_value
            else:
                print(f"Available columns for {table_name}: {list(result.columns)}")
                return None, None, None
        else:
            return None, None, None
            
    except Exception as e:
        print(f"Error getting table type for {table_name}: {e}")
        return None, None, None

def describe_table_cross_database(engine, table_name: str, schema_name: str, database_name: str = None):
    """Get table description (columns, types, etc.), handling cross-database queries."""
    try:
        # Build the full table reference
        if database_name:
            full_table_ref = f"{database_name}.{schema_name}.{table_name}"
        else:
            full_table_ref = f"{schema_name}.{table_name}"
            
        sql = f"DESC TABLE {full_table_ref}"
        
        with engine.connect() as connection:
            result = pd.read_sql(text(sql), connection)
        
        return result
        
    except Exception as e:
        print(f"Error describing table {full_table_ref}: {e}")
        return None

def find_table_location(engine, full_table_name: str):
    """Find table location, handling full database.schema.table names."""
    database_name, schema_name, table_name = parse_table_name(full_table_name)
    
    # If we have explicit database and schema, try that first
    if database_name and schema_name:
        try:
            table_type, found_schema, found_database = get_table_type_cross_database(
                engine, table_name, schema_name, database_name
            )
            if table_type:
                return found_database, found_schema, table_type
        except Exception as e:
            print(f"  Could not access {database_name}.{schema_name}: {e}")
    
    # If we have schema but no database, try current database
    if schema_name and not database_name:
        try:
            table_type, found_schema, found_database = get_table_type_cross_database(
                engine, table_name, schema_name
            )
            if table_type:
                return found_database, found_schema, table_type
        except Exception as e:
            print(f"  Could not access schema {schema_name}: {e}")
    
    # Fallback: search in default schemas
    default_schemas = ['CPS_DSCI_API', 'CPS_DSCI_BR', 'CPS_DSCI_EBV', 'CPS_BIA_BR']
    
    for schema in default_schemas:
        try:
            table_type, found_schema, found_database = get_table_type_cross_database(
                engine, table_name, schema
            )
            if table_type:
                return found_database, found_schema, table_type
        except Exception as e:
            continue
    
    # Last resort: search all schemas in current database
    try:
        table_type, found_schema, found_database = get_table_type_cross_database(engine, table_name)
        if table_type:
            return found_database, found_schema, table_type
    except Exception as e:
        pass
    
    return None, None, None

def process_all_tables(sf_env: str = 'prod', filename: str = 'main_tables_full.txt'):
    """Process all tables from the specified file and extract metadata."""
    
    # Load table names
    table_names = load_table_names(filename)
    
    # Create connection
    engine = create_sf_connection_engine(sf_env)
    
    # Results list
    all_results = []
    primary_key_counter = 1
    
    print(f"Processing {len(table_names)} tables...")
    
    for i, full_table_name in enumerate(table_names, 1):
        print(f"Processing {i}/{len(table_names)}: {full_table_name}")
        
        try:
            # Parse the table name to get components
            database_name, schema_name, table_name = parse_table_name(full_table_name)
            print(f"  Parsed as - Database: {database_name}, Schema: {schema_name}, Table: {table_name}")
            
            # Find table location and type
            found_database, found_schema, table_type = find_table_location(engine, full_table_name)
            
            if not found_schema:
                print(f"  Table {full_table_name} not found in any accessible location")
                # Add record for missing table
                all_results.append({
                    'primary_key': primary_key_counter,
                    'table_name': table_name,  # Just the table name, not full path
                    'table_type': 'NOT_FOUND',
                    'column_names': None,
                    'type': None,
                    'relationship_kind': None,
                    'primary_key_reference': None,
                    'schema_name': None
                })
                primary_key_counter += 1
                continue
            
            print(f"  Found in database: {found_database}, schema: {found_schema}, type: {table_type}")
            
            # Describe table
            desc_result = describe_table_cross_database(engine, table_name, found_schema, found_database)
            
            if desc_result is None or desc_result.empty:
                print(f"  Could not describe table {full_table_name}")
                # Add record for table that couldn't be described
                all_results.append({
                    'primary_key': primary_key_counter,
                    'table_name': table_name,  # Just the table name, not full path
                    'table_type': table_type,
                    'column_names': None,
                    'type': None,
                    'relationship_kind': None,
                    'primary_key_reference': None,
                    'schema_name': found_schema
                })
                primary_key_counter += 1
                continue
            
            # Debug: show available columns for first successful table
            if len(all_results) == 0:
                print(f"  DEBUG: Available columns in DESC result: {list(desc_result.columns)}")
                if not desc_result.empty:
                    print(f"  DEBUG: First row data: {dict(desc_result.iloc[0])}")
            
            # Process each column
            for idx, row in desc_result.iterrows():
                # Extract column information - DESC TABLE returns specific column names
                # Map 'name' to 'column_names' and 'type' to 'type'
                column_name = row['name'] if 'name' in row else 'Unknown'
                data_type = row['type'] if 'type' in row else 'Unknown'
                kind = row['kind'] if 'kind' in row else None  # This maps to relationship_kind
                # Handle primary key reference - keep the actual value ('Y' or 'N')
                primary_key_ref = row['primary key'] if 'primary key' in row else None
                
                all_results.append({
                    'primary_key': primary_key_counter,
                    'table_name': table_name,  # Just the table name, not full path
                    'table_type': table_type,
                    'column_names': column_name,
                    'type': data_type,
                    'relationship_kind': kind,
                    'primary_key_reference': primary_key_ref,
                    'schema_name': found_schema
                })
                primary_key_counter += 1
            
            print(f"  Added {len(desc_result)} columns")
            
        except Exception as e:
            print(f"  Error processing {full_table_name}: {e}")
            # Add error record
            all_results.append({
                'primary_key': primary_key_counter,
                'table_name': table_name,  # Just the table name, not full path
                'table_type': 'ERROR',
                'column_names': None,
                'type': None,
                'relationship_kind': None,
                'primary_key_reference': None,
                'schema_name': None
            })
            primary_key_counter += 1
    
    return all_results

def save_results_to_csv(results, filename: str = 'table_metadata.csv'):
    """Save results to CSV file."""
    try:
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        print(f"Total records: {len(df)}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Unique tables processed: {df['table_name'].nunique()}")
        print(f"  Total columns: {len(df[df['column_names'].notna()])}")
        
        print(f"  Tables by schema:")
        schema_counts = df['schema_name'].value_counts()
        for schema, count in schema_counts.items():
            if schema:
                print(f"    {schema}: {count} records")
        
        print(f"  Table types:")
        type_counts = df['table_type'].value_counts()
        for table_type, count in type_counts.items():
            print(f"    {table_type}: {count} records")
        
        return df
        
    except Exception as e:
        print(f"Error saving results: {e}")
        raise

if __name__ == "__main__":
    sf_env = 'prod'
    
    try:
        print("=== ENHANCED TABLE METADATA EXTRACTION ===")
        print(f"Environment: {sf_env}")
        
        # Process all tables
        results = process_all_tables(sf_env, 'main_tables_full.txt')
        
        # Save to CSV
        df = save_results_to_csv(results)
        
        print(f"\n=== EXTRACTION COMPLETE ===")
        print(f"Check 'table_metadata.csv' for complete results")
        
        # Show sample of results
        if len(df) > 0:
            print(f"\nSample results (first 10 rows):")
            print(df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"Script failed: {e}")