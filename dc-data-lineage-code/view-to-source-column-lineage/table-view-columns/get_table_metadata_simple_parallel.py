#!/usr/bin/env python3
"""
Simple parallel version of your enhanced script.
Just adds ThreadPoolExecutor to your existing working code.
"""

import pandas as pd
from sqlalchemy import create_engine, text
from common import sec
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# All your existing functions (unchanged)
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
    """Create Snowflake connection engine using existing strategy."""
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
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return None, parts[0], parts[1]
    elif len(parts) == 1:
        return None, None, parts[0]
    else:
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
        if database_name and schema_name:
            sql = f"""
            SELECT table_name, table_type, table_schema, table_catalog
            FROM {database_name}.information_schema.tables 
            WHERE UPPER(table_name) = UPPER('{table_name}') 
            AND UPPER(table_schema) = UPPER('{schema_name}')
            """
        elif schema_name:
            sql = f"""
            SELECT table_name, table_type, table_schema, table_catalog
            FROM information_schema.tables 
            WHERE UPPER(table_name) = UPPER('{table_name}') 
            AND UPPER(table_schema) = UPPER('{schema_name}')
            """
        else:
            sql = f"""
            SELECT table_name, table_type, table_schema, table_catalog
            FROM information_schema.tables 
            WHERE UPPER(table_name) = UPPER('{table_name}')
            """
        
        with engine.connect() as connection:
            result = pd.read_sql(text(sql), connection)
        
        if not result.empty:
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
                return None, None, None
        else:
            return None, None, None
            
    except Exception as e:
        print(f"Error getting table type for {table_name}: {e}")
        return None, None, None

def describe_table_cross_database(engine, table_name: str, schema_name: str, database_name: str = None):
    """Get table description (columns, types, etc.), handling cross-database queries."""
    try:
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
    
    if database_name and schema_name:
        try:
            table_type, found_schema, found_database = get_table_type_cross_database(
                engine, table_name, schema_name, database_name
            )
            if table_type:
                return found_database, found_schema, table_type
        except Exception as e:
            pass
    
    if schema_name and not database_name:
        try:
            table_type, found_schema, found_database = get_table_type_cross_database(
                engine, table_name, schema_name
            )
            if table_type:
                return found_database, found_schema, table_type
        except Exception as e:
            pass
    
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
    
    try:
        table_type, found_schema, found_database = get_table_type_cross_database(engine, table_name)
        if table_type:
            return found_database, found_schema, table_type
    except Exception as e:
        pass
    
    return None, None, None

# NEW: Single table processing function for parallel execution
def process_single_table(args):
    """Process a single table - designed for parallel execution."""
    full_table_name, sf_env, table_index, total_tables = args
    
    try:
        # Create engine for this thread
        engine = create_sf_connection_engine(sf_env)
        
        print(f"Processing {table_index}/{total_tables}: {full_table_name}")
        
        # Parse table name
        database_name, schema_name, table_name = parse_table_name(full_table_name)
        
        # Find table location and type
        found_database, found_schema, table_type = find_table_location(engine, full_table_name)
        
        if not found_schema:
            return [{
                'table_name': table_name,
                'table_type': 'NOT_FOUND',
                'column_names': None,
                'type': None,
                'relationship_kind': None,
                'primary_key_reference': None,
                'schema_name': None
            }]
        
        # Describe table
        desc_result = describe_table_cross_database(engine, table_name, found_schema, found_database)
        
        if desc_result is None or desc_result.empty:
            return [{
                'table_name': table_name,
                'table_type': table_type,
                'column_names': None,
                'type': None,
                'relationship_kind': None,
                'primary_key_reference': None,
                'schema_name': found_schema
            }]
        
        # Process each column
        table_results = []
        for idx, row in desc_result.iterrows():
            column_name = row['name'] if 'name' in row else 'Unknown'
            data_type = row['type'] if 'type' in row else 'Unknown'
            kind = row['kind'] if 'kind' in row else None
            primary_key_ref = row['primary key'] if 'primary key' in row else None
            
            table_results.append({
                'table_name': table_name,
                'table_type': table_type,
                'column_names': column_name,
                'type': data_type,
                'relationship_kind': kind,
                'primary_key_reference': primary_key_ref,
                'schema_name': found_schema
            })
        
        print(f"  Completed {full_table_name}: {len(table_results)} columns")
        return table_results
        
    except Exception as e:
        print(f"  Error processing {full_table_name}: {e}")
        _, _, table_name = parse_table_name(full_table_name)
        return [{
            'table_name': table_name,
            'table_type': 'ERROR',
            'column_names': None,
            'type': None,
            'relationship_kind': None,
            'primary_key_reference': None,
            'schema_name': None
        }]

def process_all_tables_parallel(sf_env: str = 'prod', filename: str = 'main_tables_full.txt', max_workers: int = 8):
    """Process all tables using simple parallel processing."""
    
    start_time = time.time()
    
    # Load table names
    table_names = load_table_names(filename)
    
    print(f"Processing {len(table_names)} tables with {max_workers} parallel workers...")
    
    # Prepare arguments for parallel processing
    args_list = [
        (table_name, sf_env, i+1, len(table_names)) 
        for i, table_name in enumerate(table_names)
    ]
    
    # Process tables in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_table = {
            executor.submit(process_single_table, args): args[0] 
            for args in args_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_table):
            table_name = future_to_table[future]
            try:
                table_results = future.result()
                all_results.extend(table_results)
            except Exception as e:
                print(f"Failed to process {table_name}: {e}")
    
    # Add primary keys
    for i, result in enumerate(all_results, 1):
        result['primary_key'] = i
    
    elapsed_time = time.time() - start_time
    print(f"\nParallel processing completed in {elapsed_time:.2f} seconds")
    print(f"Average time per table: {elapsed_time/len(table_names):.3f} seconds")
    
    return all_results

def save_results_to_csv(results, filename: str = 'table_metadata_parallel.csv'):
    """Save results to CSV file."""
    try:
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        print(f"Total records: {len(df)}")
        
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
    max_workers = 8  # Adjust based on your system and Snowflake connection limits
    
    try:
        print("=== SIMPLE PARALLEL TABLE METADATA EXTRACTION ===")
        print(f"Environment: {sf_env}")
        print(f"Max workers: {max_workers}")
        
        # Process all tables in parallel
        results = process_all_tables_parallel(sf_env, 'main_tables_full.txt', max_workers)
        
        # Save to CSV
        df = save_results_to_csv(results)
        
        print(f"\n=== EXTRACTION COMPLETE ===")
        print(f"Check 'table_metadata_parallel.csv' for complete results")
        
        # Show sample of results
        if len(df) > 0:
            print(f"\nSample results (first 10 rows):")
            print(df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"Script failed: {e}")