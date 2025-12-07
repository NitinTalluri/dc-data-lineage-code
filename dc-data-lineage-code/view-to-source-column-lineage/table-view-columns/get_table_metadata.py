#!/usr/bin/env python3
"""
Script to connect to Snowflake and retrieve table metadata for all tables in main_tables.txt.
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

def load_table_names(filename: str = 'main_tables.txt'):
    """Load table names from text file."""
    try:
        with open(filename, 'r') as f:
            tables = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(tables)} table names from {filename}")
        return tables
    except Exception as e:
        print(f"Error loading table names from {filename}: {e}")
        raise

def get_table_type(engine, table_name: str, schema_name: str = None):
    """Get table type from information_schema.tables."""
    try:
        if schema_name:
            sql = f"""
            SELECT table_name, table_type 
            FROM information_schema.tables 
            WHERE table_name = '{table_name}' 
            AND table_schema = '{schema_name}'
            """
        else:
            sql = f"""
            SELECT table_name, table_type, table_schema
            FROM information_schema.tables 
            WHERE table_name = '{table_name}'
            """
        
        with engine.connect() as connection:
            result = pd.read_sql(text(sql), connection)
        
        if not result.empty:
            # Handle different possible column name cases
            table_type_col = None
            schema_col = None
            
            for col in result.columns:
                if col.upper() in ['TABLE_TYPE', 'TABLETYPE']:
                    table_type_col = col
                if col.upper() in ['TABLE_SCHEMA', 'TABLESCHEMA']:
                    schema_col = col
            
            if table_type_col:
                if schema_name:
                    return result.iloc[0][table_type_col], schema_name
                else:
                    schema_value = result.iloc[0][schema_col] if schema_col else 'UNKNOWN'
                    return result.iloc[0][table_type_col], schema_value
            else:
                print(f"Available columns for {table_name}: {list(result.columns)}")
                return None, None
        else:
            return None, None
            
    except Exception as e:
        print(f"Error getting table type for {table_name}: {e}")
        return None, None

def describe_table(engine, table_name: str, schema_name: str):
    """Get table description (columns, types, etc.)."""
    try:
        sql = f"DESC TABLE {schema_name}.{table_name}"
        
        with engine.connect() as connection:
            result = pd.read_sql(text(sql), connection)
        
        return result
        
    except Exception as e:
        print(f"Error describing table {schema_name}.{table_name}: {e}")
        return None

def find_table_schema(engine, table_name: str):
    """Find which schema contains the table, prioritizing CPS_DSCI_API."""
    # First check CPS_DSCI_API
    table_type, schema = get_table_type(engine, table_name, 'CPS_DSCI_API')
    if table_type:
        return schema, table_type
    
    # If not found, check all schemas
    table_type, schema = get_table_type(engine, table_name)
    return schema, table_type

def process_all_tables(sf_env: str = 'prod'):
    """Process all tables from main_tables.txt and extract metadata."""
    
    # Load table names
    table_names = load_table_names()
    
    # Create connection
    engine = create_sf_connection_engine(sf_env)
    
    # Results list
    all_results = []
    primary_key_counter = 1
    
    print(f"Processing {len(table_names)} tables...")
    
    for i, table_name in enumerate(table_names, 1):
        print(f"Processing {i}/{len(table_names)}: {table_name}")
        
        try:
            # Find schema and table type
            schema_name, table_type = find_table_schema(engine, table_name)
            
            if not schema_name:
                print(f"  Table {table_name} not found in any schema")
                # Add record for missing table
                all_results.append({
                    'primary_key': primary_key_counter,
                    'table_name': table_name,
                    'table_type': 'NOT_FOUND',
                    'column_names': None,
                    'type': None,
                    'relationship_kind': None,
                    'primary_key_reference': None,
                    'schema_name': None
                })
                primary_key_counter += 1
                continue
            
            print(f"  Found in schema: {schema_name}, type: {table_type}")
            
            # Describe table
            desc_result = describe_table(engine, table_name, schema_name)
            
            if desc_result is None or desc_result.empty:
                print(f"  Could not describe table {table_name}")
                # Add record for table that couldn't be described
                all_results.append({
                    'primary_key': primary_key_counter,
                    'table_name': table_name,
                    'table_type': table_type,
                    'column_names': None,
                    'type': None,
                    'relationship_kind': None,
                    'primary_key_reference': None,
                    'schema_name': schema_name
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
                    'table_name': table_name,
                    'table_type': table_type,
                    'column_names': column_name,
                    'type': data_type,
                    'relationship_kind': kind,
                    'primary_key_reference': primary_key_ref,
                    'schema_name': schema_name
                })
                primary_key_counter += 1
            
            print(f"  Added {len(desc_result)} columns")
            
        except Exception as e:
            print(f"  Error processing {table_name}: {e}")
            # Add error record
            all_results.append({
                'primary_key': primary_key_counter,
                'table_name': table_name,
                'table_type': 'ERROR',
                'column_names': None,
                'type': None,
                'relationship_kind': None,
                'primary_key_reference': None,
                'schema_name': None
            })
            primary_key_counter += 1
    
    return all_results

def save_results_to_csv(results, filename: str = 'view_metadata.csv'):
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
        print("=== TABLE METADATA EXTRACTION ===")
        print(f"Environment: {sf_env}")
        
        # Process all tables
        results = process_all_tables(sf_env)
        
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