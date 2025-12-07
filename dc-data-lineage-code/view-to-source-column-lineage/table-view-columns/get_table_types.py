#!/usr/bin/env python3
"""
Simple script to get table types for tables listed in analysis.txt.
Outputs a CSV with table_name and table_type columns.
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

def load_table_names(filename: str = r'C:\dev\table-column\analysis.txt'):
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
            
            for col in result.columns:
                if col.upper() in ['TABLE_TYPE', 'TABLETYPE']:
                    table_type_col = col
                    break
            
            if table_type_col:
                return result.iloc[0][table_type_col]
            else:
                print(f"Available columns for {table_name}: {list(result.columns)}")
                return None
        else:
            return None
            
    except Exception as e:
        print(f"Error getting table type for {table_name}: {e}")
        return None

def find_table_type(engine, table_name: str):
    """Find table type, prioritizing CPS_DSCI_API schema."""
    # First check CPS_DSCI_API
    table_type = get_table_type(engine, table_name, 'CPS_DSCI_API')
    if table_type:
        return table_type
    
    # If not found, check all schemas
    table_type = get_table_type(engine, table_name)
    return table_type

def process_tables_for_types(sf_env: str = 'prod'):
    """Process all tables and get just table names and types."""
    
    # Load table names
    table_names = load_table_names()
    
    # Create connection
    engine = create_sf_connection_engine(sf_env)
    
    # Results list
    results = []
    
    print(f"Processing {len(table_names)} tables...")
    
    for i, table_name in enumerate(table_names, 1):
        print(f"Processing {i}/{len(table_names)}: {table_name}")
        
        try:
            # Find table type
            table_type = find_table_type(engine, table_name)
            
            if not table_type:
                print(f"  Table {table_name} not found")
                table_type = 'NOT_FOUND'
            else:
                print(f"  Found type: {table_type}")
            
            results.append({
                'table_name': table_name,
                'table_type': table_type
            })
            
        except Exception as e:
            print(f"  Error processing {table_name}: {e}")
            results.append({
                'table_name': table_name,
                'table_type': 'ERROR'
            })
    
    return results

def save_results_to_csv(results, filename: str = 'table_types.csv'):
    """Save results to CSV file."""
    try:
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        print(f"Total records: {len(df)}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Total tables processed: {len(df)}")
        
        print(f"  Table types:")
        type_counts = df['table_type'].value_counts()
        for table_type, count in type_counts.items():
            print(f"    {table_type}: {count} tables")
        
        return df
        
    except Exception as e:
        print(f"Error saving results: {e}")
        raise

if __name__ == "__main__":
    sf_env = 'prod'
    
    try:
        print("=== TABLE TYPE EXTRACTION ===")
        print(f"Environment: {sf_env}")
        print(f"Reading from: C:\\dev\\table-column\\analysis.txt")
        
        # Process all tables
        results = process_tables_for_types(sf_env)
        
        # Save to CSV
        df = save_results_to_csv(results)
        
        print(f"\n=== EXTRACTION COMPLETE ===")
        print(f"Check 'table_types.csv' for results")
        
        # Show sample of results
        if len(df) > 0:
            print(f"\nSample results:")
            print(df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"Script failed: {e}")