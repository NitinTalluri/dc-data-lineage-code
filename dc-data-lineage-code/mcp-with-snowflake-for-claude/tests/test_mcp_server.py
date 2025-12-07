#!/usr/bin/env python3
"""
Test script for the Snowflake Architecture MCP Server
"""

import asyncio
import json
from mcp_snowflake_server import SnowflakeArchitectureServer

async def test_connection():
    """Test basic connection and table access."""
    server = SnowflakeArchitectureServer()
    
    try:
        print("Testing Snowflake connection...")
        await server.initialize_connection()
        print("[OK] Connection successful")
        
        # Test querying a table
        print("\nTesting table query...")
        df = await server.query_table('BASE_NAMES', limit=5)
        print(f"[OK] Query successful - returned {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        if len(df) > 0:
            print("\nSample data:")
            print(df.head().to_string(index=False))
        
        # Test schema query
        print(f"\nTesting schema query...")
        schema = await server.get_table_schema('BASE_NAMES')
        print(f"[OK] Schema query successful - {len(schema)} columns")
        
        if len(schema) > 0:
            print("\nSchema columns available:", list(schema[0].keys()))
            for col in schema[:3]:  # Show first 3 columns
                # Handle different possible column name cases
                col_name = col.get('COLUMN_NAME') or col.get('column_name') or col.get('Column_Name')
                data_type = col.get('DATA_TYPE') or col.get('data_type') or col.get('Data_Type')
                print(f"  {col_name}: {data_type}")
        
        print("\n[OK] All tests passed!")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_connection())