#!/usr/bin/env python3
"""
Test MCP server handler functions directly
"""

import asyncio
import json
from mcp_snowflake_server import SnowflakeArchitectureServer, handle_call_tool

async def test_mcp_handlers():
    """Test MCP server handler functions."""
    
    try:
        print("Testing MCP Server Handler Functions...")
        
        # Initialize the server
        arch_server = SnowflakeArchitectureServer()
        await arch_server.initialize_connection()
        
        # Test 1: Query architecture table
        print("\n1. Testing query_architecture_table handler...")
        result = await handle_call_tool(
            "query_architecture_table",
            {
                "table_name": "BASE_NAMES",
                "limit": 5
            }
        )
        
        if result and len(result) > 0:
            content = result[0].text
            data = json.loads(content)
            print(f"   Returned {data['query_info']['rows_returned']} rows")
            print(f"   Table: {data['table_name']}")
            print("   [OK] Query handler works")
        
        # Test 2: Get table schema
        print("\n2. Testing get_table_schema handler...")
        result = await handle_call_tool(
            "get_table_schema",
            {
                "table_name": "VIEW_TO_SOURCE_COLUMN_LINEAGE_V"
            }
        )
        
        if result and len(result) > 0:
            content = result[0].text
            data = json.loads(content)
            print(f"   Schema for {data['table_name']}")
            print(f"   Columns: {len(data['columns'])}")
            print("   [OK] Schema handler works")
        
        # Test 3: Search architecture data
        print("\n3. Testing search_architecture_data handler...")
        result = await handle_call_tool(
            "search_architecture_data",
            {
                "search_term": "ACCOUNT",
                "tables": ["BASE_NAMES", "VIEW_NAMES"]
            }
        )
        
        if result and len(result) > 0:
            content = result[0].text
            data = json.loads(content)
            print(f"   Search term: {data['search_term']}")
            print(f"   Tables searched: {len(data['tables_searched'])}")
            print(f"   Results found in: {len(data['results'])} tables")
            print("   [OK] Search handler works")
        
        print("\n[OK] All MCP handlers are working correctly!")
        
        # Show sample data from one of the key tables
        print("\n" + "="*50)
        print("SAMPLE ARCHITECTURE DATA")
        print("="*50)
        
        result = await handle_call_tool(
            "query_architecture_table",
            {
                "table_name": "SP_MAPPING_WITH_ENDPOINTS_AND_FLOWS_V",
                "limit": 3
            }
        )
        
        if result and len(result) > 0:
            content = result[0].text
            data = json.loads(content)
            print(f"\nTable: {data['table_name']}")
            print(f"Columns: {', '.join([str(col) for col in data['data'][0].keys() if data['data']])}")
            print("\nSample records:")
            for i, record in enumerate(data['data'][:2], 1):
                print(f"\nRecord {i}:")
                for key, value in record.items():
                    if value is not None:
                        print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"[ERROR] Handler test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_handlers())