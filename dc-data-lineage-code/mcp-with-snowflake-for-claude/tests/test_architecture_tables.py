#!/usr/bin/env python3
"""
Test script to verify all architecture tables are accessible
"""

import asyncio
from mcp_snowflake_server import SnowflakeArchitectureServer, ARCHITECTURE_TABLES

async def test_all_tables():
    """Test access to all architecture tables."""
    server = SnowflakeArchitectureServer()
    
    try:
        print("Testing access to all architecture tables...")
        await server.initialize_connection()
        
        results = {}
        
        for table_name in ARCHITECTURE_TABLES:
            try:
                print(f"\nTesting {table_name}...")
                
                # Test basic query
                df = await server.query_table(table_name, limit=3)
                
                # Test schema
                schema = await server.get_table_schema(table_name)
                
                results[table_name] = {
                    'accessible': True,
                    'row_count': len(df),
                    'columns': list(df.columns),
                    'schema_columns': len(schema)
                }
                
                print(f"  [OK] {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                results[table_name] = {
                    'accessible': False,
                    'error': str(e)
                }
                print(f"  [ERROR] {e}")
        
        # Summary
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        
        accessible_count = sum(1 for r in results.values() if r.get('accessible', False))
        total_count = len(results)
        
        print(f"Accessible tables: {accessible_count}/{total_count}")
        
        print(f"\nAccessible tables:")
        for table_name, result in results.items():
            if result.get('accessible', False):
                print(f"  {table_name}: {result['row_count']} rows, {len(result['columns'])} columns")
        
        if accessible_count < total_count:
            print(f"\nInaccessible tables:")
            for table_name, result in results.items():
                if not result.get('accessible', False):
                    print(f"   {table_name}: {result.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_all_tables())