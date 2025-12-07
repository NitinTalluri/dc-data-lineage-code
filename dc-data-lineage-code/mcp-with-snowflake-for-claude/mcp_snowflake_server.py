#!/usr/bin/env python3
"""
MCP Server for Snowflake Architecture Data
Provides access to specific architecture tables as source of truth for LLM responses.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy import create_engine, text
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from common import sec

# Configuration
SNOWFLAKE_ENV = 'prod'
SCHEMA_NAME = 'CPS_DSCI_BR'

# Architecture tables to expose
ARCHITECTURE_TABLES = [
    'BASE_NAMES',
    'VIEW_NAMES', 
    'BASE_TABLE',
    'VIEW_TABLE',
    'VIEW_TO_SOURCE_COLUMN_LINEAGE',
    'VIEW_TO_SOURCE_COLUMN_LINEAGE_V',
    'ACTION_TO_ENDPOINTS_TABLES_MAPPING',
    'ACTION_TO_ENDPOINTS_TABLES_MAPPING_V',
    'THOUGHTSPOT_TABLE_V',
    'THOUGHTSPOT_TABLE',
    'SP_TABLE_COLUMN_MAPPING_V',
    'SP_TABLE_COLUMN_MAPPING',
    'SP_REFERENCE_TO_FLOWS',
    'SP_MAPPING_WITH_ENDPOINTS_AND_FLOWS_V',
    'PREFECT_REFERENCE_WITH_TABLES_COLUMNS_V',
    'PREFECT_REFERENCE_WITH_TABLES_COLUMNS'
]

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

class SnowflakeArchitectureServer:
    def __init__(self):
        self.engine = None
        self.server = Server("snowflake-architecture")
        
    async def initialize_connection(self):
        """Initialize Snowflake connection."""
        try:
            self.engine = create_sf_connection_engine(SNOWFLAKE_ENV)
            print(f"Connected to Snowflake environment: {SNOWFLAKE_ENV}")
        except Exception as e:
            print(f"Failed to initialize Snowflake connection: {e}")
            raise

    async def query_table(self, table_name: str, limit: Optional[int] = None, where_clause: Optional[str] = None) -> pd.DataFrame:
        """Query a specific architecture table."""
        if not self.engine:
            await self.initialize_connection()
            
        try:
            sql = f"SELECT * FROM {SCHEMA_NAME}.{table_name}"
            
            if where_clause:
                sql += f" WHERE {where_clause}"
                
            if limit:
                sql += f" LIMIT {limit}"
            
            with self.engine.connect() as connection:
                result = pd.read_sql(text(sql), connection)
            
            return result
            
        except Exception as e:
            print(f"Error querying table {table_name}: {e}")
            raise

    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a table."""
        if not self.engine:
            await self.initialize_connection()
            
        try:
            sql = f"""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_schema = '{SCHEMA_NAME}' 
            AND table_name = '{table_name}'
            ORDER BY ordinal_position
            """
            
            with self.engine.connect() as connection:
                result = pd.read_sql(text(sql), connection)
            
            return result.to_dict('records')
            
        except Exception as e:
            print(f"Error getting schema for table {table_name}: {e}")
            raise

# Initialize server instance
arch_server = SnowflakeArchitectureServer()

@arch_server.server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available architecture tables as resources."""
    resources = []
    
    for table_name in ARCHITECTURE_TABLES:
        resources.append(
            types.Resource(
                uri=f"snowflake://architecture/{table_name}",
                name=f"Architecture Table: {table_name}",
                description=f"Access to {table_name} table data",
                mimeType="application/json"
            )
        )
    
    return resources

@arch_server.server.list_resource_templates()
async def handle_list_resource_templates() -> list[types.ResourceTemplate]:
    """List available resource templates."""
    # Return empty list - we don't use templates, just direct resources
    return []

@arch_server.server.read_resource()
async def handle_read_resource(uri: types.AnyUrl) -> str:
    """Read data from architecture tables."""
    uri_str = str(uri)
    
    if not uri_str.startswith("snowflake://architecture/"):
        raise ValueError(f"Unknown resource URI: {uri}")
    
    table_name = uri_str.split("/")[-1]
    
    if table_name not in ARCHITECTURE_TABLES:
        raise ValueError(f"Table {table_name} not in allowed architecture tables")
    
    try:
        # Get sample data (limit to 100 rows for performance)
        df = await arch_server.query_table(table_name, limit=100)
        
        # Convert to JSON
        result = {
            "table_name": table_name,
            "schema": SCHEMA_NAME,
            "row_count": len(df),
            "columns": list(df.columns),
            "sample_data": df.to_dict('records')
        }
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to read table {table_name}: {str(e)}"})

@arch_server.server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools for querying architecture data."""
    return [
        types.Tool(
            name="query_architecture_table",
            description="Query architecture tables with optional filtering and limits",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "enum": ARCHITECTURE_TABLES,
                        "description": "Name of the architecture table to query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows to return (default: 100)",
                        "minimum": 1,
                        "maximum": 10000
                    },
                    "where_clause": {
                        "type": "string",
                        "description": "Optional WHERE clause for filtering (without 'WHERE' keyword)"
                    }
                },
                "required": ["table_name"]
            }
        ),
        types.Tool(
            name="get_table_schema",
            description="Get schema information for architecture tables",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "enum": ARCHITECTURE_TABLES,
                        "description": "Name of the architecture table"
                    }
                },
                "required": ["table_name"]
            }
        ),
        types.Tool(
            name="search_architecture_data",
            description="Search across architecture tables for specific patterns or relationships",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Term to search for across tables"
                    },
                    "tables": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ARCHITECTURE_TABLES
                        },
                        "description": "Specific tables to search (default: all tables)"
                    }
                },
                "required": ["search_term"]
            }
        )
    ]

@arch_server.server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls for architecture data queries."""
    
    if name == "query_architecture_table":
        table_name = arguments["table_name"]
        limit = arguments.get("limit", 100)
        where_clause = arguments.get("where_clause")
        
        try:
            df = await arch_server.query_table(table_name, limit, where_clause)
            
            result = {
                "table_name": table_name,
                "query_info": {
                    "limit": limit,
                    "where_clause": where_clause,
                    "rows_returned": len(df)
                },
                "data": df.to_dict('records')
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, default=str)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text", 
                text=f"Error querying table {table_name}: {str(e)}"
            )]
    
    elif name == "get_table_schema":
        table_name = arguments["table_name"]
        
        try:
            schema_info = await arch_server.get_table_schema(table_name)
            
            result = {
                "table_name": table_name,
                "schema": SCHEMA_NAME,
                "columns": schema_info
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, default=str)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error getting schema for table {table_name}: {str(e)}"
            )]
    
    elif name == "search_architecture_data":
        search_term = arguments["search_term"]
        tables_to_search = arguments.get("tables", ARCHITECTURE_TABLES)
        
        try:
            search_results = {}
            
            for table_name in tables_to_search:
                # Simple text search across all columns
                df = await arch_server.query_table(table_name, limit=10000)
                
                # Convert all columns to string and search
                matches = []
                for idx, row in df.iterrows():
                    row_str = ' '.join([str(val) for val in row.values if val is not None])
                    if search_term.lower() in row_str.lower():
                        matches.append(row.to_dict())
                
                if matches:
                    search_results[table_name] = {
                        "matches_found": len(matches),
                        "matches": matches[:10]  # Limit to first 10 matches
                    }
            
            result = {
                "search_term": search_term,
                "tables_searched": tables_to_search,
                "results": search_results
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, default=str)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error searching for '{search_term}': {str(e)}"
            )]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main entry point for the MCP server."""
    
    
    # Run the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await arch_server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="snowflake-architecture",
                server_version="1.0.0",
                capabilities=types.ServerCapabilities(
                    resources=types.ResourcesCapability(),
                    tools=types.ToolsCapability()
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())