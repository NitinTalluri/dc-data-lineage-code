# LLM with Claude - Snowflake Architecture MCP Server

A Python project that provides an MCP (Model Context Protocol) server for accessing Snowflake architecture data tables. This server enables LLM models to query live architecture data as a source of truth for application structure analysis, dependencies, and relationships.

## Features

- MCP server for Snowflake architecture data access
- Direct querying of architecture tables through LLM tools
- Schema inspection and data search capabilities
- Integration with Claude models via Continue IDE extension
- Support for multiple environments (dev, stage, prod)

## Architecture Tables Available

The server provides access to these key architecture tables:

### Core Entity Tables
- `BASE_NAMES` - Source table naming information
- `VIEW_NAMES` - View table naming information  
- `BASE_TABLE` - Source table definitions with columns
- `VIEW_TABLE` - View table definitions with columns

### Lineage & Dependencies
- `VIEW_TO_SOURCE_COLUMN_LINEAGE` - Raw column lineage data
- `VIEW_TO_SOURCE_COLUMN_LINEAGE_V` - Enhanced column lineage view

### Application Integration
- `ACTION_TO_ENDPOINTS_TABLES_MAPPING` - Raw endpoint to table mappings
- `ACTION_TO_ENDPOINTS_TABLES_MAPPING_V` - Enhanced endpoint mappings

### Business Intelligence
- `THOUGHTSPOT_TABLE` - ThoughtSpot table usage data
- `THOUGHTSPOT_TABLE_V` - Enhanced ThoughtSpot integration view

### Stored Procedures
- `SP_TABLE_COLUMN_MAPPING` - Stored procedure metadata
- `SP_TABLE_COLUMN_MAPPING_V` - Enhanced SP mappings
- `SP_REFERENCE_TO_FLOWS` - Flow reference data
- `SP_MAPPING_WITH_ENDPOINTS_AND_FLOWS_V` - Comprehensive SP mapping


### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   
   Windows:
   ```bash
   .venv\Scripts\activate
   ```
   
   Linux/Mac:
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements_mcp.txt
   ```
   
   Or using uv (if available):
   ```bash
   uv sync
   ```

### Configuration

1. **Snowflake Connection Setup**
   - Ensure you have access to Snowflake environments
   - The server uses existing connection strategy via `common.sec` module
   - Default environment is set to 'prod' but can be modified in `mcp_snowflake_server.py`

2. **Continue IDE Configuration**
   - The project includes `config_snowflake.yaml` for Continue IDE replcae this with your `config.yaml`
   - Update the `cwd` path in the config to match your project location

### Running the MCP Server

1. **Direct execution**
   ```bash
   python mcp_snowflake_server.py
   ```

2. **Via Continue IDE**
   - The server will automatically start when Continue IDE loads the configuration
   - Check the MCP server configuration in `config_snowflake.yaml`


## Usage

### Available MCP Tools

1. **query_architecture_table**
   - Query specific tables with optional filtering and limits
   - Parameters: table_name (required), limit (optional), where_clause (optional)

2. **get_table_schema**
   - Get column information and schema details for tables
   - Parameters: table_name (required)

3. **search_architecture_data**
   - Search across tables for specific patterns or relationships
   - Parameters: search_term (required), tables (optional)

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify Snowflake credentials and access
   - Check network connectivity
   - Ensure the `common.sec` module is properly configured

2. **MCP Server Not Starting**
   - Verify Python path in `config_snowflake.yaml` replcae this in config.yaml
   - Check virtual environment activation
   - Review server logs for specific errors

3. **Missing Dependencies**
   - Run `pip install -r requirements_mcp.txt`
   - Ensure all required packages are installed
