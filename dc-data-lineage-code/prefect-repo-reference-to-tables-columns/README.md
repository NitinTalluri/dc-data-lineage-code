# Table-Column Reference Analysis with Prefect Flows

This repository contains tools to analyze table and column references in Prefect flow repositories with advanced SQL alias resolution capabilities.

## Overview

The `table_column_reference_with_prefect_repos.py` script performs  analysis of database table and column usage across Prefect flow repositories. It extracts SQL queries from code, resolves table aliases, and maps column references to their actual source tables.

## Prerequisites

### 1. Clone Prefect Repositories First

**IMPORTANT**: Before running the analysis, you must first clone the Prefect repositories using the provided cloning script:

```bash
python clone_prefect_repos.py
```

This will:
- Automatically discover Prefect flow repositories from AWS CodeCommit
- add your git credentials in credentials.txt
- Clone them to a `prefect_repos/` directory
- Filter repositories by naming conventions and content analysis
- Create a list of identified Prefect repositories

### 2. Required Dependencies

```bash
pip install pandas sqlalchemy boto3
```

### 3. Required Files

- `credentials.txt` - AWS CodeCommit credentials (Username/Password format)
- `common.py` - Snowflake connection utilities with `sec` module
- Access to Snowflake database with `CPS_DSCI_BR` schema

## Features


### Function-Level Mapping
- Maps each SQL query to the specific function containing it
- Uses line number analysis for precise function attribution
- Supports both synchronous and asynchronous Python functions

## Usage

### Basic Usage

```bash
python table_column_reference_with_prefect_repos.py
```


### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--env` | `prod` | Snowflake environment (dev/stage/prod) |
| `--search-dir` | `prefect_repos` | Directory containing cloned repositories |
| `--output` | `table_column_references_with_prefect_flows.csv` | Output CSV filename |
| `--workers` | `4` | Number of parallel processing workers |

## Output Format

The script generates a CSV file with the following schema:

| Column | Description |
|--------|-------------|
| `repo_name` | Name of the repository |
| `function_name` | Function containing the table/column reference |
| `table_name` | Database table name (uppercase) |
| `column_name` | Database column name (uppercase) |
| `file_name` | Source file containing the reference |

### Example Output

```csv
repo_name,function_name,table_name,column_name,file_name
data-pipeline-flows,process_customer_data,CUSTOMERS,CUSTOMER_ID,customer_processor.py
analytics-flows,generate_report,SALES,TRANSACTION_DATE,report_generator.py
etl-flows,transform_data,PRODUCTS,PRODUCT_NAME,data_transformer.py
```

## How It Works

### 1. Snowflake Data Extraction
- Connects to Snowflake using environment-specific credentials
- Fetches table and column metadata from `BASE_TABLE` and `VIEW_TABLE`
- Builds comprehensive table-column mapping for validation

### 2. Repository Analysis
- Scans all files in the `prefect_repos` directory
- Supports multiple file extensions and encoding formats
- Skips binary files and common non-source directories

### 3. SQL Query Processing
- Extracts SQL queries using multiple pattern matching techniques
- Parses table aliases and builds alias-to-table mappings
- Resolves column references to their actual source tables

### 4. Function Attribution
- Uses Python AST parsing for precise function boundary detection
- Maps SQL queries to containing functions based on line numbers
- Provides fallback attribution for non-Python files

### 5. Validation and Filtering
- Only includes columns that actually exist in the referenced tables
- Validates table names against Snowflake metadata
- Removes duplicate references and invalid mappings

## Performance Features

### Concurrent Processing
- Multi-threaded file processing with configurable worker count
- Individual file timeout protection (30 seconds per file)
- Overall analysis timeout protection (20 minutes)

### Memory Optimization

- Processes files in chunks with progress reporting
- Efficient duplicate removal and result deduplication

### Error Handling
- Graceful handling of encoding issues (UTF-8, Latin-1, CP1252)
- Timeout protection for hanging operations
- Comprehensive logging with debug information

## Logging

The script provides detailed logging output:

- **Console Output**: Progress updates and summary statistics
- **Log File**: `table_column_analysis_with_aliases.log` with detailed debug information
- **Progress Tracking**: Regular updates every 100 processed files

### Common Issues

1. **"Search directory not found"**
   - Run `python clone_prefect_repos.py` first to create the `prefect_repos` directory

2. **Snowflake connection errors**
   - Verify `common.py` and `sec` module are available
   - Check Snowflake credentials and network connectivity

3. **No repositories found**
   - Ensure `credentials.txt` contains valid AWS CodeCommit credentials
   - Check AWS CLI configuration and permissions
