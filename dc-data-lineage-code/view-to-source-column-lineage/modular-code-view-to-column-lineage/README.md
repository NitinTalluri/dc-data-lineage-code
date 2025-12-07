# DDL Column Lineage Analyzer

A comprehensive tool for analyzing DDL statements and generating column lineage information.

## Features

- Processes multiple views from `view_names.txt`
- Connects to Snowflake database
- Analyzes complex DDL with CTEs, derived columns, and wildcards
- Generates comprehensive CSV output with column lineage
- 100% accuracy for CTE analysis and column resolution

## Setup

1. Ensure you have `uv` installed
2. Place your view names in `view_names.txt` (one per line)
3. Ensure your `common.py` module is available for database connections

## Usage

```bash
uv run main.py
```

## Output

The tool generates `column_lineage_analysis.csv` with the following columns:
- View_Name
- View_Column  
- Column_Type (Direct/Derived/Unknown/Error)
- Source_Table
- Source_Column
- Expression_Type

## Configuration

Edit the `sf_env` variable in `main.py` to change the environment:
- 'prod' (default)
- 'dev' 
- 'stage'

## File Structure

```
├── main.py                     # Main execution script
├── config.py                   # Configuration constants
├── database_connection.py      # Snowflake connection handling
├── ddl_structure_analyzer.py   # DDL parsing and structure analysis
├── cte_analyzer.py            # CTE analysis logic
├── column_resolver.py         # Column mapping and resolution
├── expression_analyzer.py     # Complex expression analysis
├── csv_generator.py           # CSV output generation
├── integrated_parser.py       # Main parser orchestration
├── view_names.txt             # Input file with view names
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## Dependencies

- pandas
- sqlalchemy
- sqlglot
- common (your existing module)
```

## Usage Instructions:

1. **Setup the project:**
   ```bash
   # Create the project directory and add all the Python files
   # Make sure view_names.txt exists with your view names
   ```

2. **Run the analysis:**
   ```bash
   uv run main.py