#!/usr/bin/env python3
"""
Script to analyze table and column references in prefect_repos directory.
This version resolves SQL aliases to map columns to their ACTUAL source tables.

first colne the prefect repos using: clone_prefect_repos.py
The script connects to Snowflake to get table and column metadata,
then scans through code files to find SQL queries, resolves any table aliases,
run using: uv run python table_column_reference_with_prefect_repos.py
"""
import os
import re
import csv
import ast
import time
import logging
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import pandas as pd
from sqlalchemy import create_engine, text

# Import Snowflake connection utilities
from common import sec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('table_column_reference_with_prefect_repos.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SnowflakeDataExtractor:
    """Handles Snowflake connection and data extraction"""
    
    def __init__(self, sf_env: str = 'prod'):
        self.sf_env = sf_env
        self.engine = None
        
    def _get_correct_schema(self) -> str:
        """Get the correct schema based on environment."""
        if self.sf_env == 'prod':
            return 'CPS_DSCI_BR'
        else:
            return 'CPS_DSCI_BR'
    
    def _check_env(self) -> str:
        """Check and return the correct connection name for environment."""
        if self.sf_env == "dev":
            return "dev_cps_dsci_etl_svc"
        elif self.sf_env == "stage":
            return "stg_cps_dsci_etl_svc"
        elif self.sf_env == "prod":
            return "prd_cps_dsci_etl_svc"
        else:
            return self.sf_env
    
    def create_connection(self):
        """Create Snowflake connection engine."""
        try:
            cn = self._check_env()
            correct_schema = self._get_correct_schema()
            self.engine = create_engine(
                sec.get_sf_pw(cn, 'CPS_DSCI_ETL_EXT2_WH', correct_schema)
            )
            logger.info(f"Successfully created Snowflake connection for {self.sf_env} environment")
            logger.info(f"Using schema: {correct_schema}")
        except Exception as e:
            logger.error(f"Failed to create Snowflake connection: {e}")
            raise
    
    def fetch_base_table_data(self) -> pd.DataFrame:
        """Fetch data from BASE_TABLE."""
        if not self.engine:
            self.create_connection()
            
        sql = """
        SELECT * FROM CPS_DSCI_BR.BASE_TABLE
        ORDER BY TABLE_NAME
        """
        
        try:
            with self.engine.connect() as connection:
                result = pd.read_sql(text(sql), connection)
                logger.info(f"Fetched {len(result)} records from BASE_TABLE")
                return result
        except Exception as e:
            logger.error(f"Error fetching BASE_TABLE data: {e}")
            raise
    
    def fetch_view_table_data(self) -> pd.DataFrame:
        """Fetch data from VIEW_TABLE."""
        if not self.engine:
            self.create_connection()
            
        sql = """
        SELECT * FROM CPS_DSCI_BR.VIEW_TABLE
        ORDER BY VIEW_NAME
        """
        
        try:
            with self.engine.connect() as connection:
                result = pd.read_sql(text(sql), connection)
                logger.info(f"Fetched {len(result)} records from VIEW_TABLE")
                return result
        except Exception as e:
            logger.error(f"Error fetching VIEW_TABLE data: {e}")
            raise
    
    def close_connection(self):
        """Close the Snowflake connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Snowflake connection closed")

class SQLAliasResolver:
    """Resolves SQL table aliases to actual table names"""
    
    @staticmethod
    def extract_sql_queries(content: str) -> List[Tuple[str, int]]:
        """
        Extract SQL queries from Python code and other files.
        Returns list of (query_text, approximate_line_number) tuples.
        """
        queries = []
        
        # Split content into lines for line number tracking
        lines = content.split('\n')
        
        # Pattern 1: Triple-quoted strings (Python)
        for match in re.finditer(r'"""(.*?)"""', content, re.DOTALL):
            query_text = match.group(1)
            # Find approximate line number
            line_num = content[:match.start()].count('\n') + 1
            if len(query_text.strip()) > 20 and any(keyword in query_text.upper() for keyword in ['SELECT', 'FROM', 'JOIN', 'INSERT', 'UPDATE', 'DELETE']):
                queries.append((query_text, line_num))
        
        for match in re.finditer(r"'''(.*?)'''", content, re.DOTALL):
            query_text = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            if len(query_text.strip()) > 20 and any(keyword in query_text.upper() for keyword in ['SELECT', 'FROM', 'JOIN', 'INSERT', 'UPDATE', 'DELETE']):
                queries.append((query_text, line_num))
        
        # Pattern 2: text() wrapped SQL (SQLAlchemy)
        for match in re.finditer(r'text\s*\(\s*["\']+(.*?)["\'\)]+', content, re.DOTALL | re.IGNORECASE):
            query_text = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            if len(query_text.strip()) > 20 and any(keyword in query_text.upper() for keyword in ['SELECT', 'FROM', 'JOIN', 'INSERT', 'UPDATE', 'DELETE']):
                queries.append((query_text, line_num))
        
        # Pattern 3: SQL variable assignments
        for match in re.finditer(r'(?:sql|query|stmt)\s*=\s*["\']+(.*?)["\'\)]+', content, re.DOTALL | re.IGNORECASE):
            query_text = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            if len(query_text.strip()) > 20 and any(keyword in query_text.upper() for keyword in ['SELECT', 'FROM', 'JOIN', 'INSERT', 'UPDATE', 'DELETE']):
                queries.append((query_text, line_num))
        
        # Pattern 4: Direct SQL strings in execute calls
        for match in re.finditer(r'\.execute\s*\(\s*["\']+(.*?)["\'\)]+', content, re.DOTALL | re.IGNORECASE):
            query_text = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            if len(query_text.strip()) > 20 and any(keyword in query_text.upper() for keyword in ['SELECT', 'FROM', 'JOIN', 'INSERT', 'UPDATE', 'DELETE']):
                queries.append((query_text, line_num))
        
        return queries
    
    @staticmethod
    def parse_table_aliases(sql_query: str) -> Dict[str, str]:
        """
        Parse SQL to find table aliases.
        Returns dict mapping: alias -> actual_table_name
        
        """
        alias_map = {}  # alias -> table_name
        
        # Normalize the SQL - remove extra whitespace and newlines
        sql_normalized = re.sub(r'\s+', ' ', sql_query.upper())
        
        # Pattern for FROM/JOIN clauses with optional AS
        # Matches: FROM/JOIN table_name [AS] alias
        patterns = [
            # FROM table_name AS alias
            r'\bFROM\s+([A-Z0-9_\.]+)\s+AS\s+([A-Z0-9_]+)',
            # FROM table_name alias (no AS keyword)
            r'\bFROM\s+([A-Z0-9_\.]+)\s+([A-Z0-9_]+)(?:\s+(?:LEFT|RIGHT|INNER|OUTER|CROSS|JOIN|WHERE|GROUP|ORDER|LIMIT|UNION|,|\)|$))',
            # JOIN table_name AS alias
            r'\bJOIN\s+([A-Z0-9_\.]+)\s+AS\s+([A-Z0-9_]+)',
            # JOIN table_name alias (no AS keyword)
            r'\bJOIN\s+([A-Z0-9_\.]+)\s+([A-Z0-9_]+)(?:\s+(?:ON|LEFT|RIGHT|INNER|OUTER|CROSS|JOIN|WHERE|GROUP|ORDER|LIMIT|UNION|,|\)|$))',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sql_normalized)
            for match in matches:
                table_name = match[0].strip()
                alias = match[1].strip()
                
                # Extract just the table name if it has schema prefix (SCHEMA.TABLE)
                if '.' in table_name:
                    table_name = table_name.split('.')[-1]
                
                # Store the mapping (alias -> table_name)
                alias_map[alias] = table_name
        
        return alias_map
    
    @staticmethod
    def extract_column_references_with_aliases(sql_query: str) -> List[Tuple[Optional[str], str]]:
        """
        Extract column references with their table aliases from SQL.
        Returns list of (alias_or_None, column_name) tuples.
        
        Handles patterns like:
        - alias.column_name (returns tuple: (alias, column_name))
        - table_name.column_name (returns tuple: (table_name, column_name))
        - column_name (no prefix, returns tuple: (None, column_name))
        """
        column_refs = []
        
        # Normalize SQL
        sql_normalized = re.sub(r'\s+', ' ', sql_query.upper())
        
        # Pattern 1: identifier.column_name (aliased references)
        pattern_aliased = r'\b([A-Z0-9_]+)\.([A-Z0-9_]+)\b'
        
        matches = re.findall(pattern_aliased, sql_normalized)
        for match in matches:
            alias_or_table = match[0].strip()
            column_name = match[1].strip()
            
            # Filter out common SQL keywords and system objects
            if alias_or_table not in ['INFORMATION_SCHEMA', 'SCHEMA', 'TABLE', 'DATABASE', 'CPS_DSCI_ARCHIVE', 'CPS_DSCI_API', 'CPS_DSCI_BR']:
                column_refs.append((alias_or_table, column_name))
        
        # Pattern 2: Non-aliased column references
        # Extract potential column names from SELECT, WHERE, JOIN ON, GROUP BY, ORDER BY clauses
        # This is more complex as we need to distinguish column names from other SQL elements
        
        # Find SELECT clause columns (between SELECT and FROM)
        select_pattern = r'SELECT\s+(.*?)\s+FROM'
        select_matches = re.findall(select_pattern, sql_normalized, re.DOTALL)
        for select_clause in select_matches:
            # Split by comma and extract column names
            columns = select_clause.split(',')
            for col in columns:
                col = col.strip()
                # Skip if it already has a dot (already captured above)
                if '.' in col:
                    continue
                # Skip SQL keywords and functions
                if any(keyword in col for keyword in ['DISTINCT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'CASE', 'WHEN', 'THEN', 'END', 'AS ', '*']):
                    continue
                # Extract just the column name (before AS if present)
                col_parts = re.split(r'\s+AS\s+', col)
                if col_parts:
                    col_name = col_parts[0].strip()
                    # Check if it looks like a column name (alphanumeric with underscores)
                    if re.match(r'^[A-Z0-9_]+$', col_name) and col_name not in ['SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NULL', 'NOT']:
                        column_refs.append((None, col_name))
        
        # Find WHERE clause columns
        where_pattern = r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|UNION|$)'
        where_matches = re.findall(where_pattern, sql_normalized, re.DOTALL)
        for where_clause in where_matches:
            # Extract column names (simple pattern: word followed by comparison operator)
            col_matches = re.findall(r'\b([A-Z0-9_]+)\s*(?:=|<|>|!=|<=|>=|<>|BETWEEN|IN|IS|LIKE)', where_clause)
            for col_name in col_matches:
                if '.' not in col_name and col_name not in ['SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NULL', 'NOT', 'TRUE', 'FALSE']:
                    column_refs.append((None, col_name))
        
        # Find JOIN ON clause columns
        join_pattern = r'ON\s+(.*?)(?:LEFT|RIGHT|INNER|OUTER|CROSS|JOIN|WHERE|GROUP|ORDER|$)'
        join_matches = re.findall(join_pattern, sql_normalized, re.DOTALL)
        for join_clause in join_matches:
            # Extract non-aliased columns in JOIN conditions
            # Look for patterns like: column_name = or = column_name
            col_matches = re.findall(r'(?:^|[^A-Z0-9_\.])([A-Z0-9_]+)(?:\s*=|\s+AND|\s+OR|$)', join_clause)
            for col_name in col_matches:
                if col_name and col_name not in ['AND', 'OR', 'ON', 'NULL', 'NOT']:
                    column_refs.append((None, col_name))
        
        return column_refs
    
    @staticmethod
    def extract_insert_references(sql_query: str) -> List[Tuple[str, List[str]]]:
        """
        Extract INSERT statement table and column references.
        Returns list of (table_name, [column_names]) tuples.
        
        Handles patterns like:
        - INSERT INTO table_name (col1, col2, col3) VALUES (...)
        - INSERT INTO schema.table_name (col1, col2) VALUES (...)
        """
        insert_refs = []
        
        # Normalize SQL
        sql_normalized = re.sub(r'\s+', ' ', sql_query.upper())
        
        # Pattern: INSERT INTO table_name (columns)
        # Matches both: INSERT INTO table_name (col1, col2) and INSERT INTO schema.table_name (col1, col2)
        insert_pattern = r'INSERT\s+INTO\s+([A-Z0-9_\.]+)\s*\((.*?)\)'
        
        matches = re.findall(insert_pattern, sql_normalized, re.DOTALL)
        for match in matches:
            table_name = match[0].strip()
            columns_str = match[1].strip()
            
            # Extract just the table name if it has schema prefix
            if '.' in table_name:
                table_name = table_name.split('.')[-1]
            
            # Parse column names from the list
            columns = []
            for col in columns_str.split(','):
                col = col.strip()
                if col and re.match(r'^[A-Z0-9_]+$', col):
                    columns.append(col)
            
            if table_name and columns:
                insert_refs.append((table_name, columns))
        
        return insert_refs
    
    @staticmethod
    def extract_update_references(sql_query: str) -> List[Tuple[str, List[str]]]:
        """
        Extract UPDATE statement table and column references.
        Returns list of (table_name, [column_names]) tuples.
        
        Handles patterns like:
        - UPDATE table_name SET col1 = val1, col2 = val2
        - UPDATE schema.table_name SET col1 = val1
        """
        update_refs = []
        
        # Normalize SQL
        sql_normalized = re.sub(r'\s+', ' ', sql_query.upper())
        
        # Pattern: UPDATE table_name SET col1 = ..., col2 = ...
        update_pattern = r'UPDATE\s+([A-Z0-9_\.]+)\s+SET\s+(.*?)(?:WHERE|$)'
        
        matches = re.findall(update_pattern, sql_normalized, re.DOTALL)
        for match in matches:
            table_name = match[0].strip()
            set_clause = match[1].strip()
            
            # Extract just the table name if it has schema prefix
            if '.' in table_name:
                table_name = table_name.split('.')[-1]
            
            # Parse column names from SET clause
            columns = []
            # Pattern: column_name = value
            col_matches = re.findall(r'([A-Z0-9_]+)\s*=', set_clause)
            for col in col_matches:
                if col and col not in ['AND', 'OR', 'WHERE']:
                    columns.append(col)
            
            if table_name and columns:
                update_refs.append((table_name, columns))
        
        return update_refs
    
    @staticmethod
    def extract_delete_references(sql_query: str) -> List[str]:
        """
        Extract DELETE statement table references.
        Returns list of table_names.
        
        Handles patterns like:
        - DELETE FROM table_name WHERE ...
        - DELETE FROM schema.table_name
        """
        delete_refs = []
        
        # Normalize SQL
        sql_normalized = re.sub(r'\s+', ' ', sql_query.upper())
        
        # Pattern: DELETE FROM table_name
        delete_pattern = r'DELETE\s+FROM\s+([A-Z0-9_\.]+)'
        
        matches = re.findall(delete_pattern, sql_normalized)
        for table_name in matches:
            table_name = table_name.strip()
            
            # Extract just the table name if it has schema prefix
            if '.' in table_name:
                table_name = table_name.split('.')[-1]
            
            if table_name:
                delete_refs.append(table_name)
        
        return delete_refs

class CodeAnalyzer:
    """Analyzes code files for table and column references with alias resolution"""
    
    def __init__(self, search_dir: str = 'prefect_repos'):
        self.search_dir = search_dir
        self.ignored_repos = {'guided-workflow-backend', 'dc-data-linage-code'}
        self.searchable_extensions = {
            '.py', '.json', '.js', '.ts', '.java', '.scala', '.r', '.sh',
            '.cfg', '.conf', '.ini', '.properties', '.sql', '.yml', '.yaml'
        }
        self.alias_resolver = SQLAliasResolver()
    
    def extract_repo_name(self, file_path: str) -> str:
        """Extract repository name from file path."""
        path_parts = Path(file_path).parts
        if len(path_parts) >= 2 and path_parts[0] == self.search_dir:
            return path_parts[1]
        return "unknown"
    
    def get_searchable_files(self) -> List[str]:
        """Get list of files to search through."""
        files_to_search = []
        
        for root, dirs, files in os.walk(self.search_dir):
            # Skip hidden directories and common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {
                '__pycache__', 'node_modules', '.git', '.venv', 'venv', 
                'env', '.pytest_cache', 'dist', 'build'
            }]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                
                # Include files with searchable extensions
                if file_ext in self.searchable_extensions or file_ext == '':
                    # Skip binary-like files
                    if not any(skip in file.lower() for skip in [
                        '.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', 
                        '.zip', '.tar', '.gz', '.jpg', '.png', '.gif', '.pdf'
                    ]):
                        files_to_search.append(file_path)
        
        logger.info(f"Found {len(files_to_search)} files to analyze")
        return files_to_search
    
    def extract_functions_from_python_file(self, content: str) -> List[Tuple[str, int, int]]:
        """
        Extract function names with their line ranges from Python file content.
        Returns list of (function_name, start_line, end_line) tuples.
        """
        functions = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get line range for this function
                    start_line = node.lineno
                    # Find end line by checking the last statement or using end_lineno
                    end_line = getattr(node, 'end_lineno', node.lineno)
                    functions.append((node.name, start_line, end_line))
                elif isinstance(node, ast.AsyncFunctionDef):
                    start_line = node.lineno
                    end_line = getattr(node, 'end_lineno', node.lineno)
                    functions.append((node.name, start_line, end_line))
        except Exception as e:
            logger.debug(f"Error parsing Python AST: {e}")
        return functions
    
    def extract_functions_from_other_files(self, content: str, file_ext: str) -> List[str]:
        """Extract function-like patterns from non-Python files."""
        functions = []
        
        # Common function patterns for different file types
        patterns = []
        
        if file_ext in ['.js', '.ts']:
            patterns = [
                r'function\s+(\w+)\s*\(',
                r'(\w+)\s*:\s*function\s*\(',
                r'const\s+(\w+)\s*=\s*\(',
                r'let\s+(\w+)\s*=\s*\(',
                r'var\s+(\w+)\s*=\s*\('
            ]
        elif file_ext in ['.java', '.scala']:
            patterns = [
                r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\('
            ]
        elif file_ext == '.r':
            patterns = [
                r'(\w+)\s*<-\s*function\s*\(',
                r'(\w+)\s*=\s*function\s*\('
            ]
        elif file_ext in ['.sh']:
            patterns = [
                r'function\s+(\w+)\s*\(',
                r'(\w+)\s*\(\s*\)\s*{'
            ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            functions.extend(matches)
        
        return functions
    
    def search_file_for_references(self, file_path: str, table_names: Set[str], 
                                 table_column_mapping: Dict[str, Set[str]]) -> List[Dict[str, Any]]:
        """
        Search a single file for table and column references with alias resolution.
        Only includes columns that actually belong to the referenced table.
        Maps each SQL query to the specific function it's in.
        """
        results = []
        
        try:
            # Skip files in ignored repositories
            repo_name = self.extract_repo_name(file_path)
            if repo_name in self.ignored_repos:
                return results
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                logger.debug(f"Could not read file {file_path} with any encoding")
                return results
            
            # Skip very large files to avoid timeout
            if len(content) > 500000:  # Skip files larger than 500KB
                logger.debug(f"Skipping large file {file_path}")
                return results
            
            # Extract functions from the file with their line ranges
            file_ext = Path(file_path).suffix.lower()
            functions_with_ranges = []
            
            if file_ext == '.py':
                functions_with_ranges = self.extract_functions_from_python_file(content)
            else:
                # For non-Python files, get function names without ranges
                function_names = self.extract_functions_from_other_files(content, file_ext)
                # Create pseudo-ranges (we'll use fallback logic for these)
                functions_with_ranges = [(name, 0, len(content.split('\n'))) for name in function_names]
            
            # Extract SQL queries from the content with line numbers
            sql_queries_with_lines = self.alias_resolver.extract_sql_queries(content)
            
            # Process each SQL query for alias-resolved references
            for sql_query, query_line_num in sql_queries_with_lines:
                # Find which function this SQL query belongs to
                containing_function = None
                
                if file_ext == '.py' and functions_with_ranges:
                    # For Python files, match based on line numbers
                    for func_name, start_line, end_line in functions_with_ranges:
                        if start_line <= query_line_num <= end_line:
                            containing_function = func_name
                            break
                
                # If we couldn't determine the function, use a default or skip
                if containing_function is None:
                    if functions_with_ranges:
                        # Use the first function as fallback for non-Python or unclear cases
                        containing_function = functions_with_ranges[0][0]
                    else:
                        containing_function = 'main_content'
                
                # Parse table aliases in this query
                alias_map = self.alias_resolver.parse_table_aliases(sql_query)
                
                # Extract column references with aliases
                column_refs = self.alias_resolver.extract_column_references_with_aliases(sql_query)
                
                # Resolve each column reference to its actual table
                for alias_or_table, column_name in column_refs:
                    # Case 1: Column has an alias/table prefix
                    if alias_or_table is not None:
                        # Try to resolve alias to actual table name
                        actual_table = alias_map.get(alias_or_table, alias_or_table)
                        
                        # Check if this is a known table
                        if actual_table.upper() in table_names:
                            actual_table_upper = actual_table.upper()
                            column_name_upper = column_name.upper()
                            
                            # Only include this reference if the column actually belongs to this table
                            if actual_table_upper in table_column_mapping:
                                if column_name_upper in table_column_mapping[actual_table_upper]:
                                    results.append({
                                        'repo_name': repo_name,
                                        'function_name': containing_function,
                                        'table_name': actual_table_upper,
                                        'column_name': column_name_upper,
                                        'file_name': os.path.basename(file_path)
                                    })
                    
                    # Case 2: Column without alias/table prefix - need to find which table it belongs to
                    else:
                        column_name_upper = column_name.upper()
                        
                        # Get all tables referenced in this SQL query (from alias_map)
                        tables_in_query = set(alias_map.values())
                        # Also add any direct table references
                        for potential_table in table_names:
                            if potential_table in sql_query.upper():
                                tables_in_query.add(potential_table)
                        
                        # Find which table(s) from this query contain this column
                        for table_name_candidate in tables_in_query:
                            table_upper = table_name_candidate.upper()
                            if table_upper in table_column_mapping:
                                if column_name_upper in table_column_mapping[table_upper]:
                                    results.append({
                                        'repo_name': repo_name,
                                        'function_name': containing_function,
                                        'table_name': table_upper,
                                        'column_name': column_name_upper,
                                        'file_name': os.path.basename(file_path)
                                    })
                
                # NEW: Process INSERT statements
                insert_refs = self.alias_resolver.extract_insert_references(sql_query)
                for table_name, columns in insert_refs:
                    table_upper = table_name.upper()
                    # Verify this is a known table
                    if table_upper in table_names and table_upper in table_column_mapping:
                        # Add each column from the INSERT statement
                        for column_name in columns:
                            column_upper = column_name.upper()
                            # Verify column belongs to this table
                            if column_upper in table_column_mapping[table_upper]:
                                results.append({
                                    'repo_name': repo_name,
                                    'function_name': containing_function,
                                    'table_name': table_upper,
                                    'column_name': column_upper,
                                    'file_name': os.path.basename(file_path)
                                })
                
                # NEW: Process UPDATE statements
                update_refs = self.alias_resolver.extract_update_references(sql_query)
                for table_name, columns in update_refs:
                    table_upper = table_name.upper()
                    # Verify this is a known table
                    if table_upper in table_names and table_upper in table_column_mapping:
                        # Add each column from the UPDATE statement
                        for column_name in columns:
                            column_upper = column_name.upper()
                            # Verify column belongs to this table
                            if column_upper in table_column_mapping[table_upper]:
                                results.append({
                                    'repo_name': repo_name,
                                    'function_name': containing_function,
                                    'table_name': table_upper,
                                    'column_name': column_upper,
                                    'file_name': os.path.basename(file_path)
                                })
                
                # NEW: Process DELETE statements
                delete_refs = self.alias_resolver.extract_delete_references(sql_query)
                for table_name in delete_refs:
                    table_upper = table_name.upper()
                    # Verify this is a known table
                    if table_upper in table_names:
                        # For DELETE, we record the table without specific columns
                        # since DELETE doesn't specify which columns (deletes entire rows)
                        results.append({
                            'repo_name': repo_name,
                            'function_name': containing_function,
                            'table_name': table_upper,
                            'column_name': '',  # No specific column for DELETE
                            'file_name': os.path.basename(file_path)
                        })
            
            # FALLBACK: If no SQL queries found, do basic word matching 
            # In this case, we still map to specific functions based on line ranges
            if not sql_queries_with_lines:
                content_lower = content.lower()
                content_words = set(re.findall(r'\b\w+\b', content_lower))
                
                # Find tables that appear in content
                found_tables = set()
                for table_name in table_names:
                    if table_name.lower() in content_words:
                        found_tables.add(table_name)
                
                # For each found table, find its columns
                # Use all functions or default if none found
                if not functions_with_ranges:
                    functions_with_ranges = [('main_content', 0, 0)]
                
                for table_name in found_tables:
                    table_upper = table_name.upper()
                    if table_upper in table_column_mapping:
                        for column_name in table_column_mapping[table_upper]:
                            if column_name.lower() in content_words:
                                # In fallback mode, we can't determine exact function
                                # so we use the first function or main_content
                                function_name = functions_with_ranges[0][0]
                                results.append({
                                    'repo_name': repo_name,
                                    'function_name': function_name,
                                    'table_name': table_upper,
                                    'column_name': column_name,
                                    'file_name': os.path.basename(file_path)
                                })
        
        except Exception as e:
            logger.debug(f"Error processing file {file_path}: {e}")
        
        return results

class TableColumnReferenceAnalyzer:
    """Main class that orchestrates the analysis with alias resolution"""
    
    def __init__(self, sf_env: str = 'prod', search_dir: str = 'prefect_repos'):
        self.sf_env = sf_env
        self.search_dir = search_dir
        self.snowflake_extractor = SnowflakeDataExtractor(sf_env)
        self.code_analyzer = CodeAnalyzer(search_dir)
        
        # Data storage
        self.base_table_data = None
        self.view_table_data = None
        self.all_table_names = set()
        self.table_column_mapping = {}  # table_name -> set(column_names)
    
    def extract_snowflake_data(self):
        """Extract data from Snowflake tables."""
        logger.info("Extracting data from Snowflake...")
        
        try:
            # Fetch base table data
            self.base_table_data = self.snowflake_extractor.fetch_base_table_data()
            
            # Fetch view table data  
            self.view_table_data = self.snowflake_extractor.fetch_view_table_data()
            
            # Process base table data
            if not self.base_table_data.empty:
                for _, row in self.base_table_data.iterrows():
                    table_name = str(row.get('table_name', '')).upper()
                    column_name = str(row.get('column_names', '')).upper()
                    
                    if table_name and table_name.lower() not in ['', 'nan', 'none']:
                        self.all_table_names.add(table_name)
                        
                        if column_name and column_name.lower() not in ['', 'nan', 'none']:
                            if table_name not in self.table_column_mapping:
                                self.table_column_mapping[table_name] = set()
                            self.table_column_mapping[table_name].add(column_name)
            
            # Process view table data
            if not self.view_table_data.empty:
                for _, row in self.view_table_data.iterrows():
                    view_name = str(row.get('view_name', '')).upper()
                    column_name = str(row.get('column_names', '')).upper()
                    
                    if view_name and view_name.lower() not in ['', 'nan', 'none']:
                        self.all_table_names.add(view_name)
                        
                        if column_name and column_name.lower() not in ['', 'nan', 'none']:
                            if view_name not in self.table_column_mapping:
                                self.table_column_mapping[view_name] = set()
                            self.table_column_mapping[view_name].add(column_name)
            
            logger.info(f"Processed {len(self.all_table_names)} unique tables/views")
            logger.info(f"Processed {sum(len(cols) for cols in self.table_column_mapping.values())} table-column mappings")
            
        except Exception as e:
            logger.error(f"Error extracting Snowflake data: {e}")
            raise
        finally:
            self.snowflake_extractor.close_connection()
    
    def search_files_with_timeout(self, file_path: str, timeout: int = 30) -> List[Dict[str, Any]]:
        """Search a single file with timeout protection."""
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.code_analyzer.search_file_for_references,
                    file_path, self.all_table_names, self.table_column_mapping
                )
                return future.result(timeout=timeout)
        except TimeoutError:
            logger.warning(f"Timeout processing file: {file_path}")
            return []
        except Exception as e:
            logger.debug(f"Error processing file {file_path}: {e}")
            return []
    
    def analyze_repository_references(self, max_workers: int = 4) -> List[Dict[str, Any]]:
        """Analyze repository files for table and column references."""
        logger.info("Analyzing repository files for references with alias resolution...")
        
        files_to_search = self.code_analyzer.get_searchable_files()
        all_results = []
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with timeout protection
            future_to_file = {
                executor.submit(self.search_files_with_timeout, file_path, 30): file_path 
                for file_path in files_to_search
            }
            
            try:
                for future in as_completed(future_to_file, timeout=1200):  # 20 minute overall timeout
                    file_path = future_to_file[future]
                    completed_count += 1
                    
                    try:
                        results = future.result(timeout=2)  # 2 second individual timeout
                        all_results.extend(results)
                        
                        if completed_count % 100 == 0:
                            logger.info(f"Processed {completed_count}/{len(files_to_search)} files, found {len(all_results)} references...")
                            
                    except Exception as e:
                        logger.debug(f"Error processing file {file_path}: {e}")
                        
            except KeyboardInterrupt:
                logger.warning("Analysis interrupted by user")
                for future in future_to_file:
                    future.cancel()
        
        logger.info(f"Found {len(all_results)} total references")
        return all_results
    
    def save_results_to_csv(self, results: List[Dict[str, Any]], output_file: str):
        """Save results to CSV file."""
        try:
            # Remove duplicates
            unique_results = []
            seen = set()
            for result in results:
                key = (result['repo_name'], result['function_name'], 
                       result['table_name'], result['column_name'], result['file_name'])
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['repo_name', 'function_name', 'table_name', 'column_name', 'file_name']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header
                writer.writeheader()
                
                # Write data rows
                for result in unique_results:
                    writer.writerow(result)
            
            logger.info(f"Results saved to {output_file} ({len(unique_results)} unique records)")
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            raise
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print analysis summary."""
        if not results:
            logger.info("No references found.")
            return
        
        # Summary statistics
        unique_repos = len(set(r['repo_name'] for r in results))
        unique_tables = len(set(r['table_name'] for r in results if r['table_name']))
        unique_columns = len(set(r['column_name'] for r in results if r['column_name']))
        unique_functions = len(set(f"{r['repo_name']}.{r['function_name']}" for r in results))
        
        logger.info("=== ANALYSIS SUMMARY ===")
        logger.info(f"Total references found: {len(results)}")
        logger.info(f"Unique repositories: {unique_repos}")
        logger.info(f"Unique tables/views: {unique_tables}")
        logger.info(f"Unique columns: {unique_columns}")
        logger.info(f"Unique functions: {unique_functions}")
        
        # Top repositories by reference count
        repo_counts = {}
        for result in results:
            repo_counts[result['repo_name']] = repo_counts.get(result['repo_name'], 0) + 1
        
        logger.info("\nTop repositories by reference count:")
        for repo, count in sorted(repo_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {repo}: {count} references")
    
    def run_analysis(self, output_file: str = 'table_column_references_with_aliases.csv', max_workers: int = 8):
        """Run the complete analysis."""
        start_time = time.time()
        
        try:
            logger.info("=== TABLE-COLUMN REFERENCE ANALYSIS WITH ALIAS RESOLUTION STARTED ===")
            logger.info(f"Snowflake Environment: {self.sf_env}")
            logger.info(f"Search Directory: {self.search_dir}")
            logger.info(f"Output File: {output_file}")
            logger.info(f"Max Workers: {max_workers}")
            
            # Step 1: Extract Snowflake data
            self.extract_snowflake_data()
            
            # Step 2: Analyze repository files with alias resolution
            results = self.analyze_repository_references(max_workers)
            
            # Step 3: Sort results
            results.sort(key=lambda x: (x['repo_name'], x['function_name'], 
                                       x['table_name'], x['column_name']))
            
            # Step 4: Save results
            self.save_results_to_csv(results, output_file)
            
            # Step 5: Print summary
            self.print_summary(results)
            
            # Execution time
            elapsed_time = time.time() - start_time
            logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
            logger.info("=== ANALYSIS COMPLETE ===")
            
            return results
            
        except KeyboardInterrupt:
            logger.warning("Analysis interrupted by user")
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze table-column references with alias resolution")
    parser.add_argument("--env", default="prod", help="Snowflake environment (dev/stage/prod)")
    parser.add_argument("--search-dir", default="prefect_repos", help="Directory to search")
    parser.add_argument("--output", default="table_column_references_with_prefect_flows.csv", help="Output CSV file")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Check if search directory exists
    if not os.path.exists(args.search_dir):
        logger.error(f"Search directory not found: {args.search_dir}")
        return
    
    # Run analysis
    analyzer = TableColumnReferenceAnalyzer(
        sf_env=args.env,
        search_dir=args.search_dir
    )
    
    analyzer.run_analysis(
        output_file=args.output,
        max_workers=args.workers
    )

if __name__ == "__main__":
    main()
