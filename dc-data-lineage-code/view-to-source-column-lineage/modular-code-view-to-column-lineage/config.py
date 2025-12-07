#!/usr/bin/env python3
"""
Configuration module for DDL analysis
"""

# SQL Keywords for expression parsing
SQL_KEYWORDS = {
    'nvl', 'case', 'when', 'then', 'else', 'end', 'and', 'or', 'not', 
    'null', 'is', 'between', 'like', 'in', 'exists', 'select', 'from', 
    'where', 'group', 'by', 'order', 'having', 'distinct', 'as',
    'current_date', 'current_timestamp'
}

# Expression types that indicate derived columns
DERIVED_EXPRESSION_TYPES = {
    'Case', 'Coalesce', 'Add', 'Mul', 'Div', 'Sub', 'Binary', 
    'Anonymous', 'Window', 'Min', 'Max', 'Sum', 'Count', 'Avg'
}

# Aggregation patterns for regex matching
AGGREGATION_PATTERNS = [
    'max', 'min', 'sum', 'count', 'avg', 'first_value', 'last_value'
]

# Default dialect for SQL parsing
DEFAULT_DIALECT = "snowflake"

# CSV Headers
CSV_HEADERS = [
    'View_Name', 'View_Column', 'Column_Type', 
    'Source_Table', 'Source_Column', 'Expression_Type'
]