#!/usr/bin/env python3
"""
DDL Structure Analysis Module
Handles parsing and basic structure analysis of DDL statements
"""

import sqlglot
from sqlglot import exp
import re


class DDLStructureAnalyzer:
    """Analyzes DDL structure to determine type and extract basic information"""
    
    def __init__(self, dialect="snowflake"):
        self.dialect = dialect
    
    def analyze_ddl_structure(self, parsed):
        """Analyze DDL structure to determine type and extract basic info"""
        ddl_info = {
            'type': 'unknown',
            'object_name': 'UNKNOWN',
            'object_columns': [],
            'is_supported': False,
            'has_query': False
        }
        
        if isinstance(parsed, exp.Create):
            if parsed.kind == "VIEW":
                ddl_info['type'] = 'view'
                ddl_info['is_supported'] = True
                ddl_info['has_query'] = True
                
                # Extract view name
                try:
                    ddl_info['object_name'] = str(parsed.this.this)
                except:
                    view_def = str(parsed.this)
                    if '(' in view_def:
                        ddl_info['object_name'] = view_def.split('(')[0].strip()
                    else:
                        ddl_info['object_name'] = view_def
                
                # Extract view columns
                view_def_str = str(parsed.this)
                if '(' in view_def_str and ')' in view_def_str:
                    start = view_def_str.find('(')
                    end = view_def_str.find(')')
                    cols_part = view_def_str[start+1:end]
                    ddl_info['object_columns'] = [col.strip() for col in cols_part.split(',')]
            
            elif parsed.kind == "TABLE":
                ddl_info['type'] = 'table'
                ddl_info['is_supported'] = True
                
                # Check if it's CREATE TABLE AS SELECT (CTAS)
                if hasattr(parsed, 'expression') and parsed.expression:
                    ddl_info['has_query'] = True
                
                # Extract table name
                try:
                    ddl_info['object_name'] = str(parsed.this.this)
                except:
                    ddl_info['object_name'] = str(parsed.this)
                
                # Extract columns
                if ddl_info['has_query']:
                    ddl_info['object_columns'] = self._extract_ctas_columns(parsed)
                else:
                    ddl_info['object_columns'] = self._extract_table_columns_from_schema(parsed)
        
        return ddl_info
    
    def _extract_table_columns_from_schema(self, parsed):
        """Extract column names from CREATE TABLE schema definition"""
        columns = []
        try:
            if hasattr(parsed, 'this') and hasattr(parsed.this, 'expressions'):
                for expr in parsed.this.expressions:
                    if isinstance(expr, exp.ColumnDef):
                        columns.append(str(expr.this))
        except:
            pass
        return columns
    
    def _extract_ctas_columns(self, parsed):
        """Extract column names from CREATE TABLE AS SELECT statement"""
        columns = []
        try:
            if hasattr(parsed, 'expression') and parsed.expression:
                select_stmt = parsed.expression
                if isinstance(select_stmt, exp.Select):
                    for expr in select_stmt.expressions:
                        if isinstance(expr, exp.Alias):
                            columns.append(str(expr.alias))
                        elif isinstance(expr, exp.Column):
                            columns.append(str(expr.name))
                        elif isinstance(expr, exp.Star):
                            return [] 
                        else:
                            expr_str = str(expr)
                            if len(expr_str) < 50:
                                columns.append(expr_str.replace(' ', '_').replace('(', '').replace(')', ''))
        except:
            pass
        return columns
    
    def detect_sql_pattern(self, parsed, analysis):
        """Improved pattern detection to choose the right approach"""
        # Check for IDENTIFIER() function
        for node in parsed.walk():
            if isinstance(node, exp.Anonymous) and str(node.this).upper() == 'IDENTIFIER':
                return 'identifier_function'
        
        # Count different SQL elements
        cte_count = 0
        wildcard_count = 0
        explicit_alias_count = 0
        select_count = 0
        
        for node in parsed.walk():
            if isinstance(node, exp.CTE):
                cte_count += 1
            elif isinstance(node, exp.Star):
                wildcard_count += 1
            elif isinstance(node, exp.Alias):
                explicit_alias_count += 1
            elif isinstance(node, exp.Select):
                select_count += 1
        
        if cte_count >= 1 and wildcard_count > 0:
            pattern = 'hybrid'
        elif cte_count >= 1 and wildcard_count == 0:
            pattern = 'nested_cte_dominant'
        elif wildcard_count > 0 and cte_count == 0:
            pattern = 'wildcard_dominant'
        else:
            pattern = 'hybrid'
        
        return pattern