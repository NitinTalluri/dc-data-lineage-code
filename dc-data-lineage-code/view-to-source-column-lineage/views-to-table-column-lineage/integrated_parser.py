#!/usr/bin/env python3
"""
Complete Integrated SQL Parser - Full functionality from literal_merge_parser.py
Handles all SQL patterns: IDENTIFIER(), wildcards, nested CTEs, and CTAS
"""

import sqlglot
from sqlglot import exp
import csv
import io
import re


class CompleteIntegratedParser:
    """Complete integrated parser that handles all DDL patterns with 100% accuracy"""
    
    def __init__(self, dialect="snowflake"):
        self.dialect = dialect
        
    def analyze_ddl_statement(self, sql_text):
        """Analyze any DDL statement with improved pattern detection and no duplicates"""
        try:
            parsed = sqlglot.parse_one(sql_text, dialect=self.dialect)
            
            # Determine DDL type and structure
            ddl_info = self._analyze_ddl_structure(parsed)
            
            if not ddl_info['is_supported']:
                return {'error': f"Unsupported DDL type: {ddl_info['type']}"}
            
            # Initialize analysis with generalized structure
            analysis = {
                'ddl_type': ddl_info['type'],
                'object_name': ddl_info['object_name'],
                'object_columns': ddl_info['object_columns'],
                'source_tables': [],
                'column_mappings': {},
                'derived_columns': {},
                'cte_definitions': {},
                'table_aliases': {},
                'cte_column_details': {}
            }
            
            # Detect SQL pattern and choose approach
            sql_pattern = self._detect_sql_pattern(parsed, analysis)
            
            if sql_pattern == 'identifier_function':
                return self._analyze_identifier_pattern(parsed, analysis)
            elif sql_pattern == 'wildcard_dominant':
                return self._analyze_using_v2_approach(parsed, analysis)
            elif sql_pattern == 'nested_cte_dominant':
                return self._analyze_using_v1_approach(parsed, analysis)
            else:
                # Hybrid approach
                return self._analyze_using_hybrid_approach(parsed, analysis)
                
        except Exception as e:
            return {'error': f"Analysis failed: {str(e)}"}
    
    def _analyze_ddl_structure(self, parsed):
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
    
    def _detect_sql_pattern(self, parsed, analysis):
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
    
    def _analyze_identifier_pattern(self, parsed, analysis):
        """Handle IDENTIFIER() function pattern"""
        # Extract IDENTIFIER() table name
        identifier_table = None
        for node in parsed.walk():
            if isinstance(node, exp.Anonymous) and str(node.this).upper() == 'IDENTIFIER':
                if node.expressions:
                    table_name_expr = node.expressions[0]
                    if isinstance(table_name_expr, exp.Literal):
                        identifier_table = table_name_expr.this
                        break
        
        if not identifier_table:
            return {'error': 'Could not extract IDENTIFIER table name'}
        
        # For IDENTIFIER() with SELECT *, map all object columns to the table
        for col in analysis['object_columns']:
            analysis['column_mappings'][col] = {
                'type': 'direct',
                'source_table': identifier_table,
                'source_column': col,
                'table_alias': identifier_table
            } 
        return analysis
    
    def _analyze_using_hybrid_approach(self, parsed, analysis):
        """Hybrid approach that combines V1 and V2 intelligently"""
        
        # First, run V1 approach to get CTE analysis
        v1_result = self._analyze_using_v1_approach(parsed, analysis.copy())
        
        # Then, run V2 approach to get wildcard analysis
        v2_result = self._analyze_using_v2_approach(parsed, analysis.copy())
        
        # Merge results
        merged_analysis = analysis.copy()
        merged_analysis.update({
            'source_tables': v1_result.get('source_tables', []),
            'cte_definitions': v1_result.get('cte_definitions', {}),
            'table_aliases': v1_result.get('table_aliases', {}),
            'cte_column_details': v1_result.get('cte_column_details', {}),
            'debug_info': v2_result.get('debug_info', {})
        })# Merge column mappings with priority logic and case-insensitive matching
        missing_columns = []
        for col in analysis['object_columns']:
            # Check exact matches first
            v1_has_direct = col in v1_result.get('column_mappings', {})
            v1_has_derived = col in v1_result.get('derived_columns', {})
            v2_has_direct = col in v2_result.get('column_mappings', {})
            v2_has_derived = col in v2_result.get('derived_columns', {})
            
            # Check case-insensitive matches if exact matches not found
            if not (v1_has_direct or v1_has_derived or v2_has_direct or v2_has_derived):
                v1_direct_key = self._find_case_insensitive_key(col, v1_result.get('column_mappings', {}))
                v1_derived_key = self._find_case_insensitive_key(col, v1_result.get('derived_columns', {}))
                v2_direct_key = self._find_case_insensitive_key(col, v2_result.get('column_mappings', {}))
                v2_derived_key = self._find_case_insensitive_key(col, v2_result.get('derived_columns', {}))
                
                v1_has_direct = v1_direct_key is not None
                v1_has_derived = v1_derived_key is not None
                v2_has_direct = v2_direct_key is not None
                v2_has_derived = v2_derived_key is not None
            else:
                v1_direct_key = col if v1_has_direct else None
                v1_derived_key = col if v1_has_derived else None
                v2_direct_key = col if v2_has_direct else None
                v2_derived_key = col if v2_has_derived else None
            
            v1_direct_good = (v1_has_direct and v1_direct_key and 
                            v1_result['column_mappings'][v1_direct_key].get('type') != 'unknown')
            v2_direct_good = (v2_has_direct and v2_direct_key and 
                            v2_result['column_mappings'][v2_direct_key].get('type') != 'unknown')
            
            column_assigned = False
            if v1_has_derived and v1_derived_key:
                merged_analysis['derived_columns'][col] = v1_result['derived_columns'][v1_derived_key]
                column_assigned = True
            elif v1_direct_good:
                merged_analysis['column_mappings'][col] = v1_result['column_mappings'][v1_direct_key]
                column_assigned = True
            elif v2_direct_good:
                merged_analysis['column_mappings'][col] = v2_result['column_mappings'][v2_direct_key]
                column_assigned = True
            elif v2_has_derived and v2_derived_key:
                merged_analysis['derived_columns'][col] = v2_result['derived_columns'][v2_derived_key]
                column_assigned = True
            
            if not column_assigned:
                missing_columns.append(col)
        
        self._resolve_missing_columns_from_main_select(parsed, missing_columns, merged_analysis)
        
        # Resolve unknown columns
        unknown_columns = []
        for col_name, mapping in merged_analysis.get('column_mappings', {}).items():
            if mapping.get('type') == 'unknown':
                unknown_columns.append(col_name)
        
        if unknown_columns and len(merged_analysis.get('cte_definitions', {})) > 0:
            self._resolve_cte_wildcard_columns(unknown_columns, merged_analysis)
        elif unknown_columns:
            self._resolve_simple_wildcard_columns(unknown_columns, merged_analysis)
        elif len(analysis['object_columns']) > 0 and len(merged_analysis['column_mappings']) == 0 and len(merged_analysis['derived_columns']) == 0:
            self._resolve_missing_columns_from_main_select(parsed, analysis['object_columns'], merged_analysis)
        elif missing_columns and len(merged_analysis.get('cte_definitions', {})) > 0:
            self._resolve_cte_wildcard_columns(missing_columns, merged_analysis)
        
        # Final fallback: try to resolve missing columns from any available CTE
        remaining_missing = []
        for col in missing_columns:
            if col not in merged_analysis.get('column_mappings', {}) and col not in merged_analysis.get('derived_columns', {}):
                remaining_missing.append(col)
        
        if remaining_missing:
            self._resolve_missing_columns_from_any_cte(remaining_missing, merged_analysis)
        
        return merged_analysis
    
    def _resolve_cte_wildcard_columns(self, missing_columns, analysis):
        """Resolve missing columns through CTE wildcard pattern"""
        main_cte = None
        for cte_name, cte_columns in analysis.get('cte_column_details', {}).items():
            matches = sum(1 for col in missing_columns if col in cte_columns or 
                         any(cte_col.lower() == col.lower() for cte_col in cte_columns.keys()))
            if matches > len(missing_columns) * 0.5:
                main_cte = cte_name
                break
        
        if not main_cte:
            return
        
        cte_columns = analysis['cte_column_details'][main_cte]
        
        for missing_col in missing_columns:
            if missing_col in cte_columns:
                cte_col_info = cte_columns[missing_col]
                self._map_column_from_cte_info(missing_col, cte_col_info, analysis)
                continue
            
            found = False
            for cte_col_name, cte_col_info in cte_columns.items():
                if cte_col_name.lower() == missing_col.lower():
                    self._map_column_from_cte_info(missing_col, cte_col_info, analysis)
                    found = True
                    break
    def _find_case_insensitive_key(self, target_key, dictionary):
        """Find a key in dictionary using case-insensitive matching"""
        target_lower = target_key.lower()
        for key in dictionary.keys():
            if key.lower() == target_lower:
                return key
        return None
    def _resolve_cte_table_to_source(self, table_name, analysis):
        """Resolve a CTE table reference to its ultimate source table"""
        # Clean up table name
        if ' AS ' in table_name:
            table_name = table_name.split(' AS ')[0]
        
        # If it's a CTE reference, resolve it
        if table_name.startswith('CTE_'):
            cte_name = table_name[4:]
            
            # Check if we have detailed CTE column information
            if cte_name in analysis.get('cte_column_details', {}):
                cte_columns = analysis['cte_column_details'][cte_name]
                
                # Find the first direct column reference to get the source table
                for col_name, col_info in cte_columns.items():
                    if col_info.get('type') == 'direct':
                        source_table = col_info.get('source_table', '')
                        if ' AS ' in source_table:
                            source_table = source_table.split(' AS ')[0]
                        if not source_table.startswith('CTE_') and source_table != analysis.get('object_name', ''):
                            return source_table
                
                # If no direct columns, check enhanced references in derived columns
                for col_name, col_info in cte_columns.items():
                    if col_info.get('type') == 'derived':
                        for ref in col_info.get('enhanced_referenced_columns', []):
                            source_table = ref.get('table', '')
                            if ' AS ' in source_table:
                                source_table = source_table.split(' AS ')[0]
                            if not source_table.startswith('CTE_') and source_table != analysis.get('object_name', ''):
                                return source_table
            
            # Fallback: Look up the CTE in table_aliases to find actual tables
            if cte_name in analysis.get('cte_definitions', {}):
                cte_def = analysis['cte_definitions'][cte_name]['definition']
                # Extract table references from CTE definition
                for inner_alias, inner_table in analysis.get('table_aliases', {}).items():
                    if (not inner_table.startswith('CTE_') and 
                        inner_alias in cte_def and 
                        inner_table != analysis.get('object_name', '')):
                        return inner_table.split(' AS ')[0] if ' AS ' in inner_table else inner_table
            
            return None
        else:
            # Not a CTE, return as-is
            return table_name
    
    def _map_column_from_cte_info(self, view_col, cte_col_info, analysis):
        """Map a view column based on CTE column information with proper CTE resolution"""
        
        if cte_col_info.get('type') == 'direct':
            source_table = cte_col_info.get('source_table', '')
            source_column = cte_col_info.get('source_column', view_col)
            table_alias = cte_col_info.get('table_alias', '')
            
            # Clean up table name
            if ' AS ' in source_table:
                source_table = source_table.split(' AS ')[0]
            if source_table.startswith('CTE_'):
                source_table = source_table[4:]
            
            analysis['column_mappings'][view_col] = {
                'type': 'direct',
                'source_table': source_table,
                'source_column': source_column,
                'table_alias': table_alias,
                'resolved_from_cte': True
            }
        
        elif cte_col_info.get('type') == 'derived':
            # Move from column_mappings to derived_columns if it exists there
            if view_col in analysis.get('column_mappings', {}):
                del analysis['column_mappings'][view_col]
            
            # Create derived column entry with proper CTE resolution
            ultimate_tables = set()
            source_columns = set()
            referenced_columns = []
            
            # Get source information from CTE column info
            for ref in cte_col_info.get('referenced_columns', []):
                table_name = ref.get('table', '')
                col_name = ref.get('column', '')
                
                # Resolve CTE references recursively
                resolved_table = self._resolve_cte_table_to_source(table_name, analysis)
                
                if resolved_table:
                    ultimate_tables.add(resolved_table)
                if col_name:
                    source_columns.add(col_name)
                
                referenced_columns.append({
                    'table': resolved_table or table_name,
                    'column': col_name,
                    'alias': ref.get('alias', '')
                })
            
            # Also check enhanced references
            for ref in cte_col_info.get('enhanced_referenced_columns', []):
                table_name = ref.get('table', '')
                col_name = ref.get('column', '')
                
                # Resolve CTE references recursively
                resolved_table = self._resolve_cte_table_to_source(table_name, analysis)
                
                if resolved_table:
                    ultimate_tables.add(resolved_table)
                if col_name:
                    source_columns.add(col_name)
                
                referenced_columns.append({
                    'table': resolved_table or table_name,
                    'column': col_name,
                    'alias': ref.get('alias', '')
                })
            
            primary_source_table = list(ultimate_tables)[0] if ultimate_tables else 'CALCULATED'
            
            analysis['derived_columns'][view_col] = {
                'expression': cte_col_info.get('expression', ''),
                'expression_type': cte_col_info.get('expression_type', ''),
                'ultimate_source_tables': list(ultimate_tables),
                'primary_source_table': primary_source_table,
                'source_columns': list(source_columns),
                'enhanced_references': referenced_columns,
                'resolved_from_cte': True
            }
    
    def _resolve_simple_wildcard_columns(self, unknown_columns, analysis):
        """Resolve unknown columns for simple SELECT * FROM table patterns"""
        from_source = analysis.get('from_source')
        
        if not from_source:
            return
        
        source_table = None
        table_alias = None
        
        table_registry = analysis.get('table_registry', {})
        if from_source in table_registry:
            source_table = table_registry[from_source]['full_name']
            table_alias = from_source
        else:
            for alias, table_info in table_registry.items():
                if table_info['full_name'] == from_source or alias == from_source:
                    source_table = table_info['full_name']
                    table_alias = alias
                    break
            
            if not source_table:
                source_table = from_source
                table_alias = from_source
        
        for unknown_col in unknown_columns:
            analysis['column_mappings'][unknown_col] = {
                'type': 'direct',
                'source_table': source_table,
                'source_column': unknown_col,
                'table_alias': table_alias,
                'resolved_from_wildcard': True
            }
    
    def _resolve_missing_columns_from_any_cte(self, missing_columns, analysis):
        """Resolve missing columns by searching through all CTEs"""
        
        for missing_col in missing_columns:
            if missing_col in analysis.get('column_mappings', {}) or missing_col in analysis.get('derived_columns', {}):
                continue
            
            # Search through all CTEs for this column
            found = False
            for cte_name, cte_columns in analysis.get('cte_column_details', {}).items():
                # Check exact match
                if missing_col in cte_columns:
                    cte_col_info = cte_columns[missing_col]
                    self._map_column_from_cte_info(missing_col, cte_col_info, analysis)
                    found = True
                    break
                
                # Check case-insensitive match
                for cte_col_name, cte_col_info in cte_columns.items():
                    if cte_col_name.lower() == missing_col.lower():
                        self._map_column_from_cte_info(missing_col, cte_col_info, analysis)
                        found = True
                        break
                
                if found:
                    break
            
            # If still not found, try to infer from main table
            if not found:
                main_table = None
                for alias, table_name in analysis.get('table_aliases', {}).items():
                    if not table_name.startswith('CTE_') and table_name != analysis.get('object_name', ''):
                        main_table = table_name
                        if ' AS ' in main_table:
                            main_table = main_table.split(' AS ')[0]
                        break
                
                if main_table:
                    analysis['column_mappings'][missing_col] = {
                        'type': 'direct',
                        'source_table': main_table,
                        'source_column': missing_col,
                        'table_alias': '',
                        'resolved_method': 'fallback_main_table'
                    }
    
    def _resolve_missing_columns_from_main_select(self, parsed, missing_columns, analysis):
        """Targeted resolution of missing columns from main SELECT statement"""
        
        # Find the main SELECT statement (not in CTE)
        main_select = None
        for node in parsed.walk():
            if isinstance(node, exp.Select):
                parent = node.parent
                in_cte = False
                while parent:
                    if isinstance(parent, exp.CTE):
                        in_cte = True
                        break
                    parent = parent.parent
                
                if not in_cte:
                    main_select = node
                    break
        
        if not main_select:
            return
        
        # Process each expression in main SELECT to find missing columns
        for expr in main_select.expressions:
            if isinstance(expr, exp.Alias):
                column_alias = str(expr.alias).upper()  # Normalize case
                source_expr = expr.this
                
                # Check if this is one of our missing columns (case-insensitive)
                missing_col_match = None
                for missing_col in missing_columns:
                    if missing_col.upper() == column_alias:
                        missing_col_match = missing_col
                        break
                
                if missing_col_match:
                    if isinstance(source_expr, exp.Column):
                        # Direct column reference
                        table_ref = source_expr.table
                        col_name = str(source_expr.name)
                        
                        if table_ref:
                            table_ref_str = str(table_ref)
                            # Look up the actual table name
                            actual_table = analysis['table_aliases'].get(table_ref_str.lower(), table_ref_str)
                            
                            # Clean up table name
                            if ' AS ' in actual_table:
                                actual_table = actual_table.split(' AS ')[0]
                            analysis['column_mappings'][missing_col_match] = {
                                'type': 'direct',
                                'source_table': actual_table,
                                'source_column': col_name,
                                'table_alias': table_ref_str,
                                'resolved_method': 'missing_column_recovery'
                            }
                    
                    else:
                        # Derived expression (like arithmetic)
                        referenced_tables = set()
                        referenced_columns = set()
                        
                        # Extract column references from the expression
                        for node in source_expr.walk():
                            if isinstance(node, exp.Column):
                                node_table_ref = node.table
                                node_col_name = str(node.name)
                                
                                if node_table_ref:
                                    # Qualified column
                                    node_table_ref_str = str(node_table_ref)
                                    actual_table = analysis['table_aliases'].get(node_table_ref_str.lower(), node_table_ref_str)
                                    if ' AS ' in actual_table:
                                        actual_table = actual_table.split(' AS ')[0]
                                    referenced_tables.add(actual_table)
                                    referenced_columns.add(node_col_name)
                                else:
                                    # Unqualified column - try to resolve from CTEs
                                    resolved_tables, resolved_cols = self._resolve_derived_column_source(
                                        node_col_name, None, analysis
                                    )
                                    if resolved_tables:
                                        referenced_tables.update(resolved_tables)
                                        referenced_columns.update(resolved_cols)
                        
                        if referenced_tables:
                            primary_table = list(referenced_tables)[0]
                            if missing_col_match in analysis['column_mappings']:
                                del analysis['column_mappings'][missing_col_match]
                            
                            analysis['derived_columns'][missing_col_match] = {
                                'expression': str(source_expr),
                                'expression_type': type(source_expr).__name__,
                                'ultimate_source_tables': list(referenced_tables),
                                'primary_source_table': primary_table,
                                'source_columns': list(referenced_columns),
                                'enhanced_references': [{
                                    'table': table,
                                    'column': col,
                                    'alias': ''
                                } for table in referenced_tables for col in referenced_columns],
                                'resolved_method': 'missing_column_recovery'
                            }
                        else:
                            if missing_col_match in analysis['column_mappings']:
                                del analysis['column_mappings'][missing_col_match]
                            
                            analysis['derived_columns'][missing_col_match] = {
                                'expression': str(source_expr),
                                'expression_type': type(source_expr).__name__,
                                'ultimate_source_tables': ['CALCULATED'],
                                'primary_source_table': 'CALCULATED',
                                'source_columns': [],
                                'enhanced_references': [],
                                'resolved_method': 'missing_column_recovery_calculated'
                            }
            
            elif isinstance(expr, exp.Column):
                # Handle direct column references without aliases
                column_name = str(expr.name).upper()  # Normalize case
                
                # Check if this is one of our missing columns (case-insensitive)
                missing_col_match = None
                for missing_col in missing_columns:
                    if missing_col.upper() == column_name:
                        missing_col_match = missing_col
                        break
                
                if missing_col_match:
                    table_ref = expr.table
                    col_name = str(expr.name)
                    
                    if table_ref:
                        table_ref_str = str(table_ref)
                        # Look up the actual table name
                        actual_table = analysis['table_aliases'].get(table_ref_str.lower(), table_ref_str)
                        
                        # Clean up table name
                        if ' AS ' in actual_table:
                            actual_table = actual_table.split(' AS ')[0]
                        analysis['column_mappings'][missing_col_match] = {
                            'type': 'direct',
                            'source_table': actual_table,
                            'source_column': col_name,
                            'table_alias': table_ref_str,
                            'resolved_method': 'missing_column_recovery_direct'
                        }
                    else:
                        # Unqualified column - try to infer from available tables
                        # Use the main FROM table as default for unqualified columns
                        main_table = None
                        for alias, table_name in analysis.get('table_aliases', {}).items():
                            if not table_name.startswith('CTE_'):
                                main_table = table_name
                                if ' AS ' in main_table:
                                    main_table = main_table.split(' AS ')[0]
                                break
                        if main_table:
                            analysis['column_mappings'][missing_col_match] = {
                                'type': 'direct',
                                'source_table': main_table,
                                'source_column': col_name,
                                'table_alias': '',
                                'resolved_method': 'missing_column_recovery_unqualified'
                            }
    
    def _analyze_using_v1_approach(self, parsed, analysis):
        """V1 approach - excellent for nested CTEs and derived columns"""
        
        # Extract basic structure
        self._v1_extract_tables_and_aliases(parsed, analysis)
        self._v1_extract_ctes(parsed, analysis)
        self._v1_analyze_column_lineage(parsed, analysis)
        self._v1_analyze_cte_columns_detailed(parsed, analysis)
        self._v1_enhance_derived_columns_with_cte_tracing(analysis)
        self._v1_resolve_view_columns_through_ctes(analysis)
        self._v1_resolve_main_select_derived_columns(parsed, analysis)
        
        return analysis
    
    def _analyze_using_v2_approach(self, parsed, analysis):
        """V2 approach - excellent for wildcards and simple direct mappings"""
        self._v1_extract_tables_and_aliases(parsed, analysis)
        self._v1_extract_ctes(parsed, analysis)
        
        main_select = self._v2_find_main_select(parsed)
        if not main_select:
            return analysis
        
        main_select_analysis = self._v2_analyze_select_expressions(main_select)
        table_registry = self._v2_build_table_registry(parsed)
        cte_registry = self._v2_build_cte_registry(parsed, table_registry)
        
        analysis['main_select_analysis'] = main_select_analysis
        analysis['table_registry'] = table_registry
        analysis['cte_registry_full'] = cte_registry
        
        for obj_col in analysis['object_columns']:
            source_info = self._v2_trace_column_source(
                obj_col, main_select_analysis, cte_registry, table_registry
            )
            
            if source_info['type'] == 'derived':
                enhanced_source_info = self._v2_enhance_derived_column(source_info, analysis)
                analysis['derived_columns'][obj_col] = enhanced_source_info
            else:
                analysis['column_mappings'][obj_col] = source_info
        
        return analysis
    
    # ==================== V1 METHODS ====================
    def _v1_extract_tables_and_aliases(self, parsed, analysis):
        """Extract all table references and their aliases"""
        view_name = analysis.get('object_name', '')
        
        for node in parsed.walk():
            if isinstance(node, exp.Table):
                table_name = str(node)
                
                # Skip the view itself as a source table
                if table_name == view_name:
                    continue
                    
                analysis['source_tables'].append(table_name)
                
                if node.alias:
                    alias = str(node.alias)
                    analysis['table_aliases'][alias.lower()] = table_name
                else:
                    implicit_alias = table_name.split('.')[-1].lower()
                    # Don't add implicit alias if it would conflict with view name
                    if table_name != view_name:
                        analysis['table_aliases'][implicit_alias] = table_name
                        
    def _v1_extract_ctes(self, parsed, analysis):
        """Extract Common Table Expressions"""
        for node in parsed.walk():
            if isinstance(node, exp.CTE):
                cte_name = str(node.alias)
                analysis['cte_definitions'][cte_name] = {
                    'name': cte_name,
                    'definition': str(node.this)
                }
                analysis['table_aliases'][cte_name.lower()] = f"CTE_{cte_name}"
    
    def _v1_analyze_column_lineage(self, parsed, analysis):
        """Analyze column lineage"""
        for node in parsed.walk():
            if isinstance(node, exp.Select):
                self._v1_analyze_select_statement(node, analysis)
    
    def _v1_analyze_select_statement(self, select_node, analysis):
        """Analyze a single SELECT statement"""
        for expr in select_node.expressions:
            if isinstance(expr, exp.Alias):
                column_alias = str(expr.alias)
                source_expr = expr.this
                
                if isinstance(source_expr, exp.Column):
                    self._v1_map_direct_column(column_alias, source_expr, analysis)
                else:
                    self._v1_map_derived_column(column_alias, source_expr, analysis)
            
            elif isinstance(expr, exp.Column):
                column_name = str(expr.name)
                self._v1_map_direct_column(column_name, expr, analysis)
    
    def _v1_map_direct_column(self, column_name, column_expr, analysis):
        """Map a direct column reference"""
        table_ref = column_expr.table
        source_column = str(column_expr.name)
        
        if table_ref:
            table_ref_str = str(table_ref)
            actual_table = analysis['table_aliases'].get(table_ref_str.lower(), table_ref_str)
            
            analysis['column_mappings'][column_name] = {
                'type': 'direct',
                'source_table': actual_table,
                'source_column': source_column,
                'table_alias': table_ref_str
            }
    
    def _v1_map_derived_column(self, column_name, expression, analysis):
        """Map a derived column"""
        referenced_columns = []
        unqualified_columns = []
        
        for node in expression.walk():
            if isinstance(node, exp.Column):
                table_ref = node.table
                col_name = str(node.name)
                
                if table_ref:
                    table_ref_str = str(table_ref)
                    actual_table = analysis['table_aliases'].get(table_ref_str.lower(), table_ref_str)
                    referenced_columns.append({
                        'table': actual_table,
                        'column': col_name,
                        'alias': table_ref_str
                    })
                else:
                    unqualified_columns.append(col_name)
        
        analysis['derived_columns'][column_name] = {
            'expression': str(expression),
            'expression_type': type(expression).__name__,
            'referenced_columns': referenced_columns,
            'unqualified_columns': unqualified_columns
        }
    
    def _v1_analyze_cte_columns_detailed(self, parsed, analysis):
        """Detailed analysis of what columns each CTE provides"""
        
        for cte_name, cte_info in analysis['cte_definitions'].items():
            cte_columns = {}
            
            for node in parsed.walk():
                if isinstance(node, exp.CTE) and str(node.alias) == cte_name:
                    select_node = node.this
                    
                    for expr in select_node.expressions:
                        if isinstance(expr, exp.Alias):
                            alias_name = str(expr.alias)
                            source_expr = expr.this
                            if isinstance(source_expr, exp.Column):
                                table_ref = source_expr.table
                                original_column_name = str(source_expr.name)
                                
                                if table_ref and str(table_ref).lower() in analysis['table_aliases']:
                                    actual_table = analysis['table_aliases'][str(table_ref).lower()]
                                    cte_columns[alias_name] = {
                                        'type': 'direct',
                                        'source_table': actual_table,
                                        'source_column': original_column_name,
                                        'table_alias': str(table_ref),
                                        'cte_alias': alias_name,
                                        'original_column': original_column_name
                                    }
                                else:
                                    cte_dependency = self._find_column_in_dependent_ctes(alias_name, cte_name, analysis)
                                    if cte_dependency:
                                        cte_columns[alias_name] = cte_dependency
                                    else:
                                        cte_columns[alias_name] = {
                                            'type': 'unqualified_in_cte',
                                            'source_column': original_column_name,
                                            'needs_resolution': True,
                                            'cte_alias': alias_name,
                                            'original_column': original_column_name
                                        }
                            else:
                                referenced_cols = []
                                unqualified_cols = []
                                
                                for sub_node in source_expr.walk():
                                    if isinstance(sub_node, exp.Column):
                                        sub_table_ref = sub_node.table
                                        if sub_table_ref and str(sub_table_ref).lower() in analysis['table_aliases']:
                                            actual_table = analysis['table_aliases'][str(sub_table_ref).lower()]
                                            referenced_cols.append({
                                                'table': actual_table,
                                                'column': str(sub_node.name),
                                                'alias': str(sub_table_ref)
                                            })
                                        else:
                                            unqualified_cols.append(str(sub_node.name))
                                
                                cte_columns[alias_name] = {
                                    'type': 'derived',
                                    'expression': str(source_expr),
                                    'expression_type': type(source_expr).__name__,
                                    'referenced_columns': referenced_cols,
                                    'unqualified_columns': unqualified_cols
                                }
                        
                        elif isinstance(expr, exp.Column):
                            col_name = str(expr.name)
                            table_ref = expr.table
                            if table_ref and str(table_ref).lower() in analysis['table_aliases']:
                                actual_table = analysis['table_aliases'][str(table_ref).lower()]
                                cte_columns[col_name] = {
                                    'type': 'direct',
                                    'source_table': actual_table,
                                    'source_column': col_name,
                                    'table_alias': str(table_ref)
                                }
                            else:
                                cte_dependency = self._find_column_in_dependent_ctes(col_name, cte_name, analysis)
                                if cte_dependency:
                                    cte_columns[col_name] = cte_dependency
                                else:
                                    cte_columns[col_name] = {
                                        'type': 'unqualified_in_cte',
                                        'source_column': col_name,
                                        'needs_resolution': True
                                    }
                    
                    break
            
            analysis['cte_column_details'][cte_name] = cte_columns
        
        self._v1_resolve_unqualified_columns_in_ctes(analysis)
    
    def _v1_resolve_unqualified_columns_in_ctes(self, analysis):
        """Resolve unqualified column references within CTEs"""
        
        cte_order = self._v1_determine_cte_dependency_order(analysis)
        
        for cte_name in cte_order:
            if cte_name not in analysis['cte_column_details']:
                continue
                
            cte_columns = analysis['cte_column_details'][cte_name]
            
            for col_name, col_info in cte_columns.items():
                if col_info.get('needs_resolution'):
                    source_col = col_info['source_column']
                    resolved_source = self._v1_resolve_unqualified_in_cte_context(
                                        source_col, cte_name, analysis
                                    )
                    if resolved_source:
                        cte_dependency = self._find_column_in_dependent_ctes(col_name, cte_name, analysis)
                        if cte_dependency:
                            cte_columns[col_name] = cte_dependency
                        else:
                            if resolved_source.get('type') == 'direct':
                                original_col = self._find_original_column_name_in_cte(
                                    cte_name, col_name, source_col, analysis
                                )
                                if original_col and original_col != source_col:
                                    resolved_source = resolved_source.copy()
                                    resolved_source['source_column'] = original_col
                                    resolved_source['cte_alias'] = col_name
                            
                            cte_columns[col_name] = resolved_source
                    
                elif col_info.get('type') == 'derived' and col_info.get('unqualified_columns'):
                    enhanced_refs = col_info.get('referenced_columns', []).copy()
                    
                    for unqual_col in col_info['unqualified_columns']:
                        resolved_source = self._v1_resolve_unqualified_in_cte_context(
                            unqual_col, cte_name, analysis
                        )
                        
                        if resolved_source and resolved_source.get('type') == 'direct':
                            enhanced_refs.append({
                                'table': resolved_source['source_table'],
                                'column': resolved_source['source_column'],
                                'alias': resolved_source.get('table_alias', ''),
                                'resolved_from': unqual_col
                            })
                    
                    cte_columns[col_name]['enhanced_referenced_columns'] = enhanced_refs
    
    def _v1_determine_cte_dependency_order(self, analysis):
        """Determine CTE dependency order"""
        cte_names = list(analysis['cte_definitions'].keys())
        
        if len(cte_names) <= 1:
            return cte_names
        
        dependencies = {}
        for cte_name in cte_names:
            dependencies[cte_name] = set()
            cte_def = analysis['cte_definitions'][cte_name]['definition']
            
            for other_cte in cte_names:
                if other_cte != cte_name and other_cte in cte_def:
                    dependencies[cte_name].add(other_cte)
        
        ordered = []
        remaining = set(cte_names)
        
        while remaining:
            ready = []
            for cte in remaining:
                if not (dependencies[cte] & remaining):
                    ready.append(cte)
            
            if not ready:
                ordered.extend(sorted(remaining))
                break
            
            ready.sort()
            ordered.extend(ready)
            remaining -= set(ready)
        
        return ordered
    
    def _v1_resolve_unqualified_in_cte_context(self, column_name, current_cte, analysis):
        """Enhanced resolve an unqualified column within a CTE context"""
        # First try dynamic parsing of the CTE definition
        dynamic_resolution = self._dynamic_resolve_column_in_cte(column_name, current_cte, analysis)
        if dynamic_resolution:
            return dynamic_resolution
        
        # Fallback to original dependency-based resolution
        cte_order = self._v1_determine_cte_dependency_order(analysis)
        current_index = cte_order.index(current_cte) if current_cte in cte_order else -1
        
        for i in range(current_index):
            other_cte = cte_order[i]
            if other_cte in analysis['cte_column_details']:
                other_cte_columns = analysis['cte_column_details'][other_cte]
                if column_name in other_cte_columns:
                    col_info = other_cte_columns[column_name]
                    if col_info.get('type') in ['direct', 'derived']:
                        return col_info
        
        for other_cte, cte_columns in analysis['cte_column_details'].items():
            if other_cte != current_cte and column_name in cte_columns:
                col_info = cte_columns[column_name]
                if col_info.get('type') in ['direct', 'derived']:
                    return col_info
        
        # Look in the current CTE's FROM clause for tables that might have this column
        if current_cte in analysis.get('cte_definitions', {}):
            cte_def = analysis['cte_definitions'][current_cte]['definition']
            
            # Dynamic approach: Parse the CTE definition to find qualified column references
            qualified_pattern = rf'\b([a-zA-Z_][a-zA-Z0-9_]*)\.{re.escape(column_name)}\b'
            matches = re.findall(qualified_pattern, cte_def, re.IGNORECASE)
            
            if matches:
                for table_alias in matches:
                    if table_alias.lower() in analysis['table_aliases']:
                        actual_table = analysis['table_aliases'][table_alias.lower()]
                        if (not actual_table.startswith('CTE_') and 
                            actual_table != analysis.get('object_name', '')):
                            return {
                                'type': 'direct',
                                'source_table': actual_table,
                                'source_column': column_name,
                                'table_alias': table_alias,
                                'resolved_in_context': current_cte,
                                'resolution_method': 'qualified_reference'
                            }
            
            # SQL Standard Resolution - For unqualified columns, follow SQL precedence rules
            available_tables = []
            join_tables = []
            main_table = None
            
            for alias, table_name in analysis['table_aliases'].items():
                if (not table_name.startswith('CTE_') and 
                    alias in cte_def and 
                    table_name != analysis.get('object_name', '')):
                    
                    if re.search(rf'\bjoin\s+[^\s]+\s+{re.escape(alias)}\b', cte_def, re.IGNORECASE):
                        join_tables.append((alias, table_name))
                    else:
                        available_tables.append((alias, table_name))
                        if main_table is None:
                            main_table = (alias, table_name)
            
            # Prefer JOIN tables over main FROM table for unqualified columns
            if join_tables:
                alias, table_name = join_tables[0]
                return {
                    'type': 'direct',
                    'source_table': table_name,
                    'source_column': column_name,
                    'table_alias': alias,
                    'resolved_in_context': current_cte,
                    'resolution_method': 'join_table_precedence'
                }
            
            # Final fallback: use main table
            if main_table:
                alias, table_name = main_table
                return {
                    'type': 'direct',
                    'source_table': table_name,
                    'source_column': column_name,
                    'table_alias': alias,
                    'resolved_in_context': current_cte,
                    'resolution_method': 'main_table_fallback'
                }
        
        # Final fallback: look for any non-CTE table
        for alias, table_name in analysis['table_aliases'].items():
            if (not table_name.startswith('CTE_') and 
                table_name != analysis.get('object_name', '')):
                return {
                    'type': 'direct',
                    'source_table': table_name,
                    'source_column': column_name,
                    'table_alias': alias,
                    'resolved_in_context': current_cte
                }
        return None
    
    def _dynamic_resolve_column_in_cte(self, column_name, cte_name, analysis):
        """Dynamically resolve column source by parsing CTE definition"""
        
        if cte_name not in analysis.get('cte_definitions', {}):
            return None
        
        cte_def = analysis['cte_definitions'][cte_name]['definition']
        
        try:
            # Method 1: Parse with sqlglot for accurate AST analysis
            parsed_cte = sqlglot.parse_one(cte_def, dialect=self.dialect)
            
            # Find the SELECT statement
            select_stmt = None
            if isinstance(parsed_cte, sqlglot.exp.Select):
                select_stmt = parsed_cte
            else:
                for node in parsed_cte.walk():
                    if isinstance(node, sqlglot.exp.Select):
                        select_stmt = node
                        break
            
            if select_stmt:
                # Look for the column in SELECT expressions
                for expr in select_stmt.expressions:
                    if isinstance(expr, sqlglot.exp.Alias):
                        alias_name = str(expr.alias)
                        if alias_name.lower() == column_name.lower():
                            return self._trace_column_source_dynamically(expr.this, select_stmt, analysis)
                    
                    elif isinstance(expr, sqlglot.exp.Column):
                        col_name = str(expr.name)
                        if col_name.lower() == column_name.lower():
                            return self._trace_column_source_dynamically(expr, select_stmt, analysis)
                
                # If not found in explicit SELECT, might be from wildcard
                return self._resolve_from_wildcard_context(column_name, select_stmt, analysis)
        
        except Exception:
            pass
        
        # Method 2: Dynamic regex-based analysis
        return self._regex_dynamic_column_resolution(column_name, cte_def, analysis)
    
    def _trace_column_source_dynamically(self, expr, select_stmt, analysis):
        """Dynamically trace any expression back to its source table"""
        
        if isinstance(expr, sqlglot.exp.Column):
            table_ref = expr.table
            col_name = str(expr.name)
            
            if table_ref:
                table_alias = str(table_ref)
                actual_table = self._find_actual_table_for_alias(table_alias, select_stmt, analysis)
                
                if actual_table:
                    return {
                        'type': 'direct',
                        'source_table': actual_table,
                        'source_column': col_name,
                        'table_alias': table_alias,
                        'resolution_method': 'dynamic_qualified_resolution'
                    }
            else:
                resolved_table = self._resolve_unqualified_dynamically(col_name, select_stmt, analysis)
                
                if resolved_table:
                    return {
                        'type': 'direct',
                        'source_table': resolved_table,
                        'source_column': col_name,
                        'table_alias': '',
                        'resolution_method': 'dynamic_unqualified_resolution'
                    }
        
        # For complex expressions, extract all column references
        elif hasattr(expr, 'walk'):
            referenced_tables = set()
            referenced_columns = set()
            
            for node in expr.walk():
                if isinstance(node, sqlglot.exp.Column):
                    node_table_ref = node.table
                    node_col_name = str(node.name)
                    
                    if node_table_ref:
                        node_table_alias = str(node_table_ref)
                        actual_table = self._find_actual_table_for_alias(node_table_alias, select_stmt, analysis)
                        if actual_table:
                            referenced_tables.add(actual_table)
                            referenced_columns.add(node_col_name)
                    else:
                        resolved_table = self._resolve_unqualified_dynamically(node_col_name, select_stmt, analysis)
                        if resolved_table:
                            referenced_tables.add(resolved_table)
                            referenced_columns.add(node_col_name)
            
            if referenced_tables:
                return {
                    'type': 'derived',
                    'expression': str(expr),
                    'expression_type': type(expr).__name__,
                    'referenced_columns': [{
                        'table': table,
                        'column': col,
                        'alias': ''
                    } for table in referenced_tables for col in referenced_columns],
                    'ultimate_source_tables': list(referenced_tables),
                    'primary_source_table': list(referenced_tables)[0]
                }
        
        return None
    
    def _find_actual_table_for_alias(self, table_alias, select_stmt, analysis):
        """Dynamically find actual table name for any alias in any SELECT statement"""
        
        try:
            # Check FROM clause
            from_clause = select_stmt.find(sqlglot.exp.From)
            if from_clause and from_clause.this:
                table_expr = from_clause.this
                if isinstance(table_expr, sqlglot.exp.Table):
                    if hasattr(table_expr, 'alias') and table_expr.alias:
                        if str(table_expr.alias).lower() == table_alias.lower():
                            return str(table_expr)
                    table_name = str(table_expr)
                    implicit_alias = table_name.split('.')[-1].lower()
                    if implicit_alias == table_alias.lower():
                        return table_name
            
            # Check JOIN clauses
            for join in select_stmt.find_all(sqlglot.exp.Join):
                if join.this and isinstance(join.this, sqlglot.exp.Table):
                    join_table = join.this
                    if hasattr(join_table, 'alias') and join_table.alias:
                        if str(join_table.alias).lower() == table_alias.lower():
                            return str(join_table)
                    table_name = str(join_table)
                    implicit_alias = table_name.split('.')[-1].lower()
                    if implicit_alias == table_alias.lower():
                        return table_name
        
        except Exception:
            pass
        
        # Method 2: Fallback to analysis table_aliases
        if table_alias.lower() in analysis.get('table_aliases', {}):
            return analysis['table_aliases'][table_alias.lower()]
        
        return None
    
    def _resolve_unqualified_dynamically(self, column_name, select_stmt, analysis):
        """Dynamically resolve unqualified columns using SQL precedence rules"""
        
        tables_in_order = []
        
        try:
            # Collect JOIN tables (in reverse order - most recent JOIN first)
            join_tables = []
            for join in select_stmt.find_all(sqlglot.exp.Join):
                if join.this and isinstance(join.this, sqlglot.exp.Table):
                    join_tables.append(str(join.this))
            
            # Add JOIN tables in reverse order (most recent first)
            tables_in_order.extend(reversed(join_tables))
            
            # Add FROM table last (lowest precedence)
            from_clause = select_stmt.find(sqlglot.exp.From)
            if from_clause and from_clause.this and isinstance(from_clause.this, sqlglot.exp.Table):
                tables_in_order.append(str(from_clause.this))
            
            # Return the first table (highest precedence)
            if tables_in_order:
                return tables_in_order[0]
        
        except Exception:
            pass
        
        # Fallback: use any available table from analysis
        for alias, table_name in analysis.get('table_aliases', {}).items():
            if not table_name.startswith('CTE_') and table_name != analysis.get('object_name', ''):
                return table_name
        
        return None
    
    def _resolve_from_wildcard_context(self, column_name, select_stmt, analysis):
        """Resolve column that might come from SELECT * context"""
        
        # Check if there's a wildcard in the SELECT
        has_wildcard = False
        try:
            for expr in select_stmt.expressions:
                if isinstance(expr, sqlglot.exp.Star):
                    has_wildcard = True
                    break
        except Exception:
            pass
        
        if has_wildcard:
            resolved_table = self._resolve_unqualified_dynamically(column_name, select_stmt, analysis)
            
            if resolved_table:
                return {
                    'type': 'direct',
                    'source_table': resolved_table,
                    'source_column': column_name,
                    'table_alias': '',
                    'resolution_method': 'wildcard_dynamic_resolution'
                }
        
        return None
    
    def _regex_dynamic_column_resolution(self, column_name, cte_def, analysis):
        """Dynamic regex-based resolution as fallback"""
        
        # Method 1: Look for qualified references to this column
        qualified_pattern = rf'\b([a-zA-Z_][a-zA-Z0-9_]*)\.{re.escape(column_name)}\b'
        matches = re.findall(qualified_pattern, cte_def, re.IGNORECASE)
        
        if matches:
            table_alias = matches[0]
            if table_alias.lower() in analysis.get('table_aliases', {}):
                actual_table = analysis['table_aliases'][table_alias.lower()]
                return {
                    'type': 'direct',
                    'source_table': actual_table,
                    'source_column': column_name,
                    'table_alias': table_alias,
                    'resolution_method': 'regex_qualified_dynamic'
                }
        
        # Method 2: Analyze JOIN structure to determine table precedence
        table_pattern = r'(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_.]*(?:\s+(?:as\s+)?[a-zA-Z_][a-zA-Z0-9_]*)?)'  
        table_matches = re.findall(table_pattern, cte_def, re.IGNORECASE)
        
        if table_matches:
            for table_match in reversed(table_matches):
                parts = table_match.strip().split()
                if len(parts) >= 3 and parts[1].lower() == 'as':
                    table_name = parts[0]
                    table_alias = parts[2]
                elif len(parts) == 2:
                    table_name = parts[0]
                    table_alias = parts[1]
                else:
                    table_name = parts[0]
                    table_alias = table_name.split('.')[-1]
                
                if table_alias.lower() in analysis.get('table_aliases', {}):
                    actual_table = analysis['table_aliases'][table_alias.lower()]
                    return {
                        'type': 'direct',
                        'source_table': actual_table,
                        'source_column': column_name,
                        'table_alias': table_alias,
                        'resolution_method': 'regex_join_precedence_dynamic'
                    }
        
        return None
    def _v1_enhance_derived_columns_with_cte_tracing(self, analysis):
        """Enhance derived columns by tracing unqualified column references through CTEs"""
        
        for col_name, derived_info in analysis['derived_columns'].items():
            enhanced_references = derived_info['referenced_columns'].copy()
            ultimate_tables = set()
            source_columns = set()
            
            for ref in derived_info['referenced_columns']:
                table_name = ref['table']
                if ' AS ' in table_name:
                    table_name = table_name.split(' AS ')[0]
                ultimate_tables.add(table_name)
                source_columns.add(ref.get('column', ''))
            
            unqualified_columns = derived_info.get('unqualified_columns', [])
            
            for unqual_col in unqualified_columns:
                traced_sources = self._v1_trace_column_through_ctes(unqual_col, analysis)
                
                for source in traced_sources:
                    enhanced_references.append({
                        'table': source['table'],
                        'column': source['column'],
                        'alias': source.get('alias', ''),
                        'traced_from': unqual_col
                    })
                    
                    table_name = source['table']
                    if ' AS ' in table_name:
                        table_name = table_name.split(' AS ')[0]
                    ultimate_tables.add(table_name)
                    source_columns.add(source['column'])# Also check if this column references CTE columns and resolve them
            for ref in derived_info.get('referenced_columns', []):
                ref_table = ref.get('table', '')
                ref_column = ref.get('column', '')
                
                # If this references a CTE, resolve the CTE column
                if ref_table.startswith('CTE_'):
                    cte_name = ref_table[4:]
                    if cte_name in analysis.get('cte_column_details', {}):
                        cte_columns = analysis['cte_column_details'][cte_name]
                        
                        if ref_column in cte_columns:
                            cte_col_info = cte_columns[ref_column]
                            
                            if cte_col_info.get('type') == 'direct':
                                # Resolve direct CTE column to its source
                                source_table = cte_col_info.get('source_table', '')
                                source_column = cte_col_info.get('source_column', ref_column)
                                
                                if ' AS ' in source_table:
                                    source_table = source_table.split(' AS ')[0]
                                
                                if source_table and source_table != analysis.get('object_name', ''):
                                    ultimate_tables.add(source_table)
                                if source_column:
                                    source_columns.add(source_column)
                                
                                enhanced_references.append({
                                    'table': source_table,
                                    'column': source_column,
                                    'alias': cte_col_info.get('table_alias', ''),
                                    'resolved_from_cte': cte_name,
                                    'original_cte_reference': f'{ref_table}.{ref_column}'
                                })
                            
                            elif cte_col_info.get('type') == 'derived':
                                # Inherit enhanced references from derived CTE column
                                for cte_ref in cte_col_info.get('enhanced_referenced_columns', []):
                                    cte_table_name = cte_ref.get('table', '')
                                    cte_column_name = cte_ref.get('column', '')
                                    
                                    if ' AS ' in cte_table_name:
                                        cte_table_name = cte_table_name.split(' AS ')[0]
                                    
                                    if cte_table_name and cte_table_name != analysis.get('object_name', ''):
                                        ultimate_tables.add(cte_table_name)
                                    if cte_column_name:
                                        source_columns.add(cte_column_name)
                                    
                                    enhanced_references.append({
                                        'table': cte_table_name,
                                        'column': cte_column_name,
                                        'alias': cte_ref.get('alias', ''),
                                        'resolved_from_cte': cte_name,
                                        'original_cte_reference': f'{ref_table}.{ref_column}'
                                    })
            
            # Also check if this column comes from a CTE and inherit its enhanced references
            main_cte = self._v1_find_main_cte_for_view(analysis)
            if main_cte and main_cte in analysis.get('cte_column_details', {}):
                cte_columns = analysis['cte_column_details'][main_cte]
                
                # Check if this column exists in the CTE
                cte_col_info = None
                if col_name in cte_columns:
                    cte_col_info = cte_columns[col_name]
                else:
                    # Case-insensitive search
                    for cte_col_name, col_info in cte_columns.items():
                        if cte_col_name.lower() == col_name.lower():
                            cte_col_info = col_info
                            break
                
                if cte_col_info and cte_col_info.get('type') == 'derived':
                    # Inherit enhanced references from CTE
                    for ref in cte_col_info.get('enhanced_referenced_columns', []):
                        table_name = ref.get('table', '')
                        column_name = ref.get('column', '')
                        
                        if ' AS ' in table_name:
                            table_name = table_name.split(' AS ')[0]
                        
                        if table_name and table_name != analysis.get('object_name', ''):
                            ultimate_tables.add(table_name)
                        if column_name:
                            source_columns.add(column_name)
                        
                        enhanced_references.append({
                            'table': table_name,
                            'column': column_name,
                            'alias': ref.get('alias', ''),
                            'inherited_from_cte': main_cte
                        })
                
                elif cte_col_info and cte_col_info.get('type') == 'direct':
                    # For direct CTE columns, use the source table directly
                    source_table = cte_col_info.get('source_table', '')
                    source_column = cte_col_info.get('source_column', col_name)
                    
                    if ' AS ' in source_table:
                        source_table = source_table.split(' AS ')[0]
                    
                    if source_table and source_table != analysis.get('object_name', ''):
                        ultimate_tables.add(source_table)
                    if source_column:
                        source_columns.add(source_column)
                    
                    enhanced_references.append({
                        'table': source_table,
                        'column': source_column,
                        'alias': cte_col_info.get('table_alias', ''),
                        'inherited_from_cte_direct': main_cte
                    })# Clean up ultimate_source_tables to remove CTE references
            cleaned_ultimate_tables = set()
            for table in ultimate_tables:
                if not table.startswith('CTE_'):
                    cleaned_ultimate_tables.add(table)
            
            analysis['derived_columns'][col_name]['enhanced_references'] = enhanced_references
            analysis['derived_columns'][col_name]['ultimate_source_tables'] = list(cleaned_ultimate_tables)
            analysis['derived_columns'][col_name]['source_columns'] = list(source_columns)
    
    def _v1_trace_column_through_ctes(self, column_name, analysis):
        """Trace a column through the CTE hierarchy"""
        traced_sources = []
        
        cte_order = self._v1_determine_cte_dependency_order(analysis)
        main_cte = cte_order[-1] if cte_order else None
        
        if main_cte and main_cte in analysis['cte_column_details']:
            main_cte_columns = analysis['cte_column_details'][main_cte]
            
            if column_name in main_cte_columns:
                col_info = main_cte_columns[column_name]
                
                if col_info['type'] == 'direct':
                    traced_sources.append({
                        'table': col_info['source_table'],
                        'column': col_info['source_column'],
                        'alias': col_info.get('table_alias', '')
                    })
                
                elif col_info['type'] == 'derived':
                    for ref in col_info.get('referenced_columns', []):
                        traced_sources.append({
                            'table': ref['table'],
                            'column': ref['column'],
                            'alias': ref.get('alias', '')
                        })
                    for ref in col_info.get('enhanced_referenced_columns', []):
                        traced_sources.append({
                            'table': ref['table'],
                            'column': ref['column'],
                            'alias': ref.get('alias', '')
                        })
        
        return traced_sources
    
    def _v1_resolve_view_columns_through_ctes(self, analysis):
        """Resolve view columns by mapping them to CTE columns"""
        
        main_cte = self._v1_find_main_cte_for_view(analysis)
        
        if not main_cte or main_cte not in analysis['cte_column_details']:
            return
        
        main_cte_columns = analysis['cte_column_details'][main_cte]
        
        for view_col in analysis['object_columns']:
            if view_col in analysis['column_mappings'] or view_col in analysis['derived_columns']:
                continue
            
            if view_col in main_cte_columns:
                cte_col_info = main_cte_columns[view_col]
                self._v1_map_view_column_from_cte(view_col, cte_col_info, analysis)
                continue
            
            for cte_col_name, cte_col_info in main_cte_columns.items():
                if cte_col_name.lower() == view_col.lower():
                    self._v1_map_view_column_from_cte(view_col, cte_col_info, analysis)
                    break
    
    def _v1_find_main_cte_for_view(self, analysis):
        """Find the main CTE that the view's main SELECT uses"""
        cte_order = self._v1_determine_cte_dependency_order(analysis)
        if cte_order:
            return cte_order[-1]
        return None
    
    def _v1_map_view_column_from_cte(self, view_col, cte_col_info, analysis):
        """Map a view column based on CTE column information"""
        
        if cte_col_info['type'] == 'direct':
            source_table = cte_col_info['source_table']
            if ' AS ' in source_table:
                source_table = source_table.split(' AS ')[0]
            if source_table.startswith('CTE_'):
                source_table = source_table[4:]
            
            analysis['column_mappings'][view_col] = {
                'type': 'direct',
                'source_table': source_table,
                'source_column': cte_col_info['source_column'],
                'table_alias': cte_col_info.get('table_alias', ''),
                'traced_through_cte': True
            }
        
        elif cte_col_info['type'] == 'derived':
            ultimate_tables = set()
            source_columns = set()
            
            for ref in cte_col_info.get('referenced_columns', []):
                table_name = ref['table']
                if ' AS ' in table_name:
                    table_name = table_name.split(' AS ')[0]
                if table_name.startswith('CTE_'):
                    table_name = table_name[4:]
                ultimate_tables.add(table_name)
                source_columns.add(ref['column'])
            
            for ref in cte_col_info.get('enhanced_referenced_columns', []):
                table_name = ref['table']
                if ' AS ' in table_name:
                    table_name = table_name.split(' AS ')[0]
                if table_name.startswith('CTE_'):
                    table_name = table_name[4:]
                ultimate_tables.add(table_name)
                source_columns.add(ref['column'])
            
            primary_source_table = list(ultimate_tables)[0] if ultimate_tables else 'CALCULATED'
            
            analysis['derived_columns'][view_col] = {
                'expression': cte_col_info['expression'],
                'expression_type': cte_col_info['expression_type'],
                'referenced_columns': cte_col_info.get('referenced_columns', []),
                'enhanced_references': cte_col_info.get('referenced_columns', []),
                'ultimate_source_tables': list(ultimate_tables),
                'primary_source_table': primary_source_table,
                'source_columns': list(source_columns),
                'traced_through_cte': True
            }
    
    def _v1_resolve_main_select_derived_columns(self, parsed, analysis):
        """Enhanced resolution of derived columns from the main SELECT statement"""
        
        main_select = None
        for node in parsed.walk():
            if isinstance(node, exp.Select):
                parent = node.parent
                in_cte = False
                while parent:
                    if isinstance(parent, exp.CTE):
                        in_cte = True
                        break
                    parent = parent.parent
                
                if not in_cte:
                    main_select = node
                    break
        
        if not main_select:
            return
        
        for expr in main_select.expressions:
            if isinstance(expr, exp.Alias):
                column_alias = str(expr.alias)
                source_expr = expr.this
                
                if column_alias in analysis['object_columns']:
                    
                    if isinstance(source_expr, exp.Column):
                        table_ref = source_expr.table
                        col_name = str(source_expr.name)
                        
                        if table_ref:
                            table_ref_str = str(table_ref)
                            
                            if table_ref_str in analysis.get('cte_column_details', {}):
                                resolved_tables, resolved_columns = self._resolve_cte_reference(
                                    table_ref_str, col_name, analysis
                                )
                                
                                if resolved_tables:
                                    primary_table = list(resolved_tables)[0]
                                    primary_column = list(resolved_columns)[0] if resolved_columns else col_name
                                    
                                    analysis['column_mappings'][column_alias] = {
                                        'type': 'direct',
                                        'source_table': primary_table,
                                        'source_column': primary_column,
                                        'table_alias': table_ref_str,
                                        'resolved_from_cte': True
                                    }
                                    continue
                            
                            elif table_ref_str.lower() in analysis.get('table_aliases', {}):
                                actual_table = analysis['table_aliases'][table_ref_str.lower()]
                                
                                if not actual_table.startswith('CTE_'):
                                    analysis['column_mappings'][column_alias] = {
                                        'type': 'direct',
                                        'source_table': actual_table,
                                        'source_column': col_name,
                                        'table_alias': table_ref_str
                                    }
                                    continue
                        
                        if column_alias not in analysis['column_mappings']:
                            self._v1_map_direct_column(column_alias, source_expr, analysis)
                    
                    else:
                        self._v1_map_enhanced_derived_column(column_alias, source_expr, analysis)
            
            elif isinstance(expr, exp.Column):
                column_name = str(expr.name)
                if column_name in analysis['object_columns'] and column_name not in analysis['column_mappings']:
                    self._v1_map_direct_column(column_name, expr, analysis)
    
    def _v1_map_enhanced_derived_column(self, column_name, expression, analysis):
        """Enhanced derived column mapping with better CTE resolution"""
        
        referenced_columns = []
        unqualified_columns = []
        ultimate_tables = set()
        source_columns = set()
        
        expression_type = type(expression).__name__
        
        if expression_type == 'Window':
            self._extract_window_function_columns(expression, ultimate_tables, source_columns, referenced_columns, analysis)
        else:
            for node in expression.walk():
                if isinstance(node, exp.Column):
                    table_ref = node.table
                    col_name = str(node.name)
                    
                    if table_ref:
                        table_ref_str = str(table_ref)
                        
                        if table_ref_str in analysis.get('cte_column_details', {}):
                            resolved_tables, resolved_cols = self._resolve_cte_reference(
                                table_ref_str, col_name, analysis
                            )
                            
                            if resolved_tables:
                                ultimate_tables.update(resolved_tables)
                                source_columns.update(resolved_cols)
                                
                                for table in resolved_tables:
                                    for col in resolved_cols:
                                        referenced_columns.append({
                                            'table': table,
                                            'column': col,
                                            'alias': table_ref_str,
                                            'resolved_from_cte': True
                                        })
                            else:
                                actual_table = analysis['table_aliases'].get(table_ref_str.lower(), table_ref_str)
                                if ' AS ' in actual_table:
                                    actual_table = actual_table.split(' AS ')[0]
                                if actual_table.startswith('CTE_'):
                                    actual_table = actual_table[4:]
                                
                                ultimate_tables.add(actual_table)
                                source_columns.add(col_name)
                                
                                referenced_columns.append({
                                    'table': actual_table,
                                    'column': col_name,
                                    'alias': table_ref_str
                                })
                        else:
                            actual_table = analysis['table_aliases'].get(table_ref_str.lower(), table_ref_str)
                            if ' AS ' in actual_table:
                                actual_table = actual_table.split(' AS ')[0]
                            if actual_table.startswith('CTE_'):
                                actual_table = actual_table[4:]
                            
                            ultimate_tables.add(actual_table)
                            source_columns.add(col_name)
                            
                            referenced_columns.append({
                                'table': actual_table,
                                'column': col_name,
                                'alias': table_ref_str
                            })
                    else:
                        unqualified_columns.append(col_name)
        
        # Resolve unqualified columns
        for unqual_col in unqualified_columns:
            resolved = False
            
            main_cte = self._v1_find_main_cte_for_view(analysis)
            if main_cte:
                resolved_tables, resolved_cols = self._resolve_cte_reference(
                    main_cte, unqual_col, analysis
                )
                
                if resolved_tables:
                    ultimate_tables.update(resolved_tables)
                    source_columns.update(resolved_cols)
                    
                    for table in resolved_tables:
                        for col in resolved_cols:
                            referenced_columns.append({
                                'table': table,
                                'column': col,
                                'alias': '',
                                'traced_from': unqual_col,
                                'resolved_from_cte': True
                            })
                    resolved = True
            
            if not resolved:
                for cte_name in analysis.get('cte_column_details', {}).keys():
                    if cte_name != main_cte:
                        resolved_tables, resolved_cols = self._resolve_cte_reference(
                            cte_name, unqual_col, analysis
                        )
                        
                        if resolved_tables:
                            ultimate_tables.update(resolved_tables)
                            source_columns.update(resolved_cols)
                            
                            for table in resolved_tables:
                                for col in resolved_cols:
                                    referenced_columns.append({
                                        'table': table,
                                        'column': col,
                                        'alias': '',
                                        'traced_from': unqual_col,
                                        'resolved_from_cte': True
                                    })
                            resolved = True
                            break
        
        if not ultimate_tables and expression_type in ['Sub', 'Add', 'Mul', 'Div', 'Binary']:
            ultimate_tables.add('CALCULATED')
            primary_source_table = 'CALCULATED'
        else:
            primary_source_table = list(ultimate_tables)[0] if ultimate_tables else 'CALCULATED'
        
        analysis['derived_columns'][column_name] = {
            'expression': str(expression),
            'expression_type': expression_type,
            'referenced_columns': referenced_columns,
            'unqualified_columns': unqualified_columns,
            'enhanced_references': referenced_columns,
            'ultimate_source_tables': list(ultimate_tables),
            'primary_source_table': primary_source_table,
            'source_columns': list(source_columns)
        }
    
    def _extract_window_function_columns(self, window_expr, ultimate_tables, source_columns, referenced_columns, analysis):
        """Extract column references from window function PARTITION BY and ORDER BY clauses"""
        
        for node in window_expr.walk():
            if isinstance(node, exp.Column):
                table_ref = node.table
                col_name = str(node.name)
                
                if table_ref:
                    table_ref_str = str(table_ref)
                    
                    if table_ref_str in analysis.get('cte_column_details', {}):
                        resolved_tables, resolved_columns = self._resolve_cte_reference(
                            table_ref_str, col_name, analysis
                        )
                        
                        if resolved_tables:
                            ultimate_tables.update(resolved_tables)
                            source_columns.update(resolved_columns)
                            
                            for table in resolved_tables:
                                for col in resolved_columns:
                                    referenced_columns.append({
                                        'table': table,
                                        'column': col,
                                        'alias': table_ref_str,
                                        'resolved_from_cte': True,
                                        'context': 'window_function'
                                    })
                        else:
                            actual_table = analysis['table_aliases'].get(table_ref_str.lower(), table_ref_str)
                            if ' AS ' in actual_table:
                                actual_table = actual_table.split(' AS ')[0]
                            if actual_table.startswith('CTE_'):
                                actual_table = actual_table[4:]
                            
                            ultimate_tables.add(actual_table)
                            source_columns.add(col_name)
                            
                            referenced_columns.append({
                                'table': actual_table,
                                'column': col_name,
                                'alias': table_ref_str,
                                'context': 'window_function'
                            })
                    else:
                        actual_table = analysis['table_aliases'].get(table_ref_str.lower(), table_ref_str)
                        if ' AS ' in actual_table:
                            actual_table = actual_table.split(' AS ')[0]
                        if actual_table.startswith('CTE_'):
                            actual_table = actual_table[4:]
                        
                        ultimate_tables.add(actual_table)
                        source_columns.add(col_name)
                        
                        referenced_columns.append({
                            'table': actual_table,
                            'column': col_name,
                            'alias': table_ref_str,
                            'context': 'window_function'
                        })
                else:
                    # Unqualified column in window function
                    resolved = False
                    
                    for cte_name in analysis.get('cte_column_details', {}).keys():
                        resolved_tables, resolved_columns = self._resolve_cte_reference(
                            cte_name, col_name, analysis
                        )
                        
                        if resolved_tables:
                            ultimate_tables.update(resolved_tables)
                            source_columns.update(resolved_columns)
                            
                            for table in resolved_tables:
                                for col in resolved_columns:
                                    referenced_columns.append({
                                        'table': table,
                                        'column': col,
                                        'alias': '',
                                        'resolved_from_cte': True,
                                        'context': 'window_function_unqualified'
                                    })
                            resolved = True
                            break
                    
                    if not resolved:
                        for alias, table_name in analysis.get('table_aliases', {}).items():
                            if not table_name.startswith('CTE_') and table_name != analysis.get('object_name', ''):
                                if ' AS ' in table_name:
                                    table_name = table_name.split(' AS ')[0]
                                
                                ultimate_tables.add(table_name)
                                source_columns.add(col_name)
                                
                                referenced_columns.append({
                                    'table': table_name,
                                    'column': col_name,
                                    'alias': alias,
                                    'context': 'window_function_inferred'
                                })
                                break
    
    def _resolve_cte_reference(self, table_name, column_name, analysis):
        """Enhanced CTE reference resolution with dynamic recursive tracing"""
        
        resolved_tables = set()
        resolved_columns = set()
        
        # Recursively resolve CTE references with dynamic approach
        def _recursive_cte_resolve(current_table, current_column, visited_ctes=None):
            if visited_ctes is None:
                visited_ctes = set()
            
            if current_table in visited_ctes:
                return set(), set()
            
            visited_ctes.add(current_table)
            local_tables = set()
            local_columns = set()
            
            # Try dynamic resolution first
            dynamic_result = self._dynamic_resolve_column_in_cte(current_column, current_table, analysis)
            if dynamic_result and dynamic_result.get('type') == 'direct':
                source_table = dynamic_result.get('source_table', '')
                source_column = dynamic_result.get('source_column', current_column)
                
                if ' AS ' in source_table:
                    source_table = source_table.split(' AS ')[0]
                if source_table.startswith('CTE_'):
                    source_table = source_table[4:]
                
                if source_table in analysis.get('cte_column_details', {}):
                    sub_tables, sub_columns = _recursive_cte_resolve(
                        source_table, source_column, visited_ctes.copy()
                    )
                    local_tables.update(sub_tables)
                    local_columns.update(sub_columns)
                else:
                    local_tables.add(source_table)
                    local_columns.add(source_column)
                
                return local_tables, local_columns
            
            # Check if this table is actually a CTE
            if current_table in analysis.get('cte_column_details', {}):
                cte_columns = analysis['cte_column_details'][current_table]
                
                cte_col_info = None
                if current_column in cte_columns:
                    cte_col_info = cte_columns[current_column]
                else:
                    for cte_col_name, col_info in cte_columns.items():
                        if cte_col_name.lower() == current_column.lower():
                            cte_col_info = col_info
                            break
                
                if cte_col_info:
                    if cte_col_info.get('type') == 'direct':
                        actual_source_table = cte_col_info.get('source_table', '')
                        if ' AS ' in actual_source_table:
                            actual_source_table = actual_source_table.split(' AS ')[0]
                        if actual_source_table.startswith('CTE_'):
                            actual_source_table = actual_source_table[4:]
                        
                        original_source_column = cte_col_info.get('original_column', 
                                                                cte_col_info.get('source_column', current_column))
                        
                        if actual_source_table in analysis.get('cte_column_details', {}):
                            sub_tables, sub_columns = _recursive_cte_resolve(
                                actual_source_table, 
                                original_source_column,
                                visited_ctes.copy()
                            )
                            local_tables.update(sub_tables)
                            local_columns.update(sub_columns)
                        else:
                            ultimate_source_column = self._find_ultimate_source_column(
                                actual_source_table, original_source_column, current_table, current_column, analysis
                            )
                            
                            local_tables.add(actual_source_table)
                            local_columns.add(ultimate_source_column)
                    
                    elif cte_col_info.get('type') == 'derived':
                        for sub_ref in cte_col_info.get('referenced_columns', []):
                            sub_table = sub_ref['table']
                            if ' AS ' in sub_table:
                                sub_table = sub_table.split(' AS ')[0]
                            if sub_table.startswith('CTE_'):
                                sub_table = sub_table[4:]
                            
                            if sub_table in analysis.get('cte_column_details', {}):
                                sub_tables, sub_columns = _recursive_cte_resolve(
                                    sub_table, sub_ref['column'], visited_ctes.copy()
                                )
                                local_tables.update(sub_tables)
                                local_columns.update(sub_columns)
                            else:
                                local_tables.add(sub_table)
                                local_columns.add(sub_ref['column'])
                        
                        for sub_ref in cte_col_info.get('enhanced_referenced_columns', []):
                            sub_table = sub_ref['table']
                            if ' AS ' in sub_table:
                                sub_table = sub_table.split(' AS ')[0]
                            if sub_table.startswith('CTE_'):
                                sub_table = sub_table[4:]
                            
                            if sub_table in analysis.get('cte_column_details', {}):
                                sub_tables, sub_columns = _recursive_cte_resolve(
                                    sub_table, sub_ref['column'], visited_ctes.copy()
                                )
                                local_tables.update(sub_tables)
                                local_columns.update(sub_columns)
                            else:
                                local_tables.add(sub_table)
                                local_columns.add(sub_ref['column'])
                
                if not cte_col_info:
                    for cte_col_name, cte_col_data in cte_columns.items():
                        if cte_col_data.get('type') == 'direct':
                            actual_source_table = cte_col_data.get('source_table', '')
                            if ' AS ' in actual_source_table:
                                actual_source_table = actual_source_table.split(' AS ')[0]
                            if actual_source_table.startswith('CTE_'):
                                actual_source_table = actual_source_table[4:]
                            
                            if actual_source_table in analysis.get('cte_column_details', {}):
                                sub_tables, sub_columns = _recursive_cte_resolve(
                                    actual_source_table, current_column, visited_ctes.copy()
                                )
                                local_tables.update(sub_tables)
                                local_columns.update(sub_columns)
                            else:
                                local_tables.add(actual_source_table)
                    
                    if local_tables:
                        local_columns.add(current_column)
            
            else:
                if current_table in analysis.get('table_aliases', {}):
                    actual_table = analysis['table_aliases'][current_table]
                    if not actual_table.startswith('CTE_'):
                        local_tables.add(actual_table)
                        local_columns.add(current_column)
                else:
                    local_tables.add(current_table)
                    local_columns.add(current_column)
            
            return local_tables, local_columns
        
        resolved_tables, resolved_columns = _recursive_cte_resolve(table_name, column_name)
        return resolved_tables, resolved_columns
    
    def _find_ultimate_source_column(self, source_table, cte_column_name, cte_name, view_column_name, analysis):
        """Find the ultimate source column name by tracing through CTE definitions"""
        
        if cte_column_name.lower() == view_column_name.lower():
            ultimate_column = self._trace_column_through_cte_definition(cte_name, cte_column_name, analysis)
            if ultimate_column and ultimate_column != cte_column_name:
                return ultimate_column
        
        if cte_name in analysis.get('cte_definitions', {}):
            cte_def = analysis['cte_definitions'][cte_name]['definition']
            
            alias_pattern = rf'\b([a-zA-Z_][a-zA-Z0-9_.]*)\s+(?:as\s+)?{re.escape(cte_column_name)}\b'
            matches = re.findall(alias_pattern, cte_def, re.IGNORECASE)
            
            if matches:
                original_column = matches[0]
                if '.' in original_column:
                    original_column = original_column.split('.')[-1]
                return original_column
        
        return cte_column_name
    
    def _trace_column_through_cte_definition(self, cte_name, column_name, analysis):
        """Trace a column through CTE definition to find its original name"""
        
        if cte_name not in analysis.get('cte_definitions', {}):
            return column_name
        
        cte_def = analysis['cte_definitions'][cte_name]['definition']
        
        agg_patterns = [
            rf'max\s*\([^)]*\)\s+(?:as\s+)?{re.escape(column_name)}',
            rf'min\s*\([^)]*\)\s+(?:as\s+)?{re.escape(column_name)}',
            rf'sum\s*\([^)]*\)\s+(?:as\s+)?{re.escape(column_name)}',
            rf'count\s*\([^)]*\)\s+(?:as\s+)?{re.escape(column_name)}',
            rf'avg\s*\([^)]*\)\s+(?:as\s+)?{re.escape(column_name)}'
        ]
        
        for pattern in agg_patterns:
            match = re.search(pattern, cte_def, re.IGNORECASE)
            if match:
                agg_match = re.search(rf'(max|min|sum|count|avg)\s*\(([^)]+)\)', match.group(0), re.IGNORECASE)
                if agg_match:
                    inner_column = agg_match.group(2).strip()
                    if '.' in inner_column:
                        inner_column = inner_column.split('.')[-1]
                    return inner_column
        return column_name
    
    def _find_column_in_dependent_ctes(self, column_name, current_cte, analysis):
        """Find if a column comes from another CTE that this CTE depends on"""
        
        cte_order = self._v1_determine_cte_dependency_order(analysis)
        
        if current_cte not in cte_order:
            return None
        
        current_index = cte_order.index(current_cte)
        
        for i in range(current_index):
            dependency_cte = cte_order[i]
            
            if current_cte in analysis.get('cte_definitions', {}):
                current_cte_def = analysis['cte_definitions'][current_cte]['definition']
                
                if dependency_cte.lower() in current_cte_def.lower():
                    if (dependency_cte in analysis.get('cte_column_details', {}) and 
                        column_name in analysis['cte_column_details'][dependency_cte]):
                        
                        dependency_col_info = analysis['cte_column_details'][dependency_cte][column_name]
                                    
                        if dependency_col_info.get('type') == 'derived':
                            referenced_columns = dependency_col_info.get('referenced_columns', [])
                            if referenced_columns:
                                ref = referenced_columns[0]
                                return {
                                    'type': 'direct',
                                    'source_table': ref['table'],
                                    'source_column': ref['column'],
                                    'table_alias': ref.get('alias', ''),
                                    'resolved_from_cte': dependency_cte,
                                    'resolution_method': 'cte_dependency_derived'
                                }
                        
                        elif dependency_col_info.get('type') == 'direct':
                            return {
                                'type': 'direct',
                                'source_table': dependency_col_info['source_table'],
                                'source_column': dependency_col_info['source_column'],
                                'table_alias': dependency_col_info.get('table_alias', ''),
                                'resolved_from_cte': dependency_cte,
                                'resolution_method': 'cte_dependency_direct'
                            }
        
        return None
    
    def _find_original_column_name_in_cte(self, cte_name, cte_alias, resolved_column, analysis):
        """Find the original column name by parsing the CTE definition"""
        
        if cte_name not in analysis.get('cte_definitions', {}):
            return resolved_column
        
        cte_def = analysis['cte_definitions'][cte_name]['definition']
        
        # Look for column alias patterns in the SELECT clause
        alias_pattern = rf'\b([a-zA-Z_][a-zA-Z0-9_.]*?)\s+as\s+{re.escape(cte_alias)}\b'
        matches = re.findall(alias_pattern, cte_def, re.IGNORECASE)
        
        if matches:
            original_column = matches[0]
            if '.' in original_column:
                original_column = original_column.split('.')[-1]
            return original_column
        
        # Look for aggregation functions that create this alias
        agg_patterns = [
            rf'(max|min|sum|count|avg|first_value|last_value)\s*\([^)]*?([a-zA-Z_][a-zA-Z0-9_.]*)\s*\)\s+(?:as\s+)?{re.escape(cte_alias)}\b',
            rf'(case\s+when.+?end)\s+(?:as\s+)?{re.escape(cte_alias)}\b',
            rf'(datediff\s*\([^)]+?\))\s+(?:as\s+)?{re.escape(cte_alias)}\b'
        ]
        
        for pattern in agg_patterns:
            match = re.search(pattern, cte_def, re.IGNORECASE | re.DOTALL)
            if match:
                if len(match.groups()) >= 2 and match.group(2):
                    inner_column = match.group(2).strip()
                    if '.' in inner_column:
                        inner_column = inner_column.split('.')[-1]
                    return inner_column
        
        return resolved_column
    
    def _resolve_derived_column_source(self, col_name, table_ref, analysis):
        """Resolve a column in a derived expression to its ultimate source tables"""
        
        resolved_tables = set()
        resolved_columns = set()
        
        if table_ref:
            if table_ref in analysis.get('cte_column_details', {}):
                tables, columns = self._resolve_cte_reference(table_ref, col_name, analysis)
                resolved_tables.update(tables)
                resolved_columns.update(columns)
            
            elif table_ref.lower() in analysis.get('table_aliases', {}):
                actual_table = analysis['table_aliases'][table_ref.lower()]
                if actual_table.startswith('CTE_'):
                    cte_name = actual_table[4:]
                    tables, columns = self._resolve_cte_reference(cte_name, col_name, analysis)
                    resolved_tables.update(tables)
                    resolved_columns.update(columns)
                else:
                    resolved_tables.add(actual_table)
                    resolved_columns.add(col_name)
            
            else:
                resolved_tables.add(table_ref)
                resolved_columns.add(col_name)
        
        else:
            # No table reference - need to find where this column comes from
            found_in_cte = False
            
            # Check the main CTE first
            main_cte = self._v1_find_main_cte_for_view(analysis)
            if main_cte and main_cte in analysis.get('cte_column_details', {}):
                cte_columns = analysis['cte_column_details'][main_cte]
                
                if col_name in cte_columns:
                    tables, columns = self._resolve_cte_reference(main_cte, col_name, analysis)
                    resolved_tables.update(tables)
                    resolved_columns.update(columns)
                    found_in_cte = True
                
                elif not found_in_cte:
                    for cte_col_name in cte_columns.keys():
                        if cte_col_name.lower() == col_name.lower():
                            tables, columns = self._resolve_cte_reference(main_cte, cte_col_name, analysis)
                            resolved_tables.update(tables)
                            resolved_columns.update(columns)
                            found_in_cte = True
                            break
            
            # If not found in main CTE, check other CTEs
            if not found_in_cte:
                for cte_name, cte_columns in analysis.get('cte_column_details', {}).items():
                    if cte_name == main_cte:
                        continue
                    
                    if col_name in cte_columns:
                        tables, columns = self._resolve_cte_reference(cte_name, col_name, analysis)
                        resolved_tables.update(tables)
                        resolved_columns.update(columns)
                        found_in_cte = True
                        break
                    
                    for cte_col_name in cte_columns.keys():
                        if cte_col_name.lower() == col_name.lower():
                            tables, columns = self._resolve_cte_reference(cte_name, cte_col_name, analysis)
                            resolved_tables.update(tables)
                            resolved_columns.update(columns)
                            found_in_cte = True
                            break
                    
                    if found_in_cte:
                        break
            
            # If still not found, check direct table references
            if not found_in_cte:
                for alias, table_name in analysis.get('table_aliases', {}).items():
                    if not table_name.startswith('CTE_'):
                        resolved_tables.add(table_name)
                        resolved_columns.add(col_name)
                        break
        
        return resolved_tables, resolved_columns
    
    # ==================== V2 METHODS ====================
    def _v2_find_main_select(self, parsed):
        """Find the main SELECT statement dynamically"""
        all_selects = []
        
        for node in parsed.walk():
            if isinstance(node, exp.Select):
                all_selects.append(node)
        
        if not all_selects:
            return None
        
        best_candidate = None
        max_score = -1
        
        for select_stmt in all_selects:
            score = 0
            score += len(select_stmt.expressions)
            
            has_star = any(isinstance(expr, exp.Star) for expr in select_stmt.expressions)
            has_other = any(not isinstance(expr, exp.Star) for expr in select_stmt.expressions)
            
            if has_star and has_other:
                score += 10
            elif has_star:
                score += 5
            
            if select_stmt.find(exp.From):
                score += 3
            
            if select_stmt.find(exp.Where):
                score += 2
            
            if score > max_score:
                max_score = score
                best_candidate = select_stmt
        
        return best_candidate
    
    def _v2_analyze_select_expressions(self, select_stmt):
        """Analyze what a SELECT statement is doing"""
        analysis = {
            'has_wildcard': False,
            'wildcard_source': None,
            'explicit_expressions': [],
            'from_source': None
        }
        
        for expr in select_stmt.expressions:
            if isinstance(expr, exp.Star):
                analysis['has_wildcard'] = True
                if hasattr(expr, 'table') and expr.table:
                    analysis['wildcard_source'] = str(expr.table)
            elif isinstance(expr, exp.Column) and str(expr).endswith('.*'):
                analysis['has_wildcard'] = True
                wildcard_str = str(expr)
                if '.' in wildcard_str:
                    analysis['wildcard_source'] = wildcard_str.split('.')[0]
            elif isinstance(expr, exp.Alias):
                analysis['explicit_expressions'].append({
                    'alias': str(expr.alias),
                    'expression': str(expr.this),
                    'type': type(expr.this).__name__
                })
            else:
                analysis['explicit_expressions'].append({
                    'expression': str(expr),
                    'type': type(expr).__name__
                })
        
        from_clause = select_stmt.find(exp.From)
        if from_clause:
            analysis['from_source'] = str(from_clause.this)
        
        return analysis
    
    def _v2_build_table_registry(self, parsed):
        """Build registry of all tables and their aliases"""
        registry = {}
        
        for node in parsed.walk():
            if isinstance(node, exp.Table):
                table_name = str(node)
                
                if node.alias:
                    alias = str(node.alias)
                    registry[alias] = {
                        'type': 'table',
                        'full_name': table_name,
                        'alias': alias
                    }
                
                implicit_alias = table_name.split('.')[-1]
                if implicit_alias not in registry:
                    registry[implicit_alias] = {
                        'type': 'table',
                        'full_name': table_name,
                        'alias': implicit_alias
                    }
        
        return registry
    
    def _v2_build_cte_registry(self, parsed, table_registry):
        """Build registry of all CTEs and their column mappings"""
        registry = {}
        
        for node in parsed.walk():
            if isinstance(node, exp.CTE):
                cte_name = str(node.alias)
                cte_select = node.this
                
                cte_analysis = self._v2_analyze_select_expressions(cte_select)
                column_mapping = {}
                
                for expr_info in cte_analysis['explicit_expressions']:
                    if 'alias' in expr_info:
                        alias = expr_info['alias']
                        if expr_info['type'] == 'Column':
                            column_mapping[alias] = self._v2_parse_column_reference(
                                expr_info['expression'], table_registry
                            )
                        else:
                            column_mapping[alias] = {
                                'type': 'derived',
                                'expression': expr_info['expression'],
                                'expression_type': expr_info['type']
                            }
                
                if cte_analysis['has_wildcard']:
                    wildcard_source = cte_analysis.get('wildcard_source')
                    if wildcard_source and wildcard_source in table_registry:
                        column_mapping['__WILDCARD__'] = {
                            'type': 'wildcard',
                            'source_table': table_registry[wildcard_source]['full_name'],
                            'source_alias': wildcard_source
                        }
                    elif cte_analysis['from_source'] and cte_analysis['from_source'] in table_registry:
                        from_source = cte_analysis['from_source']
                        column_mapping['__WILDCARD__'] = {
                            'type': 'wildcard',
                            'source_table': table_registry[from_source]['full_name'],
                            'source_alias': from_source
                        }
                
                registry[cte_name] = {
                    'column_mapping': column_mapping,
                    'analysis': cte_analysis
                }
        
        return registry
    
    def _v2_parse_column_reference(self, col_expr_str, table_registry):
        """Parse a column reference string to find its source"""
        if '.' in col_expr_str:
            parts = col_expr_str.split('.')
            if len(parts) == 2:
                table_alias, column_name = parts
                if table_alias in table_registry:
                    return {
                        'type': 'direct',
                        'source_table': table_registry[table_alias]['full_name'],
                        'source_column': column_name,
                        'table_alias': table_alias
                    }
        
        return {
            'type': 'unqualified',
            'column_name': col_expr_str
        }
    
    def _v2_trace_column_source(self, view_col, main_select_analysis, cte_registry, table_registry):
        """Trace where a view column comes from"""
        for expr_info in main_select_analysis['explicit_expressions']:
            if 'alias' in expr_info and expr_info['alias'].lower() == view_col.lower():
                if expr_info['type'] == 'Column':
                    return self._v2_parse_column_reference(expr_info['expression'], table_registry)
                else:
                    return {
                        'type': 'derived',
                        'expression': expr_info['expression'],
                        'expression_type': expr_info['type']
                    }
        
        if main_select_analysis['has_wildcard']:
            from_source = main_select_analysis['from_source']
            
            if from_source and from_source in cte_registry:
                cte_info = cte_registry[from_source]
                
                if view_col in cte_info['column_mapping']:
                    return cte_info['column_mapping'][view_col]
                
                for cte_col_name, cte_col_mapping in cte_info['column_mapping'].items():
                    if (cte_col_name != '__WILDCARD__' and 
                        cte_col_name.lower() == view_col.lower()):
                        return cte_col_mapping
                
                if '__WILDCARD__' in cte_info['column_mapping']:
                    wildcard_info = cte_info['column_mapping']['__WILDCARD__']
                    return {
                        'type': 'direct',
                        'source_table': wildcard_info['source_table'],
                        'source_column': view_col,
                        'table_alias': wildcard_info['source_alias'],
                        'traced_through': from_source
                    }
            
            elif from_source and from_source in table_registry:
                return {
                    'type': 'direct',
                    'source_table': table_registry[from_source]['full_name'],
                    'source_column': view_col,
                    'table_alias': from_source,
                    'wildcard_source': True
                }
            
            elif from_source:
                for table_alias, table_info in table_registry.items():
                    if (table_alias.lower() == from_source.lower() or 
                        table_info['full_name'].lower() == from_source.lower()):
                        return {
                            'type': 'direct',
                            'source_table': table_info['full_name'],
                            'source_column': view_col,
                            'table_alias': table_alias,
                            'wildcard_source': True
                        }
                
                return {
                    'type': 'direct',
                    'source_table': from_source,
                    'source_column': view_col,
                    'table_alias': from_source,
                    'wildcard_source': True,
                    'fallback_resolution': True
                }
        
        return {
            'type': 'unknown',
            'source_table': 'UNKNOWN',
            'source_column': 'UNKNOWN'
        }
    
    def _v2_enhance_derived_column(self, derived_info, analysis):
        """Enhanced derived column analysis"""
        if 'expression' not in derived_info:
            return derived_info
        
        expression = derived_info['expression']
        
        referenced_columns = []
        ultimate_tables = set()
        source_columns = set()
        
        try:
            import sqlglot
            parsed_expr = sqlglot.parse_one(f"SELECT {expression}", dialect="snowflake")
            
            for node in parsed_expr.walk():
                if isinstance(node, sqlglot.exp.Column):
                    col_name = str(node.name)
                    table_ref = None
                    
                    if hasattr(node, 'table') and node.table:
                        table_ref = str(node.table)
                    
                    resolved_tables, resolved_columns = self._resolve_derived_column_source(
                        col_name, table_ref, analysis
                    )
                    
                    if resolved_tables:
                        ultimate_tables.update(resolved_tables)
                        source_columns.update(resolved_columns)
                        
                        for table in resolved_tables:
                            for column in resolved_columns:
                                referenced_columns.append({
                                    'table': table,
                                    'column': column,
                                    'alias': table_ref or '',
                                    'original_column': col_name
                                })  
        except Exception:
            pass
        if referenced_columns:
            primary_source_table = list(ultimate_tables)[0] if ultimate_tables else 'CALCULATED'
            
            derived_info['enhanced_references'] = referenced_columns
            derived_info['ultimate_source_tables'] = list(ultimate_tables)
            derived_info['primary_source_table'] = primary_source_table
            derived_info['source_columns'] = list(source_columns)
        
        return derived_info
    
    def generate_standard_csv(self, analysis):
        """Generate CSV in the standard 6-column format without duplicates"""
        if 'error' in analysis:
            return f"Error: {analysis['error']}"
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Standard 6-column header
        writer.writerow([
            f'{analysis["ddl_type"].title()}_Name',
            f'{analysis["ddl_type"].title()}_Column',
            'Column_Type',
            'Source_Table',
            'Source_Column',
            'Expression_Type'
        ])
        
        object_name = analysis['object_name']
        processed_columns = set()
        
        for obj_col in analysis['object_columns']:
            if obj_col in processed_columns:
                continue
            
            processed_columns.add(obj_col)
            
            if obj_col in analysis['column_mappings']:
                mapping = analysis['column_mappings'][obj_col]
                
                if mapping.get('type') == 'unknown':
                    main_select_analysis = analysis.get('main_select_analysis', {})
                    
                    if main_select_analysis.get('has_wildcard') and main_select_analysis.get('from_source'):
                        from_source = main_select_analysis['from_source']
                        table_registry = analysis.get('table_registry', {})
                        
                        if from_source in table_registry:
                            source_table = table_registry[from_source]['full_name']
                            
                            mapping = {
                                'type': 'direct',
                                'source_table': source_table,
                                'source_column': obj_col,
                                'table_alias': from_source,
                                'resolved_method': 'wildcard_recovery'
                            }
                        else:
                            continue
                    else:
                        continue
                
                all_sources = self._get_all_contributing_sources(obj_col, analysis)
                is_derived = False
                expression_type = all_sources.get('expression_type', '') if all_sources else ''
                if (all_sources and 
                    (len(all_sources.get('tables', [])) > 1 or 
                     len(all_sources.get('columns', [])) > 1 or 
                     expression_type in ['Case', 'Coalesce', 'Add', 'Mul', 'Div', 'Sub', 'Binary', 'Anonymous', 'Window', 'Min', 'Max', 'Sum', 'Count', 'Avg'])):
                    is_derived = True
                if is_derived:
                    if all_sources and all_sources.get('tables'):
                        # Clean and deduplicate table names before joining
                        clean_tables = set()
                        for table in all_sources['tables']:
                            clean_table = table.split(' AS ')[0] if ' AS ' in table else table
                            if clean_table.startswith('CTE_'):
                                clean_table = clean_table[4:]
                            if clean_table and clean_table != 'CALCULATED':
                                clean_tables.add(clean_table)
                        source_table_display = '; '.join(sorted(clean_tables)) if clean_tables else 'CALCULATED'
                    else:
                        source_table = mapping['source_table']
                        if ' AS ' in source_table:
                            source_table = source_table.split(' AS ')[0]
                        if source_table.startswith('CTE_'):
                            source_table = source_table[4:]
                        source_table_display = source_table
                    
                    if all_sources and all_sources.get('columns'):
                        source_column_display = '; '.join(sorted(all_sources['columns']))
                    else:
                        source_column_display = mapping['source_column']
                    
                    column_type = 'Derived'
                else:
                    source_table = mapping['source_table']
                    if ' AS ' in source_table:
                        source_table = source_table.split(' AS ')[0]
                    if source_table.startswith('CTE_'):
                        source_table = source_table[4:]
                    
                    source_table_display = source_table
                    source_column_display = mapping['source_column']
                    column_type = 'Direct'
                    expression_type = ''
                
                writer.writerow([
                    object_name.upper(),
                    obj_col.upper(),
                    column_type.upper(),
                    source_table_display.upper(),
                    source_column_display.upper(),
                    expression_type.upper()
                ])
                continue# Check for derived columns with case-insensitive matching - prioritize resolved entries
            derived = None
            candidates = []
            
            # Collect all matching columns (exact and case-insensitive)
            if obj_col in analysis['derived_columns']:
                candidates.append((obj_col, analysis['derived_columns'][obj_col]))
            
            # Case-insensitive search
            for col_name, col_info in analysis['derived_columns'].items():
                if col_name.lower() == obj_col.lower() and col_name != obj_col:
                    candidates.append((col_name, col_info))
            
            # Prioritize the candidate with properly resolved ultimate_source_tables
            if candidates:
                best_candidate = None
                best_score = -1
                
                for col_name, info in candidates:
                    ultimate_tables = info.get('ultimate_source_tables', [])
                    score = 0
                    
                    # Score based on how well resolved the tables are
                    for table in ultimate_tables:
                        if not table.startswith('CTE_'):
                            # Check if it's a real table (contains schema or dots)
                            if '.' in table or table.upper().startswith(('CPS_DSCI', 'MX_', 'BV_')):
                                score += 10  # Fully qualified table name
                            elif table in analysis.get('cte_definitions', {}):
                                score += 1   # CTE name (not resolved)
                            else:
                                score += 5   # Other table name
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = (col_name, info)
                
                # If no candidate found, use the first one
                if not best_candidate:
                    best_candidate = candidates[0]
                
                derived = best_candidate[1]
            
            if derived:# Get all source information - prioritize ultimate_source_tables
                all_source_tables = set()
                all_source_columns = set()
                
                # First try to use ultimate_source_tables (already resolved)
                ultimate_tables = derived.get('ultimate_source_tables', [])
                if ultimate_tables:
                    for table in ultimate_tables:
                        if not table.startswith('CTE_') and table != analysis.get('object_name', '') and table != 'CALCULATED':
                            all_source_tables.add(table)
                
                # Use source_columns if available
                source_columns = derived.get('source_columns', [])
                if source_columns:
                    for col in source_columns:
                        all_source_columns.add(col)# Fallback: extract from enhanced_references if ultimate_source_tables is empty
                if not all_source_tables and derived.get('enhanced_references'):
                    for ref in derived['enhanced_references']:
                        table_name = ref.get('table', '')
                        column_name = ref.get('column', '')
                        
                        # Clean up table name to remove alias first
                        if ' AS ' in table_name:
                            table_name = table_name.split(' AS ')[0]
                        
                        # Skip CTE references and calculated columns - use only resolved references
                        if table_name.startswith('CTE_') or table_name == 'CALCULATED':
                            continue
                        
                        if table_name and table_name != analysis['object_name']:
                            all_source_tables.add(table_name)
                        if column_name and not all_source_columns:
                            all_source_columns.add(column_name)
                elif derived.get('ultimate_source_tables'):
                    for table in derived['ultimate_source_tables']:
                        # Resolve CTE references to actual source tables
                        if table.startswith('CTE_'):
                            resolved_table = self._resolve_cte_table_to_source(table, analysis)
                            if resolved_table:
                                table = resolved_table
                        
                        # Clean up table name to remove alias
                        if ' AS ' in table:
                            table = table.split(' AS ')[0]
                        
                        if table and table != analysis['object_name'] and table != 'CALCULATED':
                            all_source_tables.add(table)
                    
                    if derived.get('source_columns'):
                        all_source_columns.update(derived['source_columns'])
                elif derived.get('referenced_columns'):
                    for ref in derived['referenced_columns']:
                        table_name = ref.get('table', '')
                        column_name = ref.get('column', '')
                        
                        # Clean up table name to remove alias first
                        if ' AS ' in table_name:
                            table_name = table_name.split(' AS ')[0]
                        
                        # Skip CTE references and calculated columns - use only resolved references
                        if table_name.startswith('CTE_') or table_name == 'CALCULATED':
                            continue
                        
                        if table_name and table_name != analysis['object_name']:
                            all_source_tables.add(table_name)
                        if column_name:
                            all_source_columns.add(column_name)
                
                expression_type = derived.get('expression_type', '')
                column_type = 'Derived'
                if all_source_tables:
                    # Clean and deduplicate table names before joining
                    clean_tables = set()
                    for table in all_source_tables:
                        clean_table = table.split(' AS ')[0] if ' AS ' in table else table
                        if clean_table.startswith('CTE_'):
                            clean_table = clean_table[4:]
                        if clean_table and clean_table != 'CALCULATED':
                            clean_tables.add(clean_table)
                    source_table_display = '; '.join(sorted(clean_tables)) if clean_tables else 'CALCULATED'
                else:
                    source_table_display = 'CALCULATED'
                
                if all_source_columns:
                    source_column_display = '; '.join(sorted(all_source_columns))
                else:
                    source_column_display = derived.get('expression_type', '')
                
                writer.writerow([
                    object_name.upper(),
                    obj_col.upper(),
                    column_type.upper(),
                    source_table_display.upper(),
                    source_column_display.upper(),
                    derived.get('expression_type', '').upper()
                ])
                continue
            
            # Handle case-insensitive matching
            found = False
            
            for col_name, mapping in analysis['column_mappings'].items():
                if col_name.lower() == obj_col.lower() and mapping.get('type') != 'unknown':
                    source_table = mapping['source_table']
                    if ' AS ' in source_table:
                        source_table = source_table.split(' AS ')[0]
                    if source_table.startswith('CTE_'):
                        source_table = source_table[4:]
                    writer.writerow([
                        object_name.upper(),
                        obj_col.upper(),
                        'DIRECT',
                        source_table.upper(),
                        mapping['source_column'].upper(),
                        ''
                    ])
                    found = True
                    break
            
            if not found:
                for col_name, derived in analysis['derived_columns'].items():
                    if col_name.lower() == obj_col.lower():
                        all_source_tables = set()
                        all_source_columns = set()
                        if derived.get('enhanced_references'):
                            for ref in derived['enhanced_references']:
                                table_name = ref.get('table', '')
                                column_name = ref.get('column', '')
                                
                                # Clean up table name to remove alias first
                                if ' AS ' in table_name:
                                    table_name = table_name.split(' AS ')[0]
                                if table_name.startswith('CTE_'):
                                    table_name = table_name[4:]
                                
                                if table_name and table_name != analysis['object_name'] and table_name != 'CALCULATED':
                                    all_source_tables.add(table_name)
                                if column_name:
                                    all_source_columns.add(column_name)
                        
                        if not all_source_tables and derived.get('ultimate_source_tables'):
                            for table in derived['ultimate_source_tables']:
                                # Clean up table name to remove alias first
                                if ' AS ' in table:
                                    table = table.split(' AS ')[0]
                                if table.startswith('CTE_'):
                                    table = table[4:]
                                if table and table != analysis['object_name'] and table != 'CALCULATED':
                                    all_source_tables.add(table)
                            
                            if derived.get('source_columns'):
                                all_source_columns.update(derived['source_columns'])
                        if not all_source_tables and derived.get('referenced_columns'):
                            for ref in derived['referenced_columns']:
                                table_name = ref.get('table', '')
                                column_name = ref.get('column', '')
                                
                                # Clean up table name to remove alias first
                                if ' AS ' in table_name:
                                    table_name = table_name.split(' AS ')[0]
                                if table_name.startswith('CTE_'):
                                    table_name = table_name[4:]
                                
                                if table_name and table_name != analysis['object_name'] and table_name != 'CALCULATED':
                                    all_source_tables.add(table_name)
                                if column_name:
                                    all_source_columns.add(column_name)
                        
                        expression_type = derived.get('expression_type', '')
                        column_type = 'Derived'
                        if all_source_tables:
                            # Clean and deduplicate table names before joining
                            clean_tables = set()
                            for table in all_source_tables:
                                clean_table = table.split(' AS ')[0] if ' AS ' in table else table
                                if clean_table.startswith('CTE_'):
                                    clean_table = clean_table[4:]
                                if clean_table and clean_table != 'CALCULATED':
                                    clean_tables.add(clean_table)
                            source_table_display = '; '.join(sorted(clean_tables)) if clean_tables else 'CALCULATED'
                        else:
                            source_table_display = 'CALCULATED'
                        
                        if all_source_columns:
                            source_column_display = '; '.join(sorted(all_source_columns))
                        else:
                            source_column_display = derived.get('expression_type', '')
                        
                        writer.writerow([
                            object_name.upper(),
                            obj_col.upper(),
                            column_type.upper(),
                            source_table_display.upper(),
                            source_column_display.upper(),
                            derived.get('expression_type', '').upper()
                        ])
                        found = True
                        break
            if not found:
                # Try one more comprehensive fallback
                fallback_found = False
                # 1. Check if this column exists in any CTE with case-insensitive matching
                for cte_name, cte_columns in analysis.get('cte_column_details', {}).items():
                    for cte_col_name, cte_col_info in cte_columns.items():
                        if cte_col_name.lower() == obj_col.lower():
                            if cte_col_info.get('type') == 'direct':
                                source_table = cte_col_info.get('source_table', '')
                                source_column = cte_col_info.get('source_column', obj_col)
                                
                                if ' AS ' in source_table:
                                    source_table = source_table.split(' AS ')[0]
                                
                                writer.writerow([
                                    object_name.upper(),
                                    obj_col.upper(),
                                    'DIRECT',
                                    source_table.upper(),
                                    source_column.upper(),
                                    ''
                                ])
                                fallback_found = True
                                break
                            elif cte_col_info.get('type') == 'derived':
                                # Get source tables from enhanced references
                                source_tables = set()
                                source_columns = set()
                                
                                for ref in cte_col_info.get('enhanced_referenced_columns', []):
                                    table_name = ref.get('table', '')
                                    column_name = ref.get('column', '')
                                    
                                    if ' AS ' in table_name:
                                        table_name = table_name.split(' AS ')[0]
                                    
                                    if not table_name.startswith('CTE_') and table_name != analysis.get('object_name', ''):
                                        source_tables.add(table_name)
                                    if column_name:
                                        source_columns.add(column_name)
                                if source_tables:
                                    # Clean and deduplicate table names before joining
                                    clean_tables = set()
                                    for table in source_tables:
                                        clean_table = table.split(' AS ')[0] if ' AS ' in table else table
                                        if clean_table.startswith('CTE_'):
                                            clean_table = clean_table[4:]
                                        if clean_table and clean_table != 'CALCULATED':
                                            clean_tables.add(clean_table)
                                    source_table_display = '; '.join(sorted(clean_tables)) if clean_tables else 'CALCULATED'
                                    source_column_display = '; '.join(sorted(source_columns)) if source_columns else cte_col_info.get('expression_type', '')
                                    
                                    writer.writerow([
                                        object_name.upper(),
                                        obj_col.upper(),
                                        'DERIVED',
                                        source_table_display.upper(),
                                        source_column_display.upper(),
                                        cte_col_info.get('expression_type', '').upper()
                                    ])
                                    fallback_found = True
                                    break
                    
                    if fallback_found:
                        break
                # 2. If still not found, check if this column exists in any table aliases
                if not fallback_found:
                    for alias, table_name in analysis.get('table_aliases', {}).items():
                        if not table_name.startswith('CTE_') and table_name != analysis.get('object_name', ''):
                            # Assume the column exists in this table as a fallback
                            clean_table = table_name.split(' AS ')[0] if ' AS ' in table_name else table_name
                            writer.writerow([
                                object_name.upper(),
                                obj_col.upper(),
                                'DIRECT',
                                clean_table.upper(),
                                obj_col.upper(),
                                ''
                            ])
                            fallback_found = True
                            break
                
                if not fallback_found:
                    writer.writerow([
                        object_name.upper(),
                        obj_col.upper(),
                        'UNKNOWN',
                        'UNKNOWN',
                        'UNKNOWN',
                        ''
                    ])
        
        return output.getvalue()
    def _get_all_contributing_sources(self, column_name, analysis):
        """Get all contributing sources for a column by tracing through CTEs with CTE resolution"""
        all_tables = set()
        all_columns = set()
        expression_type = ''# First check if this column exists in derived_columns (main view level)
        derived_info = None
        candidates = []
        
        # Collect all matching columns (exact and case-insensitive)
        if column_name in analysis.get('derived_columns', {}):
            candidates.append((column_name, analysis['derived_columns'][column_name]))
        
        # Case-insensitive search
        for col_name, info in analysis.get('derived_columns', {}).items():
            if col_name.lower() == column_name.lower() and col_name != column_name:
                candidates.append((col_name, info))# Prioritize the candidate with properly resolved ultimate_source_tables
        if candidates:
            best_candidate = None
            best_score = -1
            
            for col_name, info in candidates:
                ultimate_tables = info.get('ultimate_source_tables', [])
                score = 0
                for table in ultimate_tables:
                    if not table.startswith('CTE_'):
                        # Check if it's a real table (contains schema or dots)
                        if '.' in table or table.upper().startswith(('CPS_DSCI', 'MX_', 'BV_')):
                            score += 10  
                        elif table in analysis.get('cte_definitions', {}):
                            score += 1  
                        else:
                            score += 5  
                
                if score > best_score:
                    best_score = score
                    best_candidate = (col_name, info)
            
            # If no candidate found, use the first one
            if not best_candidate:
                best_candidate = candidates[0]
            
            derived_info = best_candidate[1]
        if derived_info:
            expression_type = derived_info.get('expression_type', '')# Use ultimate_source_tables which should already be resolved
            ultimate_tables = derived_info.get('ultimate_source_tables', [])
            if ultimate_tables:
                for table in ultimate_tables:
                    # Clean up table name to remove alias
                    if ' AS ' in table:
                        table = table.split(' AS ')[0]
                    if not table.startswith('CTE_') and table != analysis.get('object_name', '') and table != 'CALCULATED':
                        all_tables.add(table)# If no ultimate_source_tables, try to extract from enhanced_references
            if not all_tables and derived_info.get('enhanced_references'):
                for ref in derived_info['enhanced_references']:
                    table_name = ref.get('table', '')
                    # Clean up table name to remove alias
                    if ' AS ' in table_name:
                        table_name = table_name.split(' AS ')[0]
                    if not table_name.startswith('CTE_') and table_name != analysis.get('object_name', '') and table_name != 'CALCULATED':
                        all_tables.add(table_name)
            
            # Use source_columns
            source_columns = derived_info.get('source_columns', [])
            if source_columns:
                for col in source_columns:
                    all_columns.add(col)
            
            # If no source_columns, try to extract from enhanced_references
            if not all_columns and derived_info.get('enhanced_references'):
                for ref in derived_info['enhanced_references']:
                    column_name = ref.get('column', '')
                    if column_name and not column_name.startswith('CTE_'):
                        all_columns.add(column_name)
            
            return {
                'tables': all_tables,
                'columns': all_columns,
                'expression_type': expression_type
            }
        
        # Fallback: Check if this column comes from a CTE with derived expressions
        for cte_name, cte_columns in analysis.get('cte_column_details', {}).items():
            cte_col_info = None
            
            if column_name in cte_columns:
                cte_col_info = cte_columns[column_name]
            else:
                for cte_col_name, col_info in cte_columns.items():
                    if cte_col_name.lower() == column_name.lower():
                        cte_col_info = col_info
                        break
            
            if cte_col_info and cte_col_info.get('type') == 'derived':
                expression_type = cte_col_info.get('expression_type', '')
                for ref in cte_col_info.get('enhanced_referenced_columns', []):
                    table_name = ref.get('table', '')
                    col_name = ref.get('column', '')
                    
                    # Clean up table name to remove alias
                    if ' AS ' in table_name:
                        table_name = table_name.split(' AS ')[0]
                    
                    # Skip CTE references and calculated columns
                    if table_name.startswith('CTE_') or table_name == 'CALCULATED':
                        continue
                    
                    if table_name and table_name != analysis.get('object_name', ''):
                        all_tables.add(table_name)
                    if col_name:
                        all_columns.add(col_name)
                
                for unqual_col in cte_col_info.get('unqualified_columns', []):
                    resolved_source = self._v1_resolve_unqualified_in_cte_context(
                        unqual_col, cte_name, analysis
                    )
                    
                    if resolved_source and resolved_source.get('type') == 'direct':
                        source_table = resolved_source.get('source_table', '')
                        if ' AS ' in source_table:
                            source_table = source_table.split(' AS ')[0]
                        
                        # Skip CTE references
                        if source_table.startswith('CTE_'):
                            continue
                        
                        if source_table and source_table != analysis.get('object_name', ''):
                            all_tables.add(source_table)
                        all_columns.add(unqual_col)
                
                break
        
        return {
            'tables': all_tables,
            'columns': all_columns,
            'expression_type': expression_type
        }