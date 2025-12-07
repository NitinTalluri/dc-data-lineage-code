#!/usr/bin/env python3
"""
Expression Analysis Module
Handles complex expression parsing and derived column analysis
"""

import sqlglot
from sqlglot import exp
import re
from config import SQL_KEYWORDS, DERIVED_EXPRESSION_TYPES


class ExpressionAnalyzer:
    """Handles complex expression analysis and column extraction"""
    
    def __init__(self, dialect="snowflake"):
        self.dialect = dialect
    
    def analyze_identifier_pattern(self, parsed, analysis):
        """Handle IDENTIFIER() function pattern"""
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
    
    def enhance_derived_column_with_cte_tracing(self, analysis):
        """Enhance derived columns by tracing unqualified column references through CTEs"""
        
        for col_name, derived_info in analysis['derived_columns'].items():
            enhanced_references = derived_info['referenced_columns'].copy()
            ultimate_tables = set()
            
            for ref in derived_info['referenced_columns']:
                table_name = ref['table']
                if ' AS ' in table_name:
                    table_name = table_name.split(' AS ')[0]
                ultimate_tables.add(table_name)
            
            unqualified_columns = derived_info.get('unqualified_columns', [])
            
            for unqual_col in unqualified_columns:
                traced_sources = self._trace_column_through_ctes(unqual_col, analysis)
                
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
            
            analysis['derived_columns'][col_name]['enhanced_references'] = enhanced_references
            analysis['derived_columns'][col_name]['ultimate_source_tables'] = list(ultimate_tables)
    
    def _trace_column_through_ctes(self, column_name, analysis):
        """Trace a column through the CTE hierarchy"""
        traced_sources = []
        
        from cte_analyzer import CTEAnalyzer
        cte_analyzer = CTEAnalyzer(self.dialect)
        cte_order = cte_analyzer._determine_cte_dependency_order(analysis)
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
                
                elif col_info['type'] == 'unqualified_in_cte':
                    for other_cte_name in reversed(cte_order[:-1]):
                        if other_cte_name in analysis['cte_column_details']:
                            other_cte_columns = analysis['cte_column_details'][other_cte_name]
                            if column_name in other_cte_columns:
                                other_col_info = other_cte_columns[column_name]
                                
                                if other_col_info['type'] == 'direct':
                                    traced_sources.append({
                                        'table': other_col_info['source_table'],
                                        'column': other_col_info['source_column'],
                                        'alias': other_col_info.get('table_alias', '')
                                    })
                                    break
                                    
                                elif other_col_info['type'] == 'derived':
                                    for ref in other_col_info.get('referenced_columns', []):
                                        traced_sources.append({
                                            'table': ref['table'],
                                            'column': ref['column'],
                                            'alias': ref.get('alias', '')
                                        })
                                    for ref in other_col_info.get('enhanced_referenced_columns', []):
                                        traced_sources.append({
                                            'table': ref['table'],
                                            'column': ref['column'],
                                            'alias': ref.get('alias', '')
                                        })
                                    break
                
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
    
    def extract_columns_from_expression(self, expression):
        """Extract column references from a complex expression"""
        potential_columns = []
        word_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\b'
        matches = re.findall(word_pattern, expression)
        
        expression_lower = expression.lower()
        dynamic_sql_keywords = set()
        
        common_keywords = ['case', 'when', 'then', 'else', 'end', 'and', 'or', 'not', 'null', 'is', 'between']
        for keyword in common_keywords:
            if keyword in expression_lower:
                dynamic_sql_keywords.add(keyword)
        
        string_literals = re.findall(r"'([^']+)'", expression)
        for literal in string_literals:
            words = re.findall(r'[A-Z]+', literal)
            for word in words:
                dynamic_sql_keywords.add(word.lower())
        
        for match in matches:
            match_lower = match.lower()
            
            if match_lower in dynamic_sql_keywords:
                continue
            
            if (match.startswith("'") or 
                match.isdigit() or 
                len(match) < 2 or
                match_lower in {'current_date', 'current_timestamp'}):
                continue
            
            potential_columns.append(match)
        
        seen = set()
        unique_columns = []
        for col in potential_columns:
            if col not in seen:
                seen.add(col)
                unique_columns.append(col)
        
        return unique_columns
    
    def parse_nvl_coalesce_expression(self, expression, analysis):
        """Parse NVL/COALESCE expressions to find actual source tables"""
        
        ultimate_tables = set()
        source_columns = set()
        primary_source_table = None
        
        # Extract the first argument of nvl/coalesce
        match = re.search(r'(?:nvl|coalesce)\s*\(\s*([^,]+)', expression, re.IGNORECASE)
        if match:
            col_ref = match.group(1).strip()
            
            if '.' in col_ref:
                table_alias, col_name = col_ref.split('.', 1)
                
                if table_alias in analysis.get('cte_column_details', {}):
                    from cte_analyzer import CTEAnalyzer
                    cte_analyzer = CTEAnalyzer(self.dialect)
                    resolved_tables, resolved_columns = cte_analyzer.resolve_cte_reference(
                        table_alias, col_name, analysis
                    )
                    
                    if resolved_tables:
                        ultimate_tables.update(resolved_tables)
                        source_columns.update(resolved_columns)
                        primary_source_table = list(ultimate_tables)[0]
                
                if not ultimate_tables and table_alias.lower() in analysis.get('table_aliases', {}):
                    actual_table = analysis['table_aliases'][table_alias.lower()]
                    
                    if actual_table.startswith('CTE_'):
                        cte_name = actual_table[4:]
                        from cte_analyzer import CTEAnalyzer
                        cte_analyzer = CTEAnalyzer(self.dialect)
                        resolved_tables, resolved_columns = cte_analyzer.resolve_cte_reference(
                            cte_name, col_name, analysis
                        )
                        if resolved_tables:
                            ultimate_tables.update(resolved_tables)
                            source_columns.update(resolved_columns)
                            primary_source_table = list(ultimate_tables)[0]
                    else:
                        ultimate_tables.add(actual_table)
                        source_columns.add(col_name)
                        primary_source_table = actual_table
                
                if not ultimate_tables:
                    for cte_name in analysis.get('cte_column_details', {}).keys():
                        if cte_name.lower() == table_alias.lower():
                            from cte_analyzer import CTEAnalyzer
                            cte_analyzer = CTEAnalyzer(self.dialect)
                            resolved_tables, resolved_columns = cte_analyzer.resolve_cte_reference(
                                cte_name, col_name, analysis
                            )
                            if resolved_tables:
                                ultimate_tables.update(resolved_tables)
                                source_columns.update(resolved_columns)
                                primary_source_table = list(ultimate_tables)[0]
                                break
            
            else:
                # Unqualified column - try to find in CTEs
                from column_resolver import ColumnResolver
                resolver = ColumnResolver(self.dialect)
                main_cte = resolver._find_main_cte_for_view(analysis)
                if main_cte:
                    from cte_analyzer import CTEAnalyzer
                    cte_analyzer = CTEAnalyzer(self.dialect)
                    resolved_tables, resolved_columns = cte_analyzer.resolve_cte_reference(
                        main_cte, col_ref, analysis
                    )
                    if resolved_tables:
                        ultimate_tables.update(resolved_tables)
                        source_columns.update(resolved_columns)
                        primary_source_table = list(ultimate_tables)[0]
        
        return ultimate_tables, source_columns, primary_source_table
    
    def extract_window_function_columns(self, window_expr, ultimate_tables, source_columns, referenced_columns, analysis):
        """Extract column references from window function PARTITION BY and ORDER BY clauses"""
        
        for node in window_expr.walk():
            if isinstance(node, exp.Column):
                table_ref = node.table
                col_name = str(node.name)
                
                if table_ref:
                    table_ref_str = str(table_ref)
                    
                    if table_ref_str in analysis.get('cte_column_details', {}):
                        from cte_analyzer import CTEAnalyzer
                        cte_analyzer = CTEAnalyzer(self.dialect)
                        resolved_tables, resolved_columns = cte_analyzer.resolve_cte_reference(
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
                        from cte_analyzer import CTEAnalyzer
                        cte_analyzer = CTEAnalyzer(self.dialect)
                        resolved_tables, resolved_columns = cte_analyzer.resolve_cte_reference(
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
    
    def extract_window_columns_regex(self, expression, ultimate_tables, source_columns, referenced_columns, analysis):
        """Extract columns from window function using regex as fallback"""
        
        # Extract columns from PARTITION BY clause
        partition_match = re.search(r'PARTITION\s+BY\s+([^)]+?)(?:\s+ORDER|\))', expression, re.IGNORECASE)
        if partition_match:
            partition_cols = partition_match.group(1)
            
            for col in partition_cols.split(','):
                col = col.strip().lower()
                if col:
                    source_columns.add(col)
        
        # Extract columns from ORDER BY clause
        order_match = re.search(r'ORDER\s+BY\s+([^)]+)', expression, re.IGNORECASE)
        if order_match:
            order_cols = order_match.group(1)
            
            for col in order_cols.split(','):
                col = re.sub(r'\s+(DESC|ASC|NULLS\s+(FIRST|LAST)).*$', '', col.strip(), flags=re.IGNORECASE)
                col = col.strip().lower()
                if col:
                    source_columns.add(col)
        
        # Find the table for these columns
        if source_columns:
            for alias, table_name in analysis.get('table_aliases', {}).items():
                if not table_name.startswith('CTE_') and table_name != analysis.get('object_name', ''):
                    clean_table = table_name
                    if ' AS ' in clean_table:
                        clean_table = clean_table.split(' AS ')[0]
                    
                    ultimate_tables.add(clean_table)
                    
                    for col in source_columns:
                        referenced_columns.append({
                            'table': clean_table,
                            'column': col,
                            'alias': '',
                            'context': 'window_function_regex'
                        })
                    
                    break
    
    def enhance_basic_derived_column(self, column_name, derived_info, analysis):
        """Enhance a basic derived column with proper source table resolution"""
        expression = derived_info.get('expression', '')
        expression_type = derived_info.get('expression_type', '')
        
        ultimate_tables = set()
        source_columns = set()
        referenced_columns = []
        
        if expression_type == 'Window':
            try:
                import sqlglot
                parsed_expr = sqlglot.parse_one(f"SELECT {expression}", dialect="snowflake")
                for node in parsed_expr.walk():
                    if isinstance(node, sqlglot.exp.Column):
                        col_name = str(node.name).lower()
                        
                        found_table = None
                        
                        for alias, table_name in analysis.get('table_aliases', {}).items():
                            if not table_name.startswith('CTE_') and table_name != analysis.get('object_name', ''):
                                clean_table = table_name
                                if ' AS ' in clean_table:
                                    clean_table = clean_table.split(' AS ')[0]
                                found_table = clean_table
                                break
                        
                        if not found_table:
                            for col_mapping in analysis.get('column_mappings', {}).values():
                                if col_mapping.get('type') == 'direct':
                                    source_table = col_mapping.get('source_table', '')
                                    if source_table and source_table != 'UNKNOWN':
                                        found_table = source_table
                                        break
                        
                        if found_table:
                            ultimate_tables.add(found_table)
                            source_columns.add(col_name)
                            
                            referenced_columns.append({
                                'table': found_table,
                                'column': col_name,
                                'alias': '',
                                'context': 'window_function_enhanced'
                            })
                
                if not source_columns:
                    self.extract_window_columns_regex(expression, ultimate_tables, source_columns, referenced_columns, analysis)
            
            except Exception:
                self.extract_window_columns_regex(expression, ultimate_tables, source_columns, referenced_columns, analysis)
        
        else:
            try:
                import sqlglot
                parsed_expr = sqlglot.parse_one(f"SELECT {expression}", dialect="snowflake")
                
                for node in parsed_expr.walk():
                    if isinstance(node, sqlglot.exp.Column):
                        col_name = str(node.name)
                        table_ref = node.table
                        
                        if table_ref:
                            table_ref_str = str(table_ref)
                            actual_table = analysis['table_aliases'].get(table_ref_str.lower(), table_ref_str)
                            if ' AS ' in actual_table:
                                actual_table = actual_table.split(' AS ')[0]
                            
                            ultimate_tables.add(actual_table)
                            source_columns.add(col_name)
                            
                            referenced_columns.append({
                                'table': actual_table,
                                'column': col_name,
                                'alias': table_ref_str
                            })
            
            except Exception:
                ultimate_tables.add('CALCULATED')
        
        # Update the derived column with enhanced information
        if ultimate_tables:
            primary_table = list(ultimate_tables)[0]
            derived_info['ultimate_source_tables'] = list(ultimate_tables)
            derived_info['primary_source_table'] = primary_table
            derived_info['source_columns'] = list(source_columns)
            derived_info['enhanced_references'] = referenced_columns
        else:
            derived_info['ultimate_source_tables'] = ['CALCULATED']
            derived_info['primary_source_table'] = 'CALCULATED'
            derived_info['source_columns'] = []
            derived_info['enhanced_references'] = []