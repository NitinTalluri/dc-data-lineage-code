#!/usr/bin/env python3
"""
CSV Generation Module
Handles CSV output generation in standard format
"""

import csv
import io
from config import CSV_HEADERS, DERIVED_EXPRESSION_TYPES


class CSVGenerator:
    """Handles CSV generation from analysis results"""
    
    def __init__(self):
        pass
    
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
                expression_type = all_sources.get('expression_type', '')
                
                if (len(all_sources['tables']) > 1 or 
                    len(all_sources['columns']) > 1 or 
                    expression_type in DERIVED_EXPRESSION_TYPES):
                    is_derived = True
                
                if is_derived:
                    if all_sources['tables']:
                        source_table_display = '; '.join(sorted(all_sources['tables']))
                    else:
                        source_table = mapping['source_table']
                        if ' AS ' in source_table:
                            source_table = source_table.split(' AS ')[0]
                        if source_table.startswith('CTE_'):
                            source_table = source_table[4:]
                        source_table_display = source_table
                    
                    if all_sources['columns']:
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
                continue
            
            if obj_col in analysis['derived_columns']:
                derived = analysis['derived_columns'][obj_col]
                
                if not derived.get('ultimate_source_tables') and not derived.get('enhanced_references'):
                    self._enhance_basic_derived_column(obj_col, derived, analysis)
                
                derived_expression = derived.get('expression', '')
                if ('nvl(' in derived_expression.lower() or 'coalesce(' in derived_expression.lower()) and not derived.get('enhanced_references'):
                    from expression_analyzer import ExpressionAnalyzer
                    expr_analyzer = ExpressionAnalyzer()
                    ultimate_tables, source_columns, primary_source_table = expr_analyzer.parse_nvl_coalesce_expression(
                        derived_expression, analysis
                    )
                else:
                    ultimate_tables = set()
                    source_columns = set()
                    
                    if 'ultimate_source_tables' in derived and derived['ultimate_source_tables']:
                        ultimate_tables = set(derived['ultimate_source_tables'])
                        
                        source_columns = set()
                        for ref in derived.get('enhanced_references', []):
                            source_columns.add(ref.get('column', ''))
                        
                        final_tables = set()
                        for table in ultimate_tables:
                            cte_name = None
                            if table.startswith('CTE_'):
                                cte_name = table[4:]
                            elif table in analysis.get('cte_column_details', {}):
                                cte_name = table
                            
                            if cte_name and cte_name in analysis.get('cte_column_details', {}):
                                resolved = False
                                for col in source_columns:
                                    from cte_analyzer import CTEAnalyzer
                                    cte_analyzer = CTEAnalyzer()
                                    resolved_tables, resolved_columns = cte_analyzer.resolve_cte_reference(
                                        cte_name, col, analysis
                                    )
                                    if resolved_tables:
                                        final_tables.update(resolved_tables)
                                        resolved = True
                                        break
                                
                                if not resolved:
                                    cte_cols = analysis['cte_column_details'][cte_name]
                                    for col_name, col_info in cte_cols.items():
                                        if col_info.get('type') == 'direct':
                                            source_table = col_info.get('source_table', '')
                                            if ' AS ' in source_table:
                                                source_table = source_table.split(' AS ')[0]
                                            if source_table and not source_table.startswith('CTE_'):
                                                final_tables.add(source_table)
                                                break
                            else:
                                final_tables.add(table)
                        
                        ultimate_tables = final_tables if final_tables else ultimate_tables
                
                # Get all source information
                all_source_tables = set()
                all_source_columns = set()
                
                if derived.get('enhanced_references'):
                    for ref in derived['enhanced_references']:
                        table_name = ref.get('table', '')
                        column_name = ref.get('column', '')
                        
                        if ' AS ' in table_name:
                            table_name = table_name.split(' AS ')[0]
                        if table_name.startswith('CTE_'):
                            table_name = table_name[4:]
                        
                        if table_name and table_name != analysis['object_name'] and table_name != 'CALCULATED':
                            all_source_tables.add(table_name)
                        if column_name:
                            all_source_columns.add(column_name)
                
                elif derived.get('ultimate_source_tables'):
                    for table in derived['ultimate_source_tables']:
                        if ' AS ' in table:
                            table = table.split(' AS ')[0]
                        if table.startswith('CTE_'):
                            table = table[4:]
                        if table and table != analysis['object_name'] and table != 'CALCULATED':
                            all_source_tables.add(table)
                    
                    if derived.get('source_columns'):
                        all_source_columns.update(derived['source_columns'])
                
                elif derived.get('referenced_columns'):
                    for ref in derived['referenced_columns']:
                        table_name = ref.get('table', '')
                        column_name = ref.get('column', '')
                        
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
                    source_table_display = '; '.join(sorted(all_source_tables))
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
                            source_table_display = '; '.join(sorted(all_source_tables))
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
        """Get all contributing sources for a column by tracing through CTEs"""
        all_tables = set()
        all_columns = set()
        expression_type = ''
        
        # Check if this column comes from a CTE with derived expressions
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
                
                for ref in cte_col_info.get('referenced_columns', []):
                    table_name = ref.get('table', '')
                    col_name = ref.get('column', '')
                    
                    if ' AS ' in table_name:
                        table_name = table_name.split(' AS ')[0]
                    if table_name.startswith('CTE_'):
                        table_name = table_name[4:]
                    
                    if table_name and table_name != analysis.get('object_name', ''):
                        all_tables.add(table_name)
                    if col_name:
                        all_columns.add(col_name)
                
                for unqual_col in cte_col_info.get('unqualified_columns', []):
                    from cte_analyzer import CTEAnalyzer
                    cte_analyzer = CTEAnalyzer()
                    resolved_source = cte_analyzer._resolve_unqualified_in_cte_context(
                        unqual_col, cte_name, analysis
                    )
                    
                    if resolved_source and resolved_source.get('type') == 'direct':
                        source_table = resolved_source.get('source_table', '')
                        if ' AS ' in source_table:
                            source_table = source_table.split(' AS ')[0]
                        if source_table.startswith('CTE_'):
                            source_table = source_table[4:]
                        
                        if source_table and source_table != analysis.get('object_name', ''):
                            all_tables.add(source_table)
                        all_columns.add(unqual_col)
                
                break
        
        return {
            'tables': all_tables,
            'columns': all_columns,
            'expression_type': expression_type
        }
    
    def _enhance_basic_derived_column(self, column_name, derived_info, analysis):
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
                    self._extract_window_columns_regex(expression, ultimate_tables, source_columns, referenced_columns, analysis)
            
            except Exception:
                self._extract_window_columns_regex(expression, ultimate_tables, source_columns, referenced_columns, analysis)
        
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
    
    def _extract_window_columns_regex(self, expression, ultimate_tables, source_columns, referenced_columns, analysis):
        """Extract columns from window function using regex as fallback"""
        import re
        
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