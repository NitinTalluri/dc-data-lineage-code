#!/usr/bin/env python3
"""
Column Resolution Module
Handles column mapping and derived column analysis
"""

import sqlglot
from sqlglot import exp
import re
from config import SQL_KEYWORDS, DERIVED_EXPRESSION_TYPES


class ColumnResolver:
    """Handles column resolution and mapping"""
    
    def __init__(self, dialect="snowflake"):
        self.dialect = dialect
    
    def extract_tables_and_aliases(self, parsed, analysis):
        """Extract all table references and their aliases"""
        view_name = analysis.get('object_name', '')
        
        for node in parsed.walk():
            if isinstance(node, exp.Table):
                table_name = str(node)
                
                if table_name == view_name:
                    continue
                    
                analysis['source_tables'].append(table_name)
                
                if node.alias:
                    alias = str(node.alias)
                    analysis['table_aliases'][alias.lower()] = table_name
                else:
                    implicit_alias = table_name.split('.')[-1].lower()
                    if table_name != view_name:
                        analysis['table_aliases'][implicit_alias] = table_name
    
    def analyze_column_lineage(self, parsed, analysis):
        """Analyze column lineage"""
        for node in parsed.walk():
            if isinstance(node, exp.Select):
                self._analyze_select_statement(node, analysis)
    
    def _analyze_select_statement(self, select_node, analysis):
        """Analyze a single SELECT statement"""
        for expr in select_node.expressions:
            if isinstance(expr, exp.Alias):
                column_alias = str(expr.alias)
                source_expr = expr.this
                
                if isinstance(source_expr, exp.Column):
                    self._map_direct_column(column_alias, source_expr, analysis)
                else:
                    self._map_derived_column(column_alias, source_expr, analysis)
            
            elif isinstance(expr, exp.Column):
                column_name = str(expr.name)
                self._map_direct_column(column_name, expr, analysis)
    
    def _map_direct_column(self, column_name, column_expr, analysis):
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
    
    def _map_derived_column(self, column_name, expression, analysis):
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
    
    def resolve_missing_columns_from_main_select(self, parsed, missing_columns, analysis):
        """Targeted resolution of missing columns from main SELECT statement"""
        
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
                column_alias = str(expr.alias).upper()
                source_expr = expr.this
                
                missing_col_match = None
                for missing_col in missing_columns:
                    if missing_col.upper() == column_alias:
                        missing_col_match = missing_col
                        break
                
                if missing_col_match:
                    if isinstance(source_expr, exp.Column):
                        table_ref = source_expr.table
                        col_name = str(source_expr.name)
                        
                        if table_ref:
                            table_ref_str = str(table_ref)
                            actual_table = analysis['table_aliases'].get(table_ref_str.lower(), table_ref_str)
                            
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
                        referenced_tables = set()
                        referenced_columns = set()
                        
                        for node in source_expr.walk():
                            if isinstance(node, exp.Column):
                                node_table_ref = node.table
                                node_col_name = str(node.name)
                                
                                if node_table_ref:
                                    node_table_ref_str = str(node_table_ref)
                                    actual_table = analysis['table_aliases'].get(node_table_ref_str.lower(), node_table_ref_str)
                                    if ' AS ' in actual_table:
                                        actual_table = actual_table.split(' AS ')[0]
                                    referenced_tables.add(actual_table)
                                    referenced_columns.add(node_col_name)
                                else:
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
            
            elif isinstance(expr, exp.Column):
                column_name = str(expr.name).upper()
                
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
                        actual_table = analysis['table_aliases'].get(table_ref_str.lower(), table_ref_str)
                        
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
    
    def _resolve_derived_column_source(self, col_name, table_ref, analysis):
        """Resolve a column in a derived expression to its ultimate source tables"""
        
        resolved_tables = set()
        resolved_columns = set()
        
        if table_ref:
            if table_ref in analysis.get('cte_column_details', {}):
                # Import here to avoid circular imports
                from cte_analyzer import CTEAnalyzer
                cte_analyzer = CTEAnalyzer(self.dialect)
                tables, columns = cte_analyzer.resolve_cte_reference(table_ref, col_name, analysis)
                resolved_tables.update(tables)
                resolved_columns.update(columns)
            
            elif table_ref.lower() in analysis.get('table_aliases', {}):
                actual_table = analysis['table_aliases'][table_ref.lower()]
                if actual_table.startswith('CTE_'):
                    cte_name = actual_table[4:]
                    from cte_analyzer import CTEAnalyzer
                    cte_analyzer = CTEAnalyzer(self.dialect)
                    tables, columns = cte_analyzer.resolve_cte_reference(cte_name, col_name, analysis)
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
            main_cte = self._find_main_cte_for_view(analysis)
            if main_cte and main_cte in analysis.get('cte_column_details', {}):
                cte_columns = analysis['cte_column_details'][main_cte]
                
                if col_name in cte_columns:
                    from cte_analyzer import CTEAnalyzer
                    cte_analyzer = CTEAnalyzer(self.dialect)
                    tables, columns = cte_analyzer.resolve_cte_reference(main_cte, col_name, analysis)
                    resolved_tables.update(tables)
                    resolved_columns.update(columns)
                    found_in_cte = True
                
                elif not found_in_cte:
                    for cte_col_name in cte_columns.keys():
                        if cte_col_name.lower() == col_name.lower():
                            from cte_analyzer import CTEAnalyzer
                            cte_analyzer = CTEAnalyzer(self.dialect)
                            tables, columns = cte_analyzer.resolve_cte_reference(main_cte, cte_col_name, analysis)
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
                        from cte_analyzer import CTEAnalyzer
                        cte_analyzer = CTEAnalyzer(self.dialect)
                        tables, columns = cte_analyzer.resolve_cte_reference(cte_name, col_name, analysis)
                        resolved_tables.update(tables)
                        resolved_columns.update(columns)
                        found_in_cte = True
                        break
                    
                    for cte_col_name in cte_columns.keys():
                        if cte_col_name.lower() == col_name.lower():
                            from cte_analyzer import CTEAnalyzer
                            cte_analyzer = CTEAnalyzer(self.dialect)
                            tables, columns = cte_analyzer.resolve_cte_reference(cte_name, cte_col_name, analysis)
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
    
    def _find_main_cte_for_view(self, analysis):
        """Find the main CTE that the view's main SELECT uses"""
        from cte_analyzer import CTEAnalyzer
        cte_analyzer = CTEAnalyzer(self.dialect)
        cte_order = cte_analyzer._determine_cte_dependency_order(analysis)
        if cte_order:
            return cte_order[-1]
        return None