#!/usr/bin/env python3
"""
CTE Analysis Module
Handles all CTE-related analysis including column tracing and resolution
"""

import sqlglot
from sqlglot import exp
import re
from config import AGGREGATION_PATTERNS


class CTEAnalyzer:
    """Handles CTE extraction, analysis, and column resolution"""
    
    def __init__(self, dialect="snowflake"):
        self.dialect = dialect
    
    def extract_ctes(self, parsed, analysis):
        """Extract Common Table Expressions"""
        for node in parsed.walk():
            if isinstance(node, exp.CTE):
                cte_name = str(node.alias)
                analysis['cte_definitions'][cte_name] = {
                    'name': cte_name,
                    'definition': str(node.this)
                }
                analysis['table_aliases'][cte_name.lower()] = f"CTE_{cte_name}"
    
    def analyze_cte_columns_detailed(self, parsed, analysis):
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
        
        self._resolve_unqualified_columns_in_ctes(analysis)
    
    def _resolve_unqualified_columns_in_ctes(self, analysis):
        """Resolve unqualified column references within CTEs"""
        
        cte_order = self._determine_cte_dependency_order(analysis)
        
        for cte_name in cte_order:
            if cte_name not in analysis['cte_column_details']:
                continue
                
            cte_columns = analysis['cte_column_details'][cte_name]
            
            for col_name, col_info in cte_columns.items():
                if col_info.get('needs_resolution'):
                    source_col = col_info['source_column']
                    resolved_source = self._resolve_unqualified_in_cte_context(
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
                        resolved_source = self._resolve_unqualified_in_cte_context(
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
    
    def _determine_cte_dependency_order(self, analysis):
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
    
    def _resolve_unqualified_in_cte_context(self, column_name, current_cte, analysis):
        """Enhanced resolve an unqualified column within a CTE context"""
        # First try dynamic parsing of the CTE definition
        dynamic_resolution = self._dynamic_resolve_column_in_cte(column_name, current_cte, analysis)
        if dynamic_resolution:
            return dynamic_resolution
        
        # Fallback to original dependency-based resolution
        cte_order = self._determine_cte_dependency_order(analysis)
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
    
    def _find_column_in_dependent_ctes(self, column_name, current_cte, analysis):
        """Find if a column comes from another CTE that this CTE depends on"""
        
        cte_order = self._determine_cte_dependency_order(analysis)
        
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
    
    def resolve_cte_reference(self, table_name, column_name, analysis):
        """Enhanced CTE reference resolution with dynamic recursive tracing"""
        
        resolved_tables = set()
        resolved_columns = set()
        
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