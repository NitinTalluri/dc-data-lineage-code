#!/usr/bin/env python3
"""
Main Integrated Parser Module
Combines all analysis approaches for comprehensive DDL analysis
"""

import sqlglot
from sqlglot import exp
from ddl_structure_analyzer import DDLStructureAnalyzer
from cte_analyzer import CTEAnalyzer
from column_resolver import ColumnResolver
from expression_analyzer import ExpressionAnalyzer
from csv_generator import CSVGenerator


class IntegratedDDLParser:
    """Integrated parser that handles all DDL patterns with accuracy"""
    
    def __init__(self, dialect="snowflake"):
        self.dialect = dialect
        self.structure_analyzer = DDLStructureAnalyzer(dialect)
        self.cte_analyzer = CTEAnalyzer(dialect)
        self.column_resolver = ColumnResolver(dialect)
        self.expression_analyzer = ExpressionAnalyzer(dialect)
        self.csv_generator = CSVGenerator()
        
    def analyze_ddl_statement(self, sql_text):
        """Analyze any DDL statement with improved pattern detection and no duplicates"""
        try:
            parsed = sqlglot.parse_one(sql_text, dialect=self.dialect)
            
            # Determine DDL type and structure
            ddl_info = self.structure_analyzer.analyze_ddl_structure(parsed)
            
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
            sql_pattern = self.structure_analyzer.detect_sql_pattern(parsed, analysis)
            
            if sql_pattern == 'identifier_function':
                return self.expression_analyzer.analyze_identifier_pattern(parsed, analysis)
            elif sql_pattern == 'wildcard_dominant':
                return self._analyze_using_v2_approach(parsed, analysis)
            elif sql_pattern == 'nested_cte_dominant':
                return self._analyze_using_v1_approach(parsed, analysis)
            else:
                # Hybrid approach
                return self._analyze_using_hybrid_approach(parsed, analysis)
                
        except Exception as e:
            return {'error': f"Analysis failed: {str(e)}"}
    
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
        })
        
        # Merge column mappings with priority logic
        missing_columns = []
        for col in analysis['object_columns']:
            v1_has_direct = col in v1_result.get('column_mappings', {})
            v1_has_derived = col in v1_result.get('derived_columns', {})
            v2_has_direct = col in v2_result.get('column_mappings', {})
            v2_has_derived = col in v2_result.get('derived_columns', {})
            
            v1_direct_good = v1_has_direct and v1_result['column_mappings'][col].get('type') != 'unknown'
            v2_direct_good = v2_has_direct and v2_result['column_mappings'][col].get('type') != 'unknown'
            
            column_assigned = False
            if v1_has_derived:
                merged_analysis['derived_columns'][col] = v1_result['derived_columns'][col]
                column_assigned = True
            elif v1_direct_good:
                merged_analysis['column_mappings'][col] = v1_result['column_mappings'][col]
                column_assigned = True
            elif v2_direct_good:
                merged_analysis['column_mappings'][col] = v2_result['column_mappings'][col]
                column_assigned = True
            elif v2_has_derived:
                merged_analysis['derived_columns'][col] = v2_result['derived_columns'][col]
                column_assigned = True
            
            if not column_assigned:
                missing_columns.append(col)
        
        self.column_resolver.resolve_missing_columns_from_main_select(parsed, missing_columns, merged_analysis)
        
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
            self.column_resolver.resolve_missing_columns_from_main_select(parsed, analysis['object_columns'], merged_analysis)
        elif missing_columns and len(merged_analysis.get('cte_definitions', {})) > 0:
            self._resolve_cte_wildcard_columns(missing_columns, merged_analysis)
        
        return merged_analysis
    
    def _analyze_using_v1_approach(self, parsed, analysis):
        """V1 approach - excellent for nested CTEs and derived columns"""
        
        # Extract basic structure
        self.column_resolver.extract_tables_and_aliases(parsed, analysis)
        self.cte_analyzer.extract_ctes(parsed, analysis)
        self.column_resolver.analyze_column_lineage(parsed, analysis)
        self.cte_analyzer.analyze_cte_columns_detailed(parsed, analysis)
        self.expression_analyzer.enhance_derived_column_with_cte_tracing(analysis)
        self._resolve_view_columns_through_ctes(analysis)
        self._resolve_main_select_derived_columns(parsed, analysis)
        
        return analysis
    
    def _analyze_using_v2_approach(self, parsed, analysis):
        """V2 approach - excellent for wildcards and simple direct mappings"""
        self.column_resolver.extract_tables_and_aliases(parsed, analysis)
        self.cte_analyzer.extract_ctes(parsed, analysis)
        
        main_select = self._find_main_select(parsed)
        if not main_select:
            return analysis
        
        main_select_analysis = self._analyze_select_expressions(main_select)
        table_registry = self._build_table_registry(parsed)
        cte_registry = self._build_cte_registry(parsed, table_registry)
        
        analysis['main_select_analysis'] = main_select_analysis
        analysis['table_registry'] = table_registry
        analysis['cte_registry_full'] = cte_registry
        
        for obj_col in analysis['object_columns']:
            source_info = self._trace_column_source(
                obj_col, main_select_analysis, cte_registry, table_registry
            )
            
            if source_info['type'] == 'derived':
                enhanced_source_info = self._enhance_derived_column(source_info, analysis)
                analysis['derived_columns'][obj_col] = enhanced_source_info
            else:
                analysis['column_mappings'][obj_col] = source_info
        
        return analysis
    
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
    
    def _map_column_from_cte_info(self, view_col, cte_col_info, analysis):
        """Map a view column based on CTE column information"""
        
        if cte_col_info.get('type') == 'direct':
            source_table = cte_col_info.get('source_table', '')
            source_column = cte_col_info.get('source_column', view_col)
            table_alias = cte_col_info.get('table_alias', '')
            
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
            if view_col in analysis.get('column_mappings', {}):
                del analysis['column_mappings'][view_col]
            
            ultimate_tables = set()
            source_columns = set()
            referenced_columns = []
            
            for ref in cte_col_info.get('referenced_columns', []):
                table_name = ref.get('table', '')
                col_name = ref.get('column', '')
                
                if ' AS ' in table_name:
                    table_name = table_name.split(' AS ')[0]
                if table_name.startswith('CTE_'):
                    table_name = table_name[4:]
                
                if table_name:
                    ultimate_tables.add(table_name)
                if col_name:
                    source_columns.add(col_name)
                
                referenced_columns.append({
                    'table': table_name,
                    'column': col_name,
                    'alias': ref.get('alias', '')
                })
            
            for ref in cte_col_info.get('enhanced_referenced_columns', []):
                table_name = ref.get('table', '')
                col_name = ref.get('column', '')
                
                if ' AS ' in table_name:
                    table_name = table_name.split(' AS ')[0]
                if table_name.startswith('CTE_'):
                    table_name = table_name[4:]
                
                if table_name:
                    ultimate_tables.add(table_name)
                if col_name:
                    source_columns.add(col_name)
                
                referenced_columns.append({
                    'table': table_name,
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
    
    def _resolve_view_columns_through_ctes(self, analysis):
        """Resolve view columns by mapping them to CTE columns"""
        
        main_cte = self._find_main_cte_for_view(analysis)
        
        if not main_cte or main_cte not in analysis['cte_column_details']:
            return
        
        main_cte_columns = analysis['cte_column_details'][main_cte]
        
        for view_col in analysis['object_columns']:
            if view_col in analysis['column_mappings'] or view_col in analysis['derived_columns']:
                continue
            
            if view_col in main_cte_columns:
                cte_col_info = main_cte_columns[view_col]
                self._map_view_column_from_cte(view_col, cte_col_info, analysis)
                continue
            
            for cte_col_name, cte_col_info in main_cte_columns.items():
                if cte_col_name.lower() == view_col.lower():
                    self._map_view_column_from_cte(view_col, cte_col_info, analysis)
                    break
    
    def _find_main_cte_for_view(self, analysis):
        """Find the main CTE that the view's main SELECT uses"""
        cte_order = self.cte_analyzer._determine_cte_dependency_order(analysis)
        if cte_order:
            return cte_order[-1]
        return None
    
    def _map_view_column_from_cte(self, view_col, cte_col_info, analysis):
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
    
    def _resolve_main_select_derived_columns(self, parsed, analysis):
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
                                resolved_tables, resolved_columns = self.cte_analyzer.resolve_cte_reference(
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
                            self.column_resolver._map_direct_column(column_alias, source_expr, analysis)
                    
                    else:
                        self._map_enhanced_derived_column(column_alias, source_expr, analysis)
            
            elif isinstance(expr, exp.Column):
                column_name = str(expr.name)
                if column_name in analysis['object_columns'] and column_name not in analysis['column_mappings']:
                    self.column_resolver._map_direct_column(column_name, expr, analysis)
    
    def _map_enhanced_derived_column(self, column_name, expression, analysis):
        """Enhanced derived column mapping with better CTE resolution"""
        
        referenced_columns = []
        unqualified_columns = []
        ultimate_tables = set()
        source_columns = set()
        
        expression_type = type(expression).__name__
        
        if expression_type == 'Window':
            self.expression_analyzer.extract_window_function_columns(expression, ultimate_tables, source_columns, referenced_columns, analysis)
        else:
            for node in expression.walk():
                if isinstance(node, exp.Column):
                    table_ref = node.table
                    col_name = str(node.name)
                    
                    if table_ref:
                        table_ref_str = str(table_ref)
                        
                        if table_ref_str in analysis.get('cte_column_details', {}):
                            resolved_tables, resolved_cols = self.cte_analyzer.resolve_cte_reference(
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
            
            main_cte = self._find_main_cte_for_view(analysis)
            if main_cte:
                resolved_tables, resolved_cols = self.cte_analyzer.resolve_cte_reference(
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
                        resolved_tables, resolved_cols = self.cte_analyzer.resolve_cte_reference(
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
    
    # V2 Methods
    def _find_main_select(self, parsed):
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
    
    def _analyze_select_expressions(self, select_stmt):
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
    
    def _build_table_registry(self, parsed):
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
    
    def _build_cte_registry(self, parsed, table_registry):
        """Build registry of all CTEs and their column mappings"""
        registry = {}
        
        for node in parsed.walk():
            if isinstance(node, exp.CTE):
                cte_name = str(node.alias)
                cte_select = node.this
                
                cte_analysis = self._analyze_select_expressions(cte_select)
                column_mapping = {}
                
                for expr_info in cte_analysis['explicit_expressions']:
                    if 'alias' in expr_info:
                        alias = expr_info['alias']
                        if expr_info['type'] == 'Column':
                            column_mapping[alias] = self._parse_column_reference(
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
    
    def _parse_column_reference(self, col_expr_str, table_registry):
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
    
    def _trace_column_source(self, view_col, main_select_analysis, cte_registry, table_registry):
        """Trace where a view column comes from"""
        for expr_info in main_select_analysis['explicit_expressions']:
            if 'alias' in expr_info and expr_info['alias'].lower() == view_col.lower():
                if expr_info['type'] == 'Column':
                    return self._parse_column_reference(expr_info['expression'], table_registry)
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
    
    def _enhance_derived_column(self, derived_info, analysis):
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
                    
                    resolved_tables, resolved_columns = self.column_resolver._resolve_derived_column_source(
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
    
    def generate_csv(self, analysis):
        """Generate CSV output"""
        return self.csv_generator.generate_standard_csv(analysis)