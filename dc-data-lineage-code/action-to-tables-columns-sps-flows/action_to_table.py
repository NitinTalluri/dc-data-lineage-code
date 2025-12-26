#!/usr/bin/env python3
"""
Frontend-Backend API Mapping Script with Complete Table Analysis
This script analyzes the frontend TypeScript API files and backend Python route files
to create mapping between frontend API calls and backend endpoints, including the
database tables accessed by each endpoint.

Usage:
uv run python action_to_table.py
uv run python action_to_table.py --output mapping.json
uv run python action_to_table.py --format csv --output mapping.csv
"""
import re
import json
import argparse
import ast
import os
import glob
import builtins
import logging
from pathlib import Path
from typing import List, Optional, Dict, Set, Any
from dataclasses import dataclass
from collections import defaultdict
import functools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('action_to_table.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FrontendCall:
    file: str
    function_name: str
    method: str
    url_pattern: str
    line_number: int
    raw_code: str
    
@dataclass
class BackendRoute:
    file: str
    method: str
    route_pattern: str
    line_number: int
    function_name: str
    tags: List[str]
    raw_code: str
    tables: List[str]
    stored_procedures: List[str] = None  
    flow_calls: List[str] = None  
    response_model_info: Optional[Dict[str, Any]] = None

@dataclass
class Mapping:
    frontend: FrontendCall
    backend: Optional[BackendRoute]

# Table analysis components from merged_route_analyzer_with_table_data.py
ROUTER_DECORATORS = {
    "get": "GET",
    "post": "POST",
    "put": "PUT",
    "delete": "DELETE",
    "patch": "PATCH",
    "options": "OPTIONS",
    "head": "HEAD",
}

SKIP_FUNCTIONS = {
    "ServiceException", "HTTPException", "SDPLifeCycle", "select", "text",
    "list", "any", "map", "max", "min", "Depends", "setattr"
}

BUILTIN_FUNCTIONS = {
    name for name in dir(builtins)
    if isinstance(getattr(builtins, name), type(abs))
}

SQL_KEYWORDS = {
    # Standard SQL keywords
    "select", "from", "where", "join", "inner", "left", "right", "full", "outer", "on",
    "insert", "into", "values", "update", "set", "delete", "create", "alter", "drop",
    "table", "view", "index", "and", "or", "not", "in", "is", "null", "like", "as",
    "group", "by", "order", "having", "limit", "offset", "union", "distinct", "case",
    "when", "then", "else", "end", "exists", "count", "sum", "avg", "min", "max", "for",
    "if", "with", "primary", "key", "foreign", "references", "constraint",
    # Common CTE/alias/utility words to filter
    "static", "deleted", "the", "super", "temp", "test", "backup", "current", "stg",
    "metrics", "cluster", "clusters", "details", "hdr", "info", "data", "snapshot",
    "report", "object", "json", "parquet", "thoughtspot", "contracts", "booking", "sub",
    "extension", "extensions", "input", "output", "row", "col", "column", "columns",
    "prefect" ,  "tagset_ids", "tag_ids"
}
@functools.cache
def collect_all_possible_table_names(root_dir="."):
    """Enhanced table name collection - DYNAMIC patterns with caching"""
    def _collect_tables_from_file(path):
        table_names = set()
        patterns = [
            re.compile(r'__tablename__\s*=\s*["\']([a-zA-Z0-9_]+)["\']'),
            re.compile(r'Table\s*\(\s*["\']([a-zA-Z0-9_]+)["\']'),
            re.compile(r'["\']([a-zA-Z0-9_]+)["\'],\s*V2Base\.metadata'),
        ]
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            for pattern in patterns:
                table_names.update(pattern.findall(content))
        except Exception:
            pass
        return {name.upper() for name in table_names}

    table_names = set()
    for dirpath, _, files in os.walk(root_dir):
        if '.venv' in dirpath or '__pycache__' in dirpath:
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(dirpath, file)
                table_names.update(_collect_tables_from_file(path))

    return table_names

@functools.cache
def get_table_columns(table_name: str, root_dir=".") -> List[str]:
    """Get column names for a specific table by analyzing ORM files"""
    columns = []
    
    # Search for ORM class with this table name
    for dirpath, _, files in os.walk(root_dir):
        if '.venv' in dirpath or '__pycache__' in dirpath:
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(dirpath, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    # Check if this file contains our table
                    if f'__tablename__ = "{table_name.lower()}"' in content or f"__tablename__ = '{table_name.lower()}'" in content:
                        columns.extend(_extract_columns_from_orm_file(content, table_name))
                        
                except Exception:
                    continue
    
    return sorted(list(set(columns)))

def _extract_columns_from_orm_file(content: str, table_name: str) -> List[str]:
    """Extract column names from ORM file content"""
    columns = []
    
    try:
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if this class has the target table name
                has_target_table = False
                for stmt in node.body:
                    if (isinstance(stmt, ast.Assign) and 
                        any(isinstance(target, ast.Name) and target.id == '__tablename__' for target in stmt.targets)):
                        if isinstance(stmt.value, ast.Constant) and stmt.value.value.upper() == table_name.upper():
                            has_target_table = True
                        elif isinstance(stmt.value, ast.Str) and stmt.value.s.upper() == table_name.upper():
                            has_target_table = True
                
                if has_target_table:
                    # Extract column definitions
                    for stmt in node.body:
                        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                            field_name = stmt.target.id
                            # Skip private fields and relationships
                            if not field_name.startswith('_') and not _is_relationship_field(stmt):
                                columns.append(field_name.upper())
    
    except Exception:
        pass
    
    return columns

def _is_relationship_field(stmt: ast.AnnAssign) -> bool:
    """Check if this is a relationship field (not a database column)"""
    if stmt.value and isinstance(stmt.value, ast.Call):
        if isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == 'relationship':
            return True
    return False

def extract_tables_from_sql(sql_text: str, known_tables: set = None) -> set:
    """Extract table names from SQL text with reduced nesting."""
    tables = set()
    known_tables = known_tables or set()

    cte_patterns = [
        r"with\s+([a-zA-Z0-9_]+)\s+as\s*\(",
        r",\s*([a-zA-Z0-9_]+)\s+as\s*\("
    ]
    cte_names = set(
        name
        for pattern in cte_patterns
        for name in re.findall(pattern, sql_text, flags=re.IGNORECASE)
    )

    table_patterns = [
        r'\bFROM\s+([a-zA-Z0-9_]+)(?:\s|$|\)|,)',
        r'\bJOIN\s+([a-zA-Z0-9_]+)(?:\s|$|\)|,)',
        r'\bUPDATE\s+([a-zA-Z0-9_]+)(?:\s|$|\)|,)',
        r'\bINSERT\s+INTO\s+([a-zA-Z0-9_]+)(?:\s|$|\)|,)',
        r'\bMERGE\s+INTO\s+([a-zA-Z0-9_]+)(?:\s|$|\)|,)',
        r'\bDELETE\s+FROM\s+([a-zA-Z0-9_]+)(?:\s|$|\)|,)',
    ]

    for pattern in table_patterns:
        for match in re.findall(pattern, sql_text, re.IGNORECASE):
            table_name = match.split('.')[-1] if '.' in match else match
            if table_name in known_tables or is_valid_table_candidate(table_name, cte_names):
                tables.add(table_name.upper())

    return tables

def is_valid_table_candidate(name: str, cte_names: set) -> bool:
    """Basic validation for table names."""
    if not name or len(name) < 3:
        return False
    if name in cte_names or name.lower() in SQL_KEYWORDS or name.isdigit():
        return False
    if name.isupper() or '_' in name or (name[0].isupper() and len(name) > 4):
        return True
    return False
class CompleteTableAnalyzer(ast.NodeVisitor):
    """Complete AST analyzer with caching for service methods."""
    
    def __init__(self, model_table_map=None, all_known_tables=None, all_function_definitions=None):
        self.router_prefix = ""
        self.routes_info = []
        self.flow_service_aliases = {"flow_service"}
        self.call_graph = defaultdict(set)
        self.defined_functions = set()
        self.current_function = None
        self.proc_calls = defaultdict(list)
        self.table_calls = defaultdict(set)
        self.current_route_info = None
        
        # Deep analysis capabilities
        self.function_definitions = all_function_definitions or {}
        self.analyzed_functions = set()
        self.service_methods = defaultdict(set)
        self.cross_file_functions = {}
        
        # Enhanced service tracking
        self.service_aliases = {}  # alias -> service_class mapping  
        self.imported_services = {}  # imported service tracking
        self.service_method_cache = {}  # service_class.method -> AST node
        self.service_file_cache = {}  # cache for service file analysis
        
        # NEW: Enhanced table tracking
        self.imported_table_candidates = set()  # potential table imports
        self.model_table_map = model_table_map or self._build_complete_table_map()
        self.all_known_tables = set(all_known_tables) if all_known_tables else set()
        self.service_proc_map = self.build_proc_map() if all_function_definitions else {}
        
        # Collect all service methods at initialization
        self.service_method_cache = self._collect_all_service_methods()
        
    def build_proc_map(self):
        """Build enhanced procedure mapping from function definitions"""
        proc_map = {}
        for func_name, func_node in self.function_definitions.items():
            if any(kw in func_name.lower() for kw in ['service', 'booking', 'sdp', 'assignment', 'rebuild', 'run_']):
                procs = self.extract_procs_from_node(func_node)
                if procs:
                    proc_map[func_name] = procs
                    if '.' in func_name: 
                        proc_map[func_name.split('.')[-1]] = procs
        essential_mappings = {
            'rebuild_sdp_for_booking_contract': ['dc_sdp_contract_changes'],
            'replace_assignment_responsible_user': ['replace_responsible_user'],
            'update_verified_booking_assignments': ['assign_responsible_users'],
            'rebuild_sdp': ['dc_sdp_changes'],
            'run_rebuild_sdp': ['dc_sdp_changes'],
            'process_sea_upload': ['load_sea_data'],
            'process_macd_upload': ['load_macd_data']
        }
        proc_map.update(essential_mappings)
        return proc_map
    
    def extract_procs_from_node(self, func_node):
        """Extract stored procedures from a function node"""
        procs = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                # Check for proc_name keyword arguments
                for kw in getattr(node, 'keywords', []):
                    if kw.arg == 'proc_name':
                        if isinstance(kw.value, ast.Constant): 
                            procs.append(kw.value.value)
                        elif isinstance(kw.value, ast.Str): 
                            procs.append(kw.value.s)
                
                # Check for V2ProcedureNames calls
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "V2ProcedureNames":
                    procs.append(node.func.attr)
                
                # Check for SQL strings with CALL statements
                for arg in getattr(node, 'args', []):
                    sql_text = None
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str): 
                        sql_text = arg.value
                    elif isinstance(arg, ast.Str): 
                        sql_text = arg.s
                    
                    if sql_text:
                        for pattern in [r'CALL\s+([A-Z_][A-Z0-9_]*)\s*\(', 
                                      r'CALL\s+IDENTIFIER\s*\(\s*["\']([A-Z_][A-Z0-9_]*)["\']', 
                                      r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+([A-Z_][A-Z0-9_]*)']:
                            procs.extend(re.findall(pattern, sql_text.upper()))
        return procs

    def _build_complete_table_map(self) -> dict:
        """Enhanced version - keeps all existing functionality + adds new detection"""
        mapping = {}
        patterns = ["api/**/orm/**/*.py", "**/orm/**/*.py", "**/models/**/*.py"]
        for pattern in patterns:
            for file_path in glob.glob(pattern, recursive=True):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 1. Your existing ORM Classes with __tablename__ (keep this!)
                    class_blocks = re.findall(
                        r'class\s+(\w+)[^\n]*:(.*?)(?=^class\s|\Z)',
                        content,
                        re.DOTALL | re.MULTILINE
                    )
                    for class_name, class_body in class_blocks:
                        match = re.search(r'__tablename__\s*=\s*["\']([^"\']+)["\']', class_body)
                        if match:
                            table_name = match.group(1)
                            mapping[class_name] = table_name.upper()
                        
                        # Add __table__ reference detection
                        elif not match:  # Only if __tablename__ not found
                            table_ref_match = re.search(r'__table__\s*=\s*(\w+)', class_body)
                            if table_ref_match:
                                table_var = table_ref_match.group(1)
                                # Look up the table variable in the same file
                                table_def_patterns = [
                                    rf'{re.escape(table_var)}\s*=\s*Table\s*\(\s*["\']([^"\']+)["\']',
                                    rf'{re.escape(table_var)}\s*:\s*Table\s*=\s*Table\s*\(\s*["\']([^"\']+)["\']'
                                ]
                                for pattern_def in table_def_patterns:
                                    table_def_match = re.search(pattern_def, content)
                                    if table_def_match:
                                        actual_table = table_def_match.group(1)
                                        mapping[class_name] = actual_table.upper()
                                        break
                    
                    # 2. Your existing Raw Table Objects (keep this!)
                    table_patterns = [
                        r'(\w+)\s*=\s*Table\s*\(\s*["\']([^"\']+)["\']',  # var = Table("name", ...)
                        r'(\w+)\s*:\s*Table\s*=\s*Table\s*\(\s*["\']([^"\']+)["\']',  # var: Table = Table("name", ...)
                    ]
                    for table_pattern in table_patterns:
                        table_matches = re.findall(table_pattern, content)
                        for var_name, table_name in table_matches:
                            mapping[var_name] = table_name.upper()
                            # Also add common variations
                            if var_name.startswith('v2_'):
                                base_name = var_name[3:]  # Remove 'v2_' prefix
                                mapping[base_name] = table_name.upper()
                    
                except Exception:
                    continue
        
        
        return mapping

    def _resolve_model_to_table_enhanced(self, model_name):
        """Enhanced model resolution - PURELY DYNAMIC without hardcoded mappings"""
        # First try your existing direct mapping
        if model_name in self.model_table_map:
            return self.model_table_map[model_name]
        
        #Try variations of the model name
        variations = [
            model_name,
            model_name.lower(),
            model_name.upper(),
            f"v2_{model_name.lower()}",
            f"V2{model_name}",
        ]
        
        for variation in variations:
            if variation in self.model_table_map:
                # Cache the result for future use
                self.model_table_map[model_name] = self.model_table_map[variation]
                return self.model_table_map[variation]
        
        # Cross-file import resolution 
        if model_name in self.imported_table_candidates:
            resolved_table = self._resolve_imported_table_dynamic(model_name)
            if resolved_table:
                # Cache the result
                self.model_table_map[model_name] = resolved_table
                return resolved_table
        
        return None

    def _resolve_imported_table_dynamic(self, model_name):
        """Dynamically resolve imported table references"""
        # Search for this model in all ORM files
        patterns = ["api/**/orm/**/*.py", "**/orm/**/*.py", "**/models/**/*.py"]
        for pattern in patterns:
            for file_path in glob.glob(pattern, recursive=True):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Look for class definition
                    class_pattern = rf'class\s+{re.escape(model_name)}[^\n]*:(.*?)(?=^class\s|\Z)'
                    class_match = re.search(class_pattern, content, re.DOTALL | re.MULTILINE)
                    if class_match:
                        class_body = class_match.group(1)
                        # Check for __tablename__
                        tablename_match = re.search(r'__tablename__\s*=\s*["\']([^"\']+)["\']', class_body)
                        if tablename_match:
                            return tablename_match.group(1).upper()
                        
                        # Check for __table__ reference
                        table_ref_match = re.search(r'__table__\s*=\s*(\w+)', class_body)
                        if table_ref_match:
                            table_var = table_ref_match.group(1)
                            # Look up the table variable
                            table_def_match = re.search(
                                rf'{re.escape(table_var)}\s*=\s*Table\s*\(\s*["\']([^"\']+)["\']',
                                content
                            )
                            if table_def_match:
                                return table_def_match.group(1).upper()
                    
                    # Look for raw Table object
                    table_pattern = rf'{re.escape(model_name)}\s*=\s*Table\s*\(\s*["\']([^"\']+)["\']'
                    table_match = re.search(table_pattern, content)
                    if table_match:
                        return table_match.group(1).upper()
                
                except Exception:
                    continue
        
        return None

    def _extract_names_from_call(self, call_node):
        """Extract all name references from a call node - add as NEW method"""
        names = []
        
        # Function name
        if isinstance(call_node.func, ast.Name):
            names.append(call_node.func.id)
        elif isinstance(call_node.func, ast.Attribute):
            if isinstance(call_node.func.value, ast.Name):
                names.append(call_node.func.value.id)
        
        # Arguments
        for arg in call_node.args:
            if isinstance(arg, ast.Name):
                names.append(arg.id)
            elif isinstance(arg, ast.Attribute):
                if isinstance(arg.value, ast.Name):
                    names.append(arg.value.id)
        
        # Keyword arguments
        for kw in call_node.keywords:
            if isinstance(kw.value, ast.Name):
                names.append(kw.value.id)
            elif isinstance(kw.value, ast.Attribute):
                if isinstance(kw.value.value, ast.Name):
                    names.append(kw.value.value.id)
        
        return names

    @functools.cache
    def _collect_all_service_methods(self) -> Dict[str, Any]:
        """Collect ALL service methods from entire codebase with caching."""
        service_methods = {}
        service_patterns = [
            "**/services/**/*.py",
            "**/service/**/*.py",
            "**/*service*.py",
            "**/api/**/services/**/*.py",
        ]

        for pattern in service_patterns:
            for file_path in glob.glob(pattern, recursive=True):
                if '.venv' in str(file_path) or '__pycache__' in str(file_path):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        source = f.read()
                    tree = ast.parse(source, filename=file_path)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            class_name = node.name
                            if 'service' in class_name.lower() or class_name.endswith('Service'):
                                for class_node in node.body:
                                    if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                        method_name = class_node.name
                                        keys = [
                                            f"{class_name}.{method_name}",
                                            f"{class_name.lower()}.{method_name}",
                                            method_name,
                                        ]
                                        for key in keys:
                                            service_methods[key] = {
                                                'node': class_node,
                                                'file': str(file_path),
                                                'class': class_name,
                                                'method': method_name
                                            }
                except Exception as e:
                    logger.error(f"Error parsing service file {file_path}: {e}")
                    continue
        
        return service_methods

    def _analyze_any_service_call(self, service_identifier: str, method_name: str) -> set:
        """Analyze ANY service call - the key method"""
        tables = set()
        
        # Try multiple lookup patterns
        lookup_keys = [
            f"{service_identifier}.{method_name}",
            f"{service_identifier.lower()}.{method_name}",
            method_name,
            f"{service_identifier}Service.{method_name}",  # Add Service suffix
            f"{service_identifier.replace('Service', '')}.{method_name}",  # Remove Service suffix
        ]
        
        for key in lookup_keys:
            if key in self.service_method_cache:
                service_info = self.service_method_cache[key]
                method_node = service_info['node']
                
                # Analyze this service method
                service_tables = self._analyze_service_method_node(method_node, service_info)
                tables.update(service_tables)
                break
        
        return tables

    def _analyze_service_method_node(self, method_node, service_info) -> set:
        """Analyze a specific service method AST node"""
        tables = set()
        
        # Save current context
        prev_function = self.current_function
        prev_route_info = self.current_route_info
        
        # Set context to this service method
        self.current_function = f"{service_info['class']}.{service_info['method']}"
        self.current_route_info = None
        
        try:
            # Visit the service method node
            self.generic_visit(method_node)
            
            # Collect tables found in this service method
            tables.update(self.table_calls.get(self.current_function, set()))
            
            # Also recursively analyze any functions this service method calls
            for called_func in self.call_graph.get(self.current_function, []):
                deep_tables = self.analyze_called_function(called_func)
                tables.update(deep_tables)
        
        finally:
            # Restore context
            self.current_function = prev_function
            self.current_route_info = prev_route_info
        
        return tables

    def _pre_analyze_with_body(self, body_nodes: list, service_alias: str, service_class: str):
        """Pre-analyze with statement body for service method calls"""
        for node in ast.walk(ast.Module(body=body_nodes, type_ignores=[])):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value, ast.Name) and
                    node.func.value.id == service_alias):
                    method_name = node.func.attr
                    
                    # Analyze this service method call immediately
                    service_tables = self._analyze_any_service_call(service_class, method_name)
                    
                    # Add to current function's tables
                    for table in service_tables:
                        if table and (table in self.all_known_tables or self.is_likely_real_table(table)):
                            self.table_calls[self.current_function].add(table.upper())
                            if self.current_route_info and table.upper() not in self.current_route_info["tables"]:
                                self.current_route_info["tables"].append(table.upper())

    def _extract_table_references(self, node) -> set:
        """Recursively extract table references from AST nodes - DYNAMIC"""
        tables = set()
        
        if isinstance(node, ast.Attribute):
            # Pattern: table_var.c.column_name
            if node.attr == 'c' and isinstance(node.value, ast.Name):
                table_var = node.value.id
                table = self._resolve_model_to_table_enhanced(table_var)
                if table:
                    tables.add(table.upper())
            # Recursively check the value
            tables.update(self._extract_table_references(node.value))
        
        elif isinstance(node, ast.Name):
            # Direct table variable reference
            table = self._resolve_model_to_table_enhanced(node.id)
            if table:
                tables.add(table.upper())
        
        elif isinstance(node, ast.Call):
            # Check function and arguments
            tables.update(self._extract_table_references(node.func))
            for arg in node.args:
                tables.update(self._extract_table_references(arg))
            for kw in node.keywords:
                tables.update(self._extract_table_references(kw.value))
        
        elif isinstance(node, (ast.List, ast.Tuple)):
            for elt in node.elts:
                tables.update(self._extract_table_references(elt))
        
        elif hasattr(node, '__dict__'):
            # For any other node type, recursively check all attributes
            for attr_name, attr_value in node.__dict__.items():
                if isinstance(attr_value, list):
                    for item in attr_value:
                        if isinstance(item, ast.AST):
                            tables.update(self._extract_table_references(item))
                elif isinstance(attr_value, ast.AST):
                    tables.update(self._extract_table_references(attr_value))
        
        return tables
    def analyze_called_function(self, func_name: str, depth: int = 0) -> set:
        """Recursively analyze a called function to find its table usage"""
        if depth > 5 or func_name in self.analyzed_functions or func_name in SKIP_FUNCTIONS:
            return set()
        
        if func_name in BUILTIN_FUNCTIONS:
            return set()
        
        self.analyzed_functions.add(func_name)
        tables = set()
        
        # Look for function definition
        func_node = self.function_definitions.get(func_name)
        if func_node:
            # Save current context
            prev_function = self.current_function
            prev_route_info = self.current_route_info
            
            self.current_function = func_name
            self.current_route_info = None
            
            # Check for dynamic table patterns in this function
            dynamic_tables = self._extract_dynamic_table_patterns(func_node)
            tables.update(dynamic_tables)
            
            # Analyze the function body
            self.generic_visit(func_node)
            
            # Collect tables found in this function
            tables.update(self.table_calls.get(func_name, set()))
            
            # Recursively analyze called functions
            for called_func in self.call_graph.get(func_name, []):
                deep_tables = self.analyze_called_function(called_func, depth + 1)
                tables.update(deep_tables)
            
            # Restore context
            self.current_function = prev_function
            self.current_route_info = prev_route_info
        
        return tables

    def _extract_dynamic_table_patterns(self, func_node) -> set:
        """Extract table names from dynamic patterns like match statements"""
        tables = set()
        
        for node in ast.walk(func_node):
            # Handle match statements
            if isinstance(node, ast.Match):
                match_tables = self._extract_tables_from_match(node)
                tables.update(match_tables)
            
            # Handle if/elif chains that set table names
            elif isinstance(node, ast.If):
                if_tables = self._extract_tables_from_if_chain(node)
                tables.update(if_tables)
        
        return tables

    def _extract_tables_from_if_chain(self, if_node) -> set:
        """Extract table names from if/elif chains"""
        tables = set()
        
        def extract_from_body(body):
            for stmt in body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if (isinstance(target, ast.Name) and 
                            target.id in ['table_name', 'table'] and
                            isinstance(stmt.value, ast.Constant) and
                            isinstance(stmt.value.value, str)):
                            tables.add(stmt.value.value)
        
        # Check if body
        extract_from_body(if_node.body)
        
        # Check elif/else bodies
        current = if_node
        while current.orelse:
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                # This is an elif
                current = current.orelse[0]
                extract_from_body(current.body)
            else:
                # This is an else
                extract_from_body(current.orelse)
                break
        
        return tables

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and getattr(node.value.func, "id", None) == "APIRouter":
            for kw in node.value.keywords:
                if kw.arg == "prefix":
                    if isinstance(kw.value, ast.Constant):
                        self.router_prefix = kw.value.value
                    elif isinstance(kw.value, ast.Str):
                        self.router_prefix = kw.value.s
        self.generic_visit(node)

    def visit_With(self, node: ast.With):
        """Enhanced context manager detection"""
        for item in node.items:
            # Existing flow_service detection
            if isinstance(item.context_expr, ast.Name) and item.context_expr.id == "flow_service":
                if isinstance(item.optional_vars, ast.Name):
                    self.flow_service_aliases.add(item.optional_vars.id)
            
            # NEW: Service context manager detection
            elif isinstance(item.context_expr, ast.Call):
                if isinstance(item.context_expr.func, ast.Name):
                    service_class = item.context_expr.func.id
                    if (service_class.endswith('Service') or
                        'service' in service_class.lower()):
                        if isinstance(item.optional_vars, ast.Name):
                            service_alias = item.optional_vars.id
                            # Track this service alias
                            self.service_aliases[service_alias] = service_class
                            # Pre-analyze the with body for service calls
                            self._pre_analyze_with_body(node.body, service_alias, service_class)
        
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Enhanced import tracking - DYNAMIC without hardcoded mappings"""
        if node.module:
            # Existing service import tracking
            if 'services' in node.module:
                for alias in node.names:
                    service_name = alias.name
                    if 'service' in service_name.lower() or service_name.endswith('Service'):
                        self.imported_services[alias.asname or service_name] = {
                            'module': node.module,
                            'class': service_name
                        }
            
            # Table import tracking - DYNAMIC
            elif 'orm' in node.module or 'models' in node.module:
                for alias in node.names:
                    imported_name = alias.asname or alias.name
                    # Add to potential table references for later resolution
                    self.imported_table_candidates.add(imported_name)
        
        self.generic_visit(node)

    def visit_Import(self, node):
        """Track direct service imports"""
        for alias in node.names:
            if 'service' in alias.name.lower():
                self.imported_services[alias.asname or alias.name] = {
                    'module': alias.name,
                    'class': alias.name.split('.')[-1]
                }
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Visit attribute access nodes to catch table.c.column patterns"""
        if self.current_function:
            # Pattern: table_var.c (SQLAlchemy Table column access)
            if node.attr == 'c' and isinstance(node.value, ast.Name):
                table_var = node.value.id
                table = self._resolve_model_to_table_enhanced(table_var)
                if table:
                    self.table_calls[self.current_function].add(table.upper())
                    if self.current_route_info and table.upper() not in self.current_route_info["tables"]:
                        self.current_route_info["tables"].append(table.upper())
            
            # Pattern: table_var.column_name (direct column access)
            elif isinstance(node.value, ast.Name):
                table_var = node.value.id
                table = self._resolve_model_to_table_enhanced(table_var)
                if table:
                    self.table_calls[self.current_function].add(table.upper())
                    if self.current_route_info and table.upper() not in self.current_route_info["tables"]:
                        self.current_route_info["tables"].append(table.upper())
        
        self.generic_visit(node)
    def visit_FunctionDef(self, node):
        func_name = node.name
        self.defined_functions.add(func_name)
        self.function_definitions[func_name] = node
        
        route_info = None
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and hasattr(decorator.func, "attr"):
                method_lower = decorator.func.attr.lower()
                if method_lower in ROUTER_DECORATORS:
                    path = "/"
                    if decorator.args:
                        arg0 = decorator.args[0]
                        if isinstance(arg0, ast.Constant):
                            path = arg0.value
                        elif isinstance(arg0, ast.Str):
                                                        path = arg0.s
                    
                    full_path = self.router_prefix + path if self.router_prefix else path
                    route_info = {
                        "method": ROUTER_DECORATORS[method_lower],
                        "path": full_path,
                        "function": func_name,
                        "flow_calls": [],
                        "function_calls": [],
                        "stored_procedures": [],
                        "tables": [],
                        "call_hierarchy": [],
                        "category": "Backend, Frontend"
                    }
                    self.routes_info.append(route_info)
                    break
        
        prev_function = self.current_function
        prev_route_info = self.current_route_info
        self.current_function = func_name
        self.current_route_info = route_info
        
        # Extract dynamic table patterns before visiting
        if self.current_function:
            dynamic_tables = self._extract_dynamic_table_patterns(node)
            for table in dynamic_tables:
                if table and self.is_likely_real_table(table):
                    self.table_calls[self.current_function].add(table.upper())
                    if self.current_route_info and table.upper() not in self.current_route_info["tables"]:
                        self.current_route_info["tables"].append(table.upper())
        
        self.generic_visit(node)
        
        self.current_function = prev_function
        self.current_route_info = prev_route_info

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)
    def visit_Call(self, node):
        if not self.current_function:
            self.generic_visit(node)
            return
        base_call_name = self.get_called_name(node.func)
        if base_call_name:
            if base_call_name in BUILTIN_FUNCTIONS or base_call_name in SKIP_FUNCTIONS:
                self.generic_visit(node)
                return
            
            # Enhanced procedure detection
            self.detect_procs(node)
            
            if isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                base = node.func.value
                if isinstance(base, ast.Name):
                    # Flow service calls
                    if base.id in self.flow_service_aliases:
                        if self.current_route_info:
                            self.current_route_info["flow_calls"].append(attr_name)
                    
                    # V2ProcedureNames calls
                    if base.id == "V2ProcedureNames":
                        proc_name_val = attr_name  # Just the procedure name, not the full path
                        self.proc_calls[self.current_function].append(proc_name_val)
                        if self.current_route_info:
                            self.current_route_info["stored_procedures"].append(proc_name_val)
                    
                    # Background task handling
                    if attr_name == "add_task" and base.id == "background_tasks" and node.args and isinstance(node.args[0], ast.Name):
                        bg_func = node.args[0].id
                        self.call_graph[self.current_function].add(bg_func)
                        if self.current_route_info:
                            self.current_route_info["function_calls"].append(bg_func)
                        
                        # Enhanced background task procedure mapping
                        if bg_func in self.service_proc_map:
                            for proc in self.service_proc_map[bg_func]:
                                self.proc_calls[self.current_function].append(proc)
                                if self.current_route_info:
                                    self.current_route_info["stored_procedures"].append(proc)
                
                # Bindparams handling
                if attr_name == "bindparams" and isinstance(base, ast.Call):
                    inner_func = self.get_called_name(base.func)
                    if inner_func == "make_stored_proc_statement":
                        self.call_graph[self.current_function].add(inner_func)
                        for kw in node.keywords:
                            if kw.arg == "proc_name":
                                if isinstance(kw.value, ast.Constant):
                                    proc_name = kw.value.value
                                    self.proc_calls[self.current_function].append(proc_name)
                                    if self.current_route_info:
                                        self.current_route_info["stored_procedures"].append(proc_name)
            
            elif isinstance(node.func, ast.Name):
                self.call_graph[self.current_function].add(base_call_name)
                if self.current_route_info:
                    self.current_route_info["function_calls"].append(base_call_name)
            
            # Enhanced stored procedure detection
            if base_call_name in [
                "run_stored_procedure",
                "run_v2_stored_procedure",
                "run_put_time_entries_stored_procedure",
                "make_stored_proc_statement"
            ]:
                if base_call_name == "run_put_time_entries_stored_procedure":
                    proc_name_val = "put_user_time_entries"
                    self.proc_calls[self.current_function].append(proc_name_val)
                    if self.current_route_info:
                        self.current_route_info["stored_procedures"].append(proc_name_val)
                
                for kw in node.keywords:
                    if kw.arg == "proc_name":
                        proc_name_val = self.extract_proc_name(kw.value)
                        if proc_name_val:
                            self.proc_calls[self.current_function].append(proc_name_val)
                            if self.current_route_info:
                                self.current_route_info["stored_procedures"].append(proc_name_val)
                
                if base_call_name == "make_stored_proc_statement" and node.args:
                    first_arg = node.args[0] if node.args else None
                    if first_arg:
                        proc_name_val = self.extract_proc_name(first_arg)
                        if proc_name_val:
                            self.proc_calls[self.current_function].append(proc_name_val)
                            if self.current_route_info:
                                 self.current_route_info["stored_procedures"].append(proc_name_val)
                            self.detect_database_tables(node)
        self.generic_visit(node)
    
    def detect_procs(self, node):
        """Enhanced procedure detection from the enhanced version"""
        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            base = node.func.value
            if isinstance(base, ast.Name):
                method_key = f"{base.id}.{attr_name}"
                if method_key in self.service_proc_map or attr_name in self.service_proc_map:
                    procs = self.service_proc_map.get(method_key, self.service_proc_map.get(attr_name, []))
                    for proc in procs:
                        self.proc_calls[self.current_function].append(proc)
                        if self.current_route_info: 
                            self.current_route_info["stored_procedures"].append(proc)
        
        elif isinstance(node.func, ast.Name) and node.func.id in self.service_proc_map:
            for proc in self.service_proc_map[node.func.id]:
                self.proc_calls[self.current_function].append(proc)
                if self.current_route_info: 
                    self.current_route_info["stored_procedures"].append(proc)
        
        # Check for SQL strings with procedure calls
        for arg in getattr(node, 'args', []):
            sql_text = None
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str): 
                sql_text = arg.value
            elif isinstance(arg, ast.Str): 
                sql_text = arg.s
            
            if sql_text and any(kw in sql_text.upper() for kw in ['CALL', 'PROCEDURE']):
                for pattern in [r'CALL\s+([A-Z_][A-Z0-9_]*)\s*\(', 
                              r'CALL\s+IDENTIFIER\s*\(\s*["\']([A-Z_][A-Z0-9_]*)["\']', 
                              r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+([A-Z_][A-Z0-9_]*)']:
                    for proc in re.findall(pattern, sql_text.upper()):
                        self.proc_calls[self.current_function].append(proc)
                        if self.current_route_info: 
                            self.current_route_info["stored_procedures"].append(proc)

    def detect_database_tables(self, node):
        """Enhanced database table detection with comprehensive service tracing"""
        if not self.current_function:
            return

        tables = set()

        if isinstance(node, ast.Call):
            func_name = self.get_called_name(node.func)
            
            # Check if function name maps to a table
            if func_name:
                table = self._resolve_model_to_table_enhanced(func_name)
                if table:
                    tables.add(table.upper())
            
            # ENHANCED: Service method call detection
            if isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                base = node.func.value
                
                # Pattern 1: service.method_name()
                if isinstance(base, ast.Name):
                    base_name = base.id
                    service_tables = self._analyze_any_service_call(base_name, attr_name)
                    tables.update(service_tables)
                
                # Pattern 2: SomeService().method_name()
                elif isinstance(base, ast.Call) and isinstance(base.func, ast.Name):
                    service_class = base.func.id
                    service_tables = self._analyze_any_service_call(service_class, attr_name)
                    tables.update(service_tables)
                
                # NEW: Enhanced attribute access detection
                elif attr_name == 'c' and isinstance(base, ast.Name):
                    table_var = base.id
                    table = self._resolve_model_to_table_enhanced(table_var)
                    if table:
                        tables.add(table.upper())
            
            # Enhanced: Use comprehensive model reference extraction
            for arg in node.args:
                model_tables = self.extract_model_references(arg)
                tables.update(model_tables)
                
                # NEW: Enhanced table reference extraction
                table_refs = self._extract_table_references(arg)
                tables.update(table_refs)
                
                # Also check for SQL strings
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    sql_text = arg.value
                    if self.looks_like_sql(sql_text):
                        sql_tables = extract_tables_from_sql(sql_text, self.all_known_tables)
                        tables.update(sql_tables)
            
            # Check keyword arguments
            for kw in getattr(node, 'keywords', []):
                model_tables = self.extract_model_references(kw.value)
                tables.update(model_tables)
                
                # NEW: Enhanced table reference extraction
                table_refs = self._extract_table_references(kw.value)
                tables.update(table_refs)
                
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    sql_text = kw.value.value
                    if self.looks_like_sql(sql_text):
                        sql_tables = extract_tables_from_sql(sql_text, self.all_known_tables)
                        tables.update(sql_tables)
        
        # Enhanced Name node detection
        elif isinstance(node, ast.Name):
            # Try enhanced resolution
            table = self._resolve_model_to_table_enhanced(node.id)
            if table and self.is_likely_real_table(table):
                tables.add(table.upper())
        
        # Enhanced Attribute access detection  
        elif isinstance(node, ast.Attribute):
            # Pattern: table_var.c.column_name (SQLAlchemy Table column access)
            if node.attr == 'c' and isinstance(node.value, ast.Name):
                table_var = node.value.id
                table = self._resolve_model_to_table_enhanced(table_var)
                if table and self.is_likely_real_table(table):
                    tables.add(table.upper())
            
            # Pattern: model.column_name (direct column access)
            elif isinstance(node.value, ast.Name):
                model_name = node.value.id
                table = self._resolve_model_to_table_enhanced(model_name)
                if table and self.is_likely_real_table(table):
                    tables.add(table.upper())

        # Enhanced Call detection for complex patterns
        if isinstance(node, ast.Call):
            # Extract all names from the call and try to resolve them
            all_names = self._extract_names_from_call(node)
            for name in all_names:
                table = self._resolve_model_to_table_enhanced(name)
                if table and self.is_likely_real_table(table):
                    tables.add(table.upper())

        # Store found tables
        for table in tables:
            if table and (table in self.all_known_tables or self.is_likely_real_table(table)):
                self.table_calls[self.current_function].add(table.upper())
                if self.current_route_info and table.upper() not in self.current_route_info["tables"]:
                    self.current_route_info["tables"].append(table.upper())

    def extract_model_references(self, node) -> set:
        """Recursively extract model class references from any AST node"""
        tables = set()
        
        if isinstance(node, ast.Name):
            table = self._resolve_model_to_table_enhanced(node.id)
            if table:
                tables.add(table.upper())
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                table = self._resolve_model_to_table_enhanced(node.value.id)
                if table:
                    tables.add(table.upper())
            tables.update(self.extract_model_references(node.value))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                table = self._resolve_model_to_table_enhanced(node.func.id)
                if table:
                    tables.add(table.upper())
            for arg in node.args:
                tables.update(self.extract_model_references(arg))
            for kw in node.keywords:
                tables.update(self.extract_model_references(kw.value))
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.elts:
                tables.update(self.extract_model_references(elt))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if key:
                    tables.update(self.extract_model_references(key))
                tables.update(self.extract_model_references(value))
        elif hasattr(node, '__dict__'):
            # For any other node type, recursively check all attributes
            for attr_name, attr_value in node.__dict__.items():
                if isinstance(attr_value, list):
                    for item in attr_value:
                        if isinstance(item, ast.AST):
                            tables.update(self.extract_model_references(item))
                elif isinstance(attr_value, ast.AST):
                    tables.update(self.extract_model_references(attr_value))
        
        return tables

    def looks_like_sql(self, text: str) -> bool:
        """Check if a string looks like SQL"""
        if not text or len(text) < 10:
            return False
        
        text_upper = text.upper()
        sql_keywords = ['SELECT', 'FROM', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'JOIN', 'WHERE', 'CREATE', 'ALTER', 'DROP']
        keyword_count = sum(1 for keyword in sql_keywords if keyword in text_upper)
        
        if keyword_count >= 2:
            return True
        
        # Check for SQL-like patterns
        sql_patterns = [
            r'\bSELECT\s+.*\bFROM\b',
            r'\bINSERT\s+INTO\b',
            r'\bUPDATE\s+.*\bSET\b',
            r'\bDELETE\s+FROM\b',
            r'\bWITH\s+\w+\s+AS\s*\(',
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, text_upper):
                return True
        
        return False

    def is_likely_real_table(self, name: str) -> bool:
        """Check if a name is likely a real table"""
        if not name or len(name) < 3:
            return False
        if name.lower() in SQL_KEYWORDS:
            return False
        if (name.isupper() and ('_' in name or len(name) > 5)) or \
           ('_' in name and any(c.isupper() for c in name)):
            return True
        return False

    def get_called_name(self, node):
        """Get the name of a called function"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            return self.get_called_name(node.func)
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None

    def extract_proc_name(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self.get_full_attribute_name(node)
        return None

    def get_full_attribute_name(self, node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        parts.reverse()
        return ".".join(parts)
    def visit_Constant(self, node):
        """Visit string constants that might contain SQL"""
        if isinstance(node.value, str) and self.current_function:
            sql_text = node.value
            if self.looks_like_sql(sql_text):
                tables = extract_tables_from_sql(sql_text, self.all_known_tables)
                for table in tables:
                    if self.is_likely_real_table(table):
                        self.table_calls[self.current_function].add(table.upper())
                        if self.current_route_info and table.upper() not in self.current_route_info["tables"]:
                            self.current_route_info["tables"].append(table.upper())
                
                # Enhanced procedure detection in SQL strings
                for pattern in [r'CALL\s+([A-Z_][A-Z0-9_]*)\s*\(', 
                              r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+([A-Z_][A-Z0-9_]*)']:
                    for proc in re.findall(pattern, sql_text.upper()):
                        self.proc_calls[self.current_function].append(proc)
                        if self.current_route_info: 
                            self.current_route_info["stored_procedures"].append(proc)
        self.generic_visit(node)
    def visit_Str(self, node):  # For older Python versions
        """Handle string literals in older AST format"""
        if self.current_function:
            sql_text = node.s
            if self.looks_like_sql(sql_text):
                tables = extract_tables_from_sql(sql_text, self.all_known_tables)
                for table in tables:
                    if self.is_likely_real_table(table):
                        self.table_calls[self.current_function].add(table.upper())
                        if self.current_route_info and table.upper() not in self.current_route_info["tables"]:
                            self.current_route_info["tables"].append(table.upper())
                
                # Enhanced procedure detection in SQL strings
                for pattern in [r'CALL\s+([A-Z_][A-Z0-9_]*)\s*\(', 
                              r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+([A-Z_][A-Z0-9_]*)']:
                    for proc in re.findall(pattern, sql_text.upper()):
                        self.proc_calls[self.current_function].append(proc)
                        if self.current_route_info: 
                            self.current_route_info["stored_procedures"].append(proc)
        self.generic_visit(node)

    def visit_Match(self, node):
        """Handle match statements that dynamically determine table names"""
        if self.current_function:
            # Extract table names from match case patterns
            tables = self._extract_tables_from_match(node)
            for table in tables:
                if table and self.is_likely_real_table(table):
                    self.table_calls[self.current_function].add(table.upper())
                    if self.current_route_info and table.upper() not in self.current_route_info["tables"]:
                        self.current_route_info["tables"].append(table.upper())
        self.generic_visit(node)

    def _extract_tables_from_match(self, match_node) -> set:
        """Extract table names from match statement cases"""
        tables = set()
        
        for case in match_node.cases:
            # Look for assignments in case body
            for stmt in case.body:
                if isinstance(stmt, ast.Assign):
                    # Look for table_name = "some_table" patterns
                    for target in stmt.targets:
                        if (isinstance(target, ast.Name) and 
                            target.id in ['table_name', 'table'] and
                            isinstance(stmt.value, ast.Constant) and
                            isinstance(stmt.value.value, str)):
                            table_name = stmt.value.value
                            tables.add(table_name)
                
                # Also check for direct SQL execution with table names
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    # Look for text(f"TRUNCATE TABLE {table_name}") patterns
                    if (isinstance(stmt.value.func, ast.Name) and 
                        stmt.value.func.id == 'text' and
                        stmt.value.args):
                        arg = stmt.value.args[0]
                        if isinstance(arg, ast.JoinedStr):  # f-string
                            # Extract table names from f-string
                            for value in arg.values:
                                if isinstance(value, ast.FormattedValue):
                                    # This is a variable in the f-string
                                    continue
                                elif isinstance(value, ast.Constant):
                                    text_part = value.value
                                    if 'TRUNCATE TABLE' in text_part.upper():
                                        # We know this is truncating a table
                                        # Look back at the case assignments
                                        for case_stmt in case.body:
                                            if (isinstance(case_stmt, ast.Assign) and
                                                any(isinstance(t, ast.Name) and t.id == 'table_name' 
                                                    for t in case_stmt.targets)):
                                                if (isinstance(case_stmt.value, ast.Constant) and
                                                    isinstance(case_stmt.value.value, str)):
                                                    tables.add(case_stmt.value.value)
        
        return tables

class CompleteFrontendAnalyzer:
    """Complete frontend analyzer with all patterns"""
    
    def __init__(self, frontend_path: str):
        self.src_path = Path(frontend_path) / "src"

    def extract_frontend_calls(self) -> List[FrontendCall]:
        calls = []
        if not self.src_path.exists():
            return calls
        
        # Search all .ts and .tsx files
        for ext in ["*.ts", "*.tsx"]:
            for file_path in self.src_path.rglob(ext):
                if file_path.name in ["utils.ts", "vite-env.d.ts"]:
                    continue
                calls.extend(self._parse_file(file_path))
        return calls
    def _parse_file(self, file_path: Path) -> List[FrontendCall]:
        try:
            lines = file_path.read_text(encoding='utf-8').split('\n')
        except:
            return []
        
        calls = []
        current_function = None
        current_function_start = None
        found_lines = set()  
        
        for i, line in enumerate(lines, 1):# Enhanced function tracking with scope detection
            if 'export' in line and ('const' in line or 'function' in line):
                # Skip type declarations
                if 'export type' in line or 'export interface' in line:
                    continue
                func_patterns = [
                    r'export\s+const\s+(\w+)\s*=\s*async',  # export const funcName = async
                    r'export\s+const\s+(\w+)\s*=',  # export const funcName = 
                    r'export\s+(?:async\s+)?function\s+(\w+)',  # export function funcName or export async function funcName
                ]
                for pattern in func_patterns:
                    func_match = re.search(pattern, line)
                    if func_match:
                        current_function = func_match.group(1)
                        current_function_start = i - 1  # Store function start line (0-indexed)
                        break
            
            # Track arrow functions and const declarations
            elif re.search(r'const\s+\w+\s*=\s*async', line) and not current_function:
                func_match = re.search(r'const\s+(\w+)\s*=\s*async', line)
                if func_match:
                    current_function = func_match.group(1)
                    current_function_start = i - 1  # Store function start line (0-indexed)
            
            # Primary API call detection - COMPLETE patterns from original
            if ('client.' in line or 'tsClient.' in line) and i not in found_lines:
                api_patterns = [
                    r'(?:client|tsClient)\.(get|post|put|patch|delete)',
                    r'await\s+(?:client|tsClient)\.(get|post|put|patch|delete)',
                    r'return\s+(?:await\s+)?(?:client|tsClient)\.(get|post|put|patch|delete)',
                    r'const\s+\w+\s*=\s*(?:await\s+)?(?:client|tsClient)\.(get|post|put|patch|delete)',
                    r'=\s*(?:await\s+)?(?:client|tsClient)\.(get|post|put|patch|delete)',
                    r'\.then\(\s*(?:await\s+)?(?:client|tsClient)\.(get|post|put|patch|delete)',  
                    r'Promise\.all\([^)]*(?:client|tsClient)\.(get|post|put|patch|delete)',
                ]
                for pattern in api_patterns:
                    api_match = re.search(pattern, line)
                    if api_match:
                        method = api_match.group(1).upper()
                        # Pass function scope information to URL extraction
                        url = self._extract_url_with_scope(line, lines, i-1, current_function_start, current_function)
                        if url:
                            calls.append(FrontendCall(
                                file=file_path.name,
                                function_name=current_function or "unknown",
                                method=method,
                                url_pattern=url,
                                line_number=i,
                                raw_code=line.strip()
                            ))
                            found_lines.add(i)
                        break
            
            # Fallback patterns - COMPLETE from original
            if ('client' in line or 'tsClient' in line) and i not in found_lines:
                fallback_patterns = [
                    r'(client|tsClient)\.(\w+)',  
                    r'(client|tsClient)\s*\[\s*["\'](\w+)["\']\s*\]',  
                ]
                
                for pattern in fallback_patterns:
                    fallback_match = re.search(pattern, line)
                    if fallback_match:
                        method_name = fallback_match.group(2)
                        if method_name.lower() in ['get', 'post', 'put', 'patch', 'delete']:
                            method = method_name.upper()
                            # Use scoped URL extraction for fallback too
                            url = self._extract_url_with_scope(line, lines, i-1, current_function_start, current_function)
                            if url:
                                calls.append(FrontendCall(
                                    file=file_path.name,
                                    function_name=current_function or "unknown",
                                    method=method,
                                    url_pattern=url,
                                    line_number=i,
                                    raw_code=line.strip()
                                ))
                                found_lines.add(i)
                            break
        return calls
    
    def _extract_url_with_scope(self, line: str, all_lines: List[str], line_idx: int, 
                               function_start: Optional[int], function_name: Optional[str]) -> str:
        """Enhanced URL extraction with proper function scope isolation"""
        if 'blob' in line.lower() or 'responseType:' in line:
            return ""
        
        # Determine function boundaries
        if function_start is not None:
            # Find the end of the current function
            function_end = self._find_function_end(all_lines, function_start, function_name)
            # Limit search to within the current function only
            search_start = max(line_idx, function_start)
            search_end = min(line_idx + 10, function_end, len(all_lines))
        else:
            # Fallback to limited search if no function scope
            search_start = line_idx
            search_end = min(line_idx + 5, len(all_lines))
        
        # URL patterns - same as before
        patterns = [
            r'[`"](\$\{[^}]+\}[^`"]*)[`"]',  # Template strings
            r'[`"\']([/\w\-{}$\.]+[^`"\']*)["`\']',  # Regular strings
            r'url\s*[=:]\s*[`"\']([^`"\']*)["`\']',  # url = "..." or url: "..."
            r'endpoint\s*[=:]\s*[`"\']([^`"\']*)["`\']',  # endpoint = "..."
            r'[`"\'](\/api\/[^`"\']*)["`\']',  # Any /api/ path
            r'[`"\'](\$\{V2_WORKFLOW_URL\}[^`"\']*)["`\']',  # Workflow URLs
            r'path\s*[=:]\s*[`"\']([^`"\']*)["`\']',  # path = "..."
            r'route\s*[=:]\s*[`"\']([^`"\']*)["`\']',  # route = "..."
        ]
        
        # Search within function scope only
        for i in range(search_start, search_end):
            if i >= len(all_lines):
                break
                
            current_line = all_lines[i]
            # Skip comments and empty lines
            if current_line.strip().startswith('//') or not current_line.strip():
                continue
            
            for pattern in patterns:
                match = re.search(pattern, current_line)
                if match:
                    url = match.group(1)
                    if url and (url.startswith('/') or '${' in url or 'api' in url or 'workflow' in url.lower()):
                        normalized = self._normalize_url(url)
                        if normalized and len(normalized) > 3:
                            return normalized
        
        # Enhanced variable detection for multi-line API calls
        var_patterns = [
            r'(?:client|tsClient)\.\w+(?:<[^>]*>)?\s*\(\s*(\w+)',
            r'(?:client|tsClient)\.\w+\s*\(\s*`([^`]+)`',
            r'(?:client|tsClient)\.\w+\s*\(\s*"([^"]+)"',
            r'(?:client|tsClient)\.\w+\s*\(\s*\'([^\']+)\'',
        ]
        
        # First check the current line for direct URL patterns
        for pattern in var_patterns:
            var_match = re.search(pattern, line)
            if var_match:
                var_name = var_match.group(1)
                excluded = ['data', 'body', 'payload', 'params', 'formData', 'config', 'options', 'headers', 'rest', 'args']
                if var_name not in excluded:
                    # Search for variable definition within function scope only
                    var_search_start = max(0, function_start) if function_start is not None else max(0, line_idx - 10)
                    var_search_end = min(line_idx, function_end) if function_start is not None else line_idx
                    
                    for j in range(var_search_start, var_search_end):
                        if j >= len(all_lines):
                            break
                        search_line = all_lines[j]
                        var_def_patterns = [
                            f'(?:const|let|var)\\s+{var_name}\\s*=\\s*[`"\']([^`"\']*)["`\']',
                            f'{var_name}\\s*=\\s*[`"\']([^`"\']*)["`\']',
                            f'\\b{var_name}\\s*=\\s*`([^`]+)`',
                        ]
                        for var_def_pattern in var_def_patterns:
                            var_def = re.search(var_def_pattern, search_line)
                            if var_def:
                                found_url = self._normalize_url(var_def.group(1))
                                if found_url:
                                    return found_url
                break
        if re.search(r'(?:client|tsClient)\.\w+(?:<.*?>)?\s*\(\s*$', line):
            # Look at the next few lines for variable names
            for next_line_idx in range(line_idx + 1, min(line_idx + 5, search_end)):
                if next_line_idx >= len(all_lines):
                    break
                next_line = all_lines[next_line_idx].strip()
                
                # Skip empty lines and comments
                if not next_line or next_line.startswith('//'):
                    continue
                
                # Look for variable name on this line (e.g., "url,")
                var_match = re.search(r'^(\w+),?\s*$', next_line)
                if var_match:
                    var_name = var_match.group(1)
                    excluded = ['data', 'body', 'payload', 'params', 'formData', 'config', 'options', 'headers', 'rest', 'args']
                    if var_name not in excluded:
                        # Search for variable definition within function scope
                        var_search_start = max(0, function_start) if function_start is not None else max(0, line_idx - 10)
                        var_search_end = min(line_idx, function_end) if function_start is not None else line_idx
                        
                        for j in range(var_search_start, var_search_end):
                            if j >= len(all_lines):
                                break
                            search_line = all_lines[j]
                            var_def_patterns = [
                                f'(?:const|let|var)\\s+{var_name}\\s*=\\s*[`"\']([^`"\']*)["`\']',
                                f'{var_name}\\s*=\\s*[`"\']([^`"\']*)["`\']',
                                f'\\b{var_name}\\s*=\\s*`([^`]+)`',
                            ]
                            for var_def_pattern in var_def_patterns:
                                var_def = re.search(var_def_pattern, search_line)
                                if var_def:
                                    found_url = self._normalize_url(var_def.group(1))
                                    if found_url:
                                        return found_url
                        break
                
                # Stop if we hit a closing parenthesis or semicolon (end of call)
                if ')' in next_line or ';' in next_line:
                    break
        
        return ""
    
    def _find_function_end(self, lines: List[str], function_start: int, function_name: Optional[str]) -> int:
        """Find the end of a function by tracking braces and detecting next function"""
        if function_start >= len(lines):
            return len(lines)
        
        brace_count = 0
        in_function = False
        
        for i in range(function_start, len(lines)):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('//'):
                continue
            
            # Count braces to track function scope
            brace_count += line.count('{') - line.count('}')
            
            # Mark that we've entered the function body
            if '{' in line and not in_function:
                in_function = True
            
            # If we've closed all braces and we were in a function, we've reached the end
            if in_function and brace_count <= 0:
                return i + 1
            
            # Also detect start of next function as a boundary
            if i > function_start and ('export const' in line or 'export function' in line or 'export async' in line):
                return i
        
        return len(lines)
    
    def _extract_url(self, line: str, all_lines: List[str], line_idx: int) -> str:
        # COMPLETE URL extraction from original
        if 'blob' in line.lower() or 'responseType:' in line:
            return ""
        
        # SPECIAL CASE: Handle variable-based API calls (like pool manager)
        # Look for pattern: client.method(variable)
        var_call_match = re.search(r'(?:client|tsClient)\.(\w+)\s*\(\s*(\w+)\s*\)', line)
        if var_call_match:
            var_name = var_call_match.group(2)
            # Look for variable definition in previous lines
            for i in range(max(0, line_idx - 10), line_idx):
                if i < len(all_lines):
                    prev_line = all_lines[i]
                    # Look for: const varname = `${V2_URL}/path`;
                    var_def_match = re.search(f'const\\s+{var_name}\\s*=\\s*`([^`]+)`', prev_line)
                    if var_def_match:
                        url_template = var_def_match.group(1)
                        normalized = self._normalize_url(url_template)
                        if normalized and len(normalized) > 3:
                            return normalized
        
        # URL patterns - COMPLETE from original
        patterns = [
            r'[`"](\$\{[^}]+\}[^`"]*)[`"]',  # Template strings
            r'[`"\']([/\w\-{}$\.]+[^`"\']*)[`"\']',  # Regular strings
            r'url\s*[=:]\s*[`"\'](.*?)[`"\']',  # url = "..." or url: "..."
            r'endpoint\s*[=:]\s*[`"\'](.*?)[`"\']',  # endpoint = "..."
            r'[`"\'](/api/[^`"\']*)[`"\']',  # Any /api/ path
            r'[`"\'](\$\{V2_WORKFLOW_URL\}[^`"\']*)[`"\']',  # Workflow URLs
            r'formData\.append\([^,]+,\s*[`"\'](.*?)[`"\']',  # FormData URLs
            r'path\s*[=:]\s*[`"\'](.*?)[`"\']',  # path = "..."
            r'route\s*[=:]\s*[`"\'](.*?)[`"\']',  # route = "..."
        ]# Enhanced: Search within the current function scope more precisely
        # Limit search to a smaller range to avoid picking up URLs from other functions
        search_end = min(line_idx + 15, len(all_lines))  # Reduced from 30 to 15
        
        for i in range(line_idx, search_end):
            current_line = all_lines[i]
            # Skip comments and empty lines
            if current_line.strip().startswith('//') or not current_line.strip():
                continue
            
            # Stop if we hit another function definition (avoid cross-function contamination)
            if i > line_idx and ('export const' in current_line or 'export function' in current_line or 'const ' in current_line and ' = async' in current_line):
                break
            
            for pattern in patterns:
                match = re.search(pattern, current_line)
                if match:
                    url = match.group(1)
                    if url and (url.startswith('/') or '${' in url or 'api' in url or 'workflow' in url.lower()):
                        normalized = self._normalize_url(url)
                        if normalized and len(normalized) > 3:
                            return normalized
        
        # Variable detection - COMPLETE from original
        var_patterns = [
            r'(?:client|tsClient)\.\w+(?:<[^>]*>)?\s*\(\s*(\w+)',
            r'(?:client|tsClient)\.\w+\s*\(\s*(\w+)',
            r'(?:client|tsClient)\.\w+\s*\(\s*`([^`]+)`',
            r'(?:client|tsClient)\.\w+\s*\(\s*"([^"]+)"',
            r'(?:client|tsClient)\.\w+\s*\(\s*\'([^\']+)\'',
        ]
        
        for pattern in var_patterns:
            var_match = re.search(pattern, line)
            if var_match:
                var_name = var_match.group(1)
                excluded = ['data', 'body', 'payload', 'params', 'formData', 'config', 'options', 'headers', 'rest', 'args']
                if var_name not in excluded:
                    for j in range(max(0, line_idx - 80), line_idx):
                        search_line = all_lines[j]
                        var_def_patterns = [
                            f'(?:const|let|var)\\s+{var_name}\\s*=\\s*[`"\'](.*?)[`"\']',
                            f'{var_name}\\s*=\\s*[`"\'](.*?)[`"\']',
                            f'\\b{var_name}\\s*=\\s*`([^`]+)`',
                            f'\\b{var_name}\\s*:\\s*[`"\'](.*?)[`"\']',
                        ]
                        for var_def_pattern in var_def_patterns:
                            var_def = re.search(var_def_pattern, search_line)
                            if var_def:
                                found_url = self._normalize_url(var_def.group(1))
                                if found_url:
                                    return found_url
                        break
        
        return ""

    def _extract_url_aggressive(self, line: str, all_lines: List[str], line_idx: int) -> str:
        """Ultra-aggressive URL extraction for edge cases - COMPLETE from original"""
        if 'blob' in line.lower() or 'responseType:' in line:
            return ""
        
        # More patterns for edge cases - COMPLETE from original
        aggressive_patterns = [
            r'[`"](\$\{[^}]+\}[^`"]*)[`"]',
            r'[`"\']([/\w\-{}$\.]+[^`"\']*)[`"\']',
            r'[`"\'](\/[^`"\']*)[`"\']',  # Any path starting with /
            r'[`"\']([^`"\']*api[^`"\']*)[`"\']',  # Any string containing 'api'
            r'[`"\']([^`"\']*workflow[^`"\']*)[`"\']',  # Any string containing 'workflow'
            r'[`"\']([^`"\']*\$\{[^}]+\}[^`"\']*)[`"\']',  # Any template string
        ]
        
        # Search wider range - up to 40 lines
        for i in range(line_idx, min(line_idx + 40, len(all_lines))):
            current_line = all_lines[i]
            if current_line.strip().startswith('//') or not current_line.strip():
                continue
            
            for pattern in aggressive_patterns:
                match = re.search(pattern, current_line)
                if match:
                    url = match.group(1)
                    if url and len(url) > 2:  
                        normalized = self._normalize_url(url)
                        if normalized and len(normalized) > 3 and ('api' in normalized or normalized.startswith('/')):
                            return normalized
        
        return ""

    def _normalize_url(self, url: str) -> str:
        if not url:
            return ""
        
        # Replace template variables - COMPLETE from original
        url = url.replace('${V2_URL}', '/api/v2')
        url = url.replace('${V2_WORKFLOW_URL}', '/api/v2/workflows')
        
        # Handle complex interpolations
        url = re.sub(r'\$\{[\w.]+\.(\w+)\}', r'{\1}', url)
        url = re.sub(r'\$\{(\w+)\}', lambda m: '{' + self._camel_to_snake(m.group(1)) + '}', url)
        
        # Remove query parameters
        url = re.sub(r'\?.*$', '', url)
        
        return url.strip()

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

class CompleteBackendAnalyzer:
    """Complete backend analyzer with deep table analysis"""
    
    def __init__(self, backend_path: str):
        self.routers_path = Path(backend_path) / "api" / "v2" / "routers"
        self.backend_path = Path(backend_path)# Initialize table analysis components
        self.all_known_tables = collect_all_possible_table_names(str(self.backend_path))
        logger.info(f"Found {len(self.all_known_tables)} known tables")
        
        # Collect all function definitions for deep analysis
        self.all_function_definitions = self._collect_all_function_definitions()
        logger.info(f"Collected {len(self.all_function_definitions)} function definitions")
    def _collect_all_function_definitions(self) -> Dict[str, Any]:
        """Collect all function definitions across all Python files"""
        all_function_definitions = {}
        
        for root, _, files in os.walk(str(self.backend_path)):
            if '.venv' in root or '__pycache__' in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            source = f.read()
                        tree = ast.parse(source, filename=full_path)
                        
                        # Extract function definitions (including class methods)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                all_function_definitions[node.name] = node
                            elif isinstance(node, ast.ClassDef):
                                # Extract class methods
                                for class_node in node.body:
                                    if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                        method_name = f"{node.name}.{class_node.name}"
                                        all_function_definitions[method_name] = class_node
                    except Exception:
                        continue
        
        return all_function_definitions

    def extract_backend_routes(self) -> List[BackendRoute]:
        if not self.routers_path.exists():
            return []
        
        routes = []
        for file_path in self.routers_path.rglob("*.py"):
            if file_path.name != "__init__.py":
                routes.extend(self._parse_file(file_path))
        
        return routes
    def _parse_file(self, file_path: Path) -> List[BackendRoute]:
        try:
            # file_path = Path('guided-workflow-backend/api/v2/routers/revenue.py')
            logger.info(f"Parsing file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            # Parse with COMPLETE AST analyzer for table analysis
            tree = ast.parse(source, filename=str(file_path))
            table_analyzer = CompleteTableAnalyzer(
                all_known_tables=self.all_known_tables,
                all_function_definitions=self.all_function_definitions
            )
            table_analyzer.visit(tree)
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")        
            return []
        
        routes = []
        rel_path = file_path.relative_to(self.routers_path)
        base_prefix = self._get_prefix(rel_path)
        
        # Parse route decorators
        lines = source.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('@router.'):
                decorator_lines = [line]

                j = i + 1
                while j < len(lines) and ')' not in ''.join(decorator_lines):
                    decorator_lines.append(lines[j].strip())
                    j += 1
                
                full_decorator = ' '.join(decorator_lines)
                table_column_mapping, flag = self._get_function_decorator(full_decorator, lines, j)
                #print(table_column_mapping)
                route_match = re.search(r'@router\.(get|post|put|patch|delete)\s*\(\s*["\']([^"\']*)["\']', full_decorator, re.IGNORECASE)
                if route_match:
                    method = route_match.group(1).upper()
                    route = route_match.group(2)
                    if not flag:
                        table_column_mapping = self._get_function_return_type(lines, j)

                    #print(table_column_mapping)
                    #print(flag)
                    
                    func_name = self._get_function_name(lines, j)
                    full_route = base_prefix + route if route else base_prefix# Get tables for this function with DEEP ANALYSIS
                    all_tables = set(table_analyzer.table_calls.get(func_name, set()))
                    
                    # Reset analyzed functions for each route to ensure complete analysis
                    table_analyzer.analyzed_functions = set()
                    
                    # Analyze each called function deeply
                    for called_func in table_analyzer.call_graph.get(func_name, []):
                        # Add tables from direct calls
                        all_tables.update(table_analyzer.table_calls.get(called_func, []))
                        # Perform deep recursive analysis
                        deep_tables = table_analyzer.analyze_called_function(called_func)
                        all_tables.update(deep_tables)
                    
                    tables = list(all_tables)# Get stored procedures and flow calls for this function
                    all_stored_procedures = set(table_analyzer.proc_calls.get(func_name, []))
                    all_flow_calls = set()
                    
                    # Find flow calls from route info
                    for route_info in table_analyzer.routes_info:
                        if route_info.get('function') == func_name:
                            all_flow_calls.update(route_info.get('flow_calls', []))
                            all_stored_procedures.update(route_info.get('stored_procedures', []))
                            break
                    
                    # Also collect from called functions
                    for called_func in table_analyzer.call_graph.get(func_name, []):
                        all_stored_procedures.update(table_analyzer.proc_calls.get(called_func, []))
                        # Also check for flow calls in called functions
                        for route_info in table_analyzer.routes_info:
                            if route_info.get('function') == called_func:
                                all_flow_calls.update(route_info.get('flow_calls', []))
                                break
                    
                    stored_procedures = list(all_stored_procedures)
                    flow_calls = list(all_flow_calls)
                    routes.append(BackendRoute(
                        file=str(rel_path),
                        method=method,
                        route_pattern=full_route,
                        line_number=i + 1,
                        function_name=func_name,
                        tags=[],
                        raw_code=full_decorator[:100],
                        tables=tables,
                        stored_procedures=stored_procedures,
                        flow_calls=flow_calls,
                        response_model_info=table_column_mapping  # Add column information
                    ))
                
                i = j
            else:
                i += 1
        
        return routes

    def _get_prefix(self, rel_path: Path) -> str:
        # COMPLETE prefix mapping from original
        parts = rel_path.parts[:-1]
        filename = rel_path.stem
        prefix = "/api/v2"
        
        # Add directory prefixes
        for part in ["workflows", "admin", "manager", "support", "sdp"]:
            if part in parts:
                prefix += f"/{part}"
        
        # Add specific file mappings - COMPLETE from original
        file_routes = {
            # Workflows
            "signoff": "/sign_off", "actions": "/actions", "notifications": "/notifications",
            "downloads": "/downloads", "evidence_uploads": "/evidence_uploads",
            "lookups": "/lookups", "macd": "/macd",
            # Admin
            "unverified": "/bookings/unverified", "verified": "/bookings/verified",
            "revenue": "/revenue", "contracts": "/contracts",
            "tasks": "/tasks", "subtasks": "/subtasks", "deliverables": "/deliverables",
            # Manager  
            "bookings": "/bookings", "scv": "/scv", "super_customers": "/scv",
            "users": "/users", "pool": "/pool_manager",
            # Support
            "user": "/cases", "agent": "/agent",
            # SDP
            "completions": "/completions", "time_tracking": "/time_tracking",
            # Top-level
            "engagements": "/engagements", "tags": "/tags", "tagsets": "/tagsets",
            "stakeholders": "/stakeholders", "canvas": "/canvas", "links": "/links",
            "thought_spot": "/thought_spot", "thought_spot_tag": "/thought_spot_tag",
            "dc_types": "/dc_types", "user_defined_types": "/udt",
            "documentation": "/documentation", "announcements": "/announcements",
            "file_management": "/file_management", "static": "/static"
        }
        
        for key, route in file_routes.items():
            if key in filename:
                prefix += route
                break
        
        # Handle edge cases - COMPLETE from original
        if "financial" in str(rel_path) and "/financial" not in prefix:
            prefix += "/financial"
        if filename == "tagsets":
            prefix = "/api/v2/tagsets"
        if filename == "thought_spot_tag":
            return "/api/v2/thought_spot_tag"
        if filename == "user_defined_types":
            return "/api/v2/udt"
        
        return prefix
     
    def _get_function_decorator(self, line: str, lines: List[str], start_idx: int):
        """
        Extract response model and analyze class parameters and nested class parameters
        Returns dict with response_model info and column details
        """
        result = {
            'response_model': None,
            'columns': [],
            'nested_columns': [],
            'return_type': None
        }
        

        
        # 1. Check for response_model in decorator
        if 'response_model' in line:
            resp_match = re.search(r'response_model\s*=\s*([\w\.\[\]]+)', line)
            if resp_match:
                response_model_str = resp_match.group(1)
                result['response_model'] = response_model_str
                
                
                # Analyze the response model class
                columns, nested_columns = self._analyze_model_class(response_model_str)
                result['columns'] = columns
                result['nested_columns'] = nested_columns
                
                return result, True
        
        
        # 2. If no response_model, check function return type annotation
        func_line = self._get_function_line(lines, start_idx)
        if func_line:
            return_type = self._extract_return_type(func_line)
            if return_type:
                result['return_type'] = return_type
                logger.debug(f"Found return type: {return_type}")
                
                # Analyze the return type class
                columns, nested_columns = self._analyze_model_class(return_type)
                result['columns'] = columns  
                result['nested_columns'] = nested_columns

                return result, True

        return result, False


    def _get_function_line(self, lines: List[str], start_idx: int) -> str:
        """Get the function definition line"""
        for i in range(start_idx, min(start_idx + 10, len(lines))):
            if i < len(lines) and not lines[i].strip().startswith('@'):
                line = lines[i].strip()
                if line.startswith('def ') or line.startswith('async def '):
                    return line
        return ""
    
    def _extract_return_type(self, func_line: str) -> str:
        """Extract return type annotation from function definition"""
        # Pattern: def func_name(...) -> ReturnType:
        return_match = re.search(r'->\s*([\w\.\[\]]+):', func_line)
        if return_match:
            return return_match.group(1)
        return ""
    
    def _analyze_model_class(self, model_name: str) -> tuple[list, list]:
        """
        Analyze a Pydantic model class to extract field information
        Returns tuple of (columns, nested_columns)
        """
        columns = []
        nested_columns = []
        
        # Clean up the model name (handle list[ModelName], Optional[ModelName], etc.)
        clean_model_name = self._clean_model_name(model_name)
        
        # Find the model class definition
        model_info = self._find_model_definition(clean_model_name)
        if not model_info:
            
            return columns, nested_columns
        
        # Parse the model fields
        columns, nested_columns = self._parse_model_fields(model_info)
        
        return columns, nested_columns
    
    def _clean_model_name(self, model_name: str) -> str:
        """Clean model name from type annotations like list[Model] or Optional[Model]"""
        # Remove list[], dict[], Optional[], Union[], etc.
        patterns = [
            r'list\[([\w\.]+)\]',
            r'List\[([\w\.]+)\]', 
            r'dict\[[\w\.,\s]+,\s*([\w\.]+)\]',
            r'Dict\[[\w\.,\s]+,\s*([\w\.]+)\]',
            r'Optional\[([\w\.]+)\]',
            r'Union\[[\w\.,\s]*?([\w\.]+)[\w\.,\s]*?\]'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, model_name)
            if match:
                return match.group(1)
        
        return model_name
    
    def _find_model_definition(self, model_name: str) -> dict:
        """
        Find model definition in the codebase
        Returns dict with file path and class definition
        """
        # Search patterns for model files
        model_patterns = [
            "**/models/**/*.py",
            "**/api/**/models/**/*.py", 
            "**/orm/**/*.py",
            "**/*models*.py"
        ]
        
        for pattern in model_patterns:
            for file_path in glob.glob(pattern, recursive=True):
                if '.venv' in str(file_path) or '__pycache__' in str(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Look for class definition
                    class_pattern = rf'class\s+{re.escape(model_name)}\s*\([^)]*\):\s*(.*?)(?=^class\s|\Z)'
                    class_match = re.search(class_pattern, content, re.DOTALL | re.MULTILINE)
                    
                    if class_match:
                        return {
                            'file_path': file_path,
                            'class_body': class_match.group(1),
                            'full_content': content
                        }
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
                    continue
        
        return None
    
    def _parse_model_fields(self, model_info: dict) -> tuple[list, list]:
        """
        Parse Pydantic model fields from class definition
        Returns tuple of (simple_columns, nested_columns)
        """
        columns = []
        nested_columns = []
        
        class_body = model_info['class_body']
        full_content = model_info['full_content']
        
        # Get imports to resolve types
        imports = self._extract_imports(full_content)
        
        # Pattern to match field definitions
        field_patterns = [
            # name: type = Field(...)
            r'(\w+):\s*([\w\.\[\]|]+)(?:\s*=\s*Field\([^)]*\))?',
            # name: type
            r'(\w+):\s*([\w\.\[\]|]+)(?:\s*=\s*[^#\n]+)?',
        ]
        
        # Use a set to track processed field names and avoid duplicates
        processed_fields = set()
        
        for pattern in field_patterns:
            matches = re.findall(pattern, class_body)
            for match in matches:
                field_name = match[0] if len(match) > 0 else ""
                field_type = match[1] if len(match) > 1 else ""
                
                if field_name and not field_name.startswith('_') and field_name not in processed_fields:
                    processed_fields.add(field_name)  # Mark as processed to avoid duplicates
                    
                    # Determine if this is a simple or nested field
                    if self._is_simple_type(field_type):
                        columns.append({
                            'name': field_name,
                            'type': field_type,
                            'is_nested': False
                        })
                    else:
                        # This might be a nested model
                        nested_info = {
                            'name': field_name,
                            'type': field_type,
                            'is_nested': True,
                            'nested_fields': []
                        }
                        
                        # Try to resolve nested model fields
                        clean_type = self._clean_model_name(field_type)
                        nested_model_info = self._find_model_definition(clean_type)
                        if nested_model_info:
                            nested_fields, _ = self._parse_model_fields(nested_model_info)
                            nested_info['nested_fields'] = nested_fields
                        
                        nested_columns.append(nested_info)
        
        # Handle special Pydantic patterns
        columns, nested_columns = self._handle_special_patterns(class_body, columns, nested_columns)
        
        return columns, nested_columns
    
    def _extract_imports(self, content: str) -> dict:
        """Extract import statements for type resolution"""
        imports = {}
        
        import_patterns = [
            r'from\s+([\w\.]+)\s+import\s+([\w\s,]+)',
            r'import\s+([\w\.]+)(?:\s+as\s+(\w+))?'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) == 2:
                    module, items = match
                    if 'import' in pattern:  # from module import items
                        for item in items.split(','):
                            item = item.strip()
                            imports[item] = f"{module}.{item}"
                    else:  # import module as alias
                        alias = match[1] if match[1] else match[0]
                        imports[alias] = match[0]
        
        return imports
    
    def _is_simple_type(self, type_str: str) -> bool:
        """Check if a type is a simple type (not a nested model)"""
        simple_types = {
            'str', 'int', 'float', 'bool', 'datetime', 'date', 'time',
            'Optional', 'List', 'Dict', 'Any', 'Union', 'Literal',
            'list', 'dict', 'set', 'tuple'
        }
        
        # Remove generic type parameters
        base_type = re.sub(r'\[.*?\]', '', type_str)
        
        return any(simple in base_type for simple in simple_types)
    
    def _handle_special_patterns(self, class_body: str, columns: list, nested_columns: list) -> tuple[list, list]:
        """Handle special Pydantic patterns like __root__ and discriminated unions"""
        
        # Handle __root__ fields (discriminated unions)
        root_pattern = r'__root__:\s*([\w\.\[\]|]+)(?:\s*=\s*Field\([^)]*discriminator[^)]*\))?'
        root_match = re.search(root_pattern, class_body)
        if root_match:
            root_type = root_match.group(1)
            # Parse union types
            if '|' in root_type or 'Union[' in root_type:
                union_types = self._parse_union_types(root_type)
                for union_type in union_types:
                    clean_type = self._clean_model_name(union_type)
                    nested_model_info = self._find_model_definition(clean_type)
                    if nested_model_info:
                        nested_fields, _ = self._parse_model_fields(nested_model_info)
                        nested_columns.append({
                            'name': f"__root__{clean_type}",
                            'type': union_type,
                            'is_nested': True,
                            'nested_fields': nested_fields,
                            'is_union_member': True
                        })
        
        return columns, nested_columns
    
    def _parse_union_types(self, union_str: str) -> list:
        """Parse Union types or | separated types"""
        if 'Union[' in union_str:
            # Extract types from Union[Type1, Type2, ...]
            match = re.search(r'Union\[(.*?)\]', union_str)
            if match:
                types_str = match.group(1)
                return [t.strip() for t in types_str.split(',')]
        elif '|' in union_str:
            # Handle Type1 | Type2 | Type3 syntax
            return [t.strip() for t in union_str.split('|')]
        
        return [union_str]

    def _get_function_name(self, lines: List[str], start_idx: int) -> str:
        for i in range(start_idx, min(start_idx + 5, len(lines))):
            if i < len(lines) and not lines[i].strip().startswith('@'):
                func_match = re.search(r'def\s+(\w+)', lines[i])
                if func_match:
                    return func_match.group(1)
        return "unknown"
    
    def _get_function_return_type(self, lines: str, start_idx: int) -> Optional[str]:
        
        result = {
        'response_model': None,
        'columns': [],
        'nested_columns': [],
        'return_type': None
        }
    
        for i in range(start_idx, min(start_idx + 15, len(lines))):
            if i < len(lines) and not lines[i].strip().startswith('@'):
                return_match = re.search(r'->\s*([\w\.\[\]]+):', lines[i])

                if return_match:
                    columns, nested_columns = self._analyze_model_class(return_match.group(1))
                    result['columns'] = columns  
                    result['nested_columns'] = nested_columns
                    return_type = return_match.group(1) 
                    result['return_type'] = return_type
                    return result
        return None

class APIMapper:
    def __init__(self, frontend_calls: List[FrontendCall], backend_routes: List[BackendRoute]):
        self.frontend_calls = frontend_calls
        self.backend_routes = backend_routes

    def create_mappings(self) -> List[Mapping]:
        mappings = []
        for frontend_call in self.frontend_calls:
            backend_route = self._find_match(frontend_call)
            mappings.append(Mapping(frontend=frontend_call, backend=backend_route))
           
        return mappings

    def _find_match(self, frontend_call: FrontendCall) -> Optional[BackendRoute]:
        # Exact match first
        for backend_route in self.backend_routes:
            if self._routes_match(frontend_call, backend_route):
                return backend_route
        
        # Fuzzy match
        best_match = None
        best_score = 0.6
        
        for backend_route in self.backend_routes:
            if backend_route.method == frontend_call.method:
                score = self._similarity(frontend_call.url_pattern, backend_route.route_pattern)
                if score > best_score:
                    best_score = score
                    best_match = backend_route
        
        return best_match
    def _routes_match(self, frontend: FrontendCall, backend: BackendRoute) -> bool:
        if frontend.method != backend.method:
            return False
            
        
        fe_url = frontend.url_pattern.replace('/api/v2', '').strip('/')
        be_url = backend.route_pattern.replace('/api/v2', '').strip('/')
        
        # Exact match first (most reliable)
        if fe_url == be_url:
            return True
        
        fe_parts = fe_url.split('/') if fe_url else []
        be_parts = be_url.split('/') if be_url else []
        
        if len(fe_parts) != len(be_parts):
            return False
        
        # Enhanced matching with parameter handling
        for fe_part, be_part in zip(fe_parts, be_parts):
            # Both are parameters
            if (fe_part.startswith('{') and be_part.startswith('{')):
                continue
            # Exact match
            elif fe_part == be_part:
                continue
            # One is parameter, other is not - no match
            else:
                return False
        
        return True

    def _similarity(self, str1: str, str2: str) -> float:
        parts1 = set(str1.split('/'))
        parts2 = set(str2.split('/'))
        
        if not parts1 or not parts2:
            return 0.0
        
        intersection = parts1.intersection(parts2)
        union = parts1.union(parts2)
        
        return len(intersection) / len(union)
def generate_reports(mappings: List[Mapping], output_base: str, formats: List[str]):
    if "json" in formats:
        data = []
        for m in mappings:
            backend_data = None
            if m.backend:
                # Format response model info for JSON
                response_model_columns = []
                nested_columns = []
                
                if m.backend.response_model_info:
                    response_model_columns = m.backend.response_model_info.get('columns', [])
                    nested_columns = m.backend.response_model_info.get('nested_columns', [])
                
                backend_data = {
                    "file": m.backend.file,
                    "function": m.backend.function_name,
                    "method": m.backend.method,
                    "route": m.backend.route_pattern,
                    "line": m.backend.line_number,
                    "tables": [table.upper() for table in m.backend.tables] if m.backend.tables else [],
                    "response_model": m.backend.response_model_info.get('response_model') if m.backend.response_model_info else None,
                    "return_type": m.backend.response_model_info.get('return_type') if m.backend.response_model_info else None,
                    "columns": response_model_columns,
                    "nested_columns": nested_columns
                }
            
            data.append({
                "frontend": {
                    "file": m.frontend.file,
                    "function": m.frontend.function_name,
                    "method": m.frontend.method,
                    "url": m.frontend.url_pattern,
                    "line": m.frontend.line_number
                },
                "backend": backend_data
            })
        
        with open(f"{output_base}.json", 'w') as f:
            json.dump(data, f, indent=2)

    if "csv" in formats:
        import csv
        with open(f"{output_base}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Frontend File", "Frontend Function", "HTTP Method", "Frontend URL",
                "Backend File", "Backend Function", "Backend Route", "Tables",
                "Response Model", "Response Fields", "Nested Fields", "Table Column Details",
                "Column Count", "Relationship Type"
            ])
            
            for m in mappings:
                # Format response model info for CSV
                response_model = ""
                columns_str = ""
                nested_columns_str = ""
                table_column_details = ""
                column_count = 0
                relationship_type = "unknown"
                
                if m.backend and m.backend.response_model_info:
                    response_model = m.backend.response_model_info.get('response_model', '') or m.backend.response_model_info.get('return_type', '')
                    
                    # Format columns as "name,name,name" (only column names)
                    columns = m.backend.response_model_info.get('columns', [])
                    columns_str = ",".join([col['name'] for col in columns if isinstance(col, dict) and 'name' in col])
                    column_count += len([col for col in columns if isinstance(col, dict) and 'name' in col])
                    
                    # Format nested columns (only column names from nested fields)
                    nested_columns = m.backend.response_model_info.get('nested_columns', [])
                    nested_parts = []
                    for nested in nested_columns:
                        if isinstance(nested, dict):
                            nested_fields = nested.get('nested_fields', [])
                            fields_str = ",".join([field['name'] for field in nested_fields if isinstance(field, dict) and 'name' in field])
                            if fields_str:  # Only add if there are actual fields
                                nested_parts.append(f"{nested['name']}:[{fields_str}]")
                                column_count += len([field for field in nested_fields if isinstance(field, dict) and 'name' in field])
                    nested_columns_str = ";".join(nested_parts)
                
                # Enhanced table column details
                if m.backend and m.backend.tables:
                    tables_str = ",".join(table.upper() for table in m.backend.tables)
                    
                    # Create detailed table-column mapping
                    table_details = []
                    for table in m.backend.tables:
                        table_upper = table.upper()
                        # Try to get actual database columns for this table
                        db_columns = get_table_columns(table_upper)
                        if db_columns:
                            table_details.append(f"{table_upper}:[{','.join(db_columns)}]")
                        else:
                            table_details.append(f"{table_upper}:[columns_unknown]")
                    
                    table_column_details = ";".join(table_details)
                    
                    # Determine relationship type
                    if len(m.backend.tables) == 1:
                        relationship_type = "single_table"
                    elif len(m.backend.tables) > 1:
                        relationship_type = "multi_table_join"
                else:
                    tables_str = ""
                    relationship_type = "no_tables"
                
                if not m.backend:
                    relationship_type = "unmatched"
                
                writer.writerow([
                    m.frontend.file, m.frontend.function_name, m.frontend.method, m.frontend.url_pattern,
                    m.backend.file if m.backend else "",
                    m.backend.function_name if m.backend else "",
                    m.backend.route_pattern if m.backend else "",
                    tables_str,
                    response_model,
                    columns_str,
                    nested_columns_str,
                    table_column_details,
                    str(column_count),
                    relationship_type
                ])
def print_summary(mappings: List[Mapping]):
    total = len(mappings)
    matched = sum(1 for m in mappings if m.backend)
    unmatched = total - matched
    with_tables = sum(1 for m in mappings if m.backend and m.backend.tables)
    with_response_models = sum(1 for m in mappings if m.backend and m.backend.response_model_info and 
                              (m.backend.response_model_info.get('response_model') or m.backend.response_model_info.get('return_type')))
    
    logger.info("\nAPI MAPPING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total Frontend Calls: {total}")
    logger.info(f"Matched:              {matched} ({matched/total*100:.1f}%)")
    logger.info(f"Unmatched:            {unmatched} ({unmatched/total*100:.1f}%)")
    logger.info(f"With Tables:          {with_tables} ({with_tables/total*100:.1f}%)")
    logger.info(f"With Response Models: {with_response_models} ({with_response_models/total*100:.1f}%)")
    
    # Show some examples of table mappings
    logger.info("\nSample Table Mappings:")
    logger.info("-" * 30)
    count = 0
    for m in mappings:
        if m.backend and m.backend.tables and count < 5:
            tables_str = ", ".join(m.backend.tables[:3])  # Show first 3 tables
            if len(m.backend.tables) > 3:
                tables_str += f" (+{len(m.backend.tables)-3} more)"
            logger.info(f"  {m.backend.route_pattern} -> {tables_str}")
            count += 1
    
    if count == 0:
        logger.info("  No table mappings found")
    
    # Show some examples of response model mappings
    logger.info("\nSample Response Model Mappings:")
    logger.info("-" * 35)
    count = 0
    for m in mappings:
        if (m.backend and m.backend.response_model_info and count < 5):
            info = m.backend.response_model_info
            model_name = info.get('response_model') or info.get('return_type')
            if model_name:
                columns = info.get('columns', [])
                columns_preview = ", ".join([col['name'] for col in columns[:3]])
                if len(columns) > 3:
                    columns_preview += f" (+{len(columns)-3} more)"
                logger.info(f"  {m.backend.route_pattern} -> {model_name} [{columns_preview}]")
                count += 1
    
    if count == 0:
        logger.info("  No response model mappings found")
def main():
    parser = argparse.ArgumentParser(description="Complete frontend-backend API mapping with deep table analysis")
    parser.add_argument("--frontend", default="guided-workflow", help="Frontend project path")
    parser.add_argument("--backend", default="guided-workflow-backend", help="Backend project path")
    parser.add_argument("--output", default="complete_api_mapping_with_tables", help="Output file base name")
    parser.add_argument("--format", choices=["json", "csv", "all"], default="all", help="Output format")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level based on argument
    logger.setLevel(getattr(logging, args.log_level))
    
    logger.info("Complete Frontend-Backend API Mapping Tool with Deep Table Analysis")
    logger.info("=" * 70)
    
    # Analyze frontend with COMPLETE analyzer
    logger.info("\n1. Analyzing frontend with complete patterns...")
    frontend_analyzer = CompleteFrontendAnalyzer(args.frontend)
    frontend_calls = frontend_analyzer.extract_frontend_calls()
    logger.info(f"   Found {len(frontend_calls)} frontend API calls")
    
    # Analyze backend with COMPLETE analyzer
    logger.info("\n2. Analyzing backend with deep table analysis...")
    backend_analyzer = CompleteBackendAnalyzer(args.backend)
    backend_routes = backend_analyzer.extract_backend_routes()
    logger.info(f"   Found {len(backend_routes)} backend routes")
    
    # Show table analysis summary
    routes_with_tables = sum(1 for route in backend_routes if route.tables)
    logger.info(f"   Routes with tables: {routes_with_tables}")
    
    # Create mappings
    logger.info("\n3. Creating mappings...")
    mapper = APIMapper(frontend_calls, backend_routes)
    mappings = mapper.create_mappings()
    
    # Generate reports
    logger.info("\n4. Generating reports...")
    formats = ["json", "csv"] if args.format == "all" else [args.format]
    generate_reports(mappings, args.output, formats)
    
    for fmt in formats:
        logger.info(f"   Generated: {args.output}.{fmt}")
    
    # Print summary
    print_summary(mappings)

if __name__ == "__main__":
    main()