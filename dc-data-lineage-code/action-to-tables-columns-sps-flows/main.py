#!/usr/bin/env python3
"""
uv run python main.py
This script generates a relation showing relationships between frontend actions,
backend endpoints, database tables, and to its columns using response_model.
"""

import sys
import os
import logging
import csv
import re
import ast
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from your existing action_to_table.py
from action_to_table import (
    CompleteFrontendAnalyzer, CompleteBackendAnalyzer, APIMapper,
    CompleteTableAnalyzer, collect_all_possible_table_names
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TableColumnExtractor:
    """Extracts detailed table-column information from ORM files"""
    
    def __init__(self, backend_path: str):
        self.backend_path = Path(backend_path)
        self.orm_path = self.backend_path / "api" / "v2" / "orm"
        self.models_path = self.backend_path / "api" / "v2" / "models"
        
        # Storage for extracted information
        self.table_columns = {}  # table_name -> [column_names]
        self.orm_table_map = {}  # orm_class -> table_name
        self.response_model_fields = {}  # response_model -> [field_info]
        
    def extract_all_information(self):
        """Extract all table-column and model information"""
        logger.info("Extracting detailed table-column information...")
        
        self._extract_orm_columns()
        self._extract_response_model_fields()
        
        logger.info(f"Found {len(self.table_columns)} tables with column details")
        logger.info(f"Found {len(self.response_model_fields)} response models")
        
        return {
            'table_columns': self.table_columns,
            'orm_table_map': self.orm_table_map,
            'response_model_fields': self.response_model_fields
        }
    
    def _extract_orm_columns(self):
        """Extract table and column information from ORM files"""
        if not self.orm_path.exists():
            return
            
        for orm_file in self.orm_path.rglob("*.py"):
            if orm_file.name == "__init__.py":
                continue
                
            try:
                with open(orm_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self._parse_orm_file(content, str(orm_file))
                
            except Exception as e:
                logger.warning(f"Error parsing {orm_file}: {e}")
    
    def _parse_orm_file(self, content: str, file_path: str):
        """Parse a single ORM file to extract table and column info"""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Extract table name
                    table_name = self._get_table_name_from_class(node)
                    if table_name:
                        class_name = node.name
                        table_upper = table_name.upper()
                        
                        # Map ORM class to table
                        self.orm_table_map[class_name] = table_upper
                        
                        # Extract columns
                        columns = self._extract_columns_from_class(node)
                        if columns:
                            self.table_columns[table_upper] = sorted(columns)
                            
        except Exception as e:
            logger.warning(f"Error parsing AST for {file_path}: {e}")
    
    def _get_table_name_from_class(self, class_node: ast.ClassDef) -> str:
        """Extract __tablename__ from class definition"""
        for stmt in class_node.body:
            if (isinstance(stmt, ast.Assign) and 
                any(isinstance(target, ast.Name) and target.id == '__tablename__' for target in stmt.targets)):
                if isinstance(stmt.value, ast.Constant):
                    return stmt.value.value
                elif isinstance(stmt.value, ast.Str):  
                    return stmt.value.s
        return ""
    
    def _extract_columns_from_class(self, class_node: ast.ClassDef) -> List[str]:
        """Extract column names from ORM class"""
        columns = []
        
        for stmt in class_node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                field_name = stmt.target.id
                
                # Skip private fields and relationships
                if not field_name.startswith('_') and not self._is_relationship_field(stmt):
                    columns.append(field_name.upper())
        
        return columns
    
    def _is_relationship_field(self, stmt: ast.AnnAssign) -> bool:
        """Check if this is a relationship field (not a database column)"""
        if stmt.value and isinstance(stmt.value, ast.Call):
            if isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == 'relationship':
                return True
        return False
    
    def _extract_response_model_fields(self):
        """Extract field information from response models"""
        if not self.models_path.exists():
            return
            
        for model_file in self.models_path.rglob("*.py"):
            if model_file.name == "__init__.py":
                continue
                
            try:
                with open(model_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self._parse_response_model_file(content)
                
            except Exception as e:
                logger.warning(f"Error parsing {model_file}: {e}")
    
    def _parse_response_model_file(self, content: str):
        """Parse response model file to extract field information"""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if this is a response model
                    if any(suffix in node.name for suffix in ['Read', 'Response', 'Model', 'Output']):
                        fields = []
                        for stmt in node.body:
                            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                                field_name = stmt.target.id
                                if not field_name.startswith('_'):
                                    field_type = self._extract_type_annotation(stmt.annotation)
                                    fields.append({
                                        'name': field_name,
                                        'type': field_type,
                                        'optional': self._is_optional_field(stmt.annotation)
                                    })
                        
                        if fields:
                            self.response_model_fields[node.name] = fields
                            
        except Exception as e:
            logger.warning(f"Error parsing response model AST: {e}")
    
    def _extract_type_annotation(self, annotation) -> str:
        """Extract type information from annotation"""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                return annotation.value.id
        return "unknown"
    
    def _is_optional_field(self, annotation) -> bool:
        """Check if field is Optional"""
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name) and annotation.value.id == 'Optional':
                return True
        return False

class EnhancedCSVGenerator:
    """Generates CSV with detailed table-column relationships"""
    
    def __init__(self, frontend_path: str, backend_path: str):
        self.frontend_path = frontend_path
        self.backend_path = backend_path
        self.column_extractor = TableColumnExtractor(backend_path)
        
    def generate_enhanced_csv(self, output_file: str = "action_table_column_mapping.csv"):
        """Generate the enhanced CSV with all relationship details"""
        logger.info("Generating Table-Column Relationship CSV")
        logger.info("=" * 60)
        
        # Step 1: Extract detailed column information
        logger.info("Step 1: Extracting table-column information...")
        column_info = self.column_extractor.extract_all_information()
        
        # Step 2: Analyze frontend
        logger.info("Step 2: Analyzing frontend...")
        frontend_analyzer = CompleteFrontendAnalyzer(self.frontend_path)
        frontend_calls = frontend_analyzer.extract_frontend_calls()
        logger.info(f"Found {len(frontend_calls)} frontend API calls")# Step 3: Analyze backend with enhanced logging
        logger.info("Step 3: Analyzing backend...")
        backend_analyzer = CompleteBackendAnalyzer(self.backend_path)
        backend_routes = backend_analyzer.extract_backend_routes()
        logger.info(f"Found {len(backend_routes)} backend routes")
        
        # Log routes with missing functions
        target_functions = ['rebuild_sdp_for_booking', 'create_cxea_revenue_entries', 'create_htec_revenue_entries', 'create_cogs_revenue_entries']
        found_targets = [route for route in backend_routes if any(target in route.function_name for target in target_functions)]
        logger.info(f"Found {len(found_targets)} routes with target functions")
        for route in found_targets:
            logger.info(f"  - {route.function_name} in {route.file}")
        
        # Step 4: Create mappings
        logger.info("Step 4: Creating API mappings...")
        mapper = APIMapper(frontend_calls, backend_routes)
        mappings = mapper.create_mappings()
        logger.info(f"Created {len(mappings)} mappings")
        
        # Step 5: Generate enhanced CSV
        logger.info(f"Step 5: Writing enhanced CSV: {output_file}")
        self._write_enhanced_csv(mappings, column_info, output_file)
        
        # Step 6: Print summary
        self._print_summary(output_file, column_info)
        
        logger.info(f"Enhanced CSV generated successfully: {output_file}")
    
    def _write_enhanced_csv(self, mappings, column_info, output_file: str):
        """Write the enhanced CSV file"""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)# Write header
            writer.writerow([
                "Frontend_File",
                "Frontend_Function", 
                "HTTP_Method",
                "Frontend_URL",
                "Backend_File",
                "Backend_Function",
                "Backend_Route",
                "Database_Tables",
                "Stored_Procedures",
                "Flow_Calls",
                "Response_Model",
                "Response_Fields",
                "Nested_Fields",
                "Table_Column_Details"
            ])
            
            # Write data rows
            for mapping in mappings:
                self._write_mapping_row(writer, mapping, column_info)
    
    def _write_mapping_row(self, writer, mapping, column_info):
        """Write a single mapping row"""
        frontend = mapping.frontend
        backend = mapping.backend
        if not backend:
            # Unmatched frontend call
            writer.writerow([
                frontend.file, frontend.function_name, frontend.method, frontend.url_pattern,
                "", "", "", "", "", "", "", "", "", ""
            ])
            return# Extract information
        tables = [table.upper() for table in backend.tables] if backend.tables else []
        stored_procedures = backend.stored_procedures if backend.stored_procedures else []
        flow_calls = backend.flow_calls if backend.flow_calls else []
        
        # Response model information
        response_model = ""
        response_fields = []
        nested_fields = []
        
        if backend.response_model_info:
            response_model = (backend.response_model_info.get('response_model') or 
                            backend.response_model_info.get('return_type') or "")
            
            # Extract response fields
            columns = backend.response_model_info.get('columns', [])
            for col in columns:
                if isinstance(col, dict) and 'name' in col:
                    response_fields.append(col['name'])
            
            # Extract nested fields
            nested_columns = backend.response_model_info.get('nested_columns', [])
            for nested in nested_columns:
                if isinstance(nested, dict):
                    nested_name = nested.get('name', '')
                    nested_field_list = nested.get('nested_fields', [])
                    for field in nested_field_list:
                        if isinstance(field, dict) and 'name' in field:
                            nested_fields.append(f"{nested_name}.{field['name']}")
        
        # Table-column details
        table_column_details = []
        
        for table in tables:
            columns = column_info['table_columns'].get(table, [])
            if columns:
                table_column_details.append(f"{table}:[{','.join(columns)}]")
            else:
                table_column_details.append(f"{table}:[unknown]")# Write row
        writer.writerow([
            frontend.file,
            frontend.function_name,
            frontend.method,
            frontend.url_pattern,
            backend.file,
            backend.function_name,
            backend.route_pattern,
            ";".join(tables),
            # ",".join(stored_procedures),
            ",".join([sp.upper() for sp in stored_procedures]),
            ",".join(flow_calls),
            # ",".join([fc.upper() for fc in flow_calls]),
            response_model,
            ",".join(response_fields),
            ",".join(nested_fields),
            ";".join(table_column_details)
        ])
    
    
    
    def _print_summary(self, output_file: str, column_info: Dict[str, Any]):
        """Print summary statistics"""
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            total_rows = len(rows)
            matched = len([r for r in rows if r['Backend_Function']])
            with_tables = len([r for r in rows if r['Database_Tables']])
            with_columns = len([r for r in rows if r['Table_Column_Details'] and r['Table_Column_Details'] != ''])
            with_stored_procs = len([r for r in rows if r['Stored_Procedures']])
            with_flow_calls = len([r for r in rows if r['Flow_Calls']])
            
            logger.info("Enhanced CSV Summary:")
            logger.info(f"Total API mappings: {total_rows}")
            logger.info(f"Matched endpoints: {matched} ({matched/total_rows*100:.1f}%)")
            logger.info(f"With database tables: {with_tables} ({with_tables/total_rows*100:.1f}%)")
            logger.info(f"With stored procedures: {with_stored_procs} ({with_stored_procs/total_rows*100:.1f}%)")
            logger.info(f"With flow calls: {with_flow_calls} ({with_flow_calls/total_rows*100:.1f}%)")
            logger.info(f"With column details: {with_columns} ({with_columns/total_rows*100:.1f}%)")
            logger.info(f"Total tables analyzed: {len(column_info['table_columns'])}")
            
            # Show sample table mappings
            logger.info("Sample table-column mappings:")
            count = 0
            for table, columns in column_info['table_columns'].items():
                if count < 5:
                    columns_preview = ", ".join(columns[:5])
                    if len(columns) > 5:
                        columns_preview += f" (+{len(columns)-5} more)"
                    logger.info(f"  {table}: {columns_preview}")
                    count += 1
                    
        except Exception as e:
            logger.error(f"Error generating summary: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate enhanced CSV with table-column relationships")
    parser.add_argument("--frontend", default="guided-workflow", help="Frontend project path")
    parser.add_argument("--backend", default="guided-workflow-backend", help="Backend project path")
    parser.add_argument("--output", default="action_table_column_mapping.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    # Generate the enhanced CSV
    generator = EnhancedCSVGenerator(args.frontend, args.backend)
    generator.generate_enhanced_csv(args.output)

if __name__ == "__main__":
    main()