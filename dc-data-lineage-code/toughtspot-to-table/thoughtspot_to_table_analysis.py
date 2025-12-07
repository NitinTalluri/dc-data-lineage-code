"""
ThoughtSpot Liveboard to tables Analysis
Generates CSV output with table-to-liveboard relationships details.
"""
import json
import logging
import pandas as pd
import csv
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine, text
import sys
import time
import urllib3
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Add both the dc-canvas-service directory and its src directory to path
sys.path.insert(1,"./dc-canvas-service")
sys.path.insert(1,"./dc-canvas-service/src")
from src.dc_canvas_service.services.thoughtspot.services import ThoughtSpotService
from src.dc_canvas_service.services.thoughtspot.exceptions import (
    TSSearchMetadataError,
    TSCredsNotFoundError,
    TSTokenFetchError
)

# Import common credentials module
from common import sec

logger = logging.getLogger(__name__)


class ThoughtSpotCSVAnalysisExtension:
    """
    Enhanced extension for generating CSV reports of table-to-liveboard relationships
    with detailed column usage information.
    """
    def __init__(self, thoughtspot_service: ThoughtSpotService, force_prod_urls: bool = True):
        self.ts_service = thoughtspot_service
        self.cache = {}  
        self.cache_ttl = 3600  
        self.force_prod_urls = force_prod_urls
        
        self.retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"]
        )
        
        # Disable SSL warnings for problematic connections
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        logger.info("ThoughtSpot CSV Analysis Extension initialized")
        if self.force_prod_urls:
            logger.info("Configured to force production URLs")
    
    def get_correct_schema(self, env: str) -> str:
        """Get the correct schema based on environment."""
        if env == 'prod':
            return 'CPS_DSCI_API'
        else:
            return 'CPS_DSCI_BR'
    
    def check_env(self, env: str) -> str:
        """Check and return the correct connection name for environment"""
        if env == "dev":
            cn = "dev_cps_dsci_etl_svc"
        elif env == "stage":
            cn = "stg_cps_dsci_etl_svc"
        elif env == "prod":
            cn = "prd_cps_dsci_etl_svc"
        else:
            cn = env
        return cn
    
    def create_sf_connection_engine(self, sf_env: str):
        """Create Snowflake connection engine using existing strategy."""
        try:
            cn = self.check_env(sf_env)
            correct_schema = self.get_correct_schema(sf_env)
            
            logger.info(f"Attempting to connect to {sf_env} environment using connection '{cn}' and schema '{correct_schema}'")
            
            # Get connection string
            connection_string = sec.get_sf_pw(cn, 'CPS_DSCI_ETL_EXT2_WH', correct_schema)
            
            # Log connection details 
            if 'ciscodev' in connection_string:
                logger.info(f"Using DEV Snowflake instance: cisco.us-east-1")
            elif 'cisco.us-east-1' in connection_string:
                logger.info(f"Using PROD Snowflake instance: cisco.us-east-1")
            
            engine = create_engine(connection_string)
            return engine
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to create Snowflake connection for {sf_env}: {error_msg}")
            raise
    
    def get_all_views_and_tables(self, sf_env: str = 'prod', include_views: bool = True) -> pd.DataFrame:
        """Get all views and tables from CPS_DSCI schemas."""
        
        schemas = ['CPS_DSCI_API', 'CPS_DSCI_BR']
        
        # Build table type filter
        table_types = "'BASE TABLE'"
        if include_views:
            table_types = "'BASE TABLE', 'VIEW'"
        
        # Query to get all tables and views from both schemas
        schema_list = "', '".join(schemas)
        sql_query = f"""
        SELECT TABLE_NAME, TABLE_SCHEMA, TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA IN ('{schema_list}') 
        AND TABLE_TYPE IN ({table_types})
        AND TABLE_NAME NOT LIKE 'CANVAS_%'
        ORDER BY TABLE_SCHEMA, TABLE_NAME
        """
        
        logger.info(f"Getting tables/views from CPS_DSCI schemas for {sf_env} environment...")
        logger.info(f"Target schemas: {schemas}, Include views: {include_views}")
        
        try:
            local_engine = self.create_sf_connection_engine(sf_env)
            with local_engine.connect() as connection:
                df = pd.read_sql(text(sql_query), connection)
            
            logger.info(f"Found {len(df)} total objects")
            return df
        except Exception as e:
            logger.error(f"Error getting views/tables: {e}")
            raise
    def get_table_thoughtspot_data(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive ThoughtSpot data for a single table.
        
        Args:
            table_name (str): Name of the table to analyze (can be schema.table_name or just table_name)
            
        Returns:
            Dict containing ThoughtSpot metadata and dependent objects
        """
        cache_key = f"ts_data_{table_name}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.debug(f"Returning cached ThoughtSpot data for {table_name}")
            return self.cache[cache_key]['data']
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Extract just the table name for ThoughtSpot search
                search_table_name = table_name.split('.')[-1] if '.' in table_name else table_name
                
                # Get dependent objects from ThoughtSpot with retry logic
                dependent_objects = self.ts_service.search_table_dependent_objects(search_table_name)
                
                # Get detailed object information
                object_details = self._get_object_details(dependent_objects)
                
                # Extract liveboards
                liveboards = {}
                liveboard_types = ['LIVEBOARD', 'PINBOARD']
                
                for guid, object_type in dependent_objects.items():
                    if object_type in liveboard_types:
                        details = object_details.get(guid, {})
                        
                        liveboards[guid] = {
                            'guid': guid,
                            'type': object_type,
                            'name': details.get('name', f'Liveboard {guid[:8]}...'),
                            'description': details.get('description', ''),
                            'url': details.get('url', ''),
                            'created_by': details.get('created_by', ''),
                            'created_date': details.get('created_date', ''),
                            'modified_date': details.get('modified_date', ''),
                            'tags': details.get('tags', [])
                        }
                
                # Prepare result
                result = {
                    'has_thoughtspot_objects': len(dependent_objects) > 0,
                    'dependent_objects': dependent_objects,
                    'object_details': object_details,
                    'liveboards': liveboards,
                    'last_updated': '2024-01-01T00:00:00Z', 
                    'error': None
                }
                
                # Cache the result
                self._cache_result(cache_key, result)
                
                logger.debug(f"Successfully retrieved ThoughtSpot data for {table_name}: {len(liveboards)} liveboards")
                return result
                
            except (TSSearchMetadataError, TSCredsNotFoundError, TSTokenFetchError) as e:
                logger.error(f"ThoughtSpot error for table {table_name}: {str(e)}")
                error_result = {
                    'has_thoughtspot_objects': False,
                    'dependent_objects': {},
                    'object_details': {},
                    'liveboards': {},
                    'last_updated': '2024-01-01T00:00:00Z',
                    'error': str(e)
                }
                return error_result
            
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {table_name}: {str(e)}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  
                    continue
                else:
                    logger.error(f"All attempts failed for table {table_name}: {str(e)}")
                    error_result = {
                        'has_thoughtspot_objects': False,
                        'dependent_objects': {},
                        'object_details': {},
                        'liveboards': {},
                        'last_updated': '2024-01-01T00:00:00Z',
                        'error': f"Connection error after {max_retries} attempts: {str(e)}"
                    }
                    return error_result
    
    def get_all_tables_and_views_optimized(self, sf_env: str = 'prod', 
                                          table_pattern: str = None, 
                                          include_views: bool = True) -> List[Dict[str, str]]:
        """
        Get ALL table and view names with their schema and type information.
        
        Returns:
            List of dicts with table_name, schema, and type information
        """
        try:
            # Get DataFrame using proven method
            df = self.get_all_views_and_tables(sf_env, include_views)
            
            if df.empty:
                logger.warning(f"No tables/views found for environment {sf_env}")
                return []
            
            # Apply table pattern filter if specified
            if table_pattern:
                df = df[df['TABLE_NAME'].str.contains(table_pattern, case=False, na=False)]
                logger.info(f"Applied pattern filter '{table_pattern}': {len(df)} objects remaining")
            
            # Find column names dynamically
            schema_col = None
            type_col = None
            name_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'schema' in col_lower and 'table' in col_lower:
                    schema_col = col
                elif 'type' in col_lower and 'table' in col_lower:
                    type_col = col
                elif 'name' in col_lower and 'table' in col_lower:
                    name_col = col
            
            if not all([schema_col, type_col, name_col]):
                logger.error(f"Missing required columns. Available: {list(df.columns)}")
                return []
            
            # Create table information list
            table_info_list = []
            for _, row in df.iterrows():
                table_name = row[name_col]
                table_schema = row[schema_col]
                table_type = row[type_col]
                
                table_info = {
                    'full_name': f"{table_schema}.{table_name}",
                    'table_name': table_name,
                    'schema': table_schema,
                    'type': table_type
                }
                table_info_list.append(table_info)
            
            logger.info(f"Found {len(table_info_list)} tables/views")
            return table_info_list
            
        except Exception as e:
            logger.error(f"Error getting tables/views for {sf_env}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def generate_csv_analysis(self, sf_env: str = 'prod',
                        table_pattern: str = None,
                        max_workers: int = 5,
                        include_views: bool = True,
                        output_file: str = None) -> str:
        """
        Generate CSV analysis of table-to-liveboard relationships.
        
        Args:
            sf_env: Environment ('prod', 'dev', 'stage')
            table_pattern: Optional pattern to filter table names
            max_workers: Maximum concurrent workers
            include_views: Include views in addition to base tables
            output_file: Output CSV file path (auto-generated if None)
            
        Returns:
            Path to the generated CSV file
        """
        try:
            # Auto-generate output file name if not provided
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f'{sf_env}_liveboard_analysis_{timestamp}.csv'
            
            logger.info(f"Starting CSV analysis for {sf_env} environment")
            
            # Get ALL tables AND views with metadata
            all_tables_info = self.get_all_tables_and_views_optimized(sf_env, table_pattern, include_views)
            
            logger.info(f"Found {len(all_tables_info)} total tables/views for {sf_env} environment")
            
            if not all_tables_info:
                logger.warning(f"No tables found for {sf_env} environment!")
                # Create empty CSV with headers
                self._create_empty_csv(output_file)
                return output_file
            
            # Prepare CSV data
            csv_rows = []
            
            # Process tables in batches for better performance
            batch_size = max_workers * 2
            total_batches = (len(all_tables_info) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(all_tables_info))
                batch_tables = all_tables_info[start_idx:end_idx]
                
                logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_tables)} tables)")# Get ThoughtSpot data for batch
                batch_results = self._process_table_batch([t['full_name'] for t in batch_tables], max_workers)
                
                # Process results and add to CSV rows
                for table_info in batch_tables:
                    full_name = table_info['full_name']
                    ts_data = batch_results.get(full_name, {})
                    
                    liveboards = ts_data.get('liveboards', {})
                    
                    if liveboards:
                        # Create a row for each liveboard
                        for liveboard_guid, liveboard_info in liveboards.items():
                            csv_row = {
                                'Table_name': table_info['table_name'],
                                'Liveboard_name': liveboard_info['name'],
                                'GUID': liveboard_guid,
                                'Schema': table_info['schema'],
                                'Type': table_info['type']
                            }
                            csv_rows.append(csv_row)
                            
                            logger.debug(f"Added row: {table_info['table_name']} -> {liveboard_info['name']}")
            
            # Write to CSV
            self._write_csv_file(csv_rows, output_file)
            
            logger.info(f"CSV analysis completed: {len(csv_rows)} relationships found")
            logger.info(f"Output saved to: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error in CSV analysis for {sf_env}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    def _process_table_batch(self, table_names: List[str], max_workers: int) -> Dict[str, Dict[str, Any]]:
        """Process a batch of tables concurrently with improved error handling."""
        results = {}
        
        # Reduce max_workers to avoid overwhelming the API
        effective_max_workers = min(max_workers, 5)
        
        with ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
            future_to_table = {
                executor.submit(self.get_table_thoughtspot_data, table_name): table_name
                for table_name in table_names
            }
            
            for future in as_completed(future_to_table):
                table_name = future_to_table[future]
                try:
                    results[table_name] = future.result(timeout=30)  # Add timeout
                except Exception as e:
                    logger.warning(f"Failed to process {table_name}: {str(e)}")
                    results[table_name] = {
                        'has_thoughtspot_objects': False,
                        'liveboards': {},
                        'error': str(e)
                    }
        
        return results
    def _write_csv_file(self, csv_rows: List[Dict[str, str]], output_file: str) -> None:
        """Write CSV rows to file."""
        if not csv_rows:
            self._create_empty_csv(output_file)
            return
        
        fieldnames = ['Table_name', 'Liveboard_name', 'GUID', 'Schema', 'Type']
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            
            logger.info(f"Successfully wrote {len(csv_rows)} rows to {output_file}")
            
        except Exception as e:
            logger.error(f"Error writing CSV file {output_file}: {str(e)}")
            raise
    def _create_empty_csv(self, output_file: str) -> None:
        """Create empty CSV with headers only."""
        fieldnames = ['Table_name', 'Liveboard_name', 'GUID', 'Schema', 'Type']
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        logger.info(f"Created empty CSV file: {output_file}")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key]['timestamp']
        current_time = time.time()
        
        return (current_time - cached_time) < self.cache_ttl
    
    def _cache_result(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache the result with timestamp."""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': time.time() 
        }
    
    def _get_object_details(self, dependent_objects: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Get detailed information for ThoughtSpot objects using existing search methods."""
        object_details = {}

        if not dependent_objects:
            return object_details
        
        try:
            # Use the existing search_tmls method to get metadata for all objects at once
            guids_list = list(dependent_objects.keys())
            search_results = self.ts_service.search_tmls(metadata_GUIDs=guids_list)
            
            for guid, object_type in dependent_objects.items():
                search_result = search_results.get(guid)
                
                if search_result:
                    # Extract metadata from search result
                    object_details[guid] = {
                        'name': search_result.get('metadata_name', f'Object {guid[:8]}...'),
                        'description': search_result.get('metadata_detail', {}).get('description', ''),
                        'url': self._generate_object_url(guid, object_type),
                        'created_by': search_result.get('metadata_detail', {}).get('author', {}).get('name', ''),
                        'created_date': search_result.get('metadata_detail', {}).get('created', ''),
                        'modified_date': search_result.get('metadata_detail', {}).get('modified', ''),
                        'tags': search_result.get('metadata_detail', {}).get('tags', []),
                        'type': object_type
                    }
                else:
                    # Fallback for objects that couldn't be found
                    object_details[guid] = {
                        'name': f'Object {guid[:8]}...',
                        'description': '',
                        'url': self._generate_object_url(guid, object_type),
                        'created_by': '',
                        'created_date': '',
                        'modified_date': '',
                        'tags': [],
                        'type': object_type
                    }
                    
        except Exception as e:
            logger.warning(f"Could not get detailed metadata for objects: {str(e)}")
            # Fallback: create basic details for all objects
            for guid, object_type in dependent_objects.items():
                object_details[guid] = {
                    'name': f'Object {guid[:8]}...',
                    'description': '',
                    'url': self._generate_object_url(guid, object_type),
                    'created_by': '',
                    'created_date': '',
                    'modified_date': '',
                    'tags': [],
                    'type': object_type
                }
        
        return object_details
    
    def _generate_object_url(self, guid: str, object_type: str) -> str:
        """Generate direct URL to object in ThoughtSpot."""
        base_url = self.ts_service.ts_table_base_link.replace('/data/tables', '')
        
        # Force production URL instead of dev URL if configured
        if self.force_prod_urls and 'cisco-dev.thoughtspot.cloud' in base_url:
            base_url = base_url.replace('cisco-dev.thoughtspot.cloud', 'cisco.thoughtspot.cloud')
        
        # Map object types to URL paths
        type_paths = {
            'LOGICAL_TABLE': '/data/tables',
            'WORKSHEET': '/data/worksheets', 
            'LIVEBOARD': '/pinboard',
            'ANSWER': '/saved-answer',
            'PINBOARD': '/pinboard',
            'VIEW': '/data/views'
        }
        
        path = type_paths.get(object_type, '/data/tables')
        return f"{base_url}{path}/{guid}"
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self.cache.clear()
        logger.info("ThoughtSpot analysis cache cleared")


# Utility functions for CSV analysis

def create_thoughtspot_csv_analysis_extension(settings, s3_service, force_prod_urls: bool = True) -> ThoughtSpotCSVAnalysisExtension:
    """
    Factory function to create ThoughtSpot CSV analysis extension.
    
    Args:
        settings: Application settings
        s3_service: S3 service instance
        force_prod_urls: Force production URLs instead of dev URLs (default: True)
        
    Returns:
        ThoughtSpotCSVAnalysisExtension instance
    """
    ts_service = ThoughtSpotService(settings, s3_service)
    return ThoughtSpotCSVAnalysisExtension(ts_service, force_prod_urls)


def run_csv_liveboard_analysis(settings, s3_service, 
                              sf_env: str = 'prod',
                              output_file: str = None,
                              table_pattern: str = None,
                              max_workers: int = 10,
                              force_prod_urls: bool = True,
                              include_views: bool = True) -> str:
    """
    Run CSV-focused liveboard analysis.
    
    Args:
        settings: Application settings
        s3_service: S3 service instance
        sf_env: Environment ('prod', 'dev', 'stage')
        output_file: Output CSV file path (auto-generated if None)
        table_pattern: Optional table name pattern filter
        max_workers: Maximum concurrent workers
        force_prod_urls: Force production URLs instead of dev URLs (default: True)
        include_views: Include views in addition to base tables (default: True)
        
    Returns:
        Path to the generated CSV file
    """
    try:
        # Create CSV analysis extension
        csv_extension = create_thoughtspot_csv_analysis_extension(settings, s3_service, force_prod_urls)
        
        logger.info(f"Starting CSV liveboard analysis for {sf_env} environment")
        
        # Run CSV analysis
        output_path = csv_extension.generate_csv_analysis(
            sf_env=sf_env,
            table_pattern=table_pattern,
            max_workers=max_workers,
            include_views=include_views,
            output_file=output_file
        )
        
        logger.info(f"CSV LIVEBOARD ANALYSIS COMPLETE:")
        logger.info(f"  - Environment: {sf_env}")
        logger.info(f"  - Output saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"CSV liveboard analysis failed for {sf_env}: {str(e)}")
        raise


# Example usage - CSV Liveboard Analysis
if __name__ == "__main__":
    try:
        from dc_canvas_service.common import Settings
        from dc_canvas_service.services.s3 import S3Service
        
        # Initialize services
        settings = Settings()
        s3_service = S3Service(settings)
        
        # Test environments
        environments = ['prod']
        
        for sf_env in environments:
            print(f"\n=== {sf_env.upper()} ENVIRONMENT CSV ANALYSIS ===")
            
            # Run CSV liveboard analysis
            print(f"Generating CSV analysis for {sf_env} environment...")
            
            output_file = run_csv_liveboard_analysis(
                settings=settings,
                s3_service=s3_service,
                sf_env=sf_env,
                max_workers=20,
                force_prod_urls=True,
                include_views=True
            )
            print(f"\n {sf_env.upper()} CSV ANALYSIS COMPLETE:")
            print(f"  - Environment: {sf_env}")
            print(f"  - CSV file generated: {output_file}")
            print(f"  - Format: Table_name, Liveboard_name, GUID, Schema, Type")
        
    except Exception as e:
        print(f"\n CSV liveboard analysis failed: {str(e)}")
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)