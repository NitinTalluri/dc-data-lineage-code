#!/usr/bin/env python3
"""
This script to search for stored procedure references in prefect_repos directory cloned using clone_prefect_repos.py.
Connects to Snowflake to get SP list automatically and find the references then the sql is used to relete it with existing tables in snowflake.
"""
import os
import re
import csv
import time
import logging
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from sqlalchemy import create_engine, text

# Import Snowflake connection utilities
from common import sec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sp_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Snowflake Connection Management ---
def get_correct_schema(env: str) -> str:
    """Get the correct schema based on environment."""
    if env == 'prod':
        return 'CPS_DSCI_API'
    else:
        return 'CPS_DSCI_BR'

def check_env(env: str) -> str:
    """Check and return the correct connection name for environment."""
    if env == "dev":
        return "dev_cps_dsci_etl_svc"
    elif env == "stage":
        return "stg_cps_dsci_etl_svc"
    elif env == "prod":
        return "prd_cps_dsci_etl_svc"
    else:
        return env

def create_sf_connection_engine(sf_env: str = 'prod'):
    """Create Snowflake connection engine using existing strategy."""
    try:
        cn = check_env(sf_env)
        correct_schema = get_correct_schema(sf_env)
        engine = create_engine(
            sec.get_sf_pw(cn, 'CPS_DSCI_ETL_EXT2_WH', correct_schema)
        )
        logger.info(f"Successfully created Snowflake connection for {sf_env} environment")
        return engine
    except Exception as e:
        logger.error(f"Failed to create Snowflake connection: {e}")
        raise

def fetch_stored_procedure_names(sf_env: str = 'prod') -> List[str]:
    """Fetch stored procedure names from Snowflake."""
    engine = create_sf_connection_engine(sf_env)
    sql = """
    SELECT DISTINCT
        procedure_name
    FROM CPS_DB.INFORMATION_SCHEMA.PROCEDURES
    WHERE procedure_schema LIKE '%CPS_DSCI_API%'
    ORDER BY procedure_name
    """
    
    try:
        with engine.connect() as connection:
            result = pd.read_sql(text(sql), connection)
            
            # Handle case-insensitive column access
            procedure_names = []
            if len(result.columns) > 0:
                # Get the first column (should be procedure_name)
                first_col = result.columns[0]
                procedure_names = result[first_col].tolist()
            
            logger.info(f"Fetched {len(procedure_names)} stored procedure names from Snowflake")
            return procedure_names
            
    except Exception as e:
        logger.error(f"Error fetching stored procedure names: {e}")
        raise
    finally:
        engine.dispose()

def extract_repo_name(file_path: str) -> str:
    """Extract repository name from file path."""
    path_parts = Path(file_path).parts
    if len(path_parts) >= 2 and path_parts[0] == 'prefect_repos':
        return path_parts[1]
    return "unknown"

def search_in_file(file_path: str, sp_names: List[str]) -> List[Tuple[str, str, str]]:
    """Search for SP names in a single file. Returns list of (sp_name, file_path, repo_name)."""
    results = []
    
    try:
        # Skip files in ignored repositories
        ignored_repos = {'guided-workflow-backend', 'dc-data-linage-code'}
        repo_name = extract_repo_name(file_path)
        if repo_name in ignored_repos:
            return results
        
        # Skip .sql files
        if file_path.lower().endswith('.sql'):
            return results
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            logger.debug(f"Warning: Could not read file {file_path} with any encoding")
            return results
        
        # Convert content to lowercase for case-insensitive search
        content_lower = content.lower()
        
        # Search for each SP name
        for sp_name in sp_names:
            sp_name_lower = sp_name.lower()
            
            # Use word boundaries to avoid partial matches
            # Look for the SP name as a whole word
            pattern = r'\b' + re.escape(sp_name_lower) + r'\b'
            
            if re.search(pattern, content_lower):
                results.append((sp_name, file_path, repo_name))
    
    except Exception as e:
        logger.debug(f"Error processing file {file_path}: {e}")
    
    return results

def get_searchable_files(root_dir: str) -> List[str]:
    """Get list of files to search through."""
    searchable_extensions = {
        '.py', '.json', 
        '.js', '.ts', '.java', '.scala', '.r', '.sh',
        '.cfg', '.conf', '.ini', '.properties'
    }
    
    files_to_search = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories and common non-source directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {
            '__pycache__', 'node_modules', '.git', '.venv', 'venv', 
            'env', '.pytest_cache', 'dist', 'build'
        }]
        
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = Path(file).suffix.lower()
            
            # Include files with searchable extensions or no extension
            if file_ext in searchable_extensions or file_ext == '':
                # Skip binary-like files
                if not any(skip in file.lower() for skip in [
                    '.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', 
                    '.zip', '.tar', '.gz', '.jpg', '.png', '.gif', '.pdf'
                ]):
                    files_to_search.append(file_path)
    
    return files_to_search

def search_single_file(args):
    """Search a single file for SP references - wrapper for parallel execution."""
    file_path, sp_names = args
    return search_in_file(file_path, sp_names)

def search_files_parallel(files_to_search: List[str], sp_names: List[str], max_workers: int = 8) -> List[Tuple[str, str, str]]:
    """Search files in parallel for better performance."""
    
    all_results = []
    completed_count = 0
    
    # Prepare arguments for parallel processing
    search_args = [(file_path, sp_names) for file_path in files_to_search]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(search_single_file, args): args[0] 
            for args in search_args
        }
        
        # Process completed tasks
        try:
            for future in as_completed(future_to_file, timeout=300):
                file_path = future_to_file[future]
                completed_count += 1
                
                try:
                    results = future.result(timeout=10)
                    all_results.extend(results)
                    
                    # if completed_count % 100 == 0:
                    #     pass
                    #     # logger.info(f"Processed {completed_count}/{len(files_to_search)} files...")
                        
                except Exception as e:
                    logger.debug(f"Error processing file {file_path}: {e}")
                    
        except KeyboardInterrupt:
            logger.warning("Search interrupted by user")
            # Cancel remaining futures
            for future in future_to_file:
                future.cancel()
    
    return all_results

def save_to_csv(results: List[Tuple[str, str, str]], output_file: str):
    """Save results to CSV file."""
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['SP_NAME', 'FILE_PATH', 'REPO_NAME'])
            
            # Write data rows
            for sp_name, file_path, repo_name in results:
                # Convert file path to use forward slashes for consistency
                normalized_path = file_path.replace(os.sep, '/')
                writer.writerow([sp_name, normalized_path, repo_name])
        
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")
        raise

def main():
    """Main function to orchestrate the search with Snowflake integration."""
    start_time = time.time()
    
    # Configuration
    sf_env = 'prod'  # Snowflake environment
    search_dir = 'prefect_repos'
    output_csv = 'sp_search_results.csv'
    max_workers = 8  # Parallel processing workers
    
    try:
        logger.info("=== STORED PROCEDURE REFERENCE SEARCH STARTED ===")
        logger.info(f"Snowflake Environment: {sf_env}")
        logger.info(f"Output File: {output_csv}")
        
        # Check if directories exist
        if not os.path.exists(search_dir):
            logger.error(f"Error: {search_dir} directory not found")
            return
        
        # Fetch SP names from Snowflake
        # logger.info("Fetching stored procedure names from Snowflake...")
        sp_names = fetch_stored_procedure_names(sf_env)
        if not sp_names:
            logger.error("No stored procedure names found in Snowflake.")
            return
        
        files_to_search = get_searchable_files(search_dir)
        
        # Search for SP references in parallel
        all_results = search_files_parallel(files_to_search, sp_names, max_workers)
        
        # Sort results by SP name, then by file path
        all_results.sort(key=lambda x: (x[0], x[1]))
        
        # Save results to CSV
        if all_results:
            save_to_csv(all_results, output_csv)
        
        # Summary by SP
        sp_counts = {}
        for sp_name, _, _ in all_results:
            sp_counts[sp_name] = sp_counts.get(sp_name, 0) + 1
        
        if sp_counts:
            print(f"\nSummary by SP (found {len(sp_counts)} out of {len(sp_names)} SPs with references):")
            for sp_name in sorted(sp_counts.keys()):
                print(f"  {sp_name}: {sp_counts[sp_name]} reference(s)")
            
            # Show SPs not found
            not_found = set(sp_names) - set(sp_counts.keys())
            if not_found:
                print(f"\nSPs not found in code ({len(not_found)}):")
                for sp_name in sorted(not_found):
                    print(f"  {sp_name}")
        
        # Execution time
        elapsed_time = time.time() - start_time
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
        logger.info("=== SEARCH COMPLETE ===")
        
    except KeyboardInterrupt:
        logger.warning("Search interrupted by user")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise

if __name__ == "__main__":
    main()