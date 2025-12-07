#!/usr/bin/env python3
"""
Main execution module for DDL analysis
Processes views from view_names.txt and generates comprehensive CSV output

Usage:
    uv run main.py
"""

import os
import sys
import pandas as pd

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_connection import SnowflakeConnection
from integrated_parser import IntegratedDDLParser


def load_view_names(filename='view_names.txt'):
    """Load view names from text file."""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        
        with open(file_path, 'r') as f:
            views = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(views)} view names from {filename}")
        return views
    except Exception as e:
        print(f"Error loading view names from {filename}: {e}")
        raise


def process_all_views(sf_env='prod'):
    """Process all views from view_names.txt and generate comprehensive analysis"""
    
    # Load view names
    view_names = load_view_names()
    
    # Create database connection
    db_connection = SnowflakeConnection(sf_env)
    
    # Create parser
    parser = IntegratedDDLParser()
    
    # Results list for CSV
    all_csv_rows = []
    
    print(f"Processing {len(view_names)} views...")
    
    for i, view_name in enumerate(view_names, 1):
        print(f"Processing {i}/{len(view_names)}: {view_name}")
        
        try:
            # Get DDL for the view
            ddl_text = db_connection.get_qualified_ddl(view_name)
            
            if not ddl_text:
                print(f"  Could not retrieve DDL for {view_name}")
                # Add error row
                all_csv_rows.append([
                    view_name.upper(),
                    'ERROR',
                    'ERROR',
                    'DDL_NOT_FOUND',
                    'DDL_NOT_FOUND',
                    'ERROR'
                ])
                continue
            
            print(f"  Retrieved DDL for {view_name}")
            
            # Analyze the DDL
            analysis = parser.analyze_ddl_statement(ddl_text)
            
            if 'error' in analysis:
                print(f"  Analysis error for {view_name}: {analysis['error']}")
                # Add error row
                all_csv_rows.append([
                    view_name.upper(),
                    'ERROR',
                    'ERROR',
                    'ANALYSIS_ERROR',
                    'ANALYSIS_ERROR',
                    'ERROR'
                ])
                continue
            
            # Generate CSV for this view
            csv_output = parser.generate_csv(analysis)
            
            # Parse CSV output and add to results
            csv_lines = csv_output.strip().split('\n')[1:]  # Skip header
            for line in csv_lines:
                if line.strip():
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 6:
                        all_csv_rows.append(parts)
            
            print(f"  Successfully processed {view_name} - {len(csv_lines)} columns")
            
        except Exception as e:
            print(f"  Error processing {view_name}: {e}")
            # Add error row
            all_csv_rows.append([
                view_name.upper(),
                'ERROR',
                'ERROR',
                'PROCESSING_ERROR',
                'PROCESSING_ERROR',
                'ERROR'
            ])
    
    return all_csv_rows


def save_results_to_csv(results, filename='column_lineage_analysis.csv'):
    """Save all results to a single CSV file"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        
        # Create DataFrame
        df = pd.DataFrame(results, columns=[
            'View_Name', 'View_Column', 'Column_Type', 
            'Source_Table', 'Source_Column', 'Expression_Type'
        ])
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        print(f"\nResults saved to: {file_path}")
        print(f"Total records: {len(df)}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Unique views processed: {df['View_Name'].nunique()}")
        print(f"  Total columns analyzed: {len(df)}")
        
        # Column type breakdown
        print(f"  Column types:")
        type_counts = df['Column_Type'].value_counts()
        for col_type, count in type_counts.items():
            print(f"    {col_type}: {count}")
        
        # Success rate calculation
        total_columns = len(df)
        error_columns = len(df[df['Column_Type'].isin(['ERROR', 'UNKNOWN'])])
        success_columns = total_columns - error_columns
        success_rate = (success_columns / total_columns * 100) if total_columns > 0 else 0
        
        print(f"  Success rate: {success_rate:.1f}%")
        
        return df
        
    except Exception as e:
        print(f"Error saving results: {e}")
        raise


def main():
    """Main execution function"""
    sf_env = 'prod'  # Change this to 'dev' or 'stage' as needed
    
    try:
        print("=== DDL COLUMN LINEAGE ANALYSIS ===")
        print(f"Environment: {sf_env}")
        print(f"Processing views from: view_names.txt")
        print(f"Usage: uv run main.py")
        
        # Process all views
        results = process_all_views(sf_env)
        
        # Save to CSV
        df = save_results_to_csv(results)
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Check 'column_lineage_analysis.csv' for complete results")
        
        # Show sample of results
        if len(df) > 0:
            print(f"\nSample results (first 10 rows):")
            print(df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"Script failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()