#!/usr/bin/env python3
"""
Script to map primary keys from views.csv to view_metadata.csv
This script adds a new column 'view_primary_key' to view_metadata.csv
that contains the primary key information from views.csv
"""

import pandas as pd
import sys

def main():
    try:
        # Read the CSV files
        print("Reading views.csv...")
        views_df = pd.read_csv('views.csv', header=None, names=['view_id', 'view_name'])
        
        print("Reading view_metadata.csv...")
        metadata_df = pd.read_csv('view_metadata.csv')
        
        print(f"Views loaded: {len(views_df)} rows")
        print(f"Metadata loaded: {len(metadata_df)} rows")
        
        # Display sample data to understand structure
        print("\nSample from views.csv:")
        print(views_df.head())
        
        print("\nSample from view_metadata.csv:")
        print(metadata_df.head())
        
        print("\nColumns in view_metadata.csv:")
        print(metadata_df.columns.tolist())
        
        # Create a mapping dictionary from views.csv
        # Map view_name to view_id (which appears to be the primary key)
        view_pk_mapping = dict(zip(views_df['view_name'], views_df['view_id']))
        
        print(f"\nCreated mapping for {len(view_pk_mapping)} views")
        
        # Add the new column to metadata_df by mapping table_name to view_id
        metadata_df['view_primary_key'] = metadata_df['table_name'].map(view_pk_mapping)
        
        # Check for any unmapped views
        unmapped_count = metadata_df['view_primary_key'].isna().sum()
        if unmapped_count > 0:
            print(f"\nWarning: {unmapped_count} rows could not be mapped to primary keys")
            unmapped_views = metadata_df[metadata_df['view_primary_key'].isna()]['table_name'].unique()
            print("Unmapped views:", unmapped_views[:10])  # Show first 10
        
        # Display sample of the result
        print("\nSample of updated metadata with new column:")
        sample_cols = ['table_name', 'view_primary_key', 'column_names', 'type']
        if all(col in metadata_df.columns for col in sample_cols):
            print(metadata_df[sample_cols].head(10))
        
        # Save the updated metadata to a new file
        output_file = 'view_metadata_with_pk.csv'
        metadata_df.to_csv(output_file, index=False)
        print(f"\nUpdated metadata saved to: {output_file}")
        
        # Display summary statistics
        print(f"\nSummary:")
        print(f"- Total metadata rows: {len(metadata_df)}")
        print(f"- Successfully mapped: {len(metadata_df) - unmapped_count}")
        print(f"- Unmapped: {unmapped_count}")
        print(f"- Unique views in metadata: {metadata_df['table_name'].nunique()}")
        print(f"- Unique views in views.csv: {len(views_df)}")
        
        # Show some examples of the mapping
        print(f"\nExample mappings:")
        for i, (view_name, pk) in enumerate(list(view_pk_mapping.items())[:5]):
            print(f"  {view_name} -> Primary Key: {pk}")
            
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()