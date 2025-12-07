#!/usr/bin/env python3
"""
Script to map primary keys from tables.csv to table_metadata.csv
This script adds a new column 'table_primary_key' to table_metadata.csv
that contains the primary key information from tables.csv
"""

import pandas as pd
import sys

def main():
    try:
        # Read the CSV files
        print("Reading tables.csv...")
        tables_df = pd.read_csv('tables.csv', header=None, names=['table_id', 'table_name'])
        
        print("Reading table_metadata.csv...")
        metadata_df = pd.read_csv('table_metadata.csv')
        
        print(f"Tables loaded: {len(tables_df)} rows")
        print(f"Metadata loaded: {len(metadata_df)} rows")
        
        # Display sample data to understand structure
        print("\nSample from tables.csv:")
        print(tables_df.head())
        
        print("\nSample from table_metadata.csv:")
        print(metadata_df.head())
        
        print("\nColumns in table_metadata.csv:")
        print(metadata_df.columns.tolist())
        
        # Function to extract table name from different formats
        def extract_table_name(full_name):
            """
            Extract table name from different formats:
            - database_name.schema_name.table_name -> table_name
            - schema_name.table_name -> table_name
            - table_name -> table_name
            """
            if pd.isna(full_name) or full_name == '':
                return full_name
            # Split by '.' and take the last part
            return str(full_name).split('.')[-1]
        
        # Extract table names from tables.csv for mapping
        print("\nExtracting table names from tables.csv formats...")
        tables_df['extracted_table_name'] = tables_df['table_name'].apply(extract_table_name)
        
        # Show some examples of the extraction
        print("\nExample table name extractions from tables.csv:")
        sample_extractions = tables_df[['table_name', 'extracted_table_name']].head(10)
        for _, row in sample_extractions.iterrows():
            if row['table_name'] != row['extracted_table_name']:
                print(f"  {row['table_name']} -> {row['extracted_table_name']}")
            else:
                print(f"  {row['table_name']} (no change)")
        
        # Create a mapping dictionary using extracted table names
        # Map extracted_table_name to table_id (primary key)
        table_pk_mapping = dict(zip(tables_df['extracted_table_name'], tables_df['table_id']))
        
        print(f"\nCreated mapping for {len(table_pk_mapping)} tables")
        
        # Check for duplicate extracted table names (potential conflicts)
        duplicate_tables = tables_df['extracted_table_name'].value_counts()
        duplicates = duplicate_tables[duplicate_tables > 1]
        if len(duplicates) > 0:
            print(f"\nWarning: Found {len(duplicates)} table names that appear multiple times after extraction:")
            for table_name, count in duplicates.head(5).items():
                print(f"  {table_name}: {count} occurrences")
            print("  Note: Only the last occurrence will be used in mapping")
        
        # Add the new column to metadata_df by mapping table_name to table_id
        metadata_df['table_primary_key'] = metadata_df['table_name'].map(table_pk_mapping)
        
        # Check for any unmapped tables
        unmapped_count = metadata_df['table_primary_key'].isna().sum()
        if unmapped_count > 0:
            print(f"\nWarning: {unmapped_count} rows could not be mapped to primary keys")
            unmapped_tables = metadata_df[metadata_df['table_primary_key'].isna()]['table_name'].unique()
            print("Unmapped tables:", unmapped_tables[:10])  # Show first 10
        
        # Display sample of the result
        print("\nSample of updated metadata with new column:")
        sample_cols = ['table_name', 'table_primary_key', 'column_names', 'type']
        if all(col in metadata_df.columns for col in sample_cols):
            print(metadata_df[sample_cols].head(10))
        
        # Save the updated metadata to a new file
        output_file = 'table_metadata_with_pk.csv'
        metadata_df.to_csv(output_file, index=False)
        print(f"\nUpdated metadata saved to: {output_file}")
        
        # Display summary statistics
        print(f"\nSummary:")
        print(f"- Total metadata rows: {len(metadata_df)}")
        print(f"- Successfully mapped: {len(metadata_df) - unmapped_count}")
        print(f"- Unmapped: {unmapped_count}")
        print(f"- Unique tables in metadata: {metadata_df['table_name'].nunique()}")
        print(f"- Unique tables in tables.csv: {len(tables_df)}")
        
        # Show some examples of the mapping
        print(f"\nExample mappings (extracted table name -> Primary Key):")
        for i, (table_name, pk) in enumerate(list(table_pk_mapping.items())[:5]):
            print(f"  {table_name} -> Primary Key: {pk}")
        
        # Show original to extracted mapping examples
        print(f"\nExample original to extracted table name mappings:")
        mapping_examples = tables_df[['table_name', 'extracted_table_name', 'table_id']].head(5)
        for _, row in mapping_examples.iterrows():
            print(f"  {row['table_name']} -> {row['extracted_table_name']} (ID: {row['table_id']})")
            
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

    