#!/usr/bin/env python3
"""
Database connection module for Snowflake
"""

import pandas as pd
from sqlalchemy import create_engine, text
from common import sec


class SnowflakeConnection:
    """Handles Snowflake database connections and DDL retrieval"""
    
    def __init__(self, sf_env='prod'):
        self.sf_env = sf_env
        self.engine = None
        
    def get_correct_schema(self, env: str) -> str:
        """Get the correct schema based on environment."""
        if env == 'prod':
            return 'CPS_DSCI_API'
        else:
            return 'CPS_DSCI_BR'

    def check_env(self, env: str) -> str:
        """Check and return the correct connection name for environment."""
        if env == "dev":
            cn = "dev_cps_dsci_etl_svc"
        elif env == "stage":
            cn = "stg_cps_dsci_etl_svc"
        elif env == "prod":
            cn = "prd_cps_dsci_etl_svc"
        else:
            cn = env
        return cn

    def create_connection(self):
        """Create Snowflake connection engine."""
        try:
            cn = self.check_env(self.sf_env)
            correct_schema = self.get_correct_schema(self.sf_env)
            self.engine = create_engine(
                sec.get_sf_pw(cn, 'CPS_DSCI_ETL_EXT2_WH', correct_schema)
            )
            return self.engine
            
        except Exception as e:
            print(f"Failed to create Snowflake connection: {e}")
            raise

    def get_ddl_for_view(self, view_name: str) -> str:
        """Get DDL for a specific view using GET_DDL function."""
        if not self.engine:
            self.create_connection()
            
        try:
            sql = f"SELECT GET_DDL('VIEW', '{view_name}') as ddl"
            
            with self.engine.connect() as connection:
                result = pd.read_sql(text(sql), connection)
            
            if not result.empty:
                return result.iloc[0]['ddl']
            else:
                return None
                
        except Exception as e:
            print(f"Error getting DDL for view {view_name}: {e}")
            return None

    def find_view_in_schemas(self, view_name: str) -> str:
        """Find which schema contains the view."""
        if not self.engine:
            self.create_connection()
            
        # First check CPS_DSCI_API
        try:
            sql = f"""
            SELECT table_schema 
            FROM information_schema.views 
            WHERE table_name = '{view_name}' 
            AND table_schema = 'CPS_DSCI_API'
            """
            
            with self.engine.connect() as connection:
                result = pd.read_sql(text(sql), connection)
            
            if not result.empty:
                return 'CPS_DSCI_API'
                
        except Exception:
            pass
        
        # If not found, check all schemas
        try:
            sql = f"""
            SELECT table_schema 
            FROM information_schema.views 
            WHERE table_name = '{view_name}'
            """
            
            with self.engine.connect() as connection:
                result = pd.read_sql(text(sql), connection)
            
            if not result.empty:
                return result.iloc[0]['table_schema']
                
        except Exception:
            pass
            
        return None

    def get_qualified_ddl(self, view_name: str) -> str:
        """Get DDL with proper schema qualification."""
        schema = self.find_view_in_schemas(view_name)
        
        if schema:
            qualified_name = f"{schema}.{view_name}"
            return self.get_ddl_for_view(qualified_name)
        else:
            # Try without schema qualification
            return self.get_ddl_for_view(view_name)