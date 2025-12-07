"""
DDL Column Lineage Analyzer Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Make main components available at package level
from .integrated_parser import IntegratedDDLParser
from .database_connection import SnowflakeConnection
from .csv_generator import CSVGenerator

__all__ = [
    'IntegratedDDLParser',
    'SnowflakeConnection', 
    'CSVGenerator'
]