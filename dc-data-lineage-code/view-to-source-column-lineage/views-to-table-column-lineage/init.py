"""
DDL Column Lineage Analyzer Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Make main components available at package level
from .integrated_parser import CompleteIntegratedParser
from .database_connection import SnowflakeConnection

__all__ = [
    'CompleteIntegratedParser',
    'SnowflakeConnection'
]