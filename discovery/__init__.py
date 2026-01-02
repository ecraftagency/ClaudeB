"""
Discovery module - Gathers context about database and system.

Components:
- SchemaScanner: Scans database schema (tables, indexes, stats)
- SystemScanner: Scans system hardware/OS configuration
- RuntimeScanner: Scans PostgreSQL runtime config
"""

from .schema import SchemaScanner
from .system import SystemScanner
from .runtime import RuntimeScanner

__all__ = ["SchemaScanner", "SystemScanner", "RuntimeScanner"]
