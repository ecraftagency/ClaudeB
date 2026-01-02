"""
Entry point for running pg_diagnose as a module.

Usage:
    python -m pg_diagnose -h localhost -d mydb
"""

from .cli import main

if __name__ == "__main__":
    main()
