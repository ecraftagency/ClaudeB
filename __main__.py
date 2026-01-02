"""
Entry point for running pg_diagnose as a module.

Usage:
    python -m pg_diagnose -h localhost -d mydb
"""

import sys
from pathlib import Path

# Add parent directory to path if running directly
if __package__ is None or __package__ == '':
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pg_diagnose.cli import main
else:
    from .cli import main

if __name__ == "__main__":
    main()
