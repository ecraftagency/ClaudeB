"""
Test scenarios for pg_diagnose integration testing.

Each scenario is a text file containing a sequence of commands
to simulate user input during automated testing.

Scenarios:
- happy_path.txt: Complete tuning flow to target achieved
- pause_resume.txt: Pause a session for later resume
- resume_session.txt: Resume a previously paused session
- error_recovery.txt: Handle 0 TPS and recovery
- early_exit.txt: Clean exit at various stages
- custom_input.txt: Test /custom command for DBA guidance
- snapshot_basic.txt: Auto-snapshot and restore to initial
- snapshot_checkpoint.txt: Manual checkpoints and restore
- snapshot_compare.txt: Compare snapshots to see differences

Usage:
    python -m pg_diagnose.cli \\
        --test-mode \\
        --mock-benchmark \\
        --mock-ai \\
        --test-scenario balanced_tps \\
        --input-file tests/scenarios/happy_path.txt \\
        --output-json \\
        -H localhost -p 6432 -d postgres
"""

from pathlib import Path

SCENARIOS_DIR = Path(__file__).parent

AVAILABLE_SCENARIOS = {
    # Core workflow scenarios
    'happy_path': SCENARIOS_DIR / 'happy_path.txt',
    'pause_resume': SCENARIOS_DIR / 'pause_resume.txt',
    'resume_session': SCENARIOS_DIR / 'resume_session.txt',
    'error_recovery': SCENARIOS_DIR / 'error_recovery.txt',
    'early_exit': SCENARIOS_DIR / 'early_exit.txt',
    'custom_input': SCENARIOS_DIR / 'custom_input.txt',
    # Snapshot/restore scenarios
    'snapshot_basic': SCENARIOS_DIR / 'snapshot_basic.txt',
    'snapshot_checkpoint': SCENARIOS_DIR / 'snapshot_checkpoint.txt',
    'snapshot_compare': SCENARIOS_DIR / 'snapshot_compare.txt',
}


def get_scenario_path(name: str) -> Path:
    """Get the path to a scenario file by name."""
    if name in AVAILABLE_SCENARIOS:
        return AVAILABLE_SCENARIOS[name]
    # Try as direct filename
    path = SCENARIOS_DIR / name
    if path.exists():
        return path
    path = SCENARIOS_DIR / f"{name}.txt"
    if path.exists():
        return path
    raise ValueError(f"Unknown scenario: {name}")


def list_scenarios() -> list:
    """List all available scenarios."""
    return list(AVAILABLE_SCENARIOS.keys())
