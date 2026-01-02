# pg_diagnose Integration Test Plan

## Overview

This document defines the integration test strategy for pg_diagnose. The test suite is designed to be executed by an AI agent (Claude) acting as a test user, enabling automated verification of all tool functionality.

## Test Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TEST EXECUTION FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   Sync       │────►│   Test       │────►│   Verify     │                 │
│  │   Changes    │     │   Execute    │     │   Results    │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│         │                    │                    │                          │
│         ▼                    ▼                    ▼                          │
│  rsync --delete        CLI --test-mode      JSON output                     │
│  (no cache)            (structured I/O)     assertions                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## CLI Test Interface

### New CLI Arguments for Testing

```bash
# Test mode - enables structured I/O and mock capabilities
python -m pg_diagnose --test-mode [OPTIONS]

Options:
  --test-mode           Enable test mode (structured JSON output)
  --mock-benchmark      Use mock benchmark results (no actual pgbench)
  --mock-ai             Use mock AI responses (no actual Gemini calls)
  --input-file FILE     Read commands from file instead of stdin
  --output-json         Output all responses as JSON
  --state-dump          Dump workspace/session state after each command
  --timeout SECONDS     Command timeout (default: 60)
  --workspace PATH      Use specific workspace directory
```

### Structured Output Format

```json
{
  "timestamp": "2024-01-15T14:30:00Z",
  "command": "/status",
  "result": {
    "success": true,
    "data": {
      "workspace": "postgres@localhost",
      "session": {
        "name": "balanced-tps-20240115",
        "state": "active",
        "phase": "tuning",
        "round": 2
      }
    },
    "output": "Workspace: postgres@localhost..."
  },
  "state": {
    "workspace_state": "open",
    "session_state": "active",
    "session_phase": "tuning"
  }
}
```

## Test Environment

```yaml
# Test database configuration
database:
  host: localhost
  port: 6432           # Through PgBouncer proxy
  user: postgres
  password: postgres
  database: postgres

# SSH configuration (for OS commands)
ssh:
  host: localhost
  user: ubuntu
  key: ~/.ssh/id_rsa

# Test workspace location
workspace:
  base_path: /tmp/pg_diagnose_tests
  cleanup: true        # Clean after tests
```

## Test Categories

### Level 1: Unit Tests (Component Isolation)

| ID | Component | Test | Status |
|----|-----------|------|--------|
| U1.1 | SessionState | All enum values exist | Pending |
| U1.2 | SessionPhase | All enum values exist | Pending |
| U1.3 | SessionStateMachine | Valid transitions allowed | Pending |
| U1.4 | SessionStateMachine | Invalid transitions rejected | Pending |
| U1.5 | SlashCommandHandler | Command parsing | Pending |
| U1.6 | SlashCommandHandler | All commands registered | Pending |
| U1.7 | StructuredMarkdownExporter | All sections generated | Pending |
| U1.8 | StructuredMarkdownExporter | YAML blocks valid | Pending |

### Level 2: Integration Tests (Component Interaction)

#### 2.1 Workspace Lifecycle

| ID | Test | Scenario | Expected | Status |
|----|------|----------|----------|--------|
| W2.1 | Create Workspace | Connect to new database | Workspace created, directory exists | Pending |
| W2.2 | Open Workspace | Open existing workspace | Workspace loaded correctly | Pending |
| W2.3 | List Workspaces | Multiple workspaces exist | All listed with status | Pending |
| W2.4 | Close Workspace | Active workspace | Workspace closed, sessions saved | Pending |
| W2.5 | Stale Cleanup | Old ACTIVE sessions | Marked as ABANDONED | Pending |

#### 2.2 Session Lifecycle

| ID | Test | Scenario | Expected | Status |
|----|------|----------|----------|--------|
| S2.1 | Create Session | New session in workspace | Session ACTIVE:INITIALIZING | Pending |
| S2.2 | Session Phases | Progress through phases | INIT→DISCOVER→STRATEGY→BASELINE→TUNING | Pending |
| S2.3 | Pause Session | /stop during tuning | Session PAUSED, checkpoint saved | Pending |
| S2.4 | Resume Session | /resume paused session | Session ACTIVE, continues from checkpoint | Pending |
| S2.5 | Archive Session | Target achieved | Session ARCHIVED | Pending |
| S2.6 | Fail Session | Max rounds exceeded | Session FAILED | Pending |
| S2.7 | Error State | Benchmark returns 0 TPS | Session ERROR, recoverable | Pending |
| S2.8 | Retry Session | /retry from ERROR | Session ACTIVE from checkpoint | Pending |
| S2.9 | Abandon Detection | 24h+ inactive | Session ABANDONED | Pending |

#### 2.3 Command System

| ID | Test | Command | Expected | Status |
|----|------|---------|----------|--------|
| C2.1 | Help | /help | All commands listed | Pending |
| C2.2 | Status | /status | Current state displayed | Pending |
| C2.3 | Sessions | /sessions | All sessions listed by state | Pending |
| C2.4 | Go | /go | Continue without custom input | Pending |
| C2.5 | Custom | /custom text | Custom instructions captured | Pending |
| C2.6 | Custom Prompt | /custom (no text) | Prompts for instructions | Pending |
| C2.7 | Apply | /apply | Changes applied | Pending |
| C2.8 | Skip | /skip | Round skipped | Pending |
| C2.9 | Stop | /stop | Session paused | Pending |
| C2.10 | Done | /done | Session ended | Pending |
| C2.11 | Export | /export | Markdown report generated | Pending |
| C2.12 | Quit | /quit | Tool exits, session saved | Pending |

#### 2.4 Database Operations

| ID | Test | Operation | Expected | Status |
|----|------|-----------|----------|--------|
| D2.1 | Connect | Initial connection | Connection established | Pending |
| D2.2 | Reconnect | After PostgreSQL restart | Connection re-established | Pending |
| D2.3 | Config Discovery | Read current settings | All settings captured | Pending |
| D2.4 | Apply Changes | ALTER SYSTEM SET | Settings applied | Pending |
| D2.5 | Reload Config | pg_reload_conf() | Config reloaded | Pending |

#### 2.5 Export System

| ID | Test | Export | Expected | Status |
|----|------|--------|----------|--------|
| E2.1 | Single Session | /export | Complete markdown report | Pending |
| E2.2 | Metadata Section | Parse YAML | Valid YAML, all fields present | Pending |
| E2.3 | Instructions | Parse shell blocks | Valid bash scripts | Pending |
| E2.4 | Validation | Parse validation section | Runnable verification scripts | Pending |
| E2.5 | Multi-Session | Workspace export | Merged report with comparison | Pending |

### Level 3: End-to-End Tests (Full Workflows)

| ID | Workflow | Description | Status |
|----|----------|-------------|--------|
| E3.1 | Happy Path | New session → Target achieved → Export | Pending |
| E3.2 | Pause/Resume | Start → Pause → Resume → Complete | Pending |
| E3.3 | Error Recovery | Start → 0 TPS → Retry → Complete | Pending |
| E3.4 | Failed Retry | Start → Fail → Retry different strategy | Pending |
| E3.5 | Multi-Session | Session 1 → Archive → Session 2 → Compare | Pending |
| E3.6 | Crash Recovery | Simulate crash → Resume from checkpoint | Pending |
| E3.7 | Connection Loss | Mid-tuning disconnect → Reconnect → Continue | Pending |

## Test Execution Protocol

### Phase 1: Sync

```bash
#!/bin/bash
# sync_test.sh - Sync with no cache

rsync -avz --delete \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  /path/to/pg_diagnose/ \
  ubuntu@localhost:~/pg_diagnose/

# Verify sync
ssh ubuntu@localhost "ls -la ~/pg_diagnose/"
```

### Phase 2: Execute Tests

```bash
#!/bin/bash
# run_tests.sh - Execute test suite

cd ~/pg_diagnose

# Run with test mode
python -m pg_diagnose \
  --test-mode \
  --mock-benchmark \
  --output-json \
  --input-file tests/scenarios/happy_path.txt \
  > tests/results/happy_path.json 2>&1

# Check exit code
if [ $? -eq 0 ]; then
  echo "PASS: happy_path"
else
  echo "FAIL: happy_path"
fi
```

### Phase 3: Verify Results

```python
# verify_results.py
import json

def verify_test(result_file, expectations):
    with open(result_file) as f:
        results = json.load(f)

    for expectation in expectations:
        assert expectation['field'] in results
        assert results[expectation['field']] == expectation['value']

    return True
```

## Test Scenarios (Input Files)

### Scenario: Happy Path (tests/scenarios/happy_path.txt)

```
# Connect to test database
--host localhost
--port 6432
--user postgres
--database postgres

# Select strategy (auto-select first)
1

# Accept benchmark
/run

# Accept target
/accept

# Continue through rounds
/go
/apply
/go
/apply
/go

# Export and quit
/export
/quit
```

### Scenario: Pause Resume (tests/scenarios/pause_resume.txt)

```
# Connect
--host localhost --port 6432 --user postgres --database postgres

# Start session
1
/run
/accept

# Do one round
/go
/apply

# Pause
/stop

# Resume
/resume

# Continue
/go
/apply
/done

# Export
/export
/quit
```

### Scenario: Error Recovery (tests/scenarios/error_recovery.txt)

```
# Connect
--host localhost --port 6432 --user postgres --database postgres

# Start session
1
/run

# Simulate 0 TPS (mock will trigger this)
# Error state should be set

# Retry
/retry

# Continue
/go
/apply
/done
/quit
```

## Mock System

### Mock Benchmark Results

```python
# mocks/benchmark.py
class MockBenchmark:
    """Mock benchmark for testing."""

    scenarios = {
        'normal': {'tps': 5000, 'latency_avg': 2.5},
        'improved': {'tps': 7500, 'latency_avg': 1.8},
        'zero_tps': {'tps': 0, 'latency_avg': 0},
        'target_hit': {'tps': 9500, 'latency_avg': 1.2},
    }

    def __init__(self, scenario='normal'):
        self.scenario = scenario
        self.call_count = 0

    def run(self, *args, **kwargs):
        self.call_count += 1

        # Progress through scenarios
        if self.call_count == 1:
            return self.scenarios['normal']  # Baseline
        elif self.call_count < 4:
            return self.scenarios['improved']
        else:
            return self.scenarios['target_hit']
```

### Mock AI Agent

```python
# mocks/agent.py
class MockAgent:
    """Mock AI agent for testing."""

    def get_strategies(self, *args, **kwargs):
        return [
            {
                'name': 'Mock Strategy 1',
                'hypothesis': 'Test hypothesis',
                'changes': [{'name': 'shared_buffers', 'value': '1GB'}]
            }
        ]

    def analyze_results(self, *args, **kwargs):
        return {
            'tuning_chunks': [
                {
                    'name': 'Mock Tuning',
                    'category': 'memory',
                    'apply_commands': ["ALTER SYSTEM SET shared_buffers = '2GB';"],
                    'rationale': 'Test rationale'
                }
            ]
        }
```

## Test Result Format

```json
{
  "test_run": {
    "id": "test-20240115-143000",
    "started_at": "2024-01-15T14:30:00Z",
    "completed_at": "2024-01-15T14:35:00Z",
    "duration_seconds": 300
  },
  "environment": {
    "host": "localhost",
    "port": 6432,
    "database": "postgres"
  },
  "summary": {
    "total": 45,
    "passed": 43,
    "failed": 2,
    "skipped": 0
  },
  "results": [
    {
      "id": "W2.1",
      "name": "Create Workspace",
      "status": "passed",
      "duration_ms": 150,
      "assertions": [
        {"check": "workspace_created", "passed": true},
        {"check": "directory_exists", "passed": true}
      ]
    },
    {
      "id": "S2.7",
      "name": "Error State",
      "status": "failed",
      "duration_ms": 200,
      "error": "Expected ERROR state, got ACTIVE",
      "assertions": [
        {"check": "session_state", "expected": "error", "actual": "active", "passed": false}
      ]
    }
  ]
}
```

## Continuous Integration

### Test on Every Change

```bash
#!/bin/bash
# ci_test.sh - Run before shipping changes

set -e

echo "=== pg_diagnose CI Test ==="
echo "Started: $(date)"

# Phase 1: Sync
echo -e "\n>>> Phase 1: Sync"
./scripts/sync_test.sh

# Phase 2: Unit Tests
echo -e "\n>>> Phase 2: Unit Tests"
ssh localhost "cd ~/pg_diagnose && python -m pytest tests/unit/ -v"

# Phase 3: Integration Tests
echo -e "\n>>> Phase 3: Integration Tests"
ssh localhost "cd ~/pg_diagnose && python -m pytest tests/integration/ -v"

# Phase 4: E2E Tests
echo -e "\n>>> Phase 4: E2E Tests"
ssh localhost "cd ~/pg_diagnose && python tests/e2e/run_all.py"

# Summary
echo -e "\n=== Test Complete ==="
echo "Finished: $(date)"
```

## Growth Strategy

As the tool evolves, tests should grow:

| Tool Change | Test Addition |
|-------------|---------------|
| New slash command | Add command test in C2.x |
| New session state | Add state transition tests in S2.x |
| New export section | Add export verification in E2.x |
| New workflow | Add E2E test in E3.x |
| Bug fix | Add regression test |

## Priority Order

1. **P0 (Must have before any release)**
   - W2.1-W2.2: Workspace create/open
   - S2.1-S2.4: Session create/pause/resume
   - C2.1-C2.6: Core commands
   - E3.1: Happy path

2. **P1 (Must have for stable release)**
   - S2.5-S2.8: Archive/Fail/Error/Retry
   - D2.1-D2.5: Database operations
   - E2.1-E2.4: Export verification

3. **P2 (Should have)**
   - E3.2-E3.7: All E2E scenarios
   - U1.1-U1.8: Unit tests

## Next Steps

1. Implement `--test-mode` CLI flag
2. Create mock benchmark system
3. Create mock AI agent
4. Implement structured JSON output
5. Create test scenarios
6. Run initial test suite
7. Fix any failures
8. Establish baseline
