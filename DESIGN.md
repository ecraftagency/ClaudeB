# pg_diagnose - Design Document v3.0

## Vision

**An AI-powered IDE for PostgreSQL DBAs** - like Claude Code but for database tuning.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  pg_diagnose - AI DBA Assistant                                              │
│                                                                              │
│  "I'm your AI partner for PostgreSQL optimization. I analyze your database, │
│   suggest strategies, run benchmarks, and learn from results. You stay in   │
│   control - I just make your job easier."                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Minimal command - AI handles the rest
python3 -m pg_diagnose -H db-server --ssh-host db-server

# That's it. The tool will:
# 1. Show recent workspaces (if any)
# 2. Connect and analyze the database
# 3. Suggest strategies based on your system
# 4. Guide you through tuning with AI assistance
```

## Core Concepts

### 1. Workspace (like VS Code workspace)
One workspace per database. Contains all tuning sessions, configurations, and history.

```
~/.pg_diagnose/workspaces/
└── postgres_10-0-0-230/          # Workspace for postgres@10.0.0.230
    ├── workspace.json            # System snapshot, proxy config
    ├── sessions/
    │   ├── WriteOptimize_01022026-050830/
    │   │   └── session.json      # Full session state
    │   └── MemoryTuning_01022026-060000/
    │       └── session.json
    └── exports/
        ├── workspace_report.md   # LLM-composed merged report
        ├── postgresql.conf
        └── apply_tuning.yml      # Ansible playbook
```

### 2. Session (like a project within workspace)
One session = one tuning strategy run. Auto-created when you pick a strategy.

**Session States:**
- `ACTIVE` - Currently running
- `PAUSED` - Interrupted, can resume with full context
- `ARCHIVED` - Completed (contributes to workspace export)
- `FAILED` - Did not achieve goal (excluded from export)

### 3. Checkpoints (Auto-save)
Sessions auto-save at every important moment:
- Strategy selection
- Baseline benchmark complete
- Each tuning round complete
- Target achieved or session end

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              pg_diagnose                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         User Interface Layer                         │    │
│  ├──────────────┬──────────────┬──────────────┬──────────────────────┤    │
│  │     CLI      │  Status Line │    Slash     │     Dashboard        │    │
│  │   (cli.py)   │(statusline.py)│  Commands   │   (dashboard.py)     │    │
│  └──────────────┴──────────────┴──────────────┴──────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────┴─────────────────────────────────┐      │
│  │                      Workspace Management                          │      │
│  │                        (workspace.py)                              │      │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │      │
│  │  │  Workspace   │  │   Session    │  │   Export Composer        │ │      │
│  │  │  Manager     │  │   Manager    │  │   (LLM composition)      │ │      │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                    │                                         │
│  ┌─────────────────────────────────┴─────────────────────────────────┐      │
│  │                         Core Services                              │      │
│  ├────────────────┬────────────────┬────────────────┬────────────────┤      │
│  │   Discovery    │   Telemetry    │    Runner      │     Agent      │      │
│  │  (discovery/)  │  (telemetry/)  │   (runner/)    │   (agent/)     │      │
│  │                │                │                │                │      │
│  │  • System scan │  • pgstat      │  • pgbench     │  • Gemini API  │      │
│  │  • Schema scan │  • iostat      │  • Custom SQL  │  • First Sight │      │
│  │  • Runtime     │  • vmstat      │  • Telemetry   │  • Strategies  │      │
│  └────────────────┴────────────────┴────────────────┴────────────────┘      │
│                                    │                                         │
│  ┌─────────────────────────────────┴─────────────────────────────────┐      │
│  │                         Data Layer                                 │      │
│  ├────────────────┬────────────────┬────────────────┬────────────────┤      │
│  │   Protocol     │    Tuning      │    Export      │   Context      │      │
│  │  (protocol/)   │  (tuning/)     │  (export.py)   │   Packet       │      │
│  └────────────────┴────────────────┴────────────────┴────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                               │
                    ▼                               ▼
            ┌──────────────┐                ┌──────────────┐
            │  PostgreSQL  │                │  Gemini AI   │
            │   Database   │                │    API       │
            └──────────────┘                └──────────────┘
```

## Workflow

### Startup Flow (IDE-like)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           pg_diagnose Startup                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Check for existing          │
                    │   workspaces                  │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────────┐       ┌───────────────────────┐
        │  Workspaces exist?    │  No   │  Connect to server    │
        │  Show recent list     │──────►│  Show database list   │
        └───────────┬───────────┘       └───────────────────────┘
                    │ Yes
                    ▼
        ┌───────────────────────┐
        │  User picks workspace │
        │  or 'n' for new       │
        └───────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
    Resume workspace        New database
    Show sessions           selection
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
            ┌───────────────┐
            │  Main Loop    │
            └───────────────┘
```

### Main Tuning Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Main Tuning Loop                                  │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌─────────────────────────────────────────────────────────────────┐
     │                      AI First Sight Analysis                     │
     │  • System analysis                                               │
     │  • Bottleneck detection                                          │
     │  • Strategy recommendations                                      │
     └────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │                    Strategy Selection                            │
     │  → AUTO-CREATE SESSION (checkpoint 1)                           │
     └────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │                    BASELINE Benchmark                            │
     │  → SAVE BASELINE (checkpoint 2)                                 │
     │  → AI SUGGESTS TARGET (DBA confirms)                            │
     └────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │                    AI generates Round 1 config                   │
     └────────────────────────────┬────────────────────────────────────┘
                                  │
     ┌────────────────────────────┴────────────────────────────────────┐
     │                                                                  │
     │                    TUNING ROUNDS LOOP                           │
     │   ┌──────────────────────────────────────────────────────┐      │
     │   │                                                      │      │
     │   │  ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │      │
     │   │  │ Apply       │───►│ Benchmark   │───►│ Collect  │ │      │
     │   │  │ Config      │    │ (pgbench)   │    │ Telemetry│ │      │
     │   │  └─────────────┘    └─────────────┘    └────┬─────┘ │      │
     │   │         ▲                                   │       │      │
     │   │         │           ┌─────────────┐         │       │      │
     │   │         └───────────│ AI Analysis │◄────────┘       │      │
     │   │                     └──────┬──────┘                 │      │
     │   │                            │                        │      │
     │   │                     → SAVE ROUND (checkpoint N)     │      │
     │   │                            │                        │      │
     │   │                     Target Hit? ───Yes──► Break     │      │
     │   │                            │                        │      │
     │   │                           No                        │      │
     │   │                            │                        │      │
     │   │                     3 Misses? ────Yes──► Break      │      │
     │   │                            │                        │      │
     │   │                           No ──────────────┘        │      │
     │   │                                                     │      │
     │   └─────────────────────────────────────────────────────┘      │
     │                                                                  │
     └────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │                    FINAL SUMMARY                                 │
     │  → ARCHIVE SESSION (checkpoint final)                           │
     │  → Ask: Continue with different strategy?                       │
     └────────────────────────────┬────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │ Yes                       │ No
                    ▼                           ▼
            Back to Strategy              Exit / Export
            Selection (new session)
```

## Slash Commands

Available at any prompt:

| Command | Alias | Description |
|---------|-------|-------------|
| `/help` | `/h`, `/?` | Show available commands |
| `/status` | `/s` | Current workspace/session status |
| `/sessions` | `/ls` | List all sessions in workspace |
| `/export` | | Export recommendations |
| `/close` | | Archive current session |
| `/pause` | | Pause session for later |
| `/resume [name]` | | Resume a paused session |
| `/switch [name]` | | Switch to different session |
| `/quit` | `/q` | Exit with auto-save |

## LLM Integration Points

### Current LLM Usage

| Step | LLM Function | Description |
|------|--------------|-------------|
| First Sight | `first_sight_analysis()` | Initial system analysis & strategy suggestions |
| Strategy | `generate_strategy()` | Generate benchmark plan |
| Target | `suggest_target()` | Suggest target TPS after baseline |
| Round 1 | `get_round1_config()` | Initial tuning configuration |
| Analysis | `analyze_results()` | Analyze benchmark results |
| Next Strategies | `get_next_strategies()` | New strategies after success |
| Export | `resolve_conflicts()` | Merge conflicting session configs |

### GAPS - More LLM Opportunities

| Gap | Proposed LLM Function | Impact |
|-----|----------------------|--------|
| Resume Session | `restore_context()` | Summarize previous session for DBA, suggest next action |
| Error Handling | `diagnose_error()` | When benchmark fails, explain why and suggest fixes |
| DBA Questions | `answer_question()` | Natural language Q&A about database state |
| Parameter Explain | `explain_parameter()` | Explain any PG parameter in context |
| Anomaly Explain | `explain_anomaly()` | When telemetry shows anomaly, explain what it means |
| Strategy Compare | `compare_strategies()` | Compare results of different strategies |
| Hardware Recommend | `recommend_hardware()` | Suggest hardware changes if tuning hits limits |

---

## WORKFLOW REVIEW - Issues & Improvements

### Issue 1: Session Resume is Incomplete
**Current:** When resuming a paused session, we load the data but don't restore the AI context or continue from where we left off.

**Fix:** Add `resumed_session` handling:
```python
if resumed_session:
    # LLM summarizes where we were
    context_summary = agent.restore_context(resumed_session)
    ui.print(context_summary)

    # Jump to appropriate point in workflow
    # - If baseline done but no rounds: go to Round 1
    # - If rounds done: continue to next round
```

### Issue 2: No LLM Help During Errors
**Current:** When benchmark fails or config apply fails, we just print error and abort.

**Fix:** Add AI error diagnosis:
```python
except Exception as e:
    diagnosis = agent.diagnose_error(e, context)
    ui.print(diagnosis.explanation)
    ui.print(diagnosis.suggested_fix)
```

### Issue 3: No Conversational Mode
**Current:** DBA can only interact via slash commands or menu choices.

**Fix:** Add natural language support:
```python
user_input = ui.prompt("Your choice or question: ")
if user_input.startswith('/'):
    handle_command(user_input)
elif user_input[0].isdigit():
    handle_menu_choice(user_input)
else:
    # Natural language question
    response = agent.answer_question(user_input, context)
    ui.print(response)
```

### Issue 4: Workspace Export Not Integrated
**Current:** ExportComposer exists but not wired into CLI workflow.

**Fix:** Add `/export workspace` command and auto-suggest export at session end:
```python
# At session archive
ui.print("Session archived. Export recommendations?")
if ui.prompt("[y/N]: ") == 'y':
    exports = ExportComposer(workspace, agent).export_workspace()
    ui.print(f"Exported to: {workspace.path}/exports/")
```

### Issue 5: Status Line Shows After Actions, Not Before Prompts
**Current:** Status line shows after checkpoints but not visible during prompts.

**Fix:** Integrate status into prompt itself:
```
[postgres@10.0.0.230 | WriteOptimize R2 | 5892 TPS] >
```

---

## NEXT STEPS - Prioritized

### P0: Critical (Workflow Integrity)
1. **Fix session resume** - Load session state and continue from correct point
2. **Wire up workspace export** - `/export workspace` command
3. **Handle ws_session reference** - Ensure `ws_session` is always available in the tuning loop

### P1: High (LLM Enhancement)
4. **Add `restore_context()`** - LLM summarizes session for resumed DBAs
5. **Add `diagnose_error()`** - LLM explains errors and suggests fixes
6. **Add conversational mode** - Natural language questions anywhere

### P2: Medium (UX)
7. **Status line in prompt** - Always visible workspace/session context
8. **Auto-suggest export** - Prompt to export at session end
9. **Simplify startup** - Remove most CLI flags, let AI guide

### P3: Low (Polish)
10. **Add `explain_parameter()`** - Ask AI about any PG setting
11. **Add `compare_strategies()`** - Compare session results
12. **Add `recommend_hardware()`** - When hitting hardware limits

---

## Command Line Philosophy

### Before (Complex)
```bash
python3 -m pg_diagnose -H 10.0.0.230 -p 5432 -U postgres -d postgres \
    --ssh-host 10.0.0.230 --ssh-user ubuntu \
    --save-session my-session --target-tps 6000
```

### After (Simple)
```bash
# Most common case - AI handles everything
pg_diagnose -H db-server --ssh-host db-server

# Only essential flags:
#   -H (host) - required
#   --ssh-host - for OS-level tuning
#   -d (database) - skip selection, optional
```

### Let AI Handle:
- Target TPS (AI suggests after baseline)
- Strategy selection (AI recommends)
- Session management (auto-save, auto-create)
- Risk assessment (AI evaluates each change)

---

## File Structure (Updated)

```
pg_diagnose/
├── __init__.py
├── __main__.py
├── cli.py              # Main CLI entry point
├── workspace.py        # Workspace/Session management
├── export.py           # Export + LLM composition
├── dashboard.py        # TUI dashboard components
├── commands.py         # Legacy slash commands
├── modes.py            # Operation modes (health, watch)
├── session.py          # Legacy session (deprecated)
│
├── ui/                 # User Interface components
│   ├── __init__.py
│   ├── commands.py     # Workspace slash commands
│   ├── statusline.py   # Status bar component
│   ├── console.py      # Rich console helpers
│   ├── display.py      # Result display
│   └── interaction.py  # User interaction
│
├── discovery/          # System discovery
│   ├── system.py       # Hardware/OS scanning
│   ├── schema.py       # Database schema analysis
│   └── runtime.py      # Runtime statistics
│
├── telemetry/          # Metrics collection
│   ├── collector.py
│   ├── pgstat.py
│   ├── iostat.py
│   ├── vmstat.py
│   ├── sysstat.py
│   └── aggregator.py
│
├── runner/             # Benchmark execution
│   └── benchmark.py
│
├── agent/              # AI integration
│   ├── client.py       # Gemini API client
│   ├── prompts.py      # AI prompts
│   └── parser.py       # Response parsing
│
├── protocol/           # Data structures
│   ├── context.py
│   ├── sdl.py
│   ├── tuning.py
│   ├── conclusion.py
│   ├── first_sight.py
│   └── result.py
│
└── tuning/             # Tuning execution
    ├── executor.py
    ├── service.py
    ├── snapshot.py
    └── verifier.py
```

---

## Design Principles

1. **AI-First**: Every decision should have AI assistance available
2. **Auto-Save**: Never lose work - checkpoint at every important moment
3. **Context Preservation**: Full session state for resume/export
4. **DBA in Control**: AI suggests, human decides
5. **Simple Start**: Minimal flags, AI guides the rest
