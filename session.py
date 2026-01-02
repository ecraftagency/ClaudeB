"""
Session Management - Simple project-like sessions.

Design:
- Workspace = pg_diagnose (strategy selection)
- Project = Session (one tuning run, auto-created on strategy pick)

Session auto-created: {strategy_name}_{mmddyyyy-hhmmss}

Checkpoints (auto-save):
1. Strategy selection
2. Baseline complete (with all metrics)
3. Each round complete (with all metrics)

Finished session:
- /export [format] - export to markdown/ansible/terraform
- /close - close session, return to workspace (strategy selection)
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field


SESSION_DIR = Path.home() / ".pg_diagnose" / "sessions"


@dataclass
class RoundData:
    """Complete data for one benchmark round (for LLM context restoration)."""
    round_num: int = 0
    tps: float = 0
    latency_avg_ms: float = 0
    latency_p99_ms: float = 0

    # Time series metrics during benchmark
    iostat_samples: List[Dict] = field(default_factory=list)
    vmstat_samples: List[Dict] = field(default_factory=list)
    pg_stat_samples: List[Dict] = field(default_factory=list)

    # Applied changes for this round
    changes: List[Dict] = field(default_factory=list)

    # AI analysis
    ai_analysis: str = ""
    ai_recommendations: List[str] = field(default_factory=list)

    # Raw benchmark output
    raw_output: str = ""
    timestamp: str = ""


@dataclass
class Session:
    """
    One tuning session = one project.

    Auto-created when DBA picks a strategy.
    Name format: {strategy_name}_{mmddyyyy-hhmmss}
    """
    # Identity
    name: str = ""
    created_at: str = ""
    updated_at: str = ""
    status: str = "active"  # active, completed, failed

    # Connection (for context restoration)
    db_host: str = ""
    db_port: int = 5432
    db_name: str = ""
    db_user: str = ""
    ssh_host: str = ""
    ssh_user: str = "ubuntu"

    # System context snapshot
    system_info: Dict[str, Any] = field(default_factory=dict)
    pg_version: str = ""
    initial_pg_config: Dict[str, str] = field(default_factory=dict)

    # Strategy (checkpoint 1)
    strategy_id: str = ""
    strategy_name: str = ""
    strategy_rationale: str = ""
    target_tps: float = 0
    max_rounds: int = 5

    # First sight analysis
    first_sight: str = ""
    bottleneck: str = ""
    confidence: float = 0

    # Rounds data (checkpoint 2, 3, ...)
    baseline: Optional[Dict] = None  # RoundData as dict
    rounds: List[Dict] = field(default_factory=list)  # List of RoundData

    # Progress
    current_round: int = 0
    best_tps: float = 0
    best_round: int = -1

    # Conclusion
    conclusion: str = ""
    sweet_spot_changes: List[Dict] = field(default_factory=list)


class SessionManager:
    """
    Manages sessions (projects).

    Workspace level - can list/open sessions, but only one active at a time.
    """

    def __init__(self, session_dir: Path = None):
        self.session_dir = session_dir or SESSION_DIR
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._current: Optional['SessionProject'] = None

    @property
    def current(self) -> Optional['SessionProject']:
        """Currently active session."""
        return self._current

    def create_for_strategy(self, strategy_name: str, strategy_id: str = "",
                           target_tps: float = 0, rationale: str = "") -> 'SessionProject':
        """
        Auto-create session when strategy is picked.

        Name: {strategy_name}_{mmddyyyy-hhmmss}
        """
        # Generate name
        ts = datetime.now().strftime("%m%d%Y-%H%M%S")
        safe_name = "".join(c for c in strategy_name if c.isalnum() or c in "-_")
        name = f"{safe_name}_{ts}"

        # Create session
        session = Session(
            name=name,
            created_at=datetime.now().isoformat(),
            strategy_id=strategy_id or safe_name,
            strategy_name=strategy_name,
            strategy_rationale=rationale,
            target_tps=target_tps,
        )

        project = SessionProject(self.session_dir / name, session)
        project.save()  # Checkpoint 1: strategy selection

        self._current = project
        return project

    def open(self, name: str) -> Optional['SessionProject']:
        """Open existing session."""
        path = self.session_dir / name
        if not (path / "session.json").exists():
            return None

        project = SessionProject(path)
        project.load()
        self._current = project
        return project

    def close(self):
        """Close current session, return to workspace."""
        if self._current:
            self._current.save()
        self._current = None

    def list_sessions(self) -> List[Dict]:
        """List all sessions."""
        sessions = []

        for path in self.session_dir.iterdir():
            if not path.is_dir():
                continue
            session_file = path / "session.json"
            if not session_file.exists():
                continue

            try:
                with open(session_file) as f:
                    data = json.load(f)

                baseline_tps = data.get('baseline', {}).get('tps', 0) if data.get('baseline') else 0
                best_tps = data.get('best_tps', 0)
                target_tps = data.get('target_tps', 0)

                sessions.append({
                    'name': data.get('name', path.name),
                    'status': data.get('status', 'unknown'),
                    'strategy': data.get('strategy_name', ''),
                    'baseline_tps': baseline_tps,
                    'best_tps': best_tps,
                    'target_tps': target_tps,
                    'rounds': len(data.get('rounds', [])),
                    'improvement': f"+{((best_tps - baseline_tps) / baseline_tps * 100):.1f}%" if baseline_tps > 0 else "",
                    'target_hit': best_tps >= target_tps if target_tps > 0 else False,
                    'updated_at': data.get('updated_at', ''),
                })
            except Exception:
                continue

        sessions.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        return sessions

    def delete(self, name: str) -> bool:
        """Delete a session."""
        path = self.session_dir / name
        if path.exists():
            shutil.rmtree(path)
            return True
        return False


class SessionProject:
    """
    One session project - all operations for a single tuning run.
    """

    def __init__(self, path: Path, session: Session = None):
        self.path = path
        self._session = session

    @property
    def session(self) -> Session:
        if self._session is None:
            self.load()
        return self._session

    @property
    def name(self) -> str:
        return self.session.name

    def load(self):
        """Load session from disk."""
        with open(self.path / "session.json") as f:
            data = json.load(f)
        self._session = Session(**data)

    def save(self):
        """Save session to disk (checkpoint)."""
        self.path.mkdir(parents=True, exist_ok=True)
        self.session.updated_at = datetime.now().isoformat()

        with open(self.path / "session.json", 'w') as f:
            json.dump(asdict(self.session), f, indent=2, default=str)

    # === Context setters ===

    def set_connection(self, host: str, port: int, dbname: str, user: str,
                      ssh_host: str = "", ssh_user: str = "ubuntu"):
        """Set connection info for context restoration."""
        self.session.db_host = host
        self.session.db_port = port
        self.session.db_name = dbname
        self.session.db_user = user
        self.session.ssh_host = ssh_host
        self.session.ssh_user = ssh_user

    def set_system_context(self, system_info: Dict, pg_version: str,
                          pg_config: Dict[str, str]):
        """Set system context snapshot."""
        self.session.system_info = system_info
        self.session.pg_version = pg_version
        self.session.initial_pg_config = pg_config

    def set_first_sight(self, analysis: str, bottleneck: str, confidence: float):
        """Set first sight analysis."""
        self.session.first_sight = analysis
        self.session.bottleneck = bottleneck
        self.session.confidence = confidence
        self.save()

    # === Checkpoint operations ===

    def save_baseline(self, tps: float, latency_avg: float = 0, latency_p99: float = 0,
                     iostat: List[Dict] = None, vmstat: List[Dict] = None,
                     pg_stat: List[Dict] = None, raw_output: str = "",
                     ai_analysis: str = ""):
        """
        Checkpoint 2: Baseline complete.

        Stores all time series metrics for LLM context restoration.
        """
        self.session.baseline = {
            'round_num': 0,
            'tps': tps,
            'latency_avg_ms': latency_avg,
            'latency_p99_ms': latency_p99,
            'iostat_samples': iostat or [],
            'vmstat_samples': vmstat or [],
            'pg_stat_samples': pg_stat or [],
            'raw_output': raw_output,
            'ai_analysis': ai_analysis,
            'timestamp': datetime.now().isoformat(),
            'changes': [],
        }

        self.session.best_tps = tps
        self.session.best_round = 0
        self.save()

    def save_round(self, round_num: int, tps: float, changes: List[Dict],
                  latency_avg: float = 0, latency_p99: float = 0,
                  iostat: List[Dict] = None, vmstat: List[Dict] = None,
                  pg_stat: List[Dict] = None, raw_output: str = "",
                  ai_analysis: str = "", ai_recommendations: List[str] = None):
        """
        Checkpoint 3+: Round complete.

        Stores all time series metrics for LLM context restoration.
        """
        round_data = {
            'round_num': round_num,
            'tps': tps,
            'latency_avg_ms': latency_avg,
            'latency_p99_ms': latency_p99,
            'iostat_samples': iostat or [],
            'vmstat_samples': vmstat or [],
            'pg_stat_samples': pg_stat or [],
            'changes': changes,
            'raw_output': raw_output,
            'ai_analysis': ai_analysis,
            'ai_recommendations': ai_recommendations or [],
            'timestamp': datetime.now().isoformat(),
        }

        # Update or append
        if round_num <= len(self.session.rounds):
            while len(self.session.rounds) < round_num:
                self.session.rounds.append({})
            if round_num == len(self.session.rounds):
                self.session.rounds.append(round_data)
            else:
                self.session.rounds[round_num - 1] = round_data
        else:
            self.session.rounds.append(round_data)

        self.session.current_round = round_num

        # Track best
        if tps > self.session.best_tps:
            self.session.best_tps = tps
            self.session.best_round = round_num

        self.save()

    # === Finish operations ===

    def complete(self, conclusion: str = ""):
        """Mark session as completed."""
        self.session.status = "completed"
        self.session.conclusion = conclusion

        # Collect sweet spot changes
        if self.session.best_round == 0:
            self.session.sweet_spot_changes = []
        else:
            # All changes up to best round
            all_changes = []
            for r in self.session.rounds[:self.session.best_round]:
                all_changes.extend(r.get('changes', []))
            self.session.sweet_spot_changes = all_changes

        self.save()

    def fail(self, reason: str = ""):
        """Mark session as failed."""
        self.session.status = "failed"
        self.session.conclusion = reason
        self.save()

    # === Export (slash commands) ===

    def export(self, format: str = 'markdown') -> str:
        """
        /export [format]

        Formats: markdown, json, ansible, terraform, sql
        """
        if format == 'markdown':
            return self._export_markdown()
        elif format == 'json':
            return json.dumps(asdict(self.session), indent=2, default=str)
        elif format == 'ansible':
            return self._export_ansible()
        elif format == 'terraform':
            return self._export_terraform()
        elif format == 'sql':
            return self._export_sql()
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_markdown(self) -> str:
        s = self.session
        baseline_tps = s.baseline.get('tps', 0) if s.baseline else 0
        improvement = ((s.best_tps - baseline_tps) / baseline_tps * 100) if baseline_tps > 0 else 0

        md = f"""# PostgreSQL Tuning Report: {s.strategy_name}

**Session:** {s.name}
**Created:** {s.created_at}
**Status:** {s.status}

## Summary

| Metric | Value |
|--------|-------|
| Strategy | {s.strategy_name} |
| Target TPS | {s.target_tps:,.0f} |
| Baseline TPS | {baseline_tps:,.0f} |
| Best TPS | {s.best_tps:,.0f} |
| Improvement | {improvement:+.1f}% |
| Rounds | {len(s.rounds)} |
| Best Round | {s.best_round} |

## First Sight Analysis

**Bottleneck:** {s.bottleneck} (confidence: {s.confidence:.0%})

{s.first_sight}

## TPS Timeline

| Round | TPS | Change |
|-------|-----|--------|
| Baseline | {baseline_tps:,.0f} | - |
"""
        for r in s.rounds:
            prev_tps = baseline_tps if r['round_num'] == 1 else s.rounds[r['round_num']-2].get('tps', baseline_tps)
            change = ((r['tps'] - prev_tps) / prev_tps * 100) if prev_tps > 0 else 0
            md += f"| Round {r['round_num']} | {r['tps']:,.0f} | {change:+.1f}% |\n"

        md += "\n## Applied Changes\n\n"
        for r in s.rounds:
            if r.get('changes'):
                md += f"### Round {r['round_num']}\n\n"
                for c in r['changes']:
                    md += f"**{c.get('name', 'Change')}** ({c.get('category', 'config')})\n\n"
                    for cmd in c.get('pg_configs', []):
                        md += f"```sql\n{cmd}\n```\n"
                    if c.get('os_command'):
                        md += f"```bash\n{c['os_command']}\n```\n"
                    md += "\n"

        if s.conclusion:
            md += f"\n## Conclusion\n\n{s.conclusion}\n"

        md += f"\n---\n*Generated by pg_diagnose*\n"
        return md

    def _export_sql(self) -> str:
        """Export as ALTER SYSTEM commands."""
        lines = [
            f"-- PostgreSQL Tuning: {self.session.strategy_name}",
            f"-- Session: {self.session.name}",
            f"-- Best TPS: {self.session.best_tps:,.0f}",
            "",
        ]

        for change in self.session.sweet_spot_changes:
            lines.append(f"-- {change.get('name', 'Change')}")
            for cmd in change.get('pg_configs', []):
                lines.append(f"{cmd};")
            lines.append("")

        lines.append("SELECT pg_reload_conf();")
        return "\n".join(lines)

    def _export_ansible(self) -> str:
        """Export as Ansible playbook."""
        settings = {}
        os_commands = []

        for change in self.session.sweet_spot_changes:
            for cmd in change.get('pg_configs', []):
                if 'ALTER SYSTEM SET' in cmd.upper():
                    parts = cmd.upper().replace('ALTER SYSTEM SET', '').strip()
                    if '=' in parts:
                        param, value = parts.split('=', 1)
                        settings[param.strip().lower()] = value.strip().strip("'\"")
            if change.get('os_command'):
                os_commands.append(change['os_command'])

        playbook = f"""---
# PostgreSQL Tuning Playbook
# Session: {self.session.name}
# Strategy: {self.session.strategy_name}

- name: Apply PostgreSQL Tuning
  hosts: {self.session.ssh_host or 'postgresql_servers'}
  become: yes

  tasks:
    - name: Apply PostgreSQL settings
      community.postgresql.postgresql_set:
        name: "{{{{ item.key }}}}"
        value: "{{{{ item.value }}}}"
      loop: "{{{{ postgresql_settings | dict2items }}}}"
      notify: Reload PostgreSQL
"""

        if os_commands:
            playbook += """
    - name: Apply OS settings
      shell: "{{ item }}"
      loop:
"""
            for cmd in os_commands:
                playbook += f'        - "{cmd}"\n'

        playbook += """
  handlers:
    - name: Reload PostgreSQL
      systemd:
        name: postgresql
        state: reloaded

  vars:
    postgresql_settings:
"""
        for param, value in settings.items():
            playbook += f'      {param}: "{value}"\n'

        return playbook

    def _export_terraform(self) -> str:
        """Export as Terraform RDS parameter group."""
        settings = {}
        for change in self.session.sweet_spot_changes:
            for cmd in change.get('pg_configs', []):
                if 'ALTER SYSTEM SET' in cmd.upper():
                    parts = cmd.upper().replace('ALTER SYSTEM SET', '').strip()
                    if '=' in parts:
                        param, value = parts.split('=', 1)
                        settings[param.strip().lower()] = value.strip().strip("'\"")

        tf = f'''# PostgreSQL Parameter Group
# Session: {self.session.name}
# Strategy: {self.session.strategy_name}

resource "aws_db_parameter_group" "{self.session.strategy_id or 'optimized'}" {{
  name        = "pg-{self.session.strategy_id or 'optimized'}"
  family      = "postgres16"
  description = "{self.session.strategy_name} - {self.session.best_tps:.0f} TPS"

'''
        for param, value in settings.items():
            # Skip non-RDS params
            if param in ['data_directory', 'config_file', 'hba_file']:
                continue
            tf += f'''  parameter {{
    name  = "{param}"
    value = "{value}"
  }}

'''
        tf += '''  tags = {
    ManagedBy = "pg_diagnose"
  }
}
'''
        return tf

    # === Context restoration for LLM ===

    def get_context_for_llm(self) -> str:
        """
        Get full context for LLM agent restoration.

        Returns formatted string with all session data.
        """
        s = self.session

        ctx = f"""# Session Context: {s.name}

## Strategy
- Name: {s.strategy_name}
- Target TPS: {s.target_tps}
- Rationale: {s.strategy_rationale}

## System
- DB: {s.db_name}@{s.db_host}:{s.db_port}
- PostgreSQL: {s.pg_version}

## First Sight
Bottleneck: {s.bottleneck} ({s.confidence:.0%} confidence)
{s.first_sight}

## Progress
"""
        if s.baseline:
            ctx += f"\n### Baseline: {s.baseline['tps']:.0f} TPS\n"
            if s.baseline.get('ai_analysis'):
                ctx += f"{s.baseline['ai_analysis']}\n"

        for r in s.rounds:
            ctx += f"\n### Round {r['round_num']}: {r['tps']:.0f} TPS\n"
            ctx += "Changes:\n"
            for c in r.get('changes', []):
                ctx += f"- {c.get('name')}: {', '.join(c.get('pg_configs', []))}\n"
            if r.get('ai_analysis'):
                ctx += f"\nAnalysis: {r['ai_analysis']}\n"

        ctx += f"\n## Current State\n"
        ctx += f"- Round: {s.current_round}\n"
        ctx += f"- Best TPS: {s.best_tps:.0f} (round {s.best_round})\n"
        ctx += f"- Status: {s.status}\n"

        return ctx


def display_sessions(sessions: List[Dict], console=None):
    """Display sessions list."""
    if not sessions:
        print("No sessions found.")
        return

    try:
        from rich.table import Table
        from rich import box

        if console:
            table = Table(title="Sessions (Projects)", box=box.ROUNDED)
            table.add_column("Name", style="cyan", max_width=35)
            table.add_column("Status", style="yellow")
            table.add_column("Strategy", style="blue")
            table.add_column("TPS", style="magenta")
            table.add_column("Improve", style="green")
            table.add_column("Rounds", style="dim")

            for s in sessions:
                status_map = {
                    'active': '[yellow]ACTIVE[/]',
                    'completed': '[green]DONE[/]',
                    'failed': '[red]FAILED[/]',
                }
                status = status_map.get(s['status'], s['status'])
                target = "[green]✓[/]" if s.get('target_hit') else ""

                table.add_row(
                    s['name'],
                    status,
                    s['strategy'][:20],
                    f"{s['best_tps']:.0f} {target}",
                    s.get('improvement', ''),
                    str(s['rounds']),
                )

            console.print(table)
            return
    except ImportError:
        pass

    # Fallback
    print("\nSessions:")
    print("-" * 80)
    for s in sessions:
        status = s['status'].upper()
        print(f"  {s['name'][:35]:35} [{status:8}] {s['best_tps']:.0f} TPS {s.get('improvement', '')}")
