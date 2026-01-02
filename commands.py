"""
Slash Commands - Interactive command system for pg_diagnose.

Provides:
- /status - Show current session status
- /history - View tuning history
- /rollback - Rollback last change or all changes
- /snapshot - Save current config as named snapshot
- /restore - Restore a named snapshot
- /compare - Compare current vs baseline
- /explain - AI explains a PostgreSQL parameter
- /benchmark - Run quick benchmark
- /watch - Live metrics mode
- /export - Export session report
- /help - Show available commands
"""

import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class SessionState:
    """Holds the current session state for command access."""
    # Connection info
    db_host: str = ""
    db_port: int = 5432
    db_name: str = ""
    db_user: str = ""
    db_password: str = ""

    # Connection object
    conn: Any = None

    # SSH config
    ssh_config: Optional[Dict] = None

    # Current session
    strategy_name: str = ""
    strategy_id: str = ""
    target_tps: float = 0
    current_round: int = 0

    # Tuning history
    baseline_tps: float = 0
    best_tps: float = 0
    current_tps: float = 0
    tps_history: List[float] = field(default_factory=list)
    applied_changes: List[Dict] = field(default_factory=list)

    # Baseline config (for rollback)
    baseline_config: Dict[str, str] = field(default_factory=dict)

    # Snapshots (in-memory, legacy)
    snapshots: Dict[str, Dict] = field(default_factory=dict)

    # Workspace reference (for persistent snapshots)
    workspace: Any = None
    session_name: str = ""

    # Agent reference
    agent: Any = None

    # UI reference
    ui: Any = None

    # Test mode flags
    test_mode: bool = False
    mock_snapshot: bool = False


@dataclass
class Command:
    """Represents a slash command."""
    name: str
    description: str
    usage: str
    handler: Callable
    aliases: List[str] = field(default_factory=list)


class CommandRegistry:
    """Registry for slash commands."""

    def __init__(self):
        self.commands: Dict[str, Command] = {}
        self.aliases: Dict[str, str] = {}

    def register(self, name: str, description: str, usage: str,
                 handler: Callable, aliases: List[str] = None):
        """Register a command."""
        cmd = Command(
            name=name,
            description=description,
            usage=usage,
            handler=handler,
            aliases=aliases or []
        )
        self.commands[name] = cmd

        # Register aliases
        for alias in cmd.aliases:
            self.aliases[alias] = name

    def get(self, name: str) -> Optional[Command]:
        """Get a command by name or alias."""
        if name in self.commands:
            return self.commands[name]
        if name in self.aliases:
            return self.commands[self.aliases[name]]
        return None

    def list_all(self) -> List[Command]:
        """List all registered commands."""
        return list(self.commands.values())


class CommandHandler:
    """Handles slash command execution."""

    def __init__(self, state: SessionState):
        self.state = state
        self.registry = CommandRegistry()
        self.console = Console() if RICH_AVAILABLE else None
        self._register_commands()

    def _register_commands(self):
        """Register all available commands."""
        self.registry.register(
            "help", "Show available commands", "/help [command]",
            self._cmd_help, aliases=["h", "?"]
        )
        self.registry.register(
            "status", "Show current session status", "/status",
            self._cmd_status, aliases=["s"]
        )
        self.registry.register(
            "history", "View tuning history", "/history",
            self._cmd_history, aliases=["hist"]
        )
        self.registry.register(
            "rollback", "Rollback changes", "/rollback [all|last|<n>]",
            self._cmd_rollback, aliases=["rb"]
        )
        self.registry.register(
            "snapshot", "Manage config snapshots",
            "/snapshot [name] | list | show <name> | compare <a> <b>",
            self._cmd_snapshot, aliases=["snap"]
        )
        self.registry.register(
            "restore", "Restore to snapshot",
            "/restore [name] [--dry-run]",
            self._cmd_restore
        )
        self.registry.register(
            "compare", "Compare current vs baseline", "/compare [snapshot]",
            self._cmd_compare, aliases=["diff"]
        )
        self.registry.register(
            "explain", "AI explains a parameter", "/explain <parameter>",
            self._cmd_explain
        )
        self.registry.register(
            "benchmark", "Run quick benchmark", "/benchmark [duration]",
            self._cmd_benchmark, aliases=["bench"]
        )
        self.registry.register(
            "watch", "Live metrics mode", "/watch [interval]",
            self._cmd_watch
        )
        self.registry.register(
            "export", "Export session report", "/export [format]",
            self._cmd_export
        )
        self.registry.register(
            "config", "Show current PostgreSQL config", "/config [param]",
            self._cmd_config, aliases=["cfg"]
        )
        self.registry.register(
            "queries", "Show active queries", "/queries",
            self._cmd_queries, aliases=["q"]
        )

    def has_command(self, input_str: str) -> bool:
        """Check if input is a recognized command."""
        input_str = input_str.strip()
        if not input_str.startswith('/'):
            return False
        parts = input_str[1:].split(maxsplit=1)
        cmd_name = parts[0].lower()
        return self.registry.get(cmd_name) is not None

    def parse_and_execute(self, input_str: str) -> bool:
        """
        Parse input and execute if it's a command.

        Returns True if input was a command, False otherwise.
        """
        input_str = input_str.strip()

        if not input_str.startswith('/'):
            return False

        # Parse command and args
        parts = input_str[1:].split(maxsplit=1)
        cmd_name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Find and execute command
        cmd = self.registry.get(cmd_name)
        if cmd:
            try:
                cmd.handler(args)
            except Exception as e:
                self._print_error(f"Command failed: {e}")
            return True
        else:
            # Command not recognized - don't handle it
            return False

    def _print(self, message: str = ""):
        """Print a message."""
        if self.console:
            self.console.print(message)
        else:
            print(message)

    def _print_error(self, message: str):
        """Print an error message."""
        if self.console:
            self.console.print(f"[red]Error:[/] {message}")
        else:
            print(f"Error: {message}")

    # ==================== Command Implementations ====================

    def _cmd_help(self, args: str):
        """Show help for commands."""
        if args:
            # Help for specific command
            cmd = self.registry.get(args.lower())
            if cmd:
                if self.console:
                    self.console.print(Panel(
                        f"[cyan]{cmd.usage}[/]\n\n{cmd.description}" +
                        (f"\n\n[dim]Aliases: {', '.join(cmd.aliases)}[/]" if cmd.aliases else ""),
                        title=f"/{cmd.name}",
                        border_style="blue"
                    ))
                else:
                    print(f"\n/{cmd.name}")
                    print(f"  Usage: {cmd.usage}")
                    print(f"  {cmd.description}")
                    if cmd.aliases:
                        print(f"  Aliases: {', '.join(cmd.aliases)}")
            else:
                self._print_error(f"Unknown command: {args}")
        else:
            # List all commands
            if self.console:
                table = Table(title="Available Commands", box=box.ROUNDED)
                table.add_column("Command", style="cyan")
                table.add_column("Description", style="green")
                table.add_column("Aliases", style="dim")

                for cmd in self.registry.list_all():
                    aliases = ", ".join(cmd.aliases) if cmd.aliases else "-"
                    table.add_row(f"/{cmd.name}", cmd.description, aliases)

                self.console.print(table)
            else:
                print("\nAvailable Commands:")
                print("-" * 50)
                for cmd in self.registry.list_all():
                    print(f"  /{cmd.name:12} - {cmd.description}")

    def _cmd_status(self, args: str):
        """Show current session status."""
        s = self.state

        if self.console:
            table = Table(title="Session Status", box=box.ROUNDED)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Database", f"{s.db_name}@{s.db_host}:{s.db_port}")
            table.add_row("Strategy", s.strategy_name or "[dim]Not selected[/]")
            table.add_row("Current Round", str(s.current_round))
            table.add_row("Target TPS", f"{s.target_tps:.0f}" if s.target_tps else "[dim]N/A[/]")
            table.add_row("Baseline TPS", f"{s.baseline_tps:.0f}" if s.baseline_tps else "[dim]N/A[/]")
            table.add_row("Current TPS", f"{s.current_tps:.0f}" if s.current_tps else "[dim]N/A[/]")
            table.add_row("Best TPS", f"{s.best_tps:.0f}" if s.best_tps else "[dim]N/A[/]")

            if s.target_tps and s.current_tps:
                progress = (s.current_tps / s.target_tps) * 100
                status = "[green]On Target[/]" if progress >= 90 else f"[yellow]{progress:.0f}% of target[/]"
                table.add_row("Progress", status)

            table.add_row("Changes Applied", str(len(s.applied_changes)))
            table.add_row("Snapshots", str(len(s.snapshots)))

            self.console.print(table)
        else:
            print(f"\nSession Status:")
            print(f"  Database: {s.db_name}@{s.db_host}:{s.db_port}")
            print(f"  Strategy: {s.strategy_name or 'Not selected'}")
            print(f"  Round: {s.current_round}")
            print(f"  TPS: {s.current_tps:.0f} / {s.target_tps:.0f} target")
            print(f"  Changes: {len(s.applied_changes)}")

    def _cmd_history(self, args: str):
        """Show tuning history."""
        s = self.state

        if not s.applied_changes:
            self._print("No changes applied yet.")
            return

        if self.console:
            table = Table(title="Tuning History", box=box.ROUNDED)
            table.add_column("Round", style="cyan", width=6)
            table.add_column("Change", style="green")
            table.add_column("Category", style="yellow")
            table.add_column("Commands", style="dim")

            for change in s.applied_changes:
                round_num = str(change.get('round', '?'))
                name = change.get('name', 'Unknown')
                category = change.get('category', 'config')

                # Get commands
                cmds = change.get('pg_configs', [])
                if not cmds and change.get('os_command'):
                    cmds = [change['os_command']]
                cmd_str = cmds[0][:40] + "..." if cmds and len(cmds[0]) > 40 else (cmds[0] if cmds else "-")

                table.add_row(round_num, name, category, cmd_str)

            self.console.print(table)

            # TPS timeline
            if s.tps_history:
                self._print("\n[cyan]TPS Timeline:[/]")
                timeline = " → ".join(f"{t:.0f}" for t in s.tps_history)
                self._print(f"  {timeline}")
        else:
            print("\nTuning History:")
            for change in s.applied_changes:
                print(f"  Round {change.get('round', '?')}: {change.get('name')} ({change.get('category')})")

    def _cmd_rollback(self, args: str):
        """Rollback changes."""
        s = self.state

        if not s.conn:
            self._print_error("No database connection")
            return

        if not s.applied_changes:
            self._print("No changes to rollback.")
            return

        args = args.lower().strip()

        if args == "all":
            # Rollback all changes
            self._print("[yellow]Rolling back ALL changes to baseline...[/]" if self.console else "Rolling back all changes...")
            self._rollback_to_baseline()
        elif args == "last" or args == "":
            # Rollback last change
            self._rollback_last_change()
        elif args.isdigit():
            # Rollback specific number of changes
            n = int(args)
            for _ in range(min(n, len(s.applied_changes))):
                self._rollback_last_change()
        else:
            self._print_error(f"Invalid argument: {args}")
            self._print("Usage: /rollback [all|last|<n>]")

    def _rollback_last_change(self):
        """Rollback the last applied change."""
        s = self.state

        if not s.applied_changes:
            self._print("No changes to rollback.")
            return

        last_change = s.applied_changes[-1]
        name = last_change.get('name', 'Unknown')

        self._print(f"Rolling back: {name}")

        # Get rollback commands
        rollback_cmds = last_change.get('rollback_commands', [])

        if not rollback_cmds:
            # Try to construct rollback from baseline
            pg_configs = last_change.get('pg_configs', [])
            for cmd in pg_configs:
                # Extract parameter name and get baseline value
                if 'ALTER SYSTEM SET' in cmd.upper():
                    param = cmd.split('=')[0].replace('ALTER SYSTEM SET', '').strip()
                    if param in s.baseline_config:
                        rollback_cmds.append(f"ALTER SYSTEM SET {param} = '{s.baseline_config[param]}'")
                    else:
                        rollback_cmds.append(f"ALTER SYSTEM RESET {param}")

        if rollback_cmds:
            try:
                old_autocommit = s.conn.autocommit
                s.conn.autocommit = True

                with s.conn.cursor() as cur:
                    for cmd in rollback_cmds:
                        self._print(f"  [dim]→ {cmd}[/]" if self.console else f"  → {cmd}")
                        cur.execute(cmd)
                    cur.execute("SELECT pg_reload_conf()")

                s.conn.autocommit = old_autocommit
                s.applied_changes.pop()
                self._print("[green]Rollback successful[/]" if self.console else "Rollback successful")

            except Exception as e:
                self._print_error(f"Rollback failed: {e}")
        else:
            self._print("[yellow]No rollback commands available. Manual intervention may be required.[/]" if self.console else "No rollback commands available.")
            s.applied_changes.pop()  # Remove from tracking anyway

    def _rollback_to_baseline(self):
        """Rollback all changes to baseline configuration."""
        s = self.state

        if not s.baseline_config:
            self._print_error("No baseline configuration saved")
            return

        try:
            old_autocommit = s.conn.autocommit
            s.conn.autocommit = True

            with s.conn.cursor() as cur:
                for param, value in s.baseline_config.items():
                    cmd = f"ALTER SYSTEM SET {param} = '{value}'"
                    self._print(f"  [dim]→ {cmd}[/]" if self.console else f"  → {cmd}")
                    cur.execute(cmd)
                cur.execute("SELECT pg_reload_conf()")

            s.conn.autocommit = old_autocommit
            s.applied_changes.clear()
            self._print("[green]Rolled back to baseline[/]" if self.console else "Rolled back to baseline")

        except Exception as e:
            self._print_error(f"Rollback failed: {e}")

    def _cmd_snapshot(self, args: str):
        """Manage config snapshots with subcommands.

        Subcommands:
            /snapshot [name]     - Create checkpoint with name
            /snapshot list       - List all snapshots
            /snapshot show <n>   - Show snapshot details
            /snapshot compare <a> <b> - Compare two snapshots
        """
        s = self.state

        if not args:
            # No args - show help
            self._print("Snapshot commands:")
            self._print("  /snapshot <name>        - Create checkpoint snapshot")
            self._print("  /snapshot list          - List all snapshots")
            self._print("  /snapshot show <name>   - Show snapshot details")
            self._print("  /snapshot compare <a> <b> - Compare two snapshots")
            return

        parts = args.strip().split(maxsplit=2)
        subcommand = parts[0].lower()

        # Route to subcommand handlers
        if subcommand == "list":
            self._snapshot_list()
        elif subcommand == "show":
            name = parts[1] if len(parts) > 1 else ""
            self._snapshot_show(name)
        elif subcommand == "compare":
            if len(parts) < 3:
                self._print_error("Usage: /snapshot compare <snapshot_a> <snapshot_b>")
                return
            self._snapshot_compare(parts[1], parts[2])
        else:
            # Treat as snapshot name to create
            self._snapshot_create(subcommand)

    def _get_snapshot_manager(self):
        """Get or create SnapshotManager for current workspace."""
        s = self.state

        # Check for mock mode
        if s.test_mode and s.mock_snapshot:
            from pg_diagnose.tests.mocks import MockSnapshotManager
            return MockSnapshotManager()

        # Check for workspace
        if s.workspace is None:
            return None

        from pg_diagnose.snapshot import SnapshotManager
        return SnapshotManager(
            workspace_path=s.workspace.path,
            conn=s.conn,
            ssh_config=s.ssh_config,
        )

    def _snapshot_create(self, name: str):
        """Create a checkpoint snapshot."""
        s = self.state

        if not name:
            self._print_error("Snapshot name required")
            return

        manager = self._get_snapshot_manager()
        if manager is None:
            # Fall back to in-memory snapshots
            self._snapshot_create_legacy(name)
            return

        try:
            snapshot = manager.create_checkpoint(
                name=name,
                session_name=s.session_name,
                round_num=s.current_round,
                last_tps=s.current_tps,
                last_latency=None,
            )
            self._print(f"[green]Snapshot '{name}' saved[/]" if self.console else f"Snapshot '{name}' saved")
            self._print(f"  [dim]Params: {len(snapshot.pg_settings)}, TPS: {s.current_tps:.0f}[/]" if self.console else f"  Params: {len(snapshot.pg_settings)}")
        except Exception as e:
            self._print_error(f"Failed to create snapshot: {e}")

    def _snapshot_create_legacy(self, name: str):
        """Create snapshot using legacy in-memory storage."""
        s = self.state

        if not s.conn:
            self._print_error("No database connection")
            return

        try:
            config = {}
            with s.conn.cursor() as cur:
                cur.execute("""
                    SELECT name, setting
                    FROM pg_settings
                    WHERE source IN ('configuration file', 'session', 'user')
                    OR name IN ('shared_buffers', 'effective_cache_size', 'work_mem',
                               'maintenance_work_mem', 'max_wal_size', 'checkpoint_timeout',
                               'random_page_cost', 'effective_io_concurrency')
                """)
                for row in cur.fetchall():
                    config[row[0]] = row[1]

            s.snapshots[name] = {
                'config': config,
                'timestamp': datetime.now().isoformat(),
                'tps': s.current_tps,
                'round': s.current_round,
            }

            self._print(f"[green]Snapshot '{name}' saved (in-memory)[/]" if self.console else f"Snapshot '{name}' saved")
            self._print(f"  [dim]Params: {len(config)}, TPS: {s.current_tps:.0f}[/]" if self.console else f"  Params: {len(config)}")
        except Exception as e:
            self._print_error(f"Failed to save snapshot: {e}")

    def _snapshot_list(self):
        """List all snapshots."""
        s = self.state
        manager = self._get_snapshot_manager()

        if manager:
            try:
                snapshots = manager.list_snapshots()
                if not snapshots:
                    self._print("No snapshots found.")
                    return

                self._print("Available snapshots:")
                for snap in snapshots:
                    tps_str = f"{snap.last_tps:.0f} TPS" if snap.last_tps else "no benchmark"
                    trigger = f"[{snap.trigger}]" if snap.trigger else ""
                    if self.console:
                        self._print(f"  • {snap.name} [dim]({snap.created_at[:19]}, {tps_str}) {trigger}[/]")
                    else:
                        self._print(f"  • {snap.name} ({snap.created_at[:19]}, {tps_str}) {trigger}")
            except Exception as e:
                self._print_error(f"Failed to list snapshots: {e}")
        else:
            # Fall back to in-memory
            if not s.snapshots:
                self._print("No snapshots saved.")
                return

            self._print("Available snapshots (in-memory):")
            for name, snap in s.snapshots.items():
                tps = snap.get('tps', 0)
                ts = snap.get('timestamp', '')[:19]
                if self.console:
                    self._print(f"  • {name} [dim]({ts}, {tps:.0f} TPS)[/]")
                else:
                    self._print(f"  • {name} ({ts}, {tps:.0f} TPS)")

    def _snapshot_show(self, name: str):
        """Show snapshot details."""
        if not name:
            self._print_error("Snapshot name required")
            self._print("Usage: /snapshot show <name>")
            return

        s = self.state
        manager = self._get_snapshot_manager()

        if manager:
            try:
                snapshot = manager.get(name)
                if snapshot is None:
                    self._print_error(f"Snapshot '{name}' not found")
                    return

                self._print(f"Snapshot: {snapshot.name}")
                self._print(f"  Created: {snapshot.created_at}")
                self._print(f"  Trigger: {snapshot.trigger}")
                self._print(f"  Session: {snapshot.session_name}")
                self._print(f"  Round: {snapshot.round_num}")
                if snapshot.last_tps:
                    self._print(f"  TPS: {snapshot.last_tps:.0f}")
                self._print(f"  Parameters: {len(snapshot.pg_settings)}")

                # Show key parameters
                key_params = ['shared_buffers', 'effective_cache_size', 'work_mem',
                             'max_wal_size', 'checkpoint_timeout']
                self._print("\n  Key settings:")
                for param in key_params:
                    if param in snapshot.pg_settings:
                        self._print(f"    {param} = {snapshot.pg_settings[param]}")
            except Exception as e:
                self._print_error(f"Failed to show snapshot: {e}")
        else:
            # Fall back to in-memory
            if name not in s.snapshots:
                self._print_error(f"Snapshot '{name}' not found")
                return

            snap = s.snapshots[name]
            self._print(f"Snapshot: {name}")
            self._print(f"  Timestamp: {snap.get('timestamp', 'unknown')}")
            self._print(f"  Round: {snap.get('round', 0)}")
            self._print(f"  TPS: {snap.get('tps', 0):.0f}")
            self._print(f"  Parameters: {len(snap.get('config', {}))}")

    def _snapshot_compare(self, name_a: str, name_b: str):
        """Compare two snapshots."""
        s = self.state
        manager = self._get_snapshot_manager()

        if manager:
            try:
                diff = manager.compare(name_a, name_b)
                if diff is None:
                    self._print_error(f"One or both snapshots not found: '{name_a}', '{name_b}'")
                    return

                if diff.total_differences == 0:
                    self._print(f"Snapshots '{name_a}' and '{name_b}' are identical.")
                    return

                self._print(f"Comparing: {name_a} → {name_b}")
                self._print(f"Total differences: {diff.total_differences}")
                self._print()

                if diff.changes:
                    self._print("Changed parameters:")
                    for change in diff.changes:
                        restart = " [restart]" if change.requires_restart else ""
                        if self.console:
                            self._print(f"  {change.param}: {change.current_value} → {change.snapshot_value}[dim]{restart}[/]")
                        else:
                            self._print(f"  {change.param}: {change.current_value} → {change.snapshot_value}{restart}")

                if diff.only_in_a:
                    self._print(f"\nOnly in {name_a}: {', '.join(diff.only_in_a)}")
                if diff.only_in_b:
                    self._print(f"\nOnly in {name_b}: {', '.join(diff.only_in_b)}")

            except Exception as e:
                self._print_error(f"Failed to compare snapshots: {e}")
        else:
            self._print_error("Snapshot compare requires workspace (not available in legacy mode)")

    def _cmd_restore(self, args: str):
        """Restore a snapshot with optional dry-run.

        Usage:
            /restore              - Restore to 'initial' snapshot
            /restore <name>       - Restore to named snapshot
            /restore --dry-run    - Preview restore to initial
            /restore <name> --dry-run - Preview restore to named snapshot
        """
        s = self.state
        dry_run = False
        name = "initial"  # Default to initial

        # Parse args
        if args:
            parts = args.strip().split()
            for part in parts:
                if part == "--dry-run":
                    dry_run = True
                elif part == "list":
                    # Show list of snapshots
                    self._snapshot_list()
                    return
                else:
                    name = part

        manager = self._get_snapshot_manager()
        if manager:
            self._restore_with_manager(manager, name, dry_run)
        else:
            self._restore_legacy(name, dry_run)

    def _restore_with_manager(self, manager, name: str, dry_run: bool):
        """Restore using SnapshotManager."""
        try:
            if name == "initial":
                result = manager.restore_to_initial(dry_run=dry_run)
            else:
                result = manager.restore_to(name, dry_run=dry_run)

            if not result.success:
                self._print_error(result.error or "Restore failed")
                return

            if dry_run:
                self._print(f"[yellow]Dry-run: Preview of restoring to '{name}'[/]" if self.console else f"Dry-run: Preview of restoring to '{name}'")
                if result.preview and result.preview.changes:
                    self._print(f"\nWould apply {result.preview.total_changes} change(s):")
                    for change in result.preview.changes[:10]:  # Show first 10
                        restart = " [restart]" if change.requires_restart else ""
                        if self.console:
                            self._print(f"  {change.param}: {change.current_value} → {change.snapshot_value}[dim]{restart}[/]")
                        else:
                            self._print(f"  {change.param}: {change.current_value} → {change.snapshot_value}{restart}")
                    if result.preview.total_changes > 10:
                        self._print(f"  ... and {result.preview.total_changes - 10} more")
                    if result.preview.restart_required:
                        self._print("\n[yellow]Note: Some changes require PostgreSQL restart[/]" if self.console else "\nNote: Some changes require restart")
                else:
                    self._print("No changes needed - current config matches snapshot.")
            else:
                self._print(f"[green]Restored to snapshot '{name}'[/]" if self.console else f"Restored to snapshot '{name}'")
                self._print(f"  Applied {result.changes_applied} change(s)")
                if result.restart_required:
                    self._print("[yellow]Some settings require PostgreSQL restart to take effect[/]" if self.console else "Some settings require restart")

        except Exception as e:
            self._print_error(f"Restore failed: {e}")

    def _restore_legacy(self, name: str, dry_run: bool):
        """Restore using legacy in-memory snapshots."""
        s = self.state

        if name not in s.snapshots:
            self._print_error(f"Snapshot '{name}' not found")
            self._print("Use '/snapshot list' to see available snapshots")
            return

        snap = s.snapshots[name]

        if dry_run:
            self._print(f"[yellow]Dry-run: Would restore snapshot '{name}'[/]" if self.console else f"Dry-run: Would restore '{name}'")
            self._print(f"  Would apply {len(snap.get('config', {}))} parameter(s)")
            return

        try:
            old_autocommit = s.conn.autocommit
            s.conn.autocommit = True

            with s.conn.cursor() as cur:
                for param, value in snap['config'].items():
                    try:
                        cur.execute(f"ALTER SYSTEM SET {param} = '{value}'")
                    except Exception:
                        pass  # Skip non-modifiable params
                cur.execute("SELECT pg_reload_conf()")

            s.conn.autocommit = old_autocommit
            self._print(f"[green]Restored snapshot '{name}'[/]" if self.console else f"Restored snapshot '{name}'")
            self._print("[yellow]Some settings may require a restart to take effect[/]" if self.console else "Some settings may require restart")

        except Exception as e:
            self._print_error(f"Restore failed: {e}")

    def _cmd_compare(self, args: str):
        """Compare current config vs baseline or snapshot."""
        s = self.state

        if not s.conn:
            self._print_error("No database connection")
            return

        # Get current config
        current = {}
        try:
            with s.conn.cursor() as cur:
                cur.execute("""
                    SELECT name, setting FROM pg_settings
                    WHERE name IN ('shared_buffers', 'effective_cache_size', 'work_mem',
                                  'maintenance_work_mem', 'max_wal_size', 'checkpoint_timeout',
                                  'random_page_cost', 'effective_io_concurrency', 'max_connections',
                                  'wal_buffers', 'checkpoint_completion_target')
                """)
                for row in cur.fetchall():
                    current[row[0]] = row[1]
        except Exception as e:
            self._print_error(f"Failed to get current config: {e}")
            return

        # Get comparison target
        if args and args in s.snapshots:
            compare_to = s.snapshots[args]['config']
            compare_label = f"Snapshot '{args}'"
        else:
            compare_to = s.baseline_config
            compare_label = "Baseline"

        if not compare_to:
            self._print_error("No baseline configuration to compare against")
            return

        # Find differences
        if self.console:
            table = Table(title=f"Config Diff: Current vs {compare_label}", box=box.ROUNDED)
            table.add_column("Parameter", style="cyan")
            table.add_column(compare_label, style="yellow")
            table.add_column("Current", style="green")
            table.add_column("Change", style="bold")

            for param in sorted(set(current.keys()) | set(compare_to.keys())):
                old_val = compare_to.get(param, '-')
                new_val = current.get(param, '-')

                if old_val != new_val:
                    change = "[green]→[/]" if old_val != '-' and new_val != '-' else "[yellow]+[/]" if old_val == '-' else "[red]-[/]"
                    table.add_row(param, str(old_val), str(new_val), change)

            self.console.print(table)
        else:
            print(f"\nConfig Diff: Current vs {compare_label}")
            print("-" * 60)
            for param in sorted(set(current.keys()) | set(compare_to.keys())):
                old_val = compare_to.get(param, '-')
                new_val = current.get(param, '-')
                if old_val != new_val:
                    print(f"  {param}: {old_val} → {new_val}")

    def _cmd_explain(self, args: str):
        """AI explains a PostgreSQL parameter."""
        if not args:
            self._print_error("Parameter name required")
            self._print("Usage: /explain <parameter>")
            return

        param = args.strip().lower()
        s = self.state

        if not s.agent:
            self._print_error("AI agent not available")
            return

        self._print(f"[dim]Asking AI about '{param}'...[/]" if self.console else f"Asking AI about '{param}'...")

        try:
            # Get current value
            current_value = None
            if s.conn:
                with s.conn.cursor() as cur:
                    cur.execute(f"SHOW {param}")
                    result = cur.fetchone()
                    if result:
                        current_value = result[0]

            # Ask AI
            prompt = f"""Explain the PostgreSQL parameter '{param}' in a concise way:
1. What it does (1-2 sentences)
2. Good values for different scenarios (small, medium, large workloads)
3. Current value '{current_value}' - is this reasonable?
4. Common tuning advice

Keep the response under 200 words."""

            response = s.agent._call_api(prompt)

            if self.console:
                self.console.print(Panel(
                    response,
                    title=f"[cyan]{param}[/] = {current_value}",
                    border_style="blue"
                ))
            else:
                print(f"\n{param} = {current_value}")
                print("-" * 40)
                print(response)

        except Exception as e:
            self._print_error(f"Failed to explain: {e}")

    def _cmd_benchmark(self, args: str):
        """Run a quick benchmark."""
        s = self.state

        duration = 30  # Default duration
        if args and args.isdigit():
            duration = int(args)

        if not s.conn:
            self._print_error("No database connection")
            return

        self._print(f"Running {duration}s benchmark...")

        try:
            from .runner.benchmark import BenchmarkRunner

            runner = BenchmarkRunner(
                host=s.db_host,
                port=s.db_port,
                database=s.db_name,
                user=s.db_user,
                password=s.db_password,
            )

            # Create minimal strategy
            from .protocol.sdl import StrategyDefinition, ExecutionPlan
            quick_strategy = StrategyDefinition(
                id="quick-bench",
                name="Quick Benchmark",
                hypothesis="Quick performance check",
                execution_plan=ExecutionPlan(
                    benchmark_type="pgbench",
                    scale=100,
                    clients=10,
                    threads=2,
                    duration_seconds=duration,
                    warmup_seconds=5,
                ),
            )

            result = runner.run(quick_strategy)

            if result and result.metrics:
                tps = result.metrics.tps
                lat = result.metrics.latency_avg_ms

                if self.console:
                    self.console.print(Panel(
                        f"[green]TPS:[/] {tps:.0f}\n[green]Latency:[/] {lat:.2f}ms",
                        title="Benchmark Result",
                        border_style="green"
                    ))
                else:
                    print(f"\nResult: {tps:.0f} TPS, {lat:.2f}ms latency")

                s.current_tps = tps
                s.tps_history.append(tps)
            else:
                self._print_error("Benchmark failed to return results")

        except Exception as e:
            self._print_error(f"Benchmark failed: {e}")

    def _cmd_watch(self, args: str):
        """Live metrics mode."""
        s = self.state

        interval = 5  # Default interval
        if args and args.isdigit():
            interval = int(args)

        if not s.conn:
            self._print_error("No database connection")
            return

        self._print(f"[dim]Live metrics (every {interval}s). Press Ctrl+C to stop.[/]" if self.console else f"Live metrics every {interval}s. Ctrl+C to stop.")

        try:
            while True:
                # Get metrics
                with s.conn.cursor() as cur:
                    # TPS estimate from pg_stat_database
                    cur.execute("""
                        SELECT xact_commit + xact_rollback as txn,
                               blks_hit, blks_read
                        FROM pg_stat_database
                        WHERE datname = current_database()
                    """)
                    row = cur.fetchone()
                    txn = row[0] if row else 0
                    cache_hit = (row[1] / (row[1] + row[2] + 0.001) * 100) if row else 0

                    # Active connections
                    cur.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
                    active = cur.fetchone()[0]

                    # Locks
                    cur.execute("SELECT count(*) FROM pg_locks WHERE NOT granted")
                    waiting = cur.fetchone()[0]

                timestamp = datetime.now().strftime("%H:%M:%S")

                if self.console:
                    self.console.print(
                        f"[dim]{timestamp}[/] | "
                        f"Active: [cyan]{active}[/] | "
                        f"Cache: [{'green' if cache_hit > 99 else 'yellow'}]{cache_hit:.1f}%[/] | "
                        f"Waiting: [{'green' if waiting == 0 else 'red'}]{waiting}[/]"
                    )
                else:
                    print(f"{timestamp} | Active: {active} | Cache: {cache_hit:.1f}% | Waiting: {waiting}")

                time.sleep(interval)

        except KeyboardInterrupt:
            self._print("\n[dim]Watch mode stopped[/]" if self.console else "\nWatch stopped")

    def _cmd_export(self, args: str):
        """Export session report."""
        s = self.state

        format_type = args.lower().strip() if args else "markdown"

        if format_type == "json":
            report = {
                'session': {
                    'database': f"{s.db_name}@{s.db_host}",
                    'strategy': s.strategy_name,
                    'timestamp': datetime.now().isoformat(),
                },
                'performance': {
                    'baseline_tps': s.baseline_tps,
                    'best_tps': s.best_tps,
                    'current_tps': s.current_tps,
                    'target_tps': s.target_tps,
                    'improvement_pct': ((s.best_tps - s.baseline_tps) / s.baseline_tps * 100) if s.baseline_tps else 0,
                },
                'changes': s.applied_changes,
                'tps_history': s.tps_history,
            }

            filename = f"pg_diagnose_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)

            self._print(f"[green]Exported to {filename}[/]" if self.console else f"Exported to {filename}")

        else:  # markdown
            improvement = ((s.best_tps - s.baseline_tps) / s.baseline_tps * 100) if s.baseline_tps else 0

            md = f"""# PostgreSQL Tuning Report

## Session Info
- **Database:** {s.db_name}@{s.db_host}
- **Strategy:** {s.strategy_name}
- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary
| Metric | Value |
|--------|-------|
| Baseline TPS | {s.baseline_tps:.0f} |
| Best TPS | {s.best_tps:.0f} |
| Improvement | {improvement:.1f}% |
| Target TPS | {s.target_tps:.0f} |
| Rounds | {s.current_round} |

## Applied Changes
"""
            for change in s.applied_changes:
                md += f"\n### {change.get('name')} (Round {change.get('round', '?')})\n"
                md += f"Category: {change.get('category', 'config')}\n"
                for cmd in change.get('pg_configs', []):
                    md += f"- `{cmd}`\n"
                if change.get('os_command'):
                    md += f"- `{change['os_command']}`\n"

            md += f"\n## TPS History\n{' → '.join(f'{t:.0f}' for t in s.tps_history)}\n"

            filename = f"pg_diagnose_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(filename, 'w') as f:
                f.write(md)

            self._print(f"[green]Exported to {filename}[/]" if self.console else f"Exported to {filename}")

    def _cmd_config(self, args: str):
        """Show current PostgreSQL config."""
        s = self.state

        if not s.conn:
            self._print_error("No database connection")
            return

        try:
            with s.conn.cursor() as cur:
                if args:
                    # Show specific parameter
                    cur.execute(f"SHOW {args.strip()}")
                    result = cur.fetchone()
                    if result:
                        self._print(f"[cyan]{args}[/] = [green]{result[0]}[/]" if self.console else f"{args} = {result[0]}")
                else:
                    # Show key parameters
                    params = [
                        'shared_buffers', 'effective_cache_size', 'work_mem',
                        'maintenance_work_mem', 'max_wal_size', 'checkpoint_timeout',
                        'random_page_cost', 'effective_io_concurrency', 'max_connections',
                        'wal_buffers', 'checkpoint_completion_target'
                    ]

                    if self.console:
                        table = Table(title="Key PostgreSQL Settings", box=box.ROUNDED)
                        table.add_column("Parameter", style="cyan")
                        table.add_column("Value", style="green")

                        for param in params:
                            cur.execute(f"SHOW {param}")
                            result = cur.fetchone()
                            if result:
                                table.add_row(param, result[0])

                        self.console.print(table)
                    else:
                        print("\nKey PostgreSQL Settings:")
                        for param in params:
                            cur.execute(f"SHOW {param}")
                            result = cur.fetchone()
                            if result:
                                print(f"  {param} = {result[0]}")

        except Exception as e:
            self._print_error(f"Failed to get config: {e}")

    def _cmd_queries(self, args: str):
        """Show active queries."""
        s = self.state

        if not s.conn:
            self._print_error("No database connection")
            return

        try:
            with s.conn.cursor() as cur:
                cur.execute("""
                    SELECT pid, usename, state,
                           EXTRACT(EPOCH FROM (now() - query_start))::int as duration,
                           LEFT(query, 60) as query
                    FROM pg_stat_activity
                    WHERE state != 'idle'
                      AND pid != pg_backend_pid()
                    ORDER BY query_start
                    LIMIT 20
                """)
                rows = cur.fetchall()

                if not rows:
                    self._print("No active queries.")
                    return

                if self.console:
                    table = Table(title="Active Queries", box=box.ROUNDED)
                    table.add_column("PID", style="cyan")
                    table.add_column("User", style="yellow")
                    table.add_column("State", style="green")
                    table.add_column("Duration", style="red")
                    table.add_column("Query", style="dim")

                    for row in rows:
                        duration = f"{row[3]}s" if row[3] else "-"
                        query = row[4] + "..." if row[4] and len(row[4]) >= 60 else (row[4] or "-")
                        table.add_row(str(row[0]), row[1] or "-", row[2] or "-", duration, query)

                    self.console.print(table)
                else:
                    print("\nActive Queries:")
                    for row in rows:
                        print(f"  PID {row[0]}: {row[2]} ({row[3]}s) - {row[4]}")

        except Exception as e:
            self._print_error(f"Failed to get queries: {e}")


def create_command_handler(state: SessionState) -> CommandHandler:
    """Create a command handler with the given state."""
    return CommandHandler(state)
