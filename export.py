"""
Export - Generate structured Markdown reports for agent consumption.

Philosophy: Do one thing well.
- pg_diagnose produces rich, structured Markdown
- Other agents (Terraform, Ansible, etc.) consume this Markdown
- Single source of truth, machine-parseable, human-readable

Output includes:
- Metadata (YAML block for machine parsing)
- Executive Summary (AI narrative)
- Hardware Recommendations
- OS Configuration (kernel, disk, network)
- Database Configuration (PostgreSQL, pooling, architecture)
- Concrete Instructions (step-by-step shell scripts)
- Validation Commands
- Rollback Information
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class StructuredMarkdownExporter:
    """
    Generate structured Markdown reports optimized for both humans and AI agents.

    The output is designed to be:
    1. Human-readable with clear sections
    2. Machine-parseable with YAML/code blocks
    3. Actionable with concrete instructions
    4. Verifiable with validation commands
    5. Reversible with rollback info
    """

    def __init__(self, session_data: Dict[str, Any], agent=None):
        """
        Initialize with session data.

        Args:
            session_data: Dict containing session info, changes, metrics, etc.
            agent: Optional AI agent for generating executive summary
        """
        self.data = session_data
        self.agent = agent

    def export(self) -> str:
        """Generate the complete structured Markdown report."""
        sections = [
            self._header(),
            self._metadata_section(),
            self._executive_summary(),
            self._performance_timeline(),
            self._hardware_recommendations(),
            self._os_configuration(),
            self._database_configuration(),
            self._concrete_instructions(),
            self._validation_section(),
            self._rollback_section(),
            self._footer(),
        ]

        return "\n\n".join(filter(None, sections))

    def _header(self) -> str:
        """Generate report header."""
        return """# PostgreSQL Tuning Report

> **This report is structured for both human review and automated agent consumption.**
> Code blocks with language tags are machine-parseable configuration."""

    def _metadata_section(self) -> str:
        """Generate YAML metadata block for machine parsing."""
        d = self.data

        baseline = d.get('baseline_tps', 0)
        best = d.get('best_tps', 0)
        target = d.get('target_tps', 0)
        improvement = ((best - baseline) / baseline * 100) if baseline > 0 else 0
        target_achieved = best >= target * 0.9 if target > 0 else False
        gap_pct = ((target - best) / best * 100) if best > 0 and target > 0 else 0

        return f"""## Metadata

```yaml
# Session identification
session_id: "{d.get('session_id', 'unknown')}"
generated_at: "{datetime.now().isoformat()}"
tool_version: "3.0"

# Target system
target:
  host: "{d.get('db_host', 'unknown')}"
  port: {d.get('db_port', 5432)}
  database: "{d.get('db_name', 'unknown')}"
  user: "{d.get('db_user', 'postgres')}"

# Performance results
performance:
  baseline_tps: {baseline:.0f}
  best_tps: {best:.0f}
  target_tps: {target:.0f}
  improvement_pct: {improvement:.1f}
  target_achieved: {str(target_achieved).lower()}
  gap_to_target_pct: {gap_pct:.1f}

# Session details
session:
  strategy: "{d.get('strategy_name', 'unknown')}"
  rounds_completed: {d.get('current_round', 0)}
  status: "{'completed' if target_achieved else 'partial'}"

# System context (for downstream agents)
system:
  ssh_host: "{d.get('ssh_host', '')}"
  ssh_user: "{d.get('ssh_user', 'ubuntu')}"
  ssh_port: {d.get('ssh_port', 22)}
```"""

    def _executive_summary(self) -> str:
        """Generate executive summary - AI narrative for humans."""
        d = self.data

        baseline = d.get('baseline_tps', 0)
        best = d.get('best_tps', 0)
        target = d.get('target_tps', 0)
        improvement = ((best - baseline) / baseline * 100) if baseline > 0 else 0
        target_achieved = best >= target * 0.9 if target > 0 else False

        strategy = d.get('strategy_name', 'Performance Optimization')
        rounds = d.get('current_round', 0)

        # Build summary based on results
        if target_achieved:
            status_text = f"**Target Achieved!** The optimization successfully reached the target of {target:,.0f} TPS."
            outcome = "success"
        elif improvement > 50:
            status_text = f"**Significant Improvement.** Performance improved by {improvement:.0f}% but target was not fully achieved."
            outcome = "partial_success"
        elif improvement > 20:
            status_text = f"**Moderate Improvement.** Performance improved by {improvement:.0f}%. Additional tuning or hardware upgrades may be needed."
            outcome = "needs_more"
        else:
            status_text = f"**Limited Improvement.** Only {improvement:.0f}% gain achieved. Hardware limitations likely."
            outcome = "hardware_limited"

        summary = f"""## Executive Summary

{status_text}

### Key Results

| Metric | Value |
|--------|-------|
| Strategy | {strategy} |
| Rounds Completed | {rounds} |
| Baseline TPS | {baseline:,.0f} |
| Best TPS | {best:,.0f} |
| Improvement | +{improvement:.1f}% |
| Target TPS | {target:,.0f} |
| Target Status | {'Achieved' if target_achieved else 'Not Achieved'} |

### Outcome Classification

```yaml
outcome: "{outcome}"
recommendation: "{'Apply all configurations' if target_achieved else 'Review hardware recommendations'}"
confidence: {0.9 if target_achieved else 0.7}
```"""

        return summary

    def _performance_timeline(self) -> str:
        """Generate TPS timeline with round-by-round details."""
        d = self.data
        tps_history = d.get('tps_history', [])
        changes = d.get('applied_changes', [])

        if not tps_history:
            return ""

        md = """## Performance Timeline

### TPS Progression

| Round | TPS | Change | Cumulative |
|-------|-----|--------|------------|
"""
        baseline = tps_history[0] if tps_history else 0

        for i, tps in enumerate(tps_history):
            label = "Baseline" if i == 0 else f"Round {i}"

            if i > 0:
                round_change = ((tps - tps_history[i-1]) / tps_history[i-1] * 100) if tps_history[i-1] > 0 else 0
                cumulative = ((tps - baseline) / baseline * 100) if baseline > 0 else 0
                change_str = f"+{round_change:.1f}%" if round_change >= 0 else f"{round_change:.1f}%"
                cumulative_str = f"+{cumulative:.1f}%"
            else:
                change_str = "-"
                cumulative_str = "-"

            md += f"| {label} | {tps:,.0f} | {change_str} | {cumulative_str} |\n"

        # Add changes per round
        if changes:
            md += "\n### Changes by Round\n"

            by_round = {}
            for change in changes:
                r = change.get('round', 0)
                if r not in by_round:
                    by_round[r] = []
                by_round[r].append(change)

            for round_num in sorted(by_round.keys()):
                round_label = "Initial" if round_num == 0 else f"Round {round_num}"
                md += f"\n**{round_label}:**\n"
                for change in by_round[round_num]:
                    md += f"- {change.get('name', 'Unknown')} ({change.get('category', 'config')})\n"

        return md

    def _hardware_recommendations(self) -> str:
        """Generate hardware recommendations based on gap analysis."""
        d = self.data

        best = d.get('best_tps', 0)
        target = d.get('target_tps', 0)
        target_achieved = best >= target * 0.9 if target > 0 else True

        if target_achieved:
            return """## Hardware Recommendations

```yaml
status: "no_upgrade_needed"
reason: "Target achieved with current hardware"
```

Current hardware is sufficient for the target workload. Monitor for future growth."""

        gap_pct = ((target - best) / best * 100) if best > 0 else 100

        # Determine recommendation level
        if gap_pct > 50:
            level = "major_upgrade"
            urgency = "high"
        elif gap_pct > 20:
            level = "moderate_upgrade"
            urgency = "medium"
        else:
            level = "minor_upgrade"
            urgency = "low"

        md = f"""## Hardware Recommendations

```yaml
status: "{level}"
urgency: "{urgency}"
gap_to_target: "{gap_pct:.1f}%"
```

### Current Limitations

The performance gap of {gap_pct:.1f}% suggests hardware limitations.

### Recommended Upgrades

"""
        if gap_pct > 50:
            md += """#### CPU
```yaml
current: "4 cores (estimated)"
recommended: "8-16 cores"
rationale: "Significant parallelism needed for query processing"
```

#### Memory
```yaml
current: "16 GB (estimated)"
recommended: "32-64 GB"
rationale: "Larger shared_buffers and work_mem for complex queries"
```

#### Storage
```yaml
current: "SSD/gp3"
recommended: "NVMe RAID or io2"
rationale: "WAL writes and checkpoint I/O are bottlenecks"
iops_target: 10000
throughput_target: "500 MB/s"
```

#### Instance Class (AWS)
```yaml
current: "r5.large or similar"
recommended: "r5.2xlarge or r6i.2xlarge"
alternative: "Consider Aurora PostgreSQL for managed scaling"
```
"""
        elif gap_pct > 20:
            md += """#### CPU
```yaml
recommendation: "Add 2-4 cores"
rationale: "Enable more parallel workers"
```

#### Memory
```yaml
recommendation: "Increase by 50-100%"
rationale: "Better caching, larger work_mem"
```

#### Storage
```yaml
recommendation: "Upgrade to NVMe or provisioned IOPS"
rationale: "Reduce I/O latency for checkpoints"
```
"""
        else:
            md += """#### Fine-Tuning Focus
```yaml
recommendation: "Optimize application layer"
options:
  - "Connection pooling (PgBouncer)"
  - "Query optimization"
  - "Read replicas for read-heavy workloads"
  - "Caching layer (Redis)"
```
"""

        return md

    def _os_configuration(self) -> str:
        """Generate OS-level configuration section."""
        d = self.data
        changes = d.get('applied_changes', [])

        # Extract OS changes
        os_changes = [c for c in changes if c.get('category') == 'os' or c.get('type') == 'os']

        md = """## OS Configuration

### Kernel Parameters (sysctl)

```sysctl
# Memory Management
vm.swappiness = 10
vm.dirty_ratio = 40
vm.dirty_background_ratio = 10
vm.dirty_expire_centisecs = 500
vm.dirty_writeback_centisecs = 100

# Shared Memory (for PostgreSQL)
kernel.shmmax = 17179869184
kernel.shmall = 4194304

# Network
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_fin_timeout = 10
net.ipv4.tcp_tw_reuse = 1
```

### Transparent Hugepages

```shell
# Disable THP (recommended for PostgreSQL)
echo 'never' > /sys/kernel/mm/transparent_hugepage/enabled
echo 'never' > /sys/kernel/mm/transparent_hugepage/defrag
```

### I/O Scheduler

```shell
# For NVMe drives (use 'none' or 'mq-deadline')
echo 'none' > /sys/block/nvme0n1/queue/scheduler

# For SSD drives
echo 'deadline' > /sys/block/sda/queue/scheduler
```

### File Descriptors

```shell
# /etc/security/limits.conf
postgres soft nofile 65535
postgres hard nofile 65535
postgres soft nproc 65535
postgres hard nproc 65535
```
"""

        # Add any applied OS changes
        if os_changes:
            md += "\n### Applied OS Changes\n\n"
            for change in os_changes:
                md += f"**{change.get('name', 'Change')}**\n"
                if change.get('os_command'):
                    md += f"```shell\n{change['os_command']}\n```\n\n"

        return md

    def _database_configuration(self) -> str:
        """Generate database configuration section."""
        d = self.data
        changes = d.get('applied_changes', [])
        baseline_config = d.get('baseline_config', {})

        # Extract PostgreSQL changes
        pg_settings = {}
        for change in changes:
            for cmd in change.get('pg_configs', []):
                if 'ALTER SYSTEM SET' in cmd.upper():
                    try:
                        parts = cmd.upper().replace('ALTER SYSTEM SET', '').strip()
                        if '=' in parts:
                            param, value = parts.split('=', 1)
                            param = param.strip().lower()
                            value = value.strip().strip("'\"").rstrip(';')
                            pg_settings[param] = {
                                'value': value,
                                'round': change.get('round', 0),
                                'name': change.get('name', ''),
                            }
                    except Exception:
                        pass

        md = """## Database Configuration

### PostgreSQL Settings

"""

        # Group settings by category
        categories = {
            'Memory': ['shared_buffers', 'effective_cache_size', 'work_mem', 'maintenance_work_mem', 'wal_buffers', 'huge_pages'],
            'WAL': ['max_wal_size', 'min_wal_size', 'wal_compression', 'wal_level', 'wal_writer_delay'],
            'Checkpoints': ['checkpoint_timeout', 'checkpoint_completion_target', 'checkpoint_flush_after'],
            'Parallelism': ['max_parallel_workers', 'max_parallel_workers_per_gather', 'max_worker_processes', 'parallel_tuple_cost', 'parallel_setup_cost'],
            'Planner': ['random_page_cost', 'effective_io_concurrency', 'default_statistics_target', 'seq_page_cost'],
            'Connections': ['max_connections', 'superuser_reserved_connections'],
        }

        for category, params in categories.items():
            category_settings = [(p, pg_settings[p]) for p in params if p in pg_settings]
            if category_settings:
                md += f"#### {category}\n\n```postgresql\n"
                for param, info in category_settings:
                    md += f"ALTER SYSTEM SET {param} = '{info['value']}';\n"
                md += "```\n\n"

        # Other settings
        other_params = [p for p in pg_settings if not any(p in cat for cat in categories.values())]
        if other_params:
            md += "#### Other\n\n```postgresql\n"
            for param in other_params:
                md += f"ALTER SYSTEM SET {param} = '{pg_settings[param]['value']}';\n"
            md += "```\n\n"

        # Add connection pooling recommendation
        md += """### Connection Pooling (PgBouncer)

```ini
[databases]
* = host=127.0.0.1 port=5432

[pgbouncer]
listen_addr = *
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

# Pool settings
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 50
min_pool_size = 10
reserve_pool_size = 10
reserve_pool_timeout = 3

# Timeouts
server_idle_timeout = 600
client_idle_timeout = 0
```

### Architecture Recommendations

```yaml
current_architecture: "single_node"
recommendations:
  - type: "connection_pooling"
    tool: "pgbouncer"
    status: "recommended"

  - type: "read_replicas"
    count: 2
    status: "consider_if_read_heavy"

  - type: "high_availability"
    tool: "patroni"
    status: "recommended_for_production"
```
"""

        return md

    def _concrete_instructions(self) -> str:
        """Generate step-by-step concrete instructions for automation."""
        d = self.data
        changes = d.get('applied_changes', [])

        # Build PostgreSQL ALTER commands
        alter_commands = []
        for change in changes:
            for cmd in change.get('pg_configs', []):
                if 'ALTER SYSTEM SET' in cmd.upper():
                    alter_commands.append(cmd.rstrip(';') + ';')

        md = """## Concrete Instructions

> **For Automation Agents:** This section contains executable scripts.
> Each step is independent and idempotent where possible.

### Step 1: Apply Kernel Parameters

```shell
#!/bin/bash
set -euo pipefail

echo "Applying kernel parameters..."

cat << 'EOF' > /etc/sysctl.d/99-postgresql.conf
# PostgreSQL Optimizations
vm.swappiness = 10
vm.dirty_ratio = 40
vm.dirty_background_ratio = 10
vm.dirty_expire_centisecs = 500
vm.dirty_writeback_centisecs = 100
kernel.shmmax = 17179869184
kernel.shmall = 4194304
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_fin_timeout = 10
net.ipv4.tcp_tw_reuse = 1
EOF

sysctl --system

echo "Kernel parameters applied."
```

### Step 2: Disable Transparent Hugepages

```shell
#!/bin/bash
set -euo pipefail

echo "Disabling Transparent Hugepages..."

# Immediate effect
echo 'never' > /sys/kernel/mm/transparent_hugepage/enabled
echo 'never' > /sys/kernel/mm/transparent_hugepage/defrag

# Persistent (systemd service)
cat << 'EOF' > /etc/systemd/system/disable-thp.service
[Unit]
Description=Disable Transparent Huge Pages
DefaultDependencies=no
After=sysinit.target local-fs.target
Before=basic.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo never > /sys/kernel/mm/transparent_hugepage/enabled'
ExecStart=/bin/sh -c 'echo never > /sys/kernel/mm/transparent_hugepage/defrag'

[Install]
WantedBy=basic.target
EOF

systemctl daemon-reload
systemctl enable disable-thp.service

echo "THP disabled."
```

### Step 3: Configure File Descriptors

```shell
#!/bin/bash
set -euo pipefail

echo "Configuring file descriptors..."

cat << 'EOF' >> /etc/security/limits.conf
# PostgreSQL limits
postgres soft nofile 65535
postgres hard nofile 65535
postgres soft nproc 65535
postgres hard nproc 65535
EOF

echo "File descriptors configured."
```

### Step 4: Apply PostgreSQL Configuration

```shell
#!/bin/bash
set -euo pipefail

echo "Applying PostgreSQL configuration..."

sudo -u postgres psql << 'EOSQL'
"""

        # Add all ALTER SYSTEM commands
        for cmd in alter_commands:
            md += f"{cmd}\n"

        md += """EOSQL

echo "PostgreSQL configuration applied."
```

### Step 5: Reload PostgreSQL Configuration

```shell
#!/bin/bash
set -euo pipefail

echo "Reloading PostgreSQL configuration..."

sudo -u postgres psql -c "SELECT pg_reload_conf();"

# Verify reload
sudo -u postgres psql -c "SELECT pg_conf_load_time();"

echo "Configuration reloaded."
```

### Step 6: Restart PostgreSQL (if required)

```shell
#!/bin/bash
set -euo pipefail

echo "Restarting PostgreSQL..."

# Check if restart is needed (shared_buffers, huge_pages, etc.)
RESTART_NEEDED=$(sudo -u postgres psql -t -c "
    SELECT count(*) FROM pg_settings
    WHERE pending_restart = true;
")

if [ "$RESTART_NEEDED" -gt 0 ]; then
    echo "Pending restart detected. Restarting..."
    systemctl restart postgresql
    sleep 5
    systemctl is-active postgresql
    echo "PostgreSQL restarted successfully."
else
    echo "No restart needed."
fi
```

### Combined Apply Script

```shell
#!/bin/bash
# apply_all.sh - Run all configuration steps
set -euo pipefail

echo "=== PostgreSQL Tuning Application ==="
echo "Started at: $(date)"

# Step 1: Kernel
echo -e "\\n>>> Step 1: Kernel Parameters"
# ... (include step 1 content)

# Step 2: THP
echo -e "\\n>>> Step 2: Disable THP"
# ... (include step 2 content)

# Step 3: File descriptors
echo -e "\\n>>> Step 3: File Descriptors"
# ... (include step 3 content)

# Step 4: PostgreSQL
echo -e "\\n>>> Step 4: PostgreSQL Configuration"
# ... (include step 4 content)

# Step 5: Reload
echo -e "\\n>>> Step 5: Reload Configuration"
# ... (include step 5 content)

# Step 6: Restart if needed
echo -e "\\n>>> Step 6: Restart Check"
# ... (include step 6 content)

echo -e "\\n=== All steps completed ==="
echo "Finished at: $(date)"
```
"""

        return md

    def _validation_section(self) -> str:
        """Generate validation commands to verify configuration."""
        d = self.data
        changes = d.get('applied_changes', [])

        # Extract expected values
        expected = {}
        for change in changes:
            for cmd in change.get('pg_configs', []):
                if 'ALTER SYSTEM SET' in cmd.upper():
                    try:
                        parts = cmd.upper().replace('ALTER SYSTEM SET', '').strip()
                        if '=' in parts:
                            param, value = parts.split('=', 1)
                            param = param.strip().lower()
                            value = value.strip().strip("'\"").rstrip(';')
                            expected[param] = value
                    except Exception:
                        pass

        md = """## Validation

> Run these commands to verify configuration was applied correctly.

### PostgreSQL Settings Verification

```shell
#!/bin/bash
set -euo pipefail

echo "=== PostgreSQL Configuration Validation ==="

ERRORS=0

check_setting() {
    local setting=$1
    local expected=$2
    local actual=$(sudo -u postgres psql -t -c "SHOW $setting;" | xargs)

    if [ "$actual" = "$expected" ]; then
        echo "[OK] $setting = $actual"
    else
        echo "[FAIL] $setting: expected '$expected', got '$actual'"
        ERRORS=$((ERRORS + 1))
    fi
}

# Check key settings
"""

        for param, value in list(expected.items())[:10]:  # Limit to key settings
            md += f'check_setting "{param}" "{value}"\n'

        md += """
echo ""
if [ $ERRORS -eq 0 ]; then
    echo "All settings verified successfully!"
    exit 0
else
    echo "Found $ERRORS configuration errors"
    exit 1
fi
```

### OS Configuration Verification

```shell
#!/bin/bash
set -euo pipefail

echo "=== OS Configuration Validation ==="

# Kernel parameters
echo "Checking sysctl values..."
sysctl vm.swappiness | grep "= 10" || echo "[WARN] vm.swappiness not set"
sysctl vm.dirty_ratio | grep "= 40" || echo "[WARN] vm.dirty_ratio not set"

# THP
echo "Checking THP..."
cat /sys/kernel/mm/transparent_hugepage/enabled | grep "\\[never\\]" || echo "[WARN] THP not disabled"

# File descriptors
echo "Checking file descriptors..."
ulimit -n | grep -E "^65535$" || echo "[WARN] File descriptors not configured"

echo "OS validation complete."
```

### Expected Results

```yaml
validation_checks:
  postgresql:
"""
        for param, value in list(expected.items())[:10]:
            md += f'    - setting: "{param}"\n      expected: "{value}"\n'

        md += """  os:
    - check: "sysctl vm.swappiness"
      expected: "10"
    - check: "cat /sys/kernel/mm/transparent_hugepage/enabled"
      expected: "[never]"
```
"""

        return md

    def _rollback_section(self) -> str:
        """Generate rollback instructions with previous values."""
        d = self.data
        baseline_config = d.get('baseline_config', {})

        md = """## Rollback

> Use these commands to restore previous configuration if issues arise.

### PostgreSQL Rollback

```shell
#!/bin/bash
set -euo pipefail

echo "Rolling back PostgreSQL configuration..."

# Reset all ALTER SYSTEM settings
sudo -u postgres psql -c "ALTER SYSTEM RESET ALL;"

# Reload to apply defaults
sudo -u postgres psql -c "SELECT pg_reload_conf();"

echo "Configuration reset to defaults."
```

### Previous Values

```yaml
# These were the original values before tuning
previous_configuration:
"""

        if baseline_config:
            for param, value in list(baseline_config.items())[:15]:
                md += f'  {param}: "{value}"\n'
        else:
            md += "  # Baseline configuration not captured\n"
            md += "  # Use SHOW ALL; to check current defaults\n"

        md += """```

### OS Rollback

```shell
#!/bin/bash
set -euo pipefail

echo "Rolling back OS configuration..."

# Remove custom sysctl
rm -f /etc/sysctl.d/99-postgresql.conf
sysctl --system

# Re-enable THP (if needed)
echo 'always' > /sys/kernel/mm/transparent_hugepage/enabled
systemctl disable disable-thp.service 2>/dev/null || true

echo "OS configuration rolled back."
```
"""

        return md

    def _footer(self) -> str:
        """Generate report footer."""
        return """---

## Report Metadata

```yaml
generator: "pg_diagnose"
version: "3.0"
format: "structured_markdown"
purpose: "human_review_and_agent_consumption"
```

### Usage Notes

1. **Human Review**: Read Executive Summary and Hardware Recommendations first
2. **Automation**: Use "Concrete Instructions" section for scripts
3. **Validation**: Run validation scripts after applying changes
4. **Rollback**: Keep rollback commands ready before applying

### Downstream Agent Instructions

This report can be consumed by:
- **Terraform Agent**: Extract `database_configuration` for RDS parameter groups
- **Ansible Agent**: Extract `concrete_instructions` for playbook tasks
- **Monitoring Agent**: Extract `validation` checks for health monitoring

---
*Generated by pg_diagnose v3.0 - Structured Markdown Export*
"""


def export_session(session_data: Dict, format: str = 'markdown',
                   output_path: Optional[str] = None, agent=None) -> str:
    """
    Export session data to Markdown format.

    Args:
        session_data: Session data dict
        format: Output format (only 'markdown' supported - other agents handle conversion)
        output_path: Optional file path to write to
        agent: Optional AI agent for enhanced summaries

    Returns:
        Exported content as string
    """
    exporter = StructuredMarkdownExporter(session_data, agent)
    content = exporter.export()

    if output_path:
        Path(output_path).write_text(content)

    return content


class WorkspaceExporter:
    """
    Export workspace with multiple sessions merged.
    """

    def __init__(self, workspace, agent=None):
        """
        Initialize workspace exporter.

        Args:
            workspace: Workspace object
            agent: Optional AI agent for LLM analysis
        """
        self.workspace = workspace
        self.agent = agent

    def export(self, output_dir: Path = None) -> Dict[str, str]:
        """
        Export workspace to structured Markdown.

        Returns dict of {filename: content}.
        """
        if output_dir is None:
            output_dir = self.workspace.path / "exports"
        output_dir.mkdir(exist_ok=True)

        exports = {}

        # Get all archived sessions
        archived = self.workspace.get_archived_sessions()

        if not archived:
            exports['error.md'] = "# No Archived Sessions\n\nNo sessions to export."
            return exports

        if len(archived) == 1:
            # Single session export
            session = archived[0]
            session_data = self._session_to_dict(session)
            exporter = StructuredMarkdownExporter(session_data, self.agent)
            exports['tuning_report.md'] = exporter.export()
        else:
            # Multiple sessions - merge
            exports['tuning_report.md'] = self._export_merged(archived)

        # Write files
        for filename, content in exports.items():
            (output_dir / filename).write_text(content)

        return exports

    def _session_to_dict(self, session) -> Dict[str, Any]:
        """Convert session object to dict for exporter."""
        data = session.data
        baseline = data.baseline if hasattr(data, 'baseline') else {}

        return {
            'session_id': session.name,
            'db_host': self.workspace.data.db_host if hasattr(self.workspace, 'data') else '',
            'db_port': self.workspace.data.db_port if hasattr(self.workspace, 'data') else 5432,
            'db_name': self.workspace.data.db_name if hasattr(self.workspace, 'data') else '',
            'db_user': self.workspace.data.db_user if hasattr(self.workspace, 'data') else 'postgres',
            'ssh_host': self.workspace.data.ssh_host if hasattr(self.workspace, 'data') else '',
            'ssh_user': self.workspace.data.ssh_user if hasattr(self.workspace, 'data') else 'ubuntu',
            'strategy_name': data.strategy_name,
            'baseline_tps': baseline.get('tps', 0) if baseline else 0,
            'best_tps': data.best_tps,
            'target_tps': data.target_tps,
            'current_round': len(data.rounds),
            'tps_history': [baseline.get('tps', 0)] + [r.get('tps', 0) for r in data.rounds] if baseline else [],
            'applied_changes': data.sweet_spot_changes,
            'baseline_config': {},
        }

    def _export_merged(self, sessions: List) -> str:
        """Export merged report from multiple sessions."""
        # Find best session
        best_session = max(sessions, key=lambda s: s.data.best_tps)

        # Use best session as base
        merged_data = self._session_to_dict(best_session)

        # Add session comparison header
        header = f"""# Workspace Tuning Report

## Session Comparison

| Session | Strategy | Best TPS | Improvement |
|---------|----------|----------|-------------|
"""
        for session in sessions:
            baseline = session.data.baseline.get('tps', 0) if session.data.baseline else 0
            improvement = ((session.data.best_tps - baseline) / baseline * 100) if baseline > 0 else 0
            header += f"| {session.name} | {session.data.strategy_name} | {session.data.best_tps:,.0f} | +{improvement:.1f}% |\n"

        header += f"\n**Best Session:** {best_session.data.strategy_name} ({best_session.data.best_tps:,.0f} TPS)\n"
        header += "\n---\n\n"

        # Generate full report using best session
        exporter = StructuredMarkdownExporter(merged_data, self.agent)
        report = exporter.export()

        # Insert comparison header after main header
        lines = report.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('## Metadata'):
                insert_idx = i
                break

        lines.insert(insert_idx, header)

        return '\n'.join(lines)


def compose_workspace_export(workspace, agent=None, output_dir: Path = None) -> Dict[str, str]:
    """
    Convenience function to export a workspace.

    Args:
        workspace: Workspace object
        agent: Optional AI agent
        output_dir: Output directory

    Returns:
        Dict of {filename: content}
    """
    exporter = WorkspaceExporter(workspace, agent)
    return exporter.export(output_dir)
