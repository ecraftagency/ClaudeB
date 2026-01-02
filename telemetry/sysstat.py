"""
SysStatCollector - Collects detailed system statistics.

Provides:
- Memory details from /proc/meminfo (dirty pages, writeback, etc.)
- Network statistics from /proc/net/dev and ss
- Load average from /proc/loadavg
- Disk space usage
"""

import subprocess
import re
from typing import Dict, Optional, Any


class SysStatCollector:
    """
    Collects detailed system statistics beyond vmstat.

    Supports both local and remote (SSH) collection.
    """

    def __init__(self, ssh_config: Optional[Dict] = None):
        self.ssh_config = ssh_config

    def _run_command(self, cmd: str, timeout: int = 10) -> Optional[str]:
        """Run a command locally or via SSH."""
        try:
            if self.ssh_config:
                full_cmd = [
                    "ssh", "-o", "StrictHostKeyChecking=no",
                    f"{self.ssh_config['user']}@{self.ssh_config['host']}",
                    cmd
                ]
                result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
            else:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass

        return None

    def collect_all(self) -> Dict[str, Any]:
        """Collect all system statistics."""
        return {
            'meminfo': self.get_meminfo(),
            'load_avg': self.get_load_average(),
            'network': self.get_network_stats(),
            'disk_space': self.get_disk_space(),
            'cpu_info': self.get_cpu_info(),
        }

    def get_meminfo(self) -> Dict[str, Any]:
        """Get detailed memory info from /proc/meminfo."""
        output = self._run_command("cat /proc/meminfo")
        if not output:
            return {}

        meminfo = {}
        for line in output.strip().split('\n'):
            if ':' not in line:
                continue

            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Extract numeric value (remove 'kB' suffix)
            match = re.match(r'(\d+)', value)
            if match:
                meminfo[key] = int(match.group(1))

        # Calculate useful metrics
        mem_total = meminfo.get('MemTotal', 0)
        mem_free = meminfo.get('MemFree', 0)
        mem_available = meminfo.get('MemAvailable', 0)
        buffers = meminfo.get('Buffers', 0)
        cached = meminfo.get('Cached', 0)
        dirty = meminfo.get('Dirty', 0)
        writeback = meminfo.get('Writeback', 0)
        swap_total = meminfo.get('SwapTotal', 0)
        swap_free = meminfo.get('SwapFree', 0)

        return {
            'mem_total_kb': mem_total,
            'mem_free_kb': mem_free,
            'mem_available_kb': mem_available,
            'mem_used_kb': mem_total - mem_free - buffers - cached,
            'mem_used_pct': ((mem_total - mem_available) / mem_total * 100) if mem_total > 0 else 0,
            'buffers_kb': buffers,
            'cached_kb': cached,
            'dirty_kb': dirty,
            'writeback_kb': writeback,
            'dirty_pct': (dirty / mem_total * 100) if mem_total > 0 else 0,
            'swap_total_kb': swap_total,
            'swap_free_kb': swap_free,
            'swap_used_kb': swap_total - swap_free,
            'swap_used_pct': ((swap_total - swap_free) / swap_total * 100) if swap_total > 0 else 0,
            'page_tables_kb': meminfo.get('PageTables', 0),
            'slab_kb': meminfo.get('Slab', 0),
            'slab_reclaimable_kb': meminfo.get('SReclaimable', 0),
            'slab_unreclaimable_kb': meminfo.get('SUnreclaim', 0),
            'hugepages_total': meminfo.get('HugePages_Total', 0),
            'hugepages_free': meminfo.get('HugePages_Free', 0),
            'hugepage_size_kb': meminfo.get('Hugepagesize', 0),
            'committed_as_kb': meminfo.get('Committed_AS', 0),
            'vmalloc_used_kb': meminfo.get('VmallocUsed', 0),
        }

    def get_load_average(self) -> Dict[str, float]:
        """Get system load average from /proc/loadavg."""
        output = self._run_command("cat /proc/loadavg")
        if not output:
            return {}

        parts = output.strip().split()
        if len(parts) >= 3:
            # Also parse running/total processes
            procs = parts[3].split('/') if len(parts) > 3 else ['0', '0']

            return {
                'load_1min': float(parts[0]),
                'load_5min': float(parts[1]),
                'load_15min': float(parts[2]),
                'running_processes': int(procs[0]) if len(procs) > 0 else 0,
                'total_processes': int(procs[1]) if len(procs) > 1 else 0,
            }

        return {}

    def get_network_stats(self) -> Dict[str, Any]:
        """Get network interface statistics."""
        result = {}

        # Get interface stats from /proc/net/dev
        dev_output = self._run_command("cat /proc/net/dev")
        if dev_output:
            interfaces = {}
            for line in dev_output.strip().split('\n')[2:]:  # Skip header lines
                if ':' not in line:
                    continue

                iface, data = line.split(':', 1)
                iface = iface.strip()
                parts = data.split()

                if len(parts) >= 16:
                    interfaces[iface] = {
                        'rx_bytes': int(parts[0]),
                        'rx_packets': int(parts[1]),
                        'rx_errors': int(parts[2]),
                        'rx_dropped': int(parts[3]),
                        'tx_bytes': int(parts[8]),
                        'tx_packets': int(parts[9]),
                        'tx_errors': int(parts[10]),
                        'tx_dropped': int(parts[11]),
                    }

            result['interfaces'] = interfaces

        # Get TCP socket stats using ss
        ss_output = self._run_command("ss -s")
        if ss_output:
            tcp_stats = {}
            for line in ss_output.strip().split('\n'):
                if line.startswith('TCP:'):
                    # Parse: TCP:   123 (estab 45, closed 6, ...)
                    match = re.search(r'TCP:\s+(\d+)', line)
                    if match:
                        tcp_stats['total'] = int(match.group(1))

                    estab_match = re.search(r'estab\s+(\d+)', line)
                    if estab_match:
                        tcp_stats['established'] = int(estab_match.group(1))

                    closed_match = re.search(r'closed\s+(\d+)', line)
                    if closed_match:
                        tcp_stats['closed'] = int(closed_match.group(1))

                    time_wait_match = re.search(r'timewait\s+(\d+)', line)
                    if time_wait_match:
                        tcp_stats['time_wait'] = int(time_wait_match.group(1))

            result['tcp'] = tcp_stats

        # Get TCP retransmit stats
        netstat_output = self._run_command("cat /proc/net/netstat")
        if netstat_output:
            lines = netstat_output.strip().split('\n')
            tcp_ext_keys = None
            tcp_ext_vals = None

            for i, line in enumerate(lines):
                if line.startswith('TcpExt:'):
                    if tcp_ext_keys is None:
                        tcp_ext_keys = line.split()[1:]
                    else:
                        tcp_ext_vals = line.split()[1:]

            if tcp_ext_keys and tcp_ext_vals:
                tcp_ext = dict(zip(tcp_ext_keys, [int(v) for v in tcp_ext_vals]))
                result['tcp_ext'] = {
                    'retransmits': tcp_ext.get('TCPRetransSegs', 0),
                    'fast_retransmits': tcp_ext.get('TCPFastRetrans', 0),
                    'slow_start_retransmits': tcp_ext.get('TCPSlowStartRetrans', 0),
                    'syn_retransmits': tcp_ext.get('TCPSynRetrans', 0),
                    'lost_retransmits': tcp_ext.get('TCPLostRetransmit', 0),
                    'timeouts': tcp_ext.get('TCPTimeouts', 0),
                }

        return result

    def get_disk_space(self) -> Dict[str, Any]:
        """Get disk space usage for data directories."""
        output = self._run_command("df -P")
        if not output:
            return {}

        filesystems = {}
        for line in output.strip().split('\n')[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 6:
                mount = parts[5]
                # Focus on important mounts
                if mount in ['/', '/var', '/var/lib', '/var/lib/postgresql'] or \
                   mount.startswith('/data') or mount.startswith('/pg'):
                    filesystems[mount] = {
                        'filesystem': parts[0],
                        'total_kb': int(parts[1]),
                        'used_kb': int(parts[2]),
                        'available_kb': int(parts[3]),
                        'used_pct': int(parts[4].rstrip('%')),
                    }

        return {'filesystems': filesystems}

    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        result = {}

        # Get number of CPUs
        output = self._run_command("nproc")
        if output:
            result['cpu_count'] = int(output.strip())

        # Get CPU frequency
        freq_output = self._run_command("cat /proc/cpuinfo | grep 'cpu MHz' | head -1")
        if freq_output:
            match = re.search(r':\s*([\d.]+)', freq_output)
            if match:
                result['cpu_mhz'] = float(match.group(1))

        # Get CPU model
        model_output = self._run_command("cat /proc/cpuinfo | grep 'model name' | head -1")
        if model_output:
            match = re.search(r':\s*(.+)', model_output)
            if match:
                result['cpu_model'] = match.group(1).strip()

        # Get context switches per second from /proc/stat
        stat_output = self._run_command("cat /proc/stat")
        if stat_output:
            for line in stat_output.split('\n'):
                if line.startswith('ctxt '):
                    result['context_switches'] = int(line.split()[1])
                elif line.startswith('processes '):
                    result['total_forks'] = int(line.split()[1])
                elif line.startswith('procs_running '):
                    result['procs_running'] = int(line.split()[1])
                elif line.startswith('procs_blocked '):
                    result['procs_blocked'] = int(line.split()[1])

        return result
