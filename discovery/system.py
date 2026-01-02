"""
SystemScanner - Scans system hardware and OS configuration.

Uses sysstat and standard OS tools (no Prometheus required).
"""

import os
import re
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from ..protocol.context import SystemContext


@dataclass
class SystemScannerConfig:
    """Configuration for system scanning."""
    ssh_host: Optional[str] = None
    ssh_port: int = 22
    ssh_user: str = "ubuntu"
    ssh_key: Optional[str] = None


class SystemScanner:
    """
    Scans system hardware and OS for context packet.

    Can run locally or remotely via SSH.
    """

    def __init__(self, config: Optional[SystemScannerConfig] = None):
        self.config = config or SystemScannerConfig()
        self.is_remote = self.config.ssh_host is not None

    def _run_command(self, cmd: str) -> str:
        """Run a command locally or remotely."""
        if self.is_remote:
            ssh_cmd = ["ssh"]
            if self.config.ssh_port != 22:
                ssh_cmd.extend(["-p", str(self.config.ssh_port)])
            if self.config.ssh_key:
                ssh_cmd.extend(["-i", self.config.ssh_key])
            ssh_cmd.extend([
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                f"{self.config.ssh_user}@{self.config.ssh_host}",
                cmd
            ])
            result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        return result.stdout.strip()

    def scan(self, pg_version: Optional[int] = None, pg_version_full: Optional[str] = None) -> SystemContext:
        """Perform full system scan."""
        cpu_info = self._get_cpu_info()
        memory_info = self._get_memory_info()
        storage_info = self._get_storage_info()
        os_info = self._get_os_info()

        return SystemContext(
            cpu_architecture=cpu_info.get('architecture', 'x86_64'),
            cpu_cores=cpu_info.get('cores', 1),
            cpu_model=cpu_info.get('model', 'Unknown'),
            ram_total_gb=memory_info.get('total_gb', 0),
            storage_topology=storage_info.get('topology', 'single_disk'),
            storage_devices=storage_info.get('devices', []),
            os_version=os_info.get('version', 'Unknown'),
            kernel_version=os_info.get('kernel', 'Unknown'),
            pg_version=pg_version or 16,
            pg_version_full=pg_version_full or "16.0",
        )

    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        info = {
            'architecture': 'x86_64',
            'cores': 1,
            'model': 'Unknown',
        }

        try:
            # Get architecture
            arch = self._run_command("uname -m")
            if arch:
                info['architecture'] = arch

            # Get CPU info from /proc/cpuinfo
            cpuinfo = self._run_command("cat /proc/cpuinfo")
            if cpuinfo:
                # Count cores
                cores = len(re.findall(r'^processor\s*:', cpuinfo, re.MULTILINE))
                info['cores'] = cores if cores > 0 else 1

                # Get model name
                model_match = re.search(r'model name\s*:\s*(.+)', cpuinfo)
                if model_match:
                    info['model'] = model_match.group(1).strip()
                else:
                    # Try ARM format
                    model_match = re.search(r'CPU implementer.*\nCPU architecture.*\nCPU variant.*\nCPU part.*', cpuinfo)
                    if 'aarch64' in arch or 'arm' in arch.lower():
                        info['model'] = 'ARM Processor'

            # Alternative: use nproc
            nproc = self._run_command("nproc")
            if nproc and nproc.isdigit():
                info['cores'] = int(nproc)

        except Exception as e:
            pass

        return info

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        info = {'total_gb': 0, 'used_gb': 0, 'available_gb': 0, 'cached_gb': 0}

        try:
            meminfo = self._run_command("cat /proc/meminfo")
            if meminfo:
                # Parse MemTotal
                total_match = re.search(r'MemTotal:\s*(\d+)\s*kB', meminfo)
                if total_match:
                    info['total_gb'] = round(int(total_match.group(1)) / (1024 ** 2), 2)

                # Parse MemAvailable
                avail_match = re.search(r'MemAvailable:\s*(\d+)\s*kB', meminfo)
                if avail_match:
                    info['available_gb'] = round(int(avail_match.group(1)) / (1024 ** 2), 2)

                # Parse Cached
                cached_match = re.search(r'^Cached:\s*(\d+)\s*kB', meminfo, re.MULTILINE)
                if cached_match:
                    info['cached_gb'] = round(int(cached_match.group(1)) / (1024 ** 2), 2)

                # Calculate used
                info['used_gb'] = round(info['total_gb'] - info['available_gb'], 2)

        except Exception:
            pass

        return info

    def _get_storage_info(self) -> Dict[str, Any]:
        """Get storage information including mount points and RAID arrays."""
        info = {
            'topology': 'single_disk',
            'devices': [],
            'mount_points': [],
            'raid_arrays': [],
        }

        try:
            # Get block devices using lsblk
            lsblk = self._run_command("lsblk -J -b -o NAME,TYPE,SIZE,MODEL,ROTA,MOUNTPOINT")
            if lsblk:
                import json
                try:
                    data = json.loads(lsblk)
                    devices = []
                    for dev in data.get('blockdevices', []):
                        if dev.get('type') == 'disk':
                            size_bytes = int(dev.get('size', 0))
                            is_rotational = dev.get('rota', True)
                            name = dev.get('name', '')

                            # Determine type
                            if 'nvme' in name:
                                dev_type = 'nvme'
                            elif not is_rotational:
                                dev_type = 'ssd'
                            else:
                                dev_type = 'hdd'

                            devices.append({
                                'name': name,
                                'type': dev_type,
                                'size_gb': round(size_bytes / (1024 ** 3), 2),
                                'model': dev.get('model', ''),
                            })

                    info['devices'] = devices
                except json.JSONDecodeError:
                    pass

            # Get mount points with sizes using df
            df_output = self._run_command("df -BG --output=source,size,used,avail,target 2>/dev/null | grep -E '^/dev|^md'")
            if df_output:
                mount_points = []
                for line in df_output.strip().split('\n'):
                    parts = line.split()
                    if len(parts) >= 5:
                        mount_points.append({
                            'device': parts[0],
                            'size_gb': int(parts[1].replace('G', '')) if parts[1].endswith('G') else 0,
                            'used_gb': int(parts[2].replace('G', '')) if parts[2].endswith('G') else 0,
                            'avail_gb': int(parts[3].replace('G', '')) if parts[3].endswith('G') else 0,
                            'mount': parts[4],
                        })
                info['mount_points'] = mount_points

            # Check for RAID configuration
            mdstat = self._run_command("cat /proc/mdstat 2>/dev/null || echo ''")
            if mdstat and 'md' in mdstat:
                # Parse RAID arrays
                raid_arrays = []
                current_array = None
                for line in mdstat.split('\n'):
                    if line.startswith('md'):
                        parts = line.split(':')
                        if len(parts) >= 2:
                            array_name = parts[0].strip()
                            array_info = parts[1].strip()
                            raid_level = 'unknown'
                            if 'raid10' in array_info:
                                raid_level = 'raid10'
                            elif 'raid1' in array_info:
                                raid_level = 'raid1'
                            elif 'raid0' in array_info:
                                raid_level = 'raid0'
                            elif 'raid5' in array_info:
                                raid_level = 'raid5'

                            # Extract member devices
                            members = re.findall(r'(nvme\d+n\d+p?\d*|sd[a-z]+\d*)\[\d+\]', array_info)
                            raid_arrays.append({
                                'name': array_name,
                                'level': raid_level,
                                'members': members,
                            })

                info['raid_arrays'] = raid_arrays

                # Determine topology
                if any(r['level'] == 'raid10' for r in raid_arrays):
                    info['topology'] = 'raid10'
                elif any(r['level'] == 'raid1' for r in raid_arrays):
                    info['topology'] = 'raid1'
                elif any(r['level'] == 'raid0' for r in raid_arrays):
                    info['topology'] = 'raid0'
                else:
                    info['topology'] = 'raid'

            # Check if NVMe devices form RAID
            nvme_count = sum(1 for d in info['devices'] if d['type'] == 'nvme')
            if nvme_count > 1 and 'raid' in info['topology']:
                info['topology'] = f"nvme_{info['topology']}"

        except Exception:
            pass

        return info

    def _get_os_info(self) -> Dict[str, Any]:
        """Get OS and kernel information."""
        info = {'version': 'Unknown', 'kernel': 'Unknown'}

        try:
            # Get OS version
            os_release = self._run_command("cat /etc/os-release")
            if os_release:
                pretty_name = re.search(r'PRETTY_NAME="([^"]+)"', os_release)
                if pretty_name:
                    info['version'] = pretty_name.group(1)

            # Get kernel version
            kernel = self._run_command("uname -r")
            if kernel:
                info['kernel'] = kernel

        except Exception:
            pass

        return info

    def get_load_average(self) -> List[float]:
        """Get current load average (1m, 5m, 15m)."""
        try:
            loadavg = self._run_command("cat /proc/loadavg")
            if loadavg:
                parts = loadavg.split()[:3]
                return [float(p) for p in parts]
        except Exception:
            pass
        return [0.0, 0.0, 0.0]

    def get_os_tuning(self) -> Dict[str, str]:
        """Get relevant sysctl values."""
        tuning = {}
        params = [
            'vm.swappiness',
            'vm.dirty_ratio',
            'vm.dirty_background_ratio',
            'vm.nr_hugepages',
            'vm.overcommit_memory',
            'kernel.shmmax',
            'kernel.shmall',
            'net.core.somaxconn',
            'net.ipv4.tcp_max_syn_backlog',
        ]

        for param in params:
            try:
                value = self._run_command(f"sysctl -n {param} 2>/dev/null")
                if value:
                    tuning[param] = value
            except Exception:
                pass

        return tuning

    def get_kernel_limits(self) -> Dict[str, int]:
        """Get kernel memory limits (for tuning error context)."""
        limits = {
            'shmmax': 0,
            'shmall': 0,
            'hugepages_total': 0,
            'hugepages_free': 0,
        }

        try:
            shmmax = self._run_command("sysctl -n kernel.shmmax 2>/dev/null")
            if shmmax and shmmax.isdigit():
                limits['shmmax'] = int(shmmax)

            shmall = self._run_command("sysctl -n kernel.shmall 2>/dev/null")
            if shmall and shmall.isdigit():
                limits['shmall'] = int(shmall)

            hugepages = self._run_command("cat /proc/meminfo | grep HugePages")
            if hugepages:
                total_match = re.search(r'HugePages_Total:\s*(\d+)', hugepages)
                free_match = re.search(r'HugePages_Free:\s*(\d+)', hugepages)
                if total_match:
                    limits['hugepages_total'] = int(total_match.group(1))
                if free_match:
                    limits['hugepages_free'] = int(free_match.group(1))

        except Exception:
            pass

        return limits
