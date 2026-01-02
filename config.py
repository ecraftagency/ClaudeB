"""
Configuration management for pg_diagnose.

Supports:
- TOML config files
- Environment variables
- Command-line overrides
- Sensible defaults

Priority (highest to lowest):
1. Command-line arguments
2. Environment variables
3. Config file
4. Defaults
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for older Python
    except ImportError:
        tomllib = None


# Project root (where this file lives)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Default config file locations (searched in order)
CONFIG_SEARCH_PATHS = [
    PROJECT_ROOT / "config.toml",              # Project root (primary)
    PROJECT_ROOT / "pg_diagnose.toml",         # Alternative name
    Path.cwd() / "config.toml",                # Current working directory
    Path.home() / ".pg_diagnose" / "config.toml",
    Path.home() / ".config" / "pg_diagnose" / "config.toml",
]


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    name: str = "postgres"

    def connection_string(self) -> str:
        """Generate connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class SSHConfig:
    """SSH configuration for remote operations."""
    host: str = ""
    user: str = "ubuntu"
    key: Optional[str] = None
    port: int = 22

    @property
    def is_remote(self) -> bool:
        return bool(self.host)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    duration: int = 60
    scale: int = 100
    clients: int = 32
    threads: int = 8


@dataclass
class TuningConfig:
    """Tuning loop configuration."""
    max_rounds: int = 3
    target_improvement: float = 20.0
    auto_apply: bool = False


@dataclass
class AIConfig:
    """AI provider configuration."""
    provider: str = "gemini"
    model: str = "gemini-3-flash-preview"
    api_key: Optional[str] = None

    def __post_init__(self):
        # Load API key from environment if not set
        if not self.api_key:
            env_keys = {
                "gemini": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
            }
            env_var = env_keys.get(self.provider, "GEMINI_API_KEY")
            self.api_key = os.environ.get(env_var)


@dataclass
class OutputConfig:
    """Output configuration."""
    dir: str = "./pg_diagnose_output"
    verbose: bool = True
    quiet: bool = False


@dataclass
class Config:
    """Main configuration container."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ssh: SSHConfig = field(default_factory=SSHConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Source tracking
    _config_file: Optional[Path] = None

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """
        Load configuration from file.

        Args:
            config_path: Explicit path to config file. If None, searches default locations.

        Returns:
            Config instance with loaded values
        """
        config = cls()

        # Find config file
        if config_path:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
        else:
            path = cls._find_config_file()

        if path:
            config = cls._load_from_file(path)
            config._config_file = path

        return config

    @classmethod
    def _find_config_file(cls) -> Optional[Path]:
        """Find config file in default locations."""
        for path in CONFIG_SEARCH_PATHS:
            if path.exists():
                return path
        return None

    @classmethod
    def _load_from_file(cls, path: Path) -> "Config":
        """Load config from TOML file."""
        if tomllib is None:
            raise ImportError(
                "TOML support requires Python 3.11+ or 'tomli' package. "
                "Install with: pip install tomli"
            )

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        config = cls()

        # Database
        if "database" in data:
            db = data["database"]
            config.database = DatabaseConfig(
                host=db.get("host", config.database.host),
                port=db.get("port", config.database.port),
                user=db.get("user", config.database.user),
                password=db.get("password", config.database.password),
                name=db.get("name", config.database.name),
            )

        # SSH
        if "ssh" in data:
            ssh = data["ssh"]
            config.ssh = SSHConfig(
                host=ssh.get("host", config.ssh.host),
                user=ssh.get("user", config.ssh.user),
                key=ssh.get("key") or None,
                port=ssh.get("port", config.ssh.port),
            )

        # Benchmark
        if "benchmark" in data:
            bench = data["benchmark"]
            config.benchmark = BenchmarkConfig(
                duration=bench.get("duration", config.benchmark.duration),
                scale=bench.get("scale", config.benchmark.scale),
                clients=bench.get("clients", config.benchmark.clients),
                threads=bench.get("threads", config.benchmark.threads),
            )

        # Tuning
        if "tuning" in data:
            tune = data["tuning"]
            config.tuning = TuningConfig(
                max_rounds=tune.get("max_rounds", config.tuning.max_rounds),
                target_improvement=tune.get("target_improvement", config.tuning.target_improvement),
                auto_apply=tune.get("auto_apply", config.tuning.auto_apply),
            )

        # AI
        if "ai" in data:
            ai = data["ai"]
            config.ai = AIConfig(
                provider=ai.get("provider", config.ai.provider),
                model=ai.get("model", config.ai.model),
                api_key=ai.get("api_key"),
            )

        # Output
        if "output" in data:
            out = data["output"]
            config.output = OutputConfig(
                dir=out.get("dir", config.output.dir),
                verbose=out.get("verbose", config.output.verbose),
                quiet=out.get("quiet", config.output.quiet),
            )

        return config

    def override_from_args(self, args) -> "Config":
        """
        Override config values from argparse namespace.

        Args with value None are ignored (keeping config file values).
        """
        # Database overrides
        if getattr(args, "db_host", None):
            self.database.host = args.db_host
        if getattr(args, "db_port", None):
            self.database.port = args.db_port
        if getattr(args, "db_user", None):
            self.database.user = args.db_user
        if getattr(args, "db_password", None):
            self.database.password = args.db_password
        if getattr(args, "db_name", None):
            self.database.name = args.db_name

        # SSH overrides
        if getattr(args, "ssh_host", None):
            self.ssh.host = args.ssh_host
        if getattr(args, "ssh_user", None):
            self.ssh.user = args.ssh_user
        if getattr(args, "ssh_key", None):
            self.ssh.key = args.ssh_key

        # Benchmark overrides
        if getattr(args, "benchmark_duration", None):
            self.benchmark.duration = args.benchmark_duration
        if getattr(args, "pgbench_scale", None):
            self.benchmark.scale = args.pgbench_scale
        if getattr(args, "pgbench_clients", None):
            self.benchmark.clients = args.pgbench_clients
        if getattr(args, "pgbench_threads", None):
            self.benchmark.threads = args.pgbench_threads

        # Tuning overrides
        if getattr(args, "max_rounds", None):
            self.tuning.max_rounds = args.max_rounds
        if getattr(args, "target_improvement", None):
            self.tuning.target_improvement = args.target_improvement

        # AI overrides
        if getattr(args, "gemini_api_key", None):
            self.ai.api_key = args.gemini_api_key

        # Output overrides
        if getattr(args, "quiet", None):
            self.output.quiet = args.quiet
            self.output.verbose = not args.quiet

        return self

    def validate(self) -> list:
        """
        Validate configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Database validation
        if not self.database.host:
            errors.append("Database host is required")
        if not self.database.user:
            errors.append("Database user is required")

        # SSH validation (only if remote)
        if self.ssh.is_remote:
            if self.ssh.key and not Path(self.ssh.key).exists():
                errors.append(f"SSH key file not found: {self.ssh.key}")

        # AI validation
        if not self.ai.api_key:
            errors.append(f"AI API key not set. Set {self.ai.provider.upper()}_API_KEY environment variable")

        # Benchmark validation
        if self.benchmark.duration < 10:
            errors.append("Benchmark duration must be at least 10 seconds")
        if self.benchmark.clients < 1:
            errors.append("Number of clients must be at least 1")

        return errors

    def summary(self, include_config_path: bool = True) -> str:
        """Generate human-readable config summary."""
        lines = []

        if include_config_path:
            if self._config_file:
                lines.append(f"Config: {self._config_file}")
            else:
                lines.append("Config: (defaults)")

        lines.append(f"Database: {self.database.user}@{self.database.host}:{self.database.port}/{self.database.name}")

        if self.ssh.is_remote:
            lines.append(f"SSH: {self.ssh.user}@{self.ssh.host}")
        else:
            lines.append("SSH: (local)")

        lines.append(f"Benchmark: {self.benchmark.duration}s, {self.benchmark.clients} clients, scale {self.benchmark.scale}")
        lines.append(f"Tuning: max {self.tuning.max_rounds} rounds, target +{self.tuning.target_improvement}%")
        lines.append(f"AI: {self.ai.provider} ({self.ai.model})")

        return "\n".join(lines)


def create_example_config(path: str = "config.toml"):
    """Create example config file."""
    example = Path(__file__).parent / "config.example.toml"
    target = Path(path)

    if target.exists():
        raise FileExistsError(f"Config file already exists: {path}")

    if example.exists():
        target.write_text(example.read_text())
    else:
        # Inline example if file not found
        target.write_text("""# pg_diagnose Configuration

[database]
host = "localhost"
port = 5432
user = "postgres"
password = ""
name = "postgres"

[ssh]
host = ""
user = "ubuntu"

[benchmark]
duration = 60
scale = 100
clients = 32

[tuning]
max_rounds = 3
target_improvement = 20.0

[ai]
provider = "gemini"
model = "gemini-3-flash-preview"
""")

    return target
