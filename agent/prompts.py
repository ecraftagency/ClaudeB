"""
PromptBuilder - Constructs AI prompts for different phases.

Prompts:
- STRATEGY: Request benchmark strategy (SDL)
- ANALYSIS: Analyze results and produce tuning proposals
- ERROR_RECOVERY: Handle errors and suggest fixes
"""

import json
from typing import Dict, Any, Optional


STRATEGY_PROMPT = """You are a PostgreSQL performance expert. Analyze this database context and generate a diagnostic strategy.

## System Context
{system_context}

## PostgreSQL Configuration
{runtime_context}

## Schema Overview
{schema_context}

## Previous Iteration (if any)
{previous_iteration}

Based on this information, generate a Strategy Definition Language (SDL) payload.

### SDL Schema
```json
{{
  "protocol_version": "v2",
  "id": "strategy-<uuid>",
  "name": "Human-readable strategy name",
  "hypothesis": "What bottleneck are we testing for?",
  "execution_plan": {{
    "benchmark_type": "pgbench|custom",
    "custom_sql": "SQL script for pgbench -f (if custom)",
    "scale": 100,
    "clients": 50,
    "threads": 4,
    "duration_seconds": 60,
    "warmup_seconds": 10
  }},
  "telemetry_requirements": [
    {{"source": "iostat", "metrics": ["await", "util"], "interval_ms": 1000}},
    {{"source": "vmstat", "metrics": ["cpu_wait", "context_switches"], "interval_ms": 1000}},
    {{"source": "pg_stat", "metrics": ["cache_hit_ratio", "tup_fetched"], "interval_ms": 1000}}
  ],
  "success_criteria": {{
    "target_tps": 5000,
    "max_latency_p99_ms": 50,
    "min_cache_hit_ratio": 99.0
  }}
}}
```

### Guidelines
1. Analyze the hardware (CPU cores, RAM, storage) to set realistic targets
2. Consider the schema size and complexity when designing the benchmark
3. For RAID10 or RAID0 setups with NVMe, expect higher I/O throughput
4. Set success criteria based on hardware capabilities, not wishful thinking
5. If previous_iteration shows failure, adjust strategy accordingly
6. **CRITICAL: Benchmark duration MUST NOT exceed 300 seconds (5 minutes)**. Use 60-180 seconds for most tests.
7. **CRITICAL: Check the "extensions" field in schema_context. Only use functions from INSTALLED extensions.** Do NOT use random_bytes(), gen_random_uuid(), or other extension functions unless that extension is listed. Use standard PostgreSQL functions like random(), md5(), or generate_series() instead.

Output ONLY valid JSON matching the SDL schema. No explanations outside JSON.
"""


ANALYSIS_PROMPT = """You are a PostgreSQL tuning expert. Analyze these benchmark results and produce tuning recommendations.

## Target KPIs (IMPORTANT - use these targets for evaluation)
{target_kpis}

## Original Strategy
{strategy}

## Benchmark Results
{benchmark_result}

## Telemetry Summary
{telemetry_summary}

## Current Configuration
{current_config}

## Tuning History
{tuning_history}

## DBA Feedback (if any)
{human_feedback}

Based on this data, you have TWO possible responses:

---

### OPTION 1: TuningProposal (if further tuning is possible)

If you identify software configuration improvements that can help, respond with:

```json
{{
  "protocol_version": "v2",
  "response_type": "tuning_proposal",
  "analysis_summary": "Brief analysis of bottlenecks found",
  "bottleneck_type": "cpu|io|memory|config|contention",
  "confidence": 0.85,
  "tuning_chunks": [
    {{
      "id": "chunk-001",
      "category": "memory|wal|checkpoint|connection|query",
      "name": "Human-readable name",
      "rationale": "Why this change helps",
      "apply_commands": [
        "ALTER SYSTEM SET shared_buffers = '4GB'",
        "SELECT pg_reload_conf()"
      ],
      "rollback_commands": [
        "ALTER SYSTEM SET shared_buffers = '1GB'",
        "SELECT pg_reload_conf()"
      ],
      "requires_restart": true,
      "recovery_strategy": "SQL_REVERT|FILE_RESTORE|OS_REVERT",
      "verification_command": "SHOW shared_buffers",
      "verification_expected": "4GB",
      "priority": "HIGH|MEDIUM|LOW",
      "risk_level": "LOW|MEDIUM|HIGH",
      "depends_on": []
    }}
  ],
  "expected_improvement": {{
    "tps_increase_pct": 20,
    "latency_reduction_pct": 15
  }},
  "verification_benchmark": {{
    "duration_seconds": 30,
    "clients": 50
  }}
}}
```

---

### OPTION 2: SessionConclusion (if hardware is saturated - "Game Over")

If the hardware is at capacity and no software tuning can further improve performance, respond with:

```json
{{
  "protocol_version": "v2",
  "response_type": "conclude_session",
  "conclusion_reason": "SATURATION|SUCCESS|DIMINISHING_RETURNS|MAX_ITERATIONS",
  "tuning_summary": {{
    "total_iterations": 3,
    "baseline_tps": 1000,
    "final_tps": 2500,
    "improvement_pct": 150,
    "key_changes_applied": ["Increased shared_buffers to 8GB", "Enabled wal_compression"]
  }},
  "hardware_saturation_analysis": {{
    "is_saturated": true,
    "bottleneck_resource": "CPU|IO_THROUGHPUT|IOPS|MEMORY|NETWORK",
    "evidence": [
      "CPU utilization consistently at 95%+ during benchmark",
      "No idle CPU time available",
      "All PostgreSQL processes competing for CPU resources"
    ],
    "scaling_recommendation": {{
      "action": "SCALE_UP|SCALE_OUT|OPTIMIZE_APP|NONE_NEEDED",
      "details": "Consider upgrading to a larger instance with more CPU cores, or implement read replicas to distribute query load"
    }}
  }},
  "final_report_markdown": "# PostgreSQL Tuning Complete\\n\\n## Summary\\n..."
}}
```

---

### Hardware Saturation Detection (CRITICAL)

**Detect HARDWARE SATURATION when:**

1. **CPU Saturated**: CPU user% + system% > 90%, CPU idle < 5%
2. **I/O Saturated**: Disk utilization > 95%, await > 20ms, I/O queue depth high
3. **Memory Saturated**: RAM fully utilized, swap activity detected
4. **Network Saturated**: Network bandwidth near limit (rare for DB)

**When hardware is saturated, respond with SessionConclusion, NOT TuningProposal.**

---

### DBA Feedback Handling

If `human_feedback` is provided, consider it carefully:
- **CORRECTION**: The DBA is correcting an assumption - adjust your analysis
- **NEW_DIRECTION**: The DBA wants to explore a different approach
- **CLARIFICATION**: Additional context to help your analysis

---

### Tuning Guidelines
1. Order chunks by priority (highest impact first)
2. Set `requires_restart: true` for parameters requiring PostgreSQL restart
3. Use `recovery_strategy`:
   - `SQL_REVERT`: For ALTER SYSTEM parameters
   - `FILE_RESTORE`: For direct config file edits
   - `OS_REVERT`: For sysctl/OS parameters
4. Provide accurate rollback_commands to restore original state
5. Set realistic expected_improvement based on bottleneck analysis

### Diminishing Returns Detection

If improvement from last iteration was < 2%, and you've already optimized the main bottlenecks, consider concluding with `conclusion_reason: "DIMINISHING_RETURNS"`.

Output ONLY valid JSON matching either TuningProposal or SessionConclusion schema.
"""


FIRST_SIGHT_PROMPT = """You are a PostgreSQL performance expert. Analyze this database system and provide an initial assessment with benchmark strategy options.

## System Context
{system_context}

## PostgreSQL Configuration
{runtime_context}

## Schema Overview
{schema_context}

Based on this information, provide a "First Sight" analysis.

### Response Schema
```json
{{
  "protocol_version": "v2",
  "system_overview": "Brief 2-3 sentence summary of the hardware and PostgreSQL setup",
  "schema_overview": "Brief 2-3 sentence summary of the database schema and key characteristics",
  "key_observations": [
    "Notable observation 1 (e.g., 'High-write workload detected on transactions table')",
    "Notable observation 2",
    "Notable observation 3"
  ],
  "warnings": [
    "Any concerns or potential issues (e.g., 'shared_buffers is only 128MB on a 64GB system')"
  ],
  "strategy_options": [
    {{
      "id": "strategy-oltp-baseline",
      "name": "OLTP Baseline Test",
      "goal": "Measure baseline transaction throughput",
      "hypothesis": "Establish current TPS performance under typical OLTP workload",
      "target_kpis": {{
        "primary": "TPS",
        "secondary": "P99 Latency",
        "target_tps": 5000,
        "max_latency_ms": 50
      }},
      "rationale": "Why this strategy is appropriate for this system",
      "estimated_duration_minutes": 5,
      "risk_level": "LOW"
    }},
    {{
      "id": "strategy-write-stress",
      "name": "Write-Heavy Stress Test",
      "goal": "Test WAL and checkpoint performance under heavy writes",
      "hypothesis": "Identify I/O bottlenecks in write path",
      "target_kpis": {{
        "primary": "TPS",
        "secondary": "WAL write latency",
        "target_tps": 3000
      }},
      "rationale": "Why this strategy makes sense",
      "estimated_duration_minutes": 10,
      "risk_level": "MEDIUM"
    }}
  ]
}}
```

### Guidelines

1. **System Overview**: Be concise. Mention CPU, RAM, storage type, PG version in 2-3 sentences max.

2. **Schema Overview**: Focus on database size, table count, and inferred workload pattern (OLTP, OLAP, mixed).

3. **Key Observations**: List 3-5 notable things you see. Focus on:
   - Workload patterns (high writes, read-heavy, etc.)
   - Configuration anomalies (too low/high settings)
   - Schema patterns (partitioning, bloat, missing indexes)

4. **Warnings**: Only include if there are genuine concerns. Empty array is fine.

5. **Strategy Options**: Provide 2-4 strategies. Each should:
   - Have a clear, distinct goal
   - Be appropriate for this specific system
   - Include realistic KPI targets based on hardware
   - Vary in risk level (include at least one LOW risk option)
   - **CRITICAL: estimated_duration_minutes MUST NOT exceed 5 minutes**. Use 1-3 minutes for most tests.

6. **Target KPIs**: Set realistic targets based on:
   - CPU cores (more cores = higher TPS potential)
   - RAM (affects cache hit ratio)
   - Storage (NVMe/RAID10 = higher IOPS)
   - Current config (shared_buffers, work_mem, etc.)

7. **Extensions**: Check the "extensions" field in schema_context. Note which extensions are installed - this affects what SQL functions can be used in custom benchmarks.

Output ONLY valid JSON matching the schema above. No explanations outside JSON.
"""


NEXT_STRATEGY_PROMPT = """You are a PostgreSQL performance expert. The DBA has completed a tuning session and wants to try a DIFFERENT strategy.

## System Context
{system_context}

## PostgreSQL Configuration
{runtime_context}

## Schema Overview
{schema_context}

## Previous Session Results
{previous_session}

## IMPORTANT: EXCLUDE THIS STRATEGY
The following strategy was already tested. DO NOT suggest it again:
{excluded_strategy}

Based on what was learned from the previous session, suggest NEW strategy options that:
1. Target different aspects of performance
2. Build on insights gained from the previous session
3. Explore areas not yet tested

### Response Schema
```json
{{
  "protocol_version": "v2",
  "system_overview": "Brief update based on what was learned",
  "insights_from_previous": "What the previous session revealed about the system",
  "strategy_options": [
    {{
      "id": "strategy-new-1",
      "name": "Strategy Name (must be different from excluded)",
      "goal": "What this strategy aims to test",
      "hypothesis": "What we expect to learn",
      "target_kpis": {{
        "primary": "TPS or latency",
        "target_tps": 5000,
        "max_latency_ms": 50
      }},
      "rationale": "Why this strategy is a good next step given previous results",
      "estimated_duration_minutes": 2,
      "risk_level": "LOW|MEDIUM|HIGH"
    }}
  ]
}}
```

### Guidelines
1. Suggest 2-3 NEW strategies that are DIFFERENT from the excluded strategy
2. Consider what the previous session revealed about bottlenecks
3. If previous session hit target easily, suggest more aggressive targets
4. If previous session struggled, suggest strategies to diagnose why
5. estimated_duration_minutes MUST NOT exceed 5 minutes

Output ONLY valid JSON matching the schema above.
"""


ROUND1_CONFIG_PROMPT = """You are a PostgreSQL performance expert. Based on the first sight analysis and chosen benchmark strategy, suggest initial configuration tuning to apply BEFORE running the baseline benchmark.

## System Context
{system_context}

## PostgreSQL Configuration
{runtime_context}

## First Sight Observations
{first_sight_summary}

## Chosen Strategy
{chosen_strategy}

## DBA Custom Instructions (if any)
{dba_message}

Based on this information, suggest Round 1 configuration changes to optimize for the chosen benchmark.

### Response Schema
```json
{{
  "protocol_version": "v2",
  "response_type": "round1_config",
  "rationale": "Brief explanation of why these changes are recommended before benchmarking",
  "tuning_chunks": [
    {{
      "id": "chunk-001",
      "category": "memory|wal|checkpoint|connection|query|os",
      "name": "Human-readable name",
      "description": "What this change does and why it helps for this benchmark",
      "apply_commands": [
        "ALTER SYSTEM SET shared_buffers = '4GB'",
        "SELECT pg_reload_conf()"
      ],
      "rollback_commands": [
        "ALTER SYSTEM SET shared_buffers = '1GB'",
        "SELECT pg_reload_conf()"
      ],
      "requires_restart": true,
      "verification_command": "SHOW shared_buffers",
      "verification_expected": "4GB",
      "priority": "HIGH|MEDIUM|LOW"
    }}
  ],
  "os_tuning": [
    {{
      "id": "os-001",
      "name": "Optimize dirty ratio for write workload",
      "description": "Reduce dirty_ratio to prevent large write stalls",
      "apply_command": "sudo sysctl -w vm.dirty_ratio=10",
      "rollback_command": "sudo sysctl -w vm.dirty_ratio=20",
      "persistent_file": "/etc/sysctl.d/99-pg-tuning.conf",
      "persistent_line": "vm.dirty_ratio = 10"
    }}
  ],
  "restart_required": true,
  "restart_reason": "shared_buffers change requires PostgreSQL restart"
}}
```

### Guidelines

1. **Focus on the benchmark goal**: Only suggest changes that will improve the specific benchmark type
2. **Be conservative**: This is Round 1, suggest safe optimizations that are widely applicable
3. **CRITICAL - FORBIDDEN SETTINGS in Round 1**:
   - Do NOT change shared_buffers (can crash PostgreSQL if set too high)
   - Do NOT enable or change huge_pages (requires OS-level pre-configuration)
   - Do NOT change max_connections drastically (can cause OOM)
4. **Safe memory settings**: Only adjust effective_cache_size (doesn't allocate memory)
5. **WAL settings**: For write-heavy benchmarks, optimize wal_buffers (max 64MB), checkpoint settings
6. **OS tuning**: Include sysctl changes for vm.dirty_ratio, vm.dirty_background_ratio, vm.swappiness
7. **Prefer no-restart settings**: Focus on parameters that can be changed with pg_reload_conf()
8. **Provide rollback**: Always include rollback commands to restore original state
9. **Empty tuning_chunks is acceptable**: If current config is already optimal, return empty arrays

### Common Round 1 Tunings (SAFE - No Restart Required)

**For OLTP/Write benchmarks:**
- Optimize checkpoint_completion_target (0.9)
- Tune wal_buffers (up to 64MB)
- Set synchronous_commit = off for max throughput testing
- Increase bgwriter_lru_maxpages for faster background writes
- Set wal_compression = lz4 or on

**For Read benchmarks:**
- Increase effective_cache_size (75% of RAM) - safe, doesn't allocate memory
- Tune random_page_cost (1.1 for SSD/NVMe)
- Increase work_mem for complex queries (be conservative, e.g., 64MB)

**OS Settings:**
- vm.dirty_ratio = 10-15
- vm.dirty_background_ratio = 3-5
- vm.swappiness = 1-10

**NEVER suggest in Round 1:**
- shared_buffers changes
- huge_pages changes
- max_connections reduction by more than 50%

Output ONLY valid JSON matching the schema above. No explanations outside JSON.
"""


ERROR_RECOVERY_PROMPT = """You are a PostgreSQL recovery expert. A tuning operation failed. Analyze and suggest recovery.

## Error Context
{error_packet}

## Service Logs
{service_logs}

## Original Tuning Chunk
{failed_chunk}

## System State
{system_context}

Based on this error, produce a RecoveryRecommendation.

### RecoveryRecommendation Schema
```json
{{
  "diagnosis": "What went wrong",
  "root_cause": "technical|config|resource|permission",
  "recovery_steps": [
    {{
      "step": 1,
      "action": "Description of action",
      "command": "Exact command to run",
      "verify": "How to verify success"
    }}
  ],
  "prevention": "How to avoid this in future",
  "retry_with_modifications": {{
    "should_retry": true,
    "modified_chunk": {{ ... }}
  }}
}}
```

### Common PostgreSQL Startup Failures

1. **shared_buffers too high**: Reduce to fit in available RAM, enable huge_pages
2. **Permission denied**: Check file ownership, SELinux/AppArmor policies
3. **Port already in use**: Check for zombie processes
4. **WAL corruption**: May need pg_resetwal (data loss risk)
5. **Disk full**: Free space, check pg_wal directory

Output ONLY valid JSON matching the RecoveryRecommendation schema.
"""


class PromptBuilder:
    """
    Builds prompts for different AI interaction phases.
    """

    @staticmethod
    def build_strategy_prompt(
        system_context: Dict[str, Any],
        runtime_context: Dict[str, Any],
        schema_context: Dict[str, Any],
        previous_iteration: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build prompt for strategy generation."""
        return STRATEGY_PROMPT.format(
            system_context=json.dumps(system_context, indent=2, default=str),
            runtime_context=json.dumps(runtime_context, indent=2, default=str),
            schema_context=json.dumps(schema_context, indent=2, default=str),
            previous_iteration=json.dumps(previous_iteration, indent=2, default=str)
            if previous_iteration else "None (first iteration)",
        )

    @staticmethod
    def build_analysis_prompt(
        strategy: Dict[str, Any],
        benchmark_result: Dict[str, Any],
        telemetry_summary: str,
        current_config: Dict[str, Any],
        target_kpis: Optional[Dict[str, Any]] = None,
        tuning_history: Optional[Dict[str, Any]] = None,
        human_feedback: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build prompt for result analysis (v2.3 with saturation detection)."""
        return ANALYSIS_PROMPT.format(
            target_kpis=json.dumps(target_kpis, indent=2, default=str)
            if target_kpis else "Not specified",
            strategy=json.dumps(strategy, indent=2, default=str),
            benchmark_result=json.dumps(benchmark_result, indent=2, default=str),
            telemetry_summary=telemetry_summary,
            current_config=json.dumps(current_config, indent=2, default=str),
            tuning_history=json.dumps(tuning_history, indent=2, default=str)
            if tuning_history else "None (first iteration)",
            human_feedback=json.dumps(human_feedback, indent=2, default=str)
            if human_feedback else "None",
        )

    @staticmethod
    def build_error_recovery_prompt(
        error_packet: Dict[str, Any],
        service_logs: list,
        failed_chunk: Dict[str, Any],
        system_context: Dict[str, Any],
    ) -> str:
        """Build prompt for error recovery."""
        return ERROR_RECOVERY_PROMPT.format(
            error_packet=json.dumps(error_packet, indent=2, default=str),
            service_logs='\n'.join(service_logs[-50:]),  # Last 50 lines
            failed_chunk=json.dumps(failed_chunk, indent=2, default=str),
            system_context=json.dumps(system_context, indent=2, default=str),
        )

    @staticmethod
    def build_context_packet_prompt(context_packet) -> str:
        """Build prompt from a ContextPacket dataclass."""
        from dataclasses import asdict

        packet_dict = asdict(context_packet)

        return PromptBuilder.build_strategy_prompt(
            system_context=packet_dict.get('system_context', {}),
            runtime_context=packet_dict.get('runtime_context', {}),
            schema_context=packet_dict.get('schema_context', {}),
            previous_iteration=packet_dict.get('previous_iteration'),
        )

    @staticmethod
    def build_first_sight_prompt(
        system_context: Dict[str, Any],
        runtime_context: Dict[str, Any],
        schema_context: Dict[str, Any],
    ) -> str:
        """Build prompt for first sight analysis."""
        return FIRST_SIGHT_PROMPT.format(
            system_context=json.dumps(system_context, indent=2, default=str),
            runtime_context=json.dumps(runtime_context, indent=2, default=str),
            schema_context=json.dumps(schema_context, indent=2, default=str),
        )

    @staticmethod
    def build_round1_config_prompt(
        system_context: Dict[str, Any],
        runtime_context: Dict[str, Any],
        first_sight_summary: Dict[str, Any],
        chosen_strategy: Dict[str, Any],
        dba_message: Optional[str] = None,
    ) -> str:
        """Build prompt for Round 1 configuration tuning."""
        return ROUND1_CONFIG_PROMPT.format(
            system_context=json.dumps(system_context, indent=2, default=str),
            runtime_context=json.dumps(runtime_context, indent=2, default=str),
            first_sight_summary=json.dumps(first_sight_summary, indent=2, default=str),
            chosen_strategy=json.dumps(chosen_strategy, indent=2, default=str),
            dba_message=dba_message or "None",
        )

    @staticmethod
    def build_next_strategy_prompt(
        system_context: Dict[str, Any],
        runtime_context: Dict[str, Any],
        schema_context: Dict[str, Any],
        previous_session: Dict[str, Any],
        excluded_strategy: Dict[str, Any],
    ) -> str:
        """Build prompt for getting next strategy options after a completed session."""
        return NEXT_STRATEGY_PROMPT.format(
            system_context=json.dumps(system_context, indent=2, default=str),
            runtime_context=json.dumps(runtime_context, indent=2, default=str),
            schema_context=json.dumps(schema_context, indent=2, default=str),
            previous_session=json.dumps(previous_session, indent=2, default=str),
            excluded_strategy=json.dumps(excluded_strategy, indent=2, default=str),
        )
