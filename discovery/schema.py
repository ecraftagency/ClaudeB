"""
SchemaScanner - Scans database schema for AI analysis.

Collects:
- Table statistics (rows, sizes, bloat)
- Index statistics (scans, usage)
- DDL summaries (CREATE TABLE statements)
- Heuristic hints (patterns for AI guidance)
"""

import re
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import psycopg2

from ..protocol.context import (
    SchemaContext,
    TableStats,
    IndexStats,
    HeuristicHints,
    SchemaSummary,
    ForeignKeyInfo,
    TriggerInfo,
    FunctionInfo,
    ExtensionInfo,
)


@dataclass
class SchemaScannerConfig:
    """Configuration for schema scanning."""
    max_detailed_tables: int = 50
    max_detailed_indexes: int = 100
    include_ddl: bool = True
    max_ddl_tables: int = 20


class SchemaScanner:
    """
    Scans database schema for AI analysis.

    Implements v2.1 Context Window Management for large schemas.
    """

    def __init__(self, connection, config: Optional[SchemaScannerConfig] = None):
        self.conn = connection
        self.config = config or SchemaScannerConfig()

    def scan(self, schema_filter: Optional[str] = None) -> SchemaContext:
        """
        Perform full schema scan.

        Args:
            schema_filter: Optional schema name to filter (e.g., 'public')
        """
        database_name = self._get_database_name()
        database_size = self._get_database_size()

        # Get all table stats
        all_table_stats = self._scan_tables(schema_filter)
        all_index_stats = self._scan_indexes(schema_filter)

        # Apply summarization if needed
        if len(all_table_stats) > self.config.max_detailed_tables:
            table_stats, schema_summary = self._summarize_tables(all_table_stats)
        else:
            table_stats = all_table_stats
            schema_summary = None

        # Limit indexes
        if len(all_index_stats) > self.config.max_detailed_indexes:
            # Keep indexes for detailed tables
            detailed_tables = set(table_stats.keys())
            index_stats = {
                k: v for k, v in all_index_stats.items()
                if v.table_name in detailed_tables
            }
        else:
            index_stats = all_index_stats

        # Get DDL for top tables
        ddl_summary = []
        if self.config.include_ddl:
            top_tables = list(table_stats.keys())[:self.config.max_ddl_tables]
            for table_name in top_tables:
                ddl = self._get_table_ddl(table_name, schema_filter or 'public')
                if ddl:
                    ddl_summary.append(ddl)

        # Generate heuristic hints
        heuristic_hints = self._generate_hints(all_table_stats, all_index_stats)

        # Scan foreign keys, triggers, functions, and extensions
        foreign_keys = self._scan_foreign_keys(schema_filter)
        triggers = self._scan_triggers(schema_filter)
        functions = self._scan_functions(schema_filter)
        extensions = self._scan_extensions()

        return SchemaContext(
            database_name=database_name,
            database_size_gb=database_size,
            ddl_summary=ddl_summary,
            table_statistics=table_stats,
            index_statistics=index_stats,
            heuristic_hints=heuristic_hints,
            schema_summary=schema_summary,
            foreign_keys=foreign_keys,
            triggers=triggers,
            functions=functions,
            extensions=extensions,
        )

    def _get_database_name(self) -> str:
        """Get current database name."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT current_database()")
            return cur.fetchone()[0]

    def _get_database_size(self) -> float:
        """Get database size in GB."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT pg_database_size(current_database())")
            size_bytes = cur.fetchone()[0]
            return round(size_bytes / (1024 ** 3), 2)

    def _scan_tables(self, schema_filter: Optional[str] = None) -> Dict[str, TableStats]:
        """Scan all user tables."""
        query = """
            SELECT
                s.schemaname,
                s.relname as tablename,
                s.n_live_tup as rows,
                pg_total_relation_size(s.schemaname || '.' || s.relname) as size_bytes,
                COALESCE(s.n_dead_tup::float / NULLIF(s.n_live_tup + s.n_dead_tup, 0), 0) as bloat_ratio,
                (SELECT count(*) FROM pg_indexes i WHERE i.tablename = s.relname AND i.schemaname = s.schemaname) as index_count,
                s.n_tup_ins,
                s.n_tup_upd,
                s.n_tup_del,
                s.n_dead_tup,
                s.last_vacuum::text,
                s.last_analyze::text
            FROM pg_stat_user_tables s
            WHERE 1=1
        """
        params = []
        if schema_filter:
            query += " AND s.schemaname = %s"
            params.append(schema_filter)

        query += " ORDER BY pg_total_relation_size(s.schemaname || '.' || s.relname) DESC"

        tables = {}
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            for row in cur.fetchall():
                (schemaname, tablename, rows, size_bytes, bloat_ratio,
                 index_count, n_tup_ins, n_tup_upd, n_tup_del, n_dead_tup,
                 last_vacuum, last_analyze) = row

                # Count FK relationships
                fk_in, fk_out = self._count_fk_relationships(schemaname, tablename)

                tables[tablename] = TableStats(
                    schema=schemaname,
                    rows=rows or 0,
                    size_mb=round((size_bytes or 0) / (1024 ** 2), 2),
                    bloat_ratio=round(bloat_ratio or 0, 4),
                    index_count=index_count or 0,
                    fk_inbound=fk_in,
                    fk_outbound=fk_out,
                    last_vacuum=last_vacuum,
                    last_analyze=last_analyze,
                    n_tup_ins=n_tup_ins or 0,
                    n_tup_upd=n_tup_upd or 0,
                    n_tup_del=n_tup_del or 0,
                    n_dead_tup=n_dead_tup or 0,
                )

        return tables

    def _count_fk_relationships(self, schema: str, table: str) -> tuple:
        """Count inbound and outbound FK relationships."""
        with self.conn.cursor() as cur:
            # Outbound FKs (this table references others)
            cur.execute("""
                SELECT count(*)
                FROM information_schema.table_constraints tc
                WHERE tc.table_schema = %s
                  AND tc.table_name = %s
                  AND tc.constraint_type = 'FOREIGN KEY'
            """, (schema, table))
            fk_out = cur.fetchone()[0]

            # Inbound FKs (other tables reference this one)
            cur.execute("""
                SELECT count(*)
                FROM information_schema.constraint_column_usage ccu
                JOIN information_schema.table_constraints tc
                  ON ccu.constraint_name = tc.constraint_name
                WHERE ccu.table_schema = %s
                  AND ccu.table_name = %s
                  AND tc.constraint_type = 'FOREIGN KEY'
            """, (schema, table))
            fk_in = cur.fetchone()[0]

        return fk_in, fk_out

    def _scan_indexes(self, schema_filter: Optional[str] = None) -> Dict[str, IndexStats]:
        """Scan all user indexes with full definitions."""
        query = """
            SELECT
                sui.indexrelname as indexname,
                sui.relname as tablename,
                sui.idx_scan,
                sui.idx_tup_read,
                sui.idx_tup_fetch,
                pg_relation_size(sui.indexrelid) as size_bytes,
                am.amname as index_type,
                idx.indisunique as is_unique,
                idx.indisprimary as is_primary,
                pg_get_indexdef(sui.indexrelid) as definition
            FROM pg_stat_user_indexes sui
            JOIN pg_index idx ON sui.indexrelid = idx.indexrelid
            JOIN pg_class c ON idx.indexrelid = c.oid
            JOIN pg_am am ON c.relam = am.oid
            WHERE 1=1
        """
        params = []
        if schema_filter:
            query += " AND sui.schemaname = %s"
            params.append(schema_filter)

        query += " ORDER BY sui.idx_scan DESC"

        indexes = {}
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            for row in cur.fetchall():
                (indexname, tablename, idx_scan, idx_tup_read,
                 idx_tup_fetch, size_bytes, index_type, is_unique,
                 is_primary, definition) = row

                # Extract column names from definition
                columns = self._extract_index_columns(definition)

                indexes[indexname] = IndexStats(
                    table_name=tablename,
                    index_type=index_type,
                    size_mb=round((size_bytes or 0) / (1024 ** 2), 2),
                    idx_scan=idx_scan or 0,
                    idx_tup_read=idx_tup_read or 0,
                    idx_tup_fetch=idx_tup_fetch or 0,
                    columns=columns,
                    is_unique=is_unique or False,
                    is_primary=is_primary or False,
                    definition=definition or "",
                )

        return indexes

    def _extract_index_columns(self, definition: str) -> List[str]:
        """Extract column names from index definition."""
        if not definition:
            return []
        # Pattern: ... USING btree (col1, col2, ...) or (col1 DESC, col2 ASC)
        match = re.search(r'USING\s+\w+\s*\(([^)]+)\)', definition, re.IGNORECASE)
        if match:
            cols_str = match.group(1)
            # Split by comma and clean up
            columns = []
            for col in cols_str.split(','):
                col = col.strip()
                # Remove ASC/DESC/NULLS FIRST/NULLS LAST
                col = re.sub(r'\s+(ASC|DESC|NULLS\s+FIRST|NULLS\s+LAST).*$', '', col, flags=re.IGNORECASE)
                columns.append(col.strip())
            return columns
        return []

    def _get_table_ddl(self, table_name: str, schema: str) -> Optional[str]:
        """Get CREATE TABLE statement for a table."""
        try:
            with self.conn.cursor() as cur:
                # Get column definitions
                cur.execute("""
                    SELECT
                        column_name,
                        data_type,
                        character_maximum_length,
                        numeric_precision,
                        numeric_scale,
                        is_nullable,
                        column_default
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                """, (schema, table_name))

                columns = []
                for row in cur.fetchall():
                    col_name, data_type, char_max, num_prec, num_scale, nullable, default = row
                    col_def = f"{col_name} {data_type}"
                    if char_max:
                        col_def += f"({char_max})"
                    elif num_prec and data_type in ('numeric', 'decimal'):
                        col_def += f"({num_prec},{num_scale or 0})"
                    if nullable == 'NO':
                        col_def += " NOT NULL"
                    if default:
                        col_def += f" DEFAULT {default}"
                    columns.append(col_def)

                # Get primary key
                cur.execute("""
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                      ON tc.constraint_name = kcu.constraint_name
                    WHERE tc.table_schema = %s
                      AND tc.table_name = %s
                      AND tc.constraint_type = 'PRIMARY KEY'
                    ORDER BY kcu.ordinal_position
                """, (schema, table_name))
                pk_cols = [row[0] for row in cur.fetchall()]

                if columns:
                    ddl = f"CREATE TABLE {table_name} ("
                    ddl += ", ".join(columns)
                    if pk_cols:
                        ddl += f", PRIMARY KEY ({', '.join(pk_cols)})"
                    ddl += ");"
                    return ddl

        except Exception:
            pass

        return None

    def _generate_hints(
        self,
        tables: Dict[str, TableStats],
        indexes: Dict[str, IndexStats]
    ) -> HeuristicHints:
        """Generate heuristic hints for AI guidance."""
        hints = HeuristicHints()

        # Check for version columns (optimistic locking)
        version_patterns = ['version', 'row_version', 'lock_version', 'revision']
        for table_name in tables.keys():
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        SELECT column_name FROM information_schema.columns
                        WHERE table_name = %s
                          AND lower(column_name) = ANY(%s)
                    """, (table_name, version_patterns))
                    if cur.fetchone():
                        hints.has_version_column = True
                        break
            except Exception:
                pass

        # Check for audit tables
        audit_patterns = ['audit', 'log', 'history', 'archive']
        for table_name in tables.keys():
            if any(p in table_name.lower() for p in audit_patterns):
                hints.has_audit_tables = True
                break

        # Identify high write volume tables (top 5 by inserts + updates)
        write_volumes = [
            (name, stats.n_tup_ins + stats.n_tup_upd)
            for name, stats in tables.items()
        ]
        write_volumes.sort(key=lambda x: x[1], reverse=True)
        hints.high_write_volume_tables = [name for name, _ in write_volumes[:5] if _ > 0]

        # Identify hot update tables (high update ratio)
        for name, stats in tables.items():
            total_ops = stats.n_tup_ins + stats.n_tup_upd + stats.n_tup_del
            if total_ops > 0 and stats.n_tup_upd / total_ops > 0.5:
                hints.hot_update_tables.append(name)

        # Identify unused indexes (zero scans)
        hints.unused_indexes = [
            name for name, stats in indexes.items()
            if stats.idx_scan == 0 and not name.endswith('_pkey')
        ][:10]  # Limit to 10

        # Identify tables without primary key
        for table_name, stats in tables.items():
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        SELECT 1 FROM information_schema.table_constraints
                        WHERE table_name = %s AND constraint_type = 'PRIMARY KEY'
                    """, (table_name,))
                    if not cur.fetchone():
                        hints.missing_pk_tables.append(table_name)
            except Exception:
                pass

        # Identify wide tables (>20 columns)
        for table_name in tables.keys():
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        SELECT count(*) FROM information_schema.columns
                        WHERE table_name = %s
                    """, (table_name,))
                    col_count = cur.fetchone()[0]
                    if col_count > 20:
                        hints.wide_tables.append(table_name)
            except Exception:
                pass

        # Check for partitioned tables
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT relname FROM pg_class
                    WHERE relkind = 'p'
                """)
                hints.partitioned_tables = [row[0] for row in cur.fetchall()]
        except Exception:
            pass

        return hints

    def _summarize_tables(
        self,
        all_tables: Dict[str, TableStats]
    ) -> tuple:
        """
        Summarize tables for large schemas (v2.1 Context Window Management).

        Returns:
            (top_tables, schema_summary)
        """
        # Rank tables by importance
        def importance_score(name: str, stats: TableStats) -> float:
            return (
                stats.size_mb * 0.3 +
                (stats.n_tup_ins + stats.n_tup_upd) * 0.3 +
                (stats.fk_inbound + stats.fk_outbound) * 100 * 0.2 +
                stats.bloat_ratio * 1000 * 0.2
            )

        ranked = sorted(
            all_tables.items(),
            key=lambda x: importance_score(x[0], x[1]),
            reverse=True
        )

        # Split into top and remainder
        top_tables = dict(ranked[:self.config.max_detailed_tables])
        remaining = ranked[self.config.max_detailed_tables:]

        # Categorize remaining tables
        categories = {
            'audit_tables': 0,
            'archive_tables': 0,
            'lookup_tables': 0,
            'other': 0,
        }
        for name, _ in remaining:
            name_lower = name.lower()
            if any(p in name_lower for p in ['audit', 'log', 'history']):
                categories['audit_tables'] += 1
            elif any(p in name_lower for p in ['archive', 'backup', 'old']):
                categories['archive_tables'] += 1
            elif any(p in name_lower for p in ['lookup', 'ref', 'type', 'status']):
                categories['lookup_tables'] += 1
            else:
                categories['other'] += 1

        summary = SchemaSummary(
            total_tables=len(all_tables),
            detailed_tables=len(top_tables),
            omitted_tables=len(remaining),
            omitted_total_size_gb=sum(s.size_mb for _, s in remaining) / 1024,
            omitted_total_rows=sum(s.rows for _, s in remaining),
            omitted_categories=categories,
        )

        return top_tables, summary

    def list_databases(self) -> List[Dict[str, Any]]:
        """List all databases."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    datname,
                    pg_database_size(datname) as size_bytes,
                    pg_get_userbyid(datdba) as owner
                FROM pg_database
                WHERE datistemplate = false
                ORDER BY pg_database_size(datname) DESC
            """)
            return [
                {"name": row[0], "size_mb": round(row[1] / (1024**2), 2), "owner": row[2]}
                for row in cur.fetchall()
            ]

    def list_schemas(self, database: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all schemas in the current database."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    n.nspname as schema_name,
                    pg_get_userbyid(n.nspowner) as owner,
                    count(c.relname) as table_count
                FROM pg_namespace n
                LEFT JOIN pg_class c ON c.relnamespace = n.oid AND c.relkind = 'r'
                WHERE n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
                  AND n.nspname NOT LIKE 'pg_temp%'
                GROUP BY n.nspname, n.nspowner
                ORDER BY n.nspname
            """)
            return [
                {"name": row[0], "owner": row[1], "table_count": row[2]}
                for row in cur.fetchall()
            ]

    def _scan_foreign_keys(self, schema_filter: Optional[str] = None) -> List[ForeignKeyInfo]:
        """Scan all foreign key constraints with full details."""
        query = """
            SELECT
                tc.constraint_name,
                tc.table_schema,
                tc.table_name as from_table,
                kcu.column_name as from_column,
                ccu.table_schema as to_schema,
                ccu.table_name as to_table,
                ccu.column_name as to_column,
                rc.delete_rule,
                rc.update_rule
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
            JOIN information_schema.referential_constraints rc
                ON rc.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
        """
        params = []
        if schema_filter:
            query += " AND tc.table_schema = %s"
            params.append(schema_filter)
        query += " ORDER BY tc.table_name, tc.constraint_name"

        # Group by constraint name (multi-column FKs)
        fk_map = {}
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            for row in cur.fetchall():
                (constraint_name, from_schema, from_table, from_column,
                 to_schema, to_table, to_column, delete_rule, update_rule) = row

                key = f"{from_schema}.{constraint_name}"
                if key not in fk_map:
                    fk_map[key] = {
                        'constraint_name': constraint_name,
                        'from_table': f"{from_schema}.{from_table}" if from_schema != 'public' else from_table,
                        'from_columns': [],
                        'to_table': f"{to_schema}.{to_table}" if to_schema != 'public' else to_table,
                        'to_columns': [],
                        'on_delete': delete_rule or 'NO ACTION',
                        'on_update': update_rule or 'NO ACTION',
                    }
                if from_column not in fk_map[key]['from_columns']:
                    fk_map[key]['from_columns'].append(from_column)
                if to_column not in fk_map[key]['to_columns']:
                    fk_map[key]['to_columns'].append(to_column)

        return [ForeignKeyInfo(**fk) for fk in fk_map.values()]

    def _scan_triggers(self, schema_filter: Optional[str] = None) -> List[TriggerInfo]:
        """Scan all triggers with definitions."""
        query = """
            SELECT
                t.tgname as trigger_name,
                c.relname as table_name,
                CASE
                    WHEN t.tgtype & 2 = 2 THEN 'BEFORE'
                    WHEN t.tgtype & 64 = 64 THEN 'INSTEAD OF'
                    ELSE 'AFTER'
                END as timing,
                CASE
                    WHEN t.tgtype & 4 = 4 THEN 'INSERT'
                    WHEN t.tgtype & 8 = 8 THEN 'DELETE'
                    WHEN t.tgtype & 16 = 16 THEN 'UPDATE'
                    WHEN t.tgtype & 32 = 32 THEN 'TRUNCATE'
                    ELSE 'UNKNOWN'
                END as event,
                p.proname as function_name,
                t.tgenabled != 'D' as enabled,
                pg_get_triggerdef(t.oid) as definition
            FROM pg_trigger t
            JOIN pg_class c ON t.tgrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            JOIN pg_proc p ON t.tgfoid = p.oid
            WHERE NOT t.tgisinternal
              AND n.nspname NOT IN ('pg_catalog', 'information_schema')
        """
        params = []
        if schema_filter:
            query += " AND n.nspname = %s"
            params.append(schema_filter)
        query += " ORDER BY c.relname, t.tgname"

        triggers = []
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            for row in cur.fetchall():
                (trigger_name, table_name, timing, event,
                 function_name, enabled, definition) = row

                triggers.append(TriggerInfo(
                    name=trigger_name,
                    table_name=table_name,
                    event=event,
                    timing=timing,
                    function_name=function_name,
                    enabled=enabled,
                    definition=definition or "",
                ))

        return triggers

    def _scan_functions(self, schema_filter: Optional[str] = None) -> List[FunctionInfo]:
        """Scan all stored procedures and functions."""
        query = """
            SELECT
                p.proname as name,
                n.nspname as schema,
                l.lanname as language,
                pg_get_function_result(p.oid) as return_type,
                pg_get_function_arguments(p.oid) as arguments,
                p.prorettype = 'pg_catalog.trigger'::regtype as is_trigger_func,
                CASE p.provolatile
                    WHEN 'i' THEN 'IMMUTABLE'
                    WHEN 's' THEN 'STABLE'
                    ELSE 'VOLATILE'
                END as volatility,
                pg_get_functiondef(p.oid) as definition
            FROM pg_proc p
            JOIN pg_namespace n ON p.pronamespace = n.oid
            JOIN pg_language l ON p.prolang = l.oid
            WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
              AND p.prokind IN ('f', 'p')  -- functions and procedures
        """
        params = []
        if schema_filter:
            query += " AND n.nspname = %s"
            params.append(schema_filter)
        query += " ORDER BY n.nspname, p.proname"

        functions = []
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            for row in cur.fetchall():
                (name, schema, language, return_type, arguments,
                 is_trigger_func, volatility, definition) = row

                # Parse argument types
                arg_types = []
                if arguments:
                    # Split arguments and extract types
                    for arg in arguments.split(','):
                        arg = arg.strip()
                        if arg:
                            # Format: "name type" or just "type"
                            parts = arg.split()
                            if len(parts) >= 1:
                                arg_types.append(parts[-1] if len(parts) > 1 else parts[0])

                functions.append(FunctionInfo(
                    name=name,
                    schema=schema,
                    language=language,
                    return_type=return_type or "void",
                    argument_types=arg_types,
                    is_trigger_func=is_trigger_func or False,
                    volatility=volatility,
                    definition=definition or "",
                ))

        return functions

    def _scan_extensions(self) -> List[ExtensionInfo]:
        """Scan all installed extensions."""
        query = """
            SELECT
                e.extname as name,
                e.extversion as version,
                n.nspname as schema
            FROM pg_extension e
            JOIN pg_namespace n ON e.extnamespace = n.oid
            ORDER BY e.extname
        """
        extensions = []
        try:
            with self.conn.cursor() as cur:
                cur.execute(query)
                for row in cur.fetchall():
                    name, version, schema = row
                    extensions.append(ExtensionInfo(
                        name=name,
                        version=version,
                        schema=schema,
                    ))
        except Exception:
            pass

        return extensions
