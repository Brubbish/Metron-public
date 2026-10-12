#!/usr/bin/env python3
"""Resolve a Debian package name to a source package and list source versions."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from .version_kb import group_source_versions_by_distro, resolve_query_candidates
except ImportError:  # pragma: no cover - script execution fallback
    from version_kb import group_source_versions_by_distro, resolve_query_candidates


FUZZY_THRESHOLD = 0.3
CANDIDATE_LIMIT = 100


class VersionKBQueryService:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "package_db",
        user: str = "postgres",
        password: str = "postgres",
    ):
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            cursor_factory=RealDictCursor,
        )

    def close(self) -> None:
        self.conn.close()

    def _fetch_candidates(
        self,
        *,
        table: str,
        matched_column: str,
        source_column: str,
        query: str,
        distro_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        normalized_query = query.strip().lower()
        if not normalized_query:
            return []

        params: Dict[str, Any] = {
            "query": normalized_query,
            "pattern": f"%{query.strip()}%",
            "threshold": FUZZY_THRESHOLD,
            "limit": CANDIDATE_LIMIT,
        }
        if distro_filter:
            params["distro"] = distro_filter
            distro_sql = "COALESCE(BOOL_OR(suite = %(distro)s), FALSE) AS has_distro_hit"
        else:
            distro_sql = "FALSE AS has_distro_hit"

        sql = f"""
            SELECT
                {matched_column} AS matched_name,
                {source_column} AS resolved_source_pkg,
                BOOL_OR(lower({matched_column}) = %(query)s) AS exact_match,
                MAX(
                    CASE
                        WHEN lower({matched_column}) = %(query)s THEN 1.0
                        ELSE similarity(lower({matched_column}), %(query)s)
                    END
                ) AS score,
                {distro_sql}
            FROM {table}
            WHERE lower({matched_column}) = %(query)s
               OR {matched_column} ILIKE %(pattern)s
               OR similarity(lower({matched_column}), %(query)s) >= %(threshold)s
            GROUP BY {matched_column}, {source_column}
            ORDER BY exact_match DESC, score DESC, matched_name ASC, resolved_source_pkg ASC
            LIMIT %(limit)s
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def fetch_source_candidates(self, query: str, distro_filter: Optional[str]) -> List[Dict[str, Any]]:
        return self._fetch_candidates(
            table="distro_version",
            matched_column="source_pkg",
            source_column="source_pkg",
            query=query,
            distro_filter=distro_filter,
        )

    def fetch_binary_candidates(self, query: str, distro_filter: Optional[str]) -> List[Dict[str, Any]]:
        return self._fetch_candidates(
            table="binary_package_sources",
            matched_column="binary_pkg",
            source_column="source_pkg",
            query=query,
            distro_filter=distro_filter,
        )

    def fetch_source_versions(self, source_pkg: str) -> List[Dict[str, Any]]:
        sql = """
            SELECT suite, repo_name, channel, component, source_version, binary_packages, is_primary
            FROM distro_version
            WHERE source_pkg = %s
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (source_pkg,))
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def query(self, query: str, distro_filter: Optional[str] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "query": query,
            "distro_filter": distro_filter,
            "resolved_source_pkg": None,
            "match": None,
            "versions_by_distro": {},
        }
        source_candidates = self.fetch_source_candidates(query, distro_filter)
        binary_candidates = self.fetch_binary_candidates(query, distro_filter)
        resolution = resolve_query_candidates(source_candidates, binary_candidates, distro_filter=distro_filter)
        if not resolution:
            return result

        source_pkg = resolution["resolved_source_pkg"]
        version_rows = self.fetch_source_versions(source_pkg)
        result["resolved_source_pkg"] = source_pkg
        result["match"] = resolution["match"]
        result["versions_by_distro"] = group_source_versions_by_distro(version_rows)
        return result


def format_match(match: Dict[str, Any]) -> str:
    return f"{match['matched_kind']} {match['match_type']} score={match['score']:.3f}"


def print_result(result: Dict[str, Any]) -> None:
    resolved_source_pkg = result.get("resolved_source_pkg")
    match = result.get("match")
    if not resolved_source_pkg or not match:
        print("No matching source package was found.")
        return

    matched_name = match["matched_name"]
    matched_kind = match["matched_kind"]
    if matched_kind == "source":
        print(f"Matched source package name '{matched_name}' ({format_match(match)}).")
    else:
        print(f"Matched binary package name '{matched_name}' ({format_match(match)}).")
    print(f"Resolved source package: {resolved_source_pkg}")

    versions_by_distro = result.get("versions_by_distro") or {}
    if not versions_by_distro:
        print("No source-version rows were found for that source package.")
        return

    for distro, entries in versions_by_distro.items():
        print()
        print(f"{distro}:")
        for entry in entries:
            print(f"- {entry['repo_name']}/{entry['component']}: {entry['version']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Source package name or binary package name")
    parser.add_argument("--distro", help="Prefer matches that exist in the given Debian suite")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="package_db")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = VersionKBQueryService(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )
    try:
        result = service.query(args.query, distro_filter=args.distro)
    finally:
        service.close()

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print_result(result)


if __name__ == "__main__":
    main()
