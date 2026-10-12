#!/usr/bin/env python3
"""Import package metadata from Build_RAG_Info JSON files into PostgreSQL."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import psycopg2
    from psycopg2.extras import execute_batch
except ImportError:  # pragma: no cover - optional for row-building tests
    psycopg2 = None
    execute_batch = None

try:
    from .version_kb import (
        DEFAULT_ARCH,
        build_binary_source_rows,
        build_source_version_rows,
        load_suite_json_map,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from version_kb import (
        DEFAULT_ARCH,
        build_binary_source_rows,
        build_source_version_rows,
        load_suite_json_map,
    )


class PackageDBImporter:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "package_db",
        user: str = "postgres",
        password: str = "postgres",
    ):
        if psycopg2 is None or execute_batch is None:
            raise RuntimeError("psycopg2 is required for PostgreSQL imports")
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )
        self.cursor = self.conn.cursor()

    def insert_packages(self, packages: Sequence[Tuple[str, str, str, str]]) -> None:
        sql = """
            INSERT INTO packages (os_version, pkg_name, src_name, pkg_version)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (os_version, pkg_name)
            DO UPDATE SET
                src_name = EXCLUDED.src_name,
                pkg_version = EXCLUDED.pkg_version,
                updated_at = NOW()
        """
        execute_batch(self.cursor, sql, packages, page_size=1000)
        self.conn.commit()
        print(f"Inserted {len(packages)} package rows")

    def insert_files(self, files: Sequence[Tuple[str, str, str]]) -> None:
        sql = """
            INSERT INTO files (os_version, filename, pkg_name)
            VALUES (%s, %s, %s)
            ON CONFLICT (os_version, filename)
            DO UPDATE SET
                pkg_name = EXCLUDED.pkg_name
        """
        execute_batch(self.cursor, sql, files, page_size=2000)
        self.conn.commit()
        print(f"Inserted {len(files)} file rows")

    def insert_distro_versions(self, rows: Sequence[Dict[str, Any]]) -> None:
        sql = """
            INSERT INTO distro_version (
                suite, repo_name, channel, component,
                source_pkg, source_version, binary_packages,
                is_primary, version_origin
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (suite, repo_name, component, source_pkg, source_version)
            DO UPDATE SET
                channel = EXCLUDED.channel,
                binary_packages = EXCLUDED.binary_packages,
                is_primary = EXCLUDED.is_primary,
                version_origin = EXCLUDED.version_origin,
                updated_at = NOW()
        """
        payload = [
            (
                row["suite"],
                row["repo_name"],
                row["channel"],
                row["component"],
                row["source_pkg"],
                row["source_version"],
                list(row.get("binary_packages", [])),
                bool(row.get("is_primary", False)),
                row.get("version_origin", "sources"),
            )
            for row in rows
        ]
        if payload:
            execute_batch(self.cursor, sql, payload, page_size=1000)
            self.conn.commit()
        print(f"Inserted {len(payload)} source version rows")

    def insert_binary_package_sources(self, rows: Sequence[Dict[str, Any]]) -> None:
        sql = """
            INSERT INTO binary_package_sources (
                suite, repo_name, channel, component, arch,
                binary_pkg, source_pkg, description, is_primary
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (suite, repo_name, component, arch, binary_pkg)
            DO UPDATE SET
                channel = EXCLUDED.channel,
                source_pkg = EXCLUDED.source_pkg,
                description = EXCLUDED.description,
                is_primary = EXCLUDED.is_primary,
                updated_at = NOW()
        """
        payload = [
            (
                row["suite"],
                row["repo_name"],
                row["channel"],
                row["component"],
                row["arch"],
                row["binary_pkg"],
                row["source_pkg"],
                row.get("description", ""),
                bool(row.get("is_primary", False)),
            )
            for row in rows
        ]
        if payload:
            execute_batch(self.cursor, sql, payload, page_size=1000)
            self.conn.commit()
        print(f"Inserted {len(payload)} binary source rows")

    def clear_data(self, os_version: Optional[str] = None) -> None:
        if os_version:
            self.cursor.execute("DELETE FROM files WHERE os_version = %s", (os_version,))
            self.cursor.execute("DELETE FROM packages WHERE os_version = %s", (os_version,))
            print(f"Cleared package/file data for {os_version}")
        else:
            self.cursor.execute("TRUNCATE TABLE files RESTART IDENTITY CASCADE")
            self.cursor.execute("TRUNCATE TABLE packages RESTART IDENTITY CASCADE")
            print("Cleared packages/files tables")
        self.conn.commit()

    def clear_version_kb(self, suite: Optional[str] = None) -> None:
        if suite:
            self.cursor.execute("DELETE FROM binary_package_sources WHERE suite = %s", (suite,))
            self.cursor.execute("DELETE FROM distro_version WHERE suite = %s", (suite,))
            print(f"Cleared version KB for suite {suite}")
        else:
            self.cursor.execute("TRUNCATE TABLE binary_package_sources RESTART IDENTITY CASCADE")
            self.cursor.execute("TRUNCATE TABLE distro_version RESTART IDENTITY CASCADE")
            print("Cleared version KB tables")
        self.conn.commit()

    def vacuum_analyze(self) -> None:
        old_isolation = self.conn.isolation_level
        self.conn.set_isolation_level(0)
        for table_name in (
            "packages",
            "files",
            "distro_version",
            "binary_package_sources",
        ):
            self.cursor.execute(f"VACUUM ANALYZE {table_name}")
        self.conn.set_isolation_level(old_isolation)
        print("Database vacuum/analyze completed")

    def close(self) -> None:
        self.cursor.close()
        self.conn.close()


def example_import() -> None:
    importer = PackageDBImporter(
        host="localhost",
        dbname="package_db",
        user="postgres",
        password="postgres",
    )
    try:
        packages = [
            ("ubuntu:20.04", "libssl1.1", "openssl", "1.1.1f-1ubuntu2"),
            ("ubuntu:20.04", "libssl-dev", "openssl", "1.1.1f-1ubuntu2"),
            ("ubuntu:22.04", "libssl3", "openssl", "3.0.2-0ubuntu1"),
        ]
        files = [
            ("ubuntu:20.04", "libssl.so.1.1", "libssl1.1"),
            ("ubuntu:20.04", "ssl.h", "libssl-dev"),
            ("ubuntu:22.04", "libssl.so.3", "libssl3"),
        ]
        importer.insert_packages(packages)
        importer.insert_files(files)
        importer.vacuum_analyze()
        print("Example import completed")
    finally:
        importer.close()


def import_from_file(pkg_file: str, file_list: str, os_version: str) -> None:
    importer = PackageDBImporter()
    try:
        packages: List[Tuple[str, str, str, str]] = []
        with open(pkg_file, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    packages.append((os_version, parts[0], parts[1], parts[2]))

        files: List[Tuple[str, str, str]] = []
        with open(file_list, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    files.append((os_version, parts[0], parts[1]))

        if packages:
            importer.insert_packages(packages)
        if files:
            importer.insert_files(files)
        importer.vacuum_analyze()
    finally:
        importer.close()


def import_from_json_dirs(
    pkglist_dir: str,
    pkgpath_dir: str,
    host: str = "localhost",
    port: int = 5432,
    dbname: str = "package_db",
    user: str = "postgres",
    password: str = "postgres",
) -> None:
    importer = PackageDBImporter(host=host, port=port, dbname=dbname, user=user, password=password)
    try:
        pkglist_path = Path(pkglist_dir)
        pkgpath_path = Path(pkgpath_dir)
        pkglist_files = sorted(pkglist_path.glob("*_pkglist_*.json"))
        print(f"Found {len(pkglist_files)} pkglist files")

        for pkglist_file in pkglist_files:
            match = re.match(r"(\w+)_pkglist_(\w+)", pkglist_file.stem)
            if not match:
                print(f"Skip {pkglist_file.name}: invalid name")
                continue
            suite = match.group(1)
            arch = match.group(2)
            os_version = f"debian:{suite}"
            with pkglist_file.open("r", encoding="utf-8") as handle:
                pkglist_data = json.load(handle)

            packages: List[Tuple[str, str, str, str]] = []
            for pkg_name, pkg_info in pkglist_data.items():
                src_name = str(pkg_info.get("source") or pkg_info.get("source_pkg") or pkg_name).strip() or pkg_name
                if "(" in src_name:
                    src_name = src_name.split("(", 1)[0].strip()
                version_data = pkg_info.get("version", "")
                if isinstance(version_data, list):
                    pkg_version = json.dumps(version_data, ensure_ascii=False)
                else:
                    pkg_version = str(version_data)
                packages.append((os_version, pkg_name, src_name, pkg_version))

            pkgpath_file = pkgpath_path / f"{suite}_pkgpath_{arch}.json"
            files: List[Tuple[str, str, str]] = []
            if pkgpath_file.exists():
                with pkgpath_file.open("r", encoding="utf-8") as handle:
                    pkgpath_data = json.load(handle)
                files = [(os_version, filename, pkg_name) for filename, pkg_name in pkgpath_data.items()]

            if packages:
                importer.insert_packages(packages)
            if files:
                importer.insert_files(files)

        importer.vacuum_analyze()
    finally:
        importer.close()


def load_version_kb_rows(base_dir: str, arch: str = DEFAULT_ARCH) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_path = Path(base_dir)
    pkg_payload_by_suite = load_suite_json_map(base_path / "pkglist", f"*_pkglist_{arch}.json")
    source_payload_by_suite = load_suite_json_map(base_path / "sourcelist", "*_sourcelist.json")
    source_rows = build_source_version_rows(source_payload_by_suite)
    binary_rows = build_binary_source_rows(pkg_payload_by_suite, arch=arch)
    return source_rows, binary_rows


def import_version_kb_from_json_dirs(
    base_dir: str,
    arch: str = DEFAULT_ARCH,
    host: str = "localhost",
    port: int = 5432,
    dbname: str = "package_db",
    user: str = "postgres",
    password: str = "postgres",
    clear_first: bool = True,
) -> None:
    importer = PackageDBImporter(host=host, port=port, dbname=dbname, user=user, password=password)
    try:
        source_rows, binary_rows = load_version_kb_rows(base_dir, arch=arch)
        if clear_first:
            importer.clear_version_kb()
        importer.insert_distro_versions(source_rows)
        importer.insert_binary_package_sources(binary_rows)
        importer.vacuum_analyze()
    finally:
        importer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pkg_file", nargs="?", help="Legacy packages.txt path")
    parser.add_argument("file_list", nargs="?", help="Legacy files.txt path")
    parser.add_argument("os_version", nargs="?", help="Legacy os_version value")
    parser.add_argument("--import-json", dest="import_json", help="Import legacy pkglist/pkgpath JSON directories")
    parser.add_argument("--import-version-kb", dest="import_version_kb", help="Import pkglist/sourcelist JSON into version KB tables")
    parser.add_argument("--arch", default=DEFAULT_ARCH, help="Architecture for version KB imports")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="package_db")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear version KB tables before import")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.import_version_kb:
        import_version_kb_from_json_dirs(
            base_dir=args.import_version_kb,
            arch=args.arch,
            host=args.host,
            port=args.port,
            dbname=args.dbname,
            user=args.user,
            password=args.password,
            clear_first=not args.no_clear,
        )
        return

    if args.import_json:
        pkglist_dir = os.path.join(args.import_json, "pkglist")
        pkgpath_dir = os.path.join(args.import_json, "pkgpath")
        import_from_json_dirs(
            pkglist_dir=pkglist_dir,
            pkgpath_dir=pkgpath_dir,
            host=args.host,
            port=args.port,
            dbname=args.dbname,
            user=args.user,
            password=args.password,
        )
        return

    if args.pkg_file and args.file_list and args.os_version:
        import_from_file(args.pkg_file, args.file_list, args.os_version)
        return

    example_import()


if __name__ == "__main__":
    main()
