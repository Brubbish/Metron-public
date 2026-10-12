import asyncio
import signal

from repo_parse.doc_scanner_parse import (
    DocumentSource,
    call_enhanced_dep_extractor,
    _terminate_process_group,
    _structural_build_steps,
    _dependencies_from_enhanced_payload,
    merge_document_sources,
    read_multiple_documents_parallel,
)
from core.build_artifact_discovery.summarizers import _parse_debian_control


def test_merge_document_sources_keeps_input_order_and_combines_source_labels(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    readme = repo / "README.md"
    install = repo / "INSTALL"
    readme.write_text("Build with cmake.\n", encoding="utf-8")
    install.write_text("Run make.\n", encoding="utf-8")

    merged = merge_document_sources(
        str(repo),
        [
            ("regex_doc", [str(readme), str(install)]),
            ("input_json", ["README.md"]),
            ("directory_ref", ["missing.md"]),
        ],
    )

    assert [source.source_type for source in merged] == ["regex_doc+input_json", "regex_doc"]
    assert [source.path for source in merged] == [str(readme.resolve()), str(install.resolve())]


def test_dependencies_from_enhanced_payload_reads_suite_packages():
    payload = {
        "CMakeLists.txt": [
            {
                "software_name": "pkgconf",
                "packages": ["pkgconf"],
                "suite_packages": ["pkg-config"],
                "sources": ["build_driver"],
            }
        ]
    }

    deps = _dependencies_from_enhanced_payload(payload)

    assert deps[0]["unified_name"] == "pkg-config"
    assert deps[0]["name"] == "pkgconf"


def test_subordinate_build_files_do_not_create_structural_entries():
    for filename in ("meson.options", "meson_options.txt", "SConscript", "wscript_build", "wscript_configure"):
        assert _structural_build_steps(
            {"file": filename, "path": f"sub/{filename}"},
            {f"sub/{filename}"},
        ) == []


def test_debian_control_preserves_package_alternatives(tmp_path):
    control = tmp_path / "control"
    control.write_text(
        "Source: demo\nBuild-Depends: cmake, missing-dev | available-dev (>= 1.0)\n",
        encoding="utf-8",
    )

    summary = _parse_debian_control(str(control))

    assert summary.apt_packages == ["cmake"]
    assert summary.apt_package_alternatives == [["missing-dev", "available-dev"]]


def test_extractor_cleanup_kills_process_group(monkeypatch):
    calls = []

    class FakeProcess:
        pid = 321
        returncode = None

        async def wait(self):
            calls.append("wait")
            self.returncode = -signal.SIGKILL

    monkeypatch.setattr(
        "repo_parse.doc_scanner_parse.os.killpg",
        lambda pid, sig: calls.append((pid, sig)),
    )

    asyncio.run(_terminate_process_group(FakeProcess()))

    assert calls == [(321, signal.SIGKILL), "wait"]


def test_extractor_cleanup_kills_group_after_leader_exits(monkeypatch):
    calls = []

    class FakeProcess:
        pid = 654
        returncode = 1

        async def wait(self):
            calls.append("unexpected wait")

    monkeypatch.setattr(
        "repo_parse.doc_scanner_parse.os.killpg",
        lambda pid, sig: calls.append((pid, sig)),
    )

    asyncio.run(_terminate_process_group(FakeProcess()))

    assert calls == [(654, signal.SIGKILL)]


def test_extractor_cancellation_reaps_process_group(monkeypatch, tmp_path):
    calls = []

    class FakeProcess:
        pid = 777
        returncode = None
        started = asyncio.Event()
        blocked = asyncio.Event()

        async def communicate(self):
            self.started.set()
            await self.blocked.wait()

        async def wait(self):
            calls.append("wait")
            self.returncode = -signal.SIGKILL

        def kill(self):
            calls.append("kill")

    process = FakeProcess()

    async def fake_create_subprocess_exec(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(
        "repo_parse.doc_scanner_parse.os.killpg",
        lambda pid, sig: calls.append((pid, sig)),
    )

    async def scenario():
        task = asyncio.create_task(
            call_enhanced_dep_extractor(str(tmp_path), str(tmp_path / "result.json"))
        )
        await process.started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())

    assert calls == [(777, signal.SIGKILL), "wait"]


def test_merge_document_sources_skips_invalid_null_byte_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    readme = repo / "README.md"
    readme.write_text("Build with cmake.\n", encoding="utf-8")

    merged = merge_document_sources(
        str(repo),
        [
            ("directory_ref", ["bad\x00path.md", "README.md"]),
        ],
    )

    assert [source.path for source in merged] == [str(readme.resolve())]


def test_merge_document_sources_skips_unreasonable_long_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    readme = repo / "README.md"
    readme.write_text("Build with cmake.\n", encoding="utf-8")

    merged = merge_document_sources(
        str(repo),
        [
            ("directory_ref", ["x" * 5000, "README.md"]),
        ],
    )

    assert [source.path for source in merged] == [str(readme.resolve())]


def test_read_multiple_documents_keeps_sections_and_order(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    first = repo / "README.md"
    second = repo / "docs.md"
    first.write_text("Build with cmake. See https://example.test/build\n", encoding="utf-8")
    second.write_text("Install deps with apt-get install zlib1g-dev\n", encoding="utf-8")

    async def fake_regex_get_urls(paths, compile_path):
        if paths[0].endswith("README.md"):
            return ["https://example.test/build"], ["docs.md"]
        return ["https://example.test/build"], []

    monkeypatch.setattr("repo_parse.doc_scanner_parse.regex_get_urls", fake_regex_get_urls)

    docs, urls, files = asyncio.run(
        read_multiple_documents_parallel(
            [
                DocumentSource(str(first), "regex_doc"),
                DocumentSource(str(second), "input_json"),
            ],
            str(repo),
        )
    )

    assert docs.index("Source: regex_doc | File: README.md") < docs.index("Source: input_json | File: docs.md")
    assert urls == ["https://example.test/build"]
    assert files == ["docs.md"]
