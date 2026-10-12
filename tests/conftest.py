"""Shared pytest configuration.

Unit tests import modules that read ``LLM_Config/models.txt`` at import time
— i.e. during test collection, before any fixture can run. The real config
holds API keys and is gitignored, so when it is absent we materialize a
throwaway dummy config (pointed at localhost so any accidental real LLM call
fails fast) and remove it again when the test process exits. Tests never
issue real LLM requests.
"""

import atexit
import os

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_TXT = os.path.join(_REPO_ROOT, "LLM_Config", "models.txt")

_DUMMY_CONTENT = (
    "dummy/test-model,http://127.0.0.1:9/v1,dummy-key,"
    "default|direct|readrepo|phase2|phase3|check\n"
)


def _ensure_dummy_model_config() -> None:
    if os.path.exists(_MODELS_TXT):
        return
    os.makedirs(os.path.dirname(_MODELS_TXT), exist_ok=True)
    with open(_MODELS_TXT, "w", encoding="utf-8") as handle:
        handle.write(_DUMMY_CONTENT)
    atexit.register(_remove_dummy_model_config)


def _remove_dummy_model_config() -> None:
    try:
        os.remove(_MODELS_TXT)
    except OSError:
        pass


_ensure_dummy_model_config()


# ---------------------------------------------------------------------------
# Integration-driver fixtures
#
# test_doc_scanner*.py and test_file_matcher.py are manual integration
# drivers: they expect a real C/C++ project checkout and hand-picked inputs.
# Point METRON_TEST_REPO (and optionally METRON_TEST_REPOS for the batch
# driver) at a local project to enable them; otherwise they skip.
# ---------------------------------------------------------------------------


@pytest.fixture
def repo_path():
    path = os.environ.get("METRON_TEST_REPO", "")
    if not path or not os.path.isdir(path):
        pytest.skip("set METRON_TEST_REPO to a local C/C++ project to run this integration driver")
    return path


@pytest.fixture
def repo_paths():
    raw = os.environ.get("METRON_TEST_REPOS", "")
    paths = [p for p in raw.split(os.pathsep) if p and os.path.isdir(p)]
    if not paths:
        pytest.skip(
            "set METRON_TEST_REPOS to a list of local C/C++ projects "
            "(os.pathsep-separated) to run this integration driver"
        )
    return paths


def _manual_input(name: str):
    pytest.skip(f"manual integration driver: provide {name} directly")


@pytest.fixture
def file_list():
    _manual_input("a list of documents")


@pytest.fixture
def urls():
    _manual_input("a list of URLs")


@pytest.fixture
def docs_content():
    _manual_input("parsed document content")


@pytest.fixture
def file_refs():
    _manual_input("file references")


@pytest.fixture
def visited_files():
    _manual_input("the visited-file set")
