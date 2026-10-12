"""Known source-dependency hints mined from successful/manual builds.

This registry is intentionally small and conservative.  It does not try to
solve a build by itself; it gives Phase 1/2 concrete clone targets and
placement notes for dependency families that are commonly missed by apt or by
generic document extraction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SourceDepRegistryEntry:
    owner: Optional[str]
    repo: str
    name: str
    clone_url: str
    reason: str
    instructions: str = ""
    build_order: tuple[str, ...] = ()
    setup_commands: tuple[str, ...] = ()
    checkout_ref: str = ""
    checkout_before_project_date: bool = False
    checkout_branch: str = ""
    recommended_sysuses: tuple[str, ...] = ()


_REGISTRY: tuple[SourceDepRegistryEntry, ...] = (
    SourceDepRegistryEntry(
        owner="asterisk",
        repo="asterisk",
        name="menuselect",
        clone_url="https://github.com/asterisk/menuselect.git",
        reason="old Asterisk commits often omit the menuselect external tree",
        instructions=(
            "Clone menuselect when the build fails with 'menuselect: No such "
            "file or directory'. Place or symlink it where the Makefile expects "
            "it, commonly /work/menuselect or /work/build_tools/menuselect."
        ),
        checkout_before_project_date=True,
        recommended_sysuses=("etch", "lenny", "squeeze"),
    ),
    SourceDepRegistryEntry(
        owner="alliedmodders",
        repo="amxmodx",
        name="hlsdk",
        clone_url="https://github.com/alliedmodders/hlsdk.git",
        reason="AMX Mod X expects Half-Life SDK headers such as extdll.h",
        instructions=(
            "If extdll.h is missing, clone or symlink this SDK to the sibling "
            "path expected by the Makefile, typically /work/../hlsdk."
        ),
        checkout_before_project_date=True,
        recommended_sysuses=("etch", "lenny", "squeeze"),
    ),
    SourceDepRegistryEntry(
        owner="alliedmodders",
        repo="amxmodx",
        name="metamod",
        clone_url="https://github.com/alliedmodders/metamod-hl1.git",
        reason="AMX Mod X expects Metamod headers beside the main checkout",
        instructions=(
            "If Metamod headers are missing, clone or symlink this dependency "
            "to the sibling path expected by the Makefile, typically "
            "/work/../metamod."
        ),
        checkout_before_project_date=True,
        recommended_sysuses=("etch", "lenny", "squeeze"),
    ),
    SourceDepRegistryEntry(
        owner="hyprwm",
        repo="Hyprland",
        name="wlroots",
        clone_url="https://gitlab.freedesktop.org/wlroots/wlroots.git",
        reason="Hyprland commits require a narrow wlroots header/API window",
        instructions=(
            "Use a wlroots revision from before the project commit date when "
            "headers like wlr/render/allocator.h or wlr_subcompositor.h are "
            "missing. Prefer the in-tree subproject path if present."
        ),
        checkout_before_project_date=True,
        recommended_sysuses=("bullseye", "bookworm"),
    ),
    SourceDepRegistryEntry(
        owner="01org",
        repo="tpm2.0-tools",
        name="tpm2-tss",
        clone_url="https://github.com/01org/tpm2-tss.git",
        reason="old tpm2-tools commits must match the tpm2-tss header/API era",
        instructions=(
            "Build and install tpm2-tss before tpm2-tools when TPM2 symbols or "
            "generated source files are missing. Pick a tpm2-tss revision before "
            "the tpm2-tools project commit date."
        ),
        checkout_before_project_date=True,
        recommended_sysuses=("jessie", "stretch", "buster"),
    ),
    SourceDepRegistryEntry(
        owner="BelledonneCommunications",
        repo="belle-sip",
        name="antlr3",
        clone_url="https://github.com/BelledonneCommunications/antlr3.git",
        reason="belle-sip requires antlr3 C runtime newer than old Debian packages",
        instructions=(
            "Use this antlr3 source when libantlr3c-dev is too old or missing "
            "symbols such as antlr3StringStreamNew. Build/install it before "
            "running belle-sip configure."
        ),
        checkout_before_project_date=True,
        recommended_sysuses=("wheezy", "jessie", "stretch"),
    ),
    SourceDepRegistryEntry(
        owner="any1",
        repo="neatvnc",
        name="aml",
        clone_url="https://github.com/any1/aml.git",
        reason="neatvnc requires aml >= 0.3.0 and Debian may provide aml 0.2.x",
        instructions=(
            "Build and install aml from source when Meson reports aml < 0.3.0. "
            "Do not substitute aml1 unless pkg-config confirms it provides the "
            "same dependency name expected by neatvnc."
        ),
        checkout_before_project_date=True,
        recommended_sysuses=("bookworm", "trixie"),
    ),
    SourceDepRegistryEntry(
        owner="facebook",
        repo="wangle",
        name="folly",
        clone_url="https://github.com/facebook/folly.git",
        reason="wangle depends on a matching folly source tree",
        instructions=(
            "Build/install folly before wangle when CMake cannot find Folly or "
            "folly headers. Use a revision before the wangle commit date."
        ),
        checkout_before_project_date=True,
        recommended_sysuses=("jessie", "stretch", "buster", "bullseye"),
    ),
    SourceDepRegistryEntry(
        owner="facebook",
        repo="wangle",
        name="fizz",
        clone_url="https://github.com/facebookincubator/fizz.git",
        reason="wangle TLS code often requires matching fizz",
        instructions=(
            "Build/install fizz after folly when wangle CMake requests Fizz. "
            "Use a revision before the wangle commit date."
        ),
        checkout_before_project_date=True,
        recommended_sysuses=("jessie", "stretch", "buster", "bullseye"),
    ),
)


def infer_project_identity(repo_path: str) -> tuple[str, str]:
    """Infer (owner, repo) from known local checkout layouts."""
    path = os.path.abspath(repo_path)
    base = os.path.basename(path.rstrip(os.sep))
    parent = os.path.basename(os.path.dirname(path.rstrip(os.sep)))

    if "__" in base:
        parts = base.split("__")
        if len(parts) >= 2:
            owner_part = parts[0].split("_", 1)[-1]
            return owner_part, parts[1]

    return parent, base


def match_source_dep_registry(repo_path: str) -> list[SourceDepRegistryEntry]:
    """Return registry entries that match the current project checkout."""
    owner, repo = infer_project_identity(repo_path)
    owner_l = owner.lower()
    repo_l = repo.lower()
    matches: list[SourceDepRegistryEntry] = []
    for entry in _REGISTRY:
        if entry.repo.lower() != repo_l:
            continue
        if entry.owner and entry.owner.lower() != owner_l:
            continue
        matches.append(entry)
    return matches


def registry_sysuse_hints(repo_path: str) -> list[str]:
    """Return ordered sysuse hints for matched registry entries."""
    out: list[str] = []
    for entry in match_source_dep_registry(repo_path):
        for sysuse in entry.recommended_sysuses:
            if sysuse not in out:
                out.append(sysuse)
    return out
