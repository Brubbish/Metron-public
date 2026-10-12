"""Shared text predicates for failure extraction and Matrix classification."""

from __future__ import annotations

import re


PACKAGE_MANAGER_FAILURE_PATTERN = (
    r"unable to locate package|has no installation candidate|unmet dependencies|"
    r"held broken packages|unable to correct problems|not going to be installed|"
    r"dpkg: error|conflicts with|conflicts:|"
    r"Packages were downgraded and -y was used without --allow-downgrades|"
    r"packages? would be downgraded|allow-downgrades"
)

MISSING_HEADER_PATTERN = (
    r"(?:fatal\s+error|error):\s*[A-Za-z0-9_./+\-]+\.(?:h|hh|hpp|hxx):\s*"
    r"No such file(?: or directory)?|"
    r"configure:\s*error:[^\n]*(?:[A-Za-z0-9_./+\-]+\.(?:h|hh|hpp|hxx)\s+not found|"
    r"development headers? (?:were |was )?not found)|"
    r"\b[A-Za-z0-9_.+\-]+\s+headers?\s+(?:were\s+|was\s+)?not found"
)

MISSING_TOOL_PATTERN = (
    r"(?:(?:/bin/)?(?:ba)?sh|/usr/bin/env):[^\n]*(?:command not found|not found|No such file or directory)|"
    r"[^\s:]+\.sh:\s*line\s+\d+:[^\n]*(?:command not found|not found)|"
    r"exec:\s*[^\s:]+:\s*not found|"
    r"make(?:\[\d+\])?:[^\n]*(?:command not found|Command not found)|"
    r"make(?:\[\d+\])?:\s*/[^\s:]+:\s*No such file or directory|"
    r"autoreconf:[^\n]*failed to run\s+[^\s:]+:\s*No such file or directory|"
    r"Program\(s\)\s+\[[^]]+\]\s+not found or not executable|"
    r"Program\s+[\"'][^\"']+[\"']\s+not found|"
    r"[A-Za-z0-9_.+\-]+\s+was not found\.\s*Please specify[^\n]*executable|"
    r"(?:testing|checking for)\s+[A-Za-z0-9_.+\-]+(?:\.\.\.)?\s*(?::\s*)?not found|"
    r"Cannot continue without\s+[A-Za-z0-9_.+\-]+|"
    r"Can(?:not|'t) find\s+[A-Za-z0-9_.+\-]+(?:\s+interpreter)?|"
    r"Could not find\s+[A-Za-z0-9_.+\-]+\.\s*Please specify the path|"
    r"[A-Za-z0-9_.+\-]+\s+not found,\s*please install|"
    r"must have\s+[`'\"]?[A-Za-z0-9_.+\-]+[`'\"]?\s+installed|"
    r"(?:command|executable)\s+[\"'][^\"']+[\"']\s+(?:was\s+)?not found|"
    r"unable to find\s+(?:program|executable|tool)\s+[A-Za-z0-9_.+\-]+|"
    r"you need to install\s+[A-Za-z0-9_.+\-]+|"
    r"[A-Za-z0-9_.+\-]+(?:ize)?:[^\n]*(?:command not found|not found)|"
    r"No\s+[A-Za-z0-9_.+\-]+(?:\s+or\s+[A-Za-z0-9_.+\-]+)+\s+present"
)

COMPILER_POLICY_PATTERN = (
    r"unrecognized (?:command[- ]line )?(?:argument|option)|"
    r"unknown (?:compiler )?(?:argument|option)|"
    r"warnings? (?:being )?treated as errors?|"
    r"does not support (?:c|c\+\+)\+?\d+|"
    r"unsupported (?:compiler|language standard)"
)

TOOL_CONFIGURATION_PATTERN = (
    r"externally-managed-environment|could not create named generator|"
    r"unknown (?:cmake|meson) generator|unsupported (?:toolchain|generator)|"
    r"unknown option(?::|\s)|unrecognized option(?::|\s)|"
    r"\b[A-Z][A-Z0-9_]+\s+is set to an invalid directory|"
    r"\bCMAKE_(?:C|CXX)_COMPILER\b[\s\S]{0,240}(?:does not exist|not a full path|was not found)|"
    r"\b[A-Za-z0-9_.+\-]+:\s*could not find (?:an?\s+|the\s+)?[^\n]+ installation|"
    r"usage:\s*meson(?:\.py)?\b|"
    r"toolchain[^\n]*does not support target"
)

VERSION_REQUIREMENT_PATTERN = (
    r"version(?:\s+of\s+[^\n:]+)?\s+is\s+too old|"
    r"version\s+[0-9][0-9A-Za-z.+:~\-]*\s+(?:is\s+)?too old|"
    r"version\s+too old|too old[^\n]*version|"
    r"\brequires?\b[^\n]*\bversion\b|"
    r"need[^\n]+(?:but\s+)?found|"
    r"invalid version of dependency|"
    r"\b[A-Za-z0-9_.+\-]+\s+version\s+[0-9][0-9A-Za-z.+\-]*\s+or higher\s+is required|"
    r"version mismatch error|"
    r"\bversion\b[^\n]+\brequires?\b"
)

API_COMPATIBILITY_PATTERN = (
    r"no member named|"
    r"(?:was not declared|is not declared) in this scope|"
    r"incomplete type|storage size of [^\n]+ isn't known|"
    r"no matching function for call|"
    r"cannot convert [^\n]+|"
    r"too (?:few|many) arguments to function"
)

INFRASTRUCTURE_FAILURE_PATTERN = (
    r"authorization failed|invalid api key|rate limit|tls handshake|"
    r"temporary failure (?:in name resolution|resolving)|"
    r"could not resolve|failed to download[^\n]*(?:timed out|urlopen error|EOF)|"
    r"cannot connect to the docker daemon|failed to start container|"
    r"failed to connect|connection timed out"
)

CAUSAL_DIAGNOSTIC_PATTERN = (
    rf"{PACKAGE_MANAGER_FAILURE_PATTERN}|{MISSING_HEADER_PATTERN}|"
    rf"{MISSING_TOOL_PATTERN}|{COMPILER_POLICY_PATTERN}|"
    rf"{TOOL_CONFIGURATION_PATTERN}|{VERSION_REQUIREMENT_PATTERN}|"
    rf"{API_COMPATIBILITY_PATTERN}|"
    rf"{INFRASTRUCTURE_FAILURE_PATTERN}|"
    r"configure:\s*error|cmake error|meson\.build.*error|"
    r"dependency\s+[\"'][^\"']+[\"']\s+(?:was\s+)?not found|"
    r"configure:\s*error:[^\n]*(?:not found|is required|missing required)|"
    r"(?:perl|python|ruby) module\s+[A-Za-z0-9_.+\-]+\s+not found|"
    r"No\s+[A-Za-z0-9_.+\-]+(?:\s+or\s+[A-Za-z0-9_.+\-]+)+\s+present|"
    r"no package\s+[\"']|could not find a package configuration|"
    r"cannot find\s+-l|undefined reference|multiple definition|"
    r"toolchain[^\n]*does not support target|"
    r"no rule to make target[^\n]+needed by|"
    r"no targets specified and no makefile found|does not appear to contain CMakeLists|"
    r"target[^\n]+does not exist|no such target|no PAL version enabled|"
    r"make(?:\[\d+\])?:\s*[^\n:]*makefile[^\n:]*:\s*No such file"
)


def has_missing_tool(text: str) -> bool:
    return bool(extract_missing_tools(text))


def extract_missing_tools(text: str) -> list[str]:
    value = str(text or "")
    patterns = (
        r"(?:(?:/bin/)?(?:ba)?sh|make(?:\[\d+\])?):(?:\s*line\s+\d+:|\s*\d+:)?\s*"
        r"([^\s:]+):\s*(?:command not found|not found)",
        r"[^\s:]+\.sh:\s*line\s+\d+:\s*([^\s:]+):\s*(?:command not found|not found)",
        r"exec:\s*([^\s:]+):\s*not found",
        r"make(?:\[\d+\])?:\s*(/[^\s:]+):\s*No such file or directory",
        r"/usr/bin/env:\s*[\"']?([^\s\"':]+)[\"']?:\s*No such file",
        r"autoreconf:[^\n]*failed to run\s+([^\s:]+):\s*No such file",
        r"Program\(s\)\s+\[[\"']([^\"']+)[\"']\]\s+not found or not executable",
        r"Program\s+[\"']([^\"']+)[\"']\s+not found",
        r"([A-Za-z0-9_.+\-]+)\s+was not found\.\s*Please specify[^\n]*executable",
        r"(?:testing|checking for)\s+([A-Za-z0-9_.+\-]+)(?:\.\.\.)?\s*(?::\s*)?not found",
        r"Cannot continue without\s+([A-Za-z0-9_.+\-]+)",
        r"Can(?:not|'t) find\s+([A-Za-z0-9_.+\-]+)(?:\s+interpreter)?",
        r"Could not find\s+([A-Za-z0-9_.+\-]+)\.\s*Please specify the path",
        r"([A-Za-z0-9_.+\-]+)\s+not found,\s*please install",
        r"must have\s+[`'\"]?([A-Za-z0-9_.+\-]+)[`'\"]?\s+installed",
        r"(?:command|executable)\s+[\"']([^\"']+)[\"']\s+(?:was\s+)?not found",
        r"unable to find\s+(?:program|executable|tool)\s+([A-Za-z0-9_.+\-]+)",
        r"you need to install\s+([A-Za-z0-9_.+\-]+)",
    )
    result: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, value, re.I):
            name = str(match.group(1) or "").strip().rstrip(".,;:")
            if _valid_missing_tool_name(name) and name not in result:
                result.append(name)
    for match in re.finditer(
        r"No\s+([A-Za-z0-9_.+\-]+)\s+or\s+([A-Za-z0-9_.+\-]+)(?:\s+or\s+([A-Za-z0-9_.+\-]+))*\s+present",
        value,
        re.I,
    ):
        for name in match.groups():
            if _valid_missing_tool_name(name) and name not in result:
                result.append(name)
    return result


_NON_TOOL_TOKENS = {
    "source", "command", "exec", "test", "true", "false", "cxx", "cc",
    "cpp", "cflags", "cxxflags", "ldflags",
}


def _valid_missing_tool_name(value: str) -> bool:
    name = str(value or "").strip().strip("`'\"")
    if not name or name.lower() in _NON_TOOL_TOKENS:
        return False
    if name.startswith(("-", ".", "/")) or "/" in name:
        return False
    return not bool(re.search(r"[<>{}$()]", name))


def extract_missing_headers(text: str) -> list[str]:
    value = str(text or "")
    patterns = (
        r"(?:fatal\s+error|error):\s*([A-Za-z0-9_./+\-]+\.(?:h|hh|hpp|hxx)):\s*No such file",
        r"configure:\s*error:[^\n]*?([A-Za-z0-9_./+\-]+\.(?:h|hh|hpp|hxx))\s+not found",
        r"configure:\s*error:\s*(?:the\s+)?([A-Za-z0-9_.+\-]+)\s+development headers? (?:were |was )?not found",
        r"(?:configure:\s*error:\s*)?([A-Za-z0-9_.+\-]+)\s+headers? (?:were |was )?not found",
    )
    result: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, value, re.I):
            name = str(match.group(1) or "").strip()
            if name and name not in result:
                result.append(name)
    return result


def has_compiler_policy(text: str) -> bool:
    return bool(re.search(COMPILER_POLICY_PATTERN, str(text or ""), re.I))


def has_tool_configuration_failure(text: str) -> bool:
    return bool(re.search(TOOL_CONFIGURATION_PATTERN, str(text or ""), re.I))


def has_version_requirement_failure(text: str) -> bool:
    return bool(re.search(VERSION_REQUIREMENT_PATTERN, str(text or ""), re.I))


def extract_version_relation(text: str, failed_tool: str = "") -> dict[str, str]:
    """Extract a conservative observed/required relation from one diagnostic."""
    value = str(text or "")
    patterns = (
        r"(?P<subject>[A-Za-z][A-Za-z0-9_.+\-]*)\s+version\s+(?P<observed>[0-9][0-9A-Za-z.+:~\-]*)"
        r"[^\n]*(?:requires?|required|need(?:s|ed)?)\s*(?:version\s*)?(?P<op>>=|>|==|=)?\s*"
        r"(?P<required>[0-9][0-9A-Za-z.+:~\-]*)",
        r"(?P<subject>[A-Za-z][A-Za-z0-9_.+\-]*)[^\n]*(?:requires?|required|need(?:s|ed)?)\s*"
        r"(?:version\s*)?(?P<op>>=|>|==|=)?\s*(?P<required>[0-9][0-9A-Za-z.+:~\-]*)"
        r"[^\n]*(?:found|current|but)\D*(?P<observed>[0-9][0-9A-Za-z.+:~\-]*)",
        r"version\s+(?P<observed>[0-9][0-9A-Za-z.+:~\-]*)\s+(?:is\s+)?too old[^\n]*"
        r"(?P<op>>=|>|==|=)?\s*(?P<required>[0-9][0-9A-Za-z.+:~\-]*)",
    )
    for pattern in patterns:
        match = re.search(pattern, value, re.I)
        if not match:
            continue
        groups = match.groupdict()
        return {
            "subject": str(groups.get("subject") or failed_tool or "build_tool").lower(),
            "observed_version": str(groups.get("observed") or ""),
            "required_operator": str(groups.get("op") or ">="),
            "required_version": str(groups.get("required") or ""),
            "evidence": match.group(0)[:400],
        }
    if has_version_requirement_failure(value):
        evidence = next(
            (line[:400] for line in value.splitlines() if re.search(VERSION_REQUIREMENT_PATTERN, line, re.I)),
            "",
        )
        return {
            "subject": str(failed_tool or "build_tool").lower(),
            "observed_version": "",
            "required_operator": "",
            "required_version": "",
            "evidence": evidence,
        }
    return {}


def has_causal_diagnostic(text: str) -> bool:
    return bool(re.search(CAUSAL_DIAGNOSTIC_PATTERN, str(text or ""), re.I))


def is_wrong_entry_or_target(text: str) -> bool:
    value = str(text or "")
    if re.search(r"no rule to make target[^\n]+needed by", value, re.I):
        return False
    return bool(re.search(
        r"no targets specified and no makefile found|"
        r"does not appear to contain CMakeLists\.txt|"
        r"source directory[^\n]+does not (?:exist|appear to contain)|"
        r"make(?:\[\d+\])?:\s*[^\n:]*makefile[^\n:]*:\s*No such file|"
        r"(?:^|\n)[^\n]*Makefile[^:\n]*:\s*[^\n]*:\s*No such file|"
        r"no rule to make target(?![^\n]+needed by)|"
        r"target[^\n]+does not exist|no such target|no PAL version enabled|"
        r"ci (?:job|environment)[^\n]*must be set|"
        r"\b[A-Za-z_][A-Za-z0-9_]*:\s*must be set by (?:travis|ci)|"
        r"could not identify ci type|"
        r"(?:ba)?sh:\s*line\s+\d+:\s*(?:/work/|/tests?/|\./(?!configure\b))[^\s:]+:\s*(?:No such file|Permission denied)|"
        r"fatal:\s*destination path\s+['\"][^'\"]+['\"]\s+already exists|"
        r"tar:\s*[^\n:]+:\s*Cannot open:\s*No such file",
        value,
        re.I,
    ))
