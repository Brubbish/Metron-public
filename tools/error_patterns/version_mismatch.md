# Category: Version Mismatch / Requirement Not Met

**Symptoms**:
```
CMake Error: ... requires version >= X.Y
configure: error: library version too old
Dependency ... found: NO (found X.Y.Z, need >= A.B.C)
No package 'old_pkg_name' found (after building latest version of a dependency from source)
```

**Diagnosis**:
- Installed version too old
- Multiple versions installed, wrong one selected
- Cloned latest version of a source dependency but the main project expects an older API/pkg-config name

**Fix pattern**:
1. Check installed version: `<tool> --version` or `pkg-config --modversion <lib>`
2. If too old: try PPA / backport / build from source
3. If multiple versions: rerun the failing command with `PKG_CONFIG_PATH` or `CMAKE_PREFIX_PATH` passed inline to the newer version path; do not persist a global override
4. If a cloned source dependency's latest version is incompatible (API changed, pkg-config name changed): checkout to an older commit matching the main project's era: `cd <dep> && git checkout $(git rev-list -n1 --before="<project_commit_date>" HEAD)` then rebuild
