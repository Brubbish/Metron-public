# Category: Git Submodule Failure

**Symptoms**:
```
fatal: repository 'git://...' not found
fatal: clone of 'git://...' into submodule path '...' failed
Failed to clone '...' a second time, aborting
git submodule update --init failed
```

**Diagnosis**:
- Submodule URLs may be outdated (git:// protocol deprecated, project renamed/moved)
- Network issues or firewall blocking git:// protocol
- Submodules not initialized in the source tree

**Fix pattern**:
1. Check what submodules exist: `cat .gitmodules`
2. If URLs use `git://`, try replacing with `https://`: `git config submodule.<name>.url https://...`
3. If the project was renamed (e.g., bro → zeek), find the new URL on GitHub
4. Try: `git submodule sync && git submodule update --init --recursive`
5. If a submodule is a well-known project, clone it manually into the expected path
6. If submodule contains only cmake modules, check if the same modules are available via apt (e.g., `cmake-extras`)
