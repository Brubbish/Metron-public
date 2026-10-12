# Category: Permission / Path Error

**Symptoms**:
```
Permission denied
cannot create directory
install: cannot create regular file
```

**Diagnosis**:
- Install prefix is system path and no sudo
- Build directory ownership issue

**Fix pattern**:
- If this is a source dependency, keep the install prefix at `/usr/local` and fix the install permissions or command invocation inside the container
- If this is the main project, prefer a normal build without system install unless the docs explicitly require installation
- Create the required build directory first (for example `mkdir -p build`) before rebuilding
