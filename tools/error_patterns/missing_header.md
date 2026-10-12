# Category: Missing Header / Include File

**Symptoms**:
```
fatal error: <header.h>: No such file or directory
cannot find include file: header.h
```

**Diagnosis**:
- The `-dev` or `-devel` package for that library is not installed
- The include path is wrong (need `-I<path>` flag)

**Fix pattern**:
1. Identify which package provides the header: `apt-file search <header.h>` or query the package-file KB
2. Install the `-dev` package: `apt-get install -y lib<name>-dev`
3. If already installed, add include path: `CFLAGS="-I/usr/include/<subdir>"`
