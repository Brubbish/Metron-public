# Category: Pkg-config Failure

**Symptoms**:
```
Package '<name>' not found
No package '<name>' found
```

**Diagnosis**:
- `.pc` file not installed (need `-dev` package)
- `.pc` file in non-standard path

**Fix pattern**:
1. Install dev package: `apt-get install -y lib<name>-dev`
2. If installed but not found: `find /usr/local /usr -name "<name>.pc"` then rerun only the failing command with `PKG_CONFIG_PATH=<dir>:$PKG_CONFIG_PATH`
