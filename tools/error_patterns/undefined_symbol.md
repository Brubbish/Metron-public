# Category: Undefined Symbol / Linker Error

**Symptoms**:
```
undefined reference to `function_name'
error: ld returned 1 exit status
cannot find -l<libname>
```

**Diagnosis**:
- Library not linked (missing `-l<name>` flag)
- Library not installed (missing `.so` / `.a` file)
- ABI mismatch (header version != library version)

**Fix pattern**:
1. Find the library: `ldconfig -p | grep <name>` or `find /usr -name "lib<name>*"`
2. Install if missing: `apt-get install -y lib<name>-dev`
3. Add link flag: `LDFLAGS="-L<path> -l<name>"`
4. If ABI mismatch: upgrade or downgrade the library package
