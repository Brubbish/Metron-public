# Category: Compiler Error (Source Code Issue)

**Symptoms**:
```
error: use of undeclared identifier
error: implicit declaration of function
error: no member named 'X' in 'Y'
```

**Diagnosis**:
- API changed between versions
- Missing `#include` in source
- Incompatible C/C++ standard required

**Fix pattern**:
1. Check if a specific C/C++ standard is needed: try `-std=c++14`, `-std=c++17`, `-std=c99`
2. If API change: check if there's a compatibility patch or build flag to disable new feature
3. Flag as `CannotSolve` if source code modification is required and no patch exists
