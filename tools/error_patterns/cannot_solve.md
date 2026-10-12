# CannotSolve Conditions

Mark as `CannotSolve` and exit Phase 3 when:
- Same error repeats 3 times despite different fix attempts
- Error requires modifying source code with no available patch
- Required dependency version unavailable in the current environment
- Build environment fundamentally incompatible (e.g., wrong architecture)
