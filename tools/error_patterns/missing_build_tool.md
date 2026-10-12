# Category: Missing Build Tool

**Symptoms**:
```
make: command not found
cmake: No such file
ninja: not found
autoreconf: not found
```

**Fix pattern**:
```bash
apt-get install -y cmake           # cmake
apt-get install -y ninja-build     # ninja
apt-get install -y autoconf automake libtool  # autotools
apt-get install -y build-essential # make, gcc, g++
```
