#!/usr/bin/env bash
set -euo pipefail

# Pin the release and verify the downloaded archive before compiling.
elpa_release=2024.05.001
elpa_sha256=9caf41a3e600e2f6f4ce1931bd54185179dade9c171556d0c9b41bbc6940f2f6
ci_deps_root="$PWD/ci-deps"
mkdir -p "$ci_deps_root/elpa-build"
cd "$ci_deps_root"
curl --fail --location --retry 3 \
    "https://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/$elpa_release/elpa-$elpa_release.tar.gz" \
    --output elpa.tar.gz
printf '%s  %s\n' "$elpa_sha256" elpa.tar.gz | sha256sum --check
tar -xzf elpa.tar.gz
cd elpa-build
"../elpa-$elpa_release/configure" \
    --prefix="$ci_deps_root/elpa-install" \
    --with-mpi=yes --enable-openmp --enable-skew-symmetric-support \
    --disable-avx512 \
    CC=mpicc CXX=mpicxx FC=mpifort \
    CFLAGS='-O2' CXXFLAGS='-O2' FCFLAGS='-O2' \
    LIBS='-lscalapack-openmpi -lopenblas -lm'
make -j2
make install
# Capture nm output before grep so pipefail cannot report an early-reader SIGPIPE.
nm -D "$ci_deps_root/elpa-install/lib/libelpa_openmp.so" > exported_symbols.txt
grep -q elpa_skew_eigenvectors exported_symbols.txt
