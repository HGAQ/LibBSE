#!/usr/bin/env bash
set -eo pipefail

# The runner owns this environment file, outside the checked-out repository.
: "${LIBBSE_CI_ENV:?Set the repository variable LIBBSE_CI_ENV}"
test -f "$LIBBSE_CI_ENV"
source "$LIBBSE_CI_ENV"
set -u
: "${LIBRPA_SOURCE:?Set LIBRPA_SOURCE in the runner environment}"
: "${LIBRI_HEADERS:?Set LIBRI_HEADERS in the runner environment}"
: "${ELPA_PREFIX:?Set ELPA_PREFIX in the runner environment}"
: "${REGRESSION_FIXTURES:?Set REGRESSION_FIXTURES in the runner environment}"
: "${REGRESSION_REFERENCE:?Set REGRESSION_REFERENCE in the runner environment}"
: "${REGRESSION_REFERENCE_SHA256:?Set the approved reference SHA256}"

# Never rebuild or silently replace the reference with the candidate.
printf '%s  %s\n' "$REGRESSION_REFERENCE_SHA256" "$REGRESSION_REFERENCE" | sha256sum --check
python3 -c 'import sys, numpy; assert sys.version_info >= (3, 11)'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cmake -S . -B build-ci \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_Fortran_COMPILER=mpifort \
    -DBLA_VENDOR=OpenBLAS \
    -DLIBRPA_USE_CMAKE_INC=OFF \
    -DLIBBSE_BUILD_TESTING=ON \
    -DLIBRPA_INCLUDE_DIR="$LIBRPA_SOURCE/include" \
    -DCEREAL_INCLUDE_DIR="$LIBRPA_SOURCE/thirdparty/cereal-1.3.0/include" \
    -DLIBCOMM_INCLUDE_DIR="$LIBRPA_SOURCE/thirdparty/LibComm/include" \
    -DLIBRI_INCLUDE_DIR="$LIBRI_HEADERS" \
    -DEXTERNAL_ELPA_DIR="$ELPA_PREFIX" \
    -DLIBBSE_REGRESSION_FIXTURES="$REGRESSION_FIXTURES" \
    -DLIBBSE_REGRESSION_REFERENCE_EXE="$REGRESSION_REFERENCE"
cmake --build build-ci --parallel 4 2>&1 | tee build-ci/build.log
ctest --test-dir build-ci --parallel 1 --no-tests=error \
    --output-on-failure --output-junit regression.xml 2>&1 | tee build-ci/ctest.log
