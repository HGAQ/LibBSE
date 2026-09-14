#!/usr/bin/env bash
set -euo pipefail

# Required executables can be overridden from the environment.
: "${AIMS_EXE:?set AIMS_EXE to the FHI-aims executable}"
: "${LIBRPA_EXE:?set LIBRPA_EXE to chi0_main.exe}"
: "${LIBBSE_EXE:?set LIBBSE_EXE to the LibBSE executable}"
: "${MOMMAT_CONVERTER:?set MOMMAT_CONVERTER to tools/aims_mommat_to_velocity.py}"

NPROCS=${NPROCS:-4}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMP_NUM_THREADS
# FHI-aims' periodic RI/momentum path allocates large thread-local arrays and
# aborts early on clusters whose batch jobs inherit a small stack limit.
ulimit -s unlimited

example_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$example_dir"

mkdir -p aims_export
cp control.in geometry.in aims_export/
(
  cd aims_export
  mpirun -np "$NPROCS" "$AIMS_EXE" > aims.out 2> aims.err
)

# The momentum matrix is an optical/BSE input, not a GW input. Convert it
# before LibRPA so the three-stage directory is already complete.
"$MOMMAT_CONVERTER" aims_export aims_export/velocity_matrix

mpirun -np "$NPROCS" "$LIBRPA_EXE" > librpa.out 2> librpa.err
mpirun -np "$NPROCS" "$LIBBSE_EXE" > libbse.out 2> libbse.err

test -s energy_qp
test -n "$(find librpa.d -name 'Wc_Mu_*_ifreq_0.mtx' -print -quit)"
test -s libbse.d/oscillator_strength_singlet_tda.dat
test -s libbse.d/spectrum_singlet_tda.dat
