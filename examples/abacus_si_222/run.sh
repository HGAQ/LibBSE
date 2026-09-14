#!/usr/bin/env bash
set -euo pipefail

: "${ABACUS_EXE:?set ABACUS_EXE to the ABACUS executable}"
: "${LIBRPA_EXE:?set LIBRPA_EXE to chi0_main.exe}"
: "${LIBBSE_EXE:?set LIBBSE_EXE to the LibBSE executable}"

NPROCS=${NPROCS:-4}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMP_NUM_THREADS

example_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$example_dir"

for atomic_file in Si_ONCV_PBE-1.0.upf Si_gga_8au_100Ry_3s3p2d.orb \
                   Si_3s3p2d1f1g_pca1e-6.abfs; do
  if [[ ! -e "$atomic_file" && ! -L "$atomic_file"
        && -n "${ABACUS_ATOMIC_DATA_DIR:-}" ]]; then
    ln -s "$ABACUS_ATOMIC_DATA_DIR/$atomic_file" "$atomic_file"
  fi
  test -s "$atomic_file" || {
    echo "missing ABACUS atomic-data file: $atomic_file" >&2
    exit 1
  }
done

cp INPUT.scf INPUT
mpirun -np "$NPROCS" "$ABACUS_EXE" > abacus-scf.out 2> abacus-scf.err
test -s OUT.librpa/velocity_matrix

mpirun -np "$NPROCS" "$LIBRPA_EXE" > librpa.out 2> librpa.err
mpirun -np "$NPROCS" "$LIBBSE_EXE" > libbse.out 2> libbse.err

test -s energy_qp
test -n "$(find librpa.d -name 'Wc_Mu_*_ifreq_0.mtx' -print -quit)"
test -s libbse.d/oscillator_strength_singlet_tda.dat
test -s libbse.d/spectrum_singlet_tda.dat
