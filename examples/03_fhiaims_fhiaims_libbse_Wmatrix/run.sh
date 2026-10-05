#!/usr/bin/env bash
#SBATCH --job-name=03_Si_BSE
#SBATCH --partition=inter
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=workflow-%j.out
#SBATCH --error=workflow-%j.err
set -euo pipefail
# Submit from the example directory; direct invocation works from any directory.
cd "${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
source ./env.sh
ulimit -s unlimited
fail() { echo "ERROR: $*" >&2; exit 1; }
need_file() { [[ -s "$1" ]] || fail "Missing or empty file: $1"; }
need_exe() { command -v "$1" >/dev/null || fail "Executable not found: $1 (edit env.sh)"; }
need_glob() {
    local pattern=$1 file
    while IFS= read -r file; do
        if [[ -s "$file" ]]; then return 0; fi
    done < <(compgen -G "$pattern" || true)
    fail "No nonempty files matching: $pattern"
}
[[ $# == 0 || ( $# == 1 && $1 == --check ) ]] || fail "Usage: ./run.sh [--check]"
[[ $NPROCS =~ ^[1-9][0-9]*$ ]] || fail "NPROCS must be a positive integer"
need_exe "$MPIEXEC"
need_exe "$LIBBSE_EXE"
need_file libbse.in
need_exe "$AIMS_EXPORT_EXE"
need_exe "$PYTHON"
"$PYTHON" -c 'import numpy, h5py'
for file in control.in geometry.in aims_mommat_to_velocity.py; do need_file "$file"; done
need_exe "$AIMS_GW_EXE"
need_file control_gw.in
if [[ ${1:-} == --check ]]; then
    echo "Input files and environment checks passed; no calculation started."
    exit 0
fi
for output in aims_export libbse.d aims_gw; do
    [[ ! -e "$output" && ! -L "$output" ]] || fail "Existing output: $output; use a fresh copy of this example."
done
echo "Running FHI-aims interface export"
mkdir aims_export
cp control.in geometry.in aims_export/
# Serial HDF5 export on node-local storage avoids shared-filesystem MPI I/O stalls.
export_dir="$PWD/aims_export"
(
    scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/libbse-aims-${SLURM_JOB_ID:-local}.XXXXXX")
    finish_export() {
        local status=$?
        cp -a "$scratch_dir/." "$export_dir/" || exit 1
        rm -rf -- "$scratch_dir"
        exit "$status"
    }
    trap finish_export EXIT
    cp control.in geometry.in "$scratch_dir/"
    cd "$scratch_dir"
    export OMP_NUM_THREADS=${AIMS_EXPORT_THREADS:-4}
    export OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1
    "$MPIEXEC" -np 1 --bind-to none "$AIMS_EXPORT_EXE" > aims.out 2> aims.err
)
need_file aims_export/basis_out
need_file aims_export/band_out
"$PYTHON" aims_mommat_to_velocity.py aims_export aims_export/velocity_matrix
need_file aims_export/velocity_matrix
echo "Running native FHI-aims GW"
mkdir aims_gw
cp geometry.in aims_gw/
cp control_gw.in aims_gw/control.in
(cd aims_gw; "$MPIEXEC" -np "$NPROCS" "$AIMS_GW_EXE" > aims.out 2> aims.err)
need_file aims_gw/energy_qp
need_glob "aims_gw/periodic_gw_w_q_*_rank_*.dat"
echo "Running LibBSE"
"$MPIEXEC" -np "$NPROCS" "$LIBBSE_EXE" > libbse.out 2> libbse.err
need_file libbse.d/oscillator_strength_singlet_tda.dat
need_file libbse.d/spectrum_singlet_tda.dat
echo "Workflow completed: libbse.d"
