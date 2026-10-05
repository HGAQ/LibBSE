#!/usr/bin/env bash
#SBATCH --job-name=07_Si_interp
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
need_exe "$PYTHON"
"$PYTHON" -c 'import numpy'
need_exe "$ABACUS_EXE"
need_exe "$LIBRPA_EXE"
for file in INPUT_scf INPUT_nscf KPT_scf KPT_nscf preprocess_abacus_for_librpa_band.py band_to_energy_qp.py STRU Si_ONCV_PBE-1.0.upf Si_gga_8au_100Ry_3s3p2d.orb Si_3s3p2d1f1g_pca1e-6.abfs librpa.in; do need_file "$file"; done
if [[ ${1:-} == --check ]]; then
    echo "Input files and environment checks passed; no calculation started."
    exit 0
fi
for output in OUT.scf OUT.nscf OUT.librpa librpa.d libbse.d energy_qp GW_band_spin_1.dat; do
    [[ ! -e "$output" && ! -L "$output" ]] || fail "Existing output: $output; use a fresh copy of this example."
done
cp INPUT_scf INPUT
cp KPT_scf KPT
echo "Running ABACUS SCF and interface export"
"$MPIEXEC" -np "$NPROCS" "$ABACUS_EXE" > abacus.out 2> abacus.err
need_file OUT.scf/vxc_out.dat
need_file OUT.librpa/velocity_matrix
echo "Running ABACUS fine-grid NSCF"
cp INPUT_nscf INPUT
cp KPT_nscf KPT
"$MPIEXEC" -np "$NPROCS" "$ABACUS_EXE" > abacus_nscf.out 2> abacus_nscf.err
"$PYTHON" preprocess_abacus_for_librpa_band.py -i OUT.nscf -o OUT.librpa
echo "Running LibRPA G0W0 (including fine-grid band points)"
# The compressed-Wc MPI path in this build requires a single LibRPA rank.
OMP_NUM_THREADS=$((NPROCS * OMP_NUM_THREADS)) OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
"$MPIEXEC" -np 1 --bind-to none "$LIBRPA_EXE" > librpa.out 2> librpa.err
need_file GW_band_spin_1.dat
"$PYTHON" band_to_energy_qp.py librpa . energy_qp_fine
need_file energy_qp_fine
need_glob "librpa.d/Wc_Mu_*_ifreq_0.mtx"
echo "Running LibBSE"
"$MPIEXEC" -np "$NPROCS" "$LIBBSE_EXE" > libbse.out 2> libbse.err
need_file libbse.d/oscillator_strength_singlet_tda.dat
need_file libbse.d/spectrum_singlet_tda.dat
echo "Workflow completed: libbse.d"
