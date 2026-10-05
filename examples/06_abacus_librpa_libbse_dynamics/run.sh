#!/usr/bin/env bash
#SBATCH --job-name=06_Si_BSE
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
need_file prepare_chi0_input.py
need_exe "$ABACUS_EXE"
need_exe "$LIBRPA_EXE"
for file in INPUT STRU KPT Si_ONCV_PBE-1.0.upf Si_gga_8au_100Ry_3s3p2d.orb Si_3s3p2d1f1g_pca1e-6.abfs librpa.in; do need_file "$file"; done
if [[ ${1:-} == --check ]]; then
    echo "Input files and environment checks passed; no calculation started."
    exit 0
fi
for output in OUT.scf OUT.librpa OUT.librpa_chi0 librpa.d libbse.d energy_qp; do
    [[ ! -e "$output" && ! -L "$output" ]] || fail "Existing output: $output; use a fresh copy of this example."
done
echo "Running ABACUS SCF and interface export"
"$MPIEXEC" -np "$NPROCS" "$ABACUS_EXE" > abacus.out 2> abacus.err
need_file OUT.scf/vxc_out.dat
need_file OUT.librpa/velocity_matrix
"$PYTHON" prepare_chi0_input.py OUT.librpa OUT.librpa_chi0
echo "Running LibRPA G0W0"
"$MPIEXEC" -np "$NPROCS" "$LIBRPA_EXE" > librpa.out 2> librpa.err
need_file energy_qp
need_file librpa.d/chi0_static/chi0_rf.info
need_glob "librpa.d/chi0_static/Chi0_Mu_*_ifreq_0.mtx"
echo "Running LibBSE"
"$MPIEXEC" -np "$NPROCS" "$LIBBSE_EXE" > libbse.out 2> libbse.err
need_file libbse.d/oscillator_strength_singlet_tda.dat
need_file libbse.d/spectrum_singlet_tda.dat
need_file libbse.d/dynamical_singlet.dat
need_file libbse.d/static_excitation_singlet.dat
echo "Workflow completed: libbse.d"
