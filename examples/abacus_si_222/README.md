# ABACUS 2x2x2 silicon example

This is the ABACUS counterpart of `fhi_aims_si_333`. It deliberately uses one
2x2x2 regular grid throughout so the end-to-end interface test is compact:

1. ABACUS SCF writes `OUT.librpa`, including the mandatory
   `velocity_matrix` enabled by `rpa_out_vel 1`.
2. Standalone LibRPA runs G0W0 and writes `energy_qp` plus static Wc blocks.
3. LibBSE solves the singlet TDA problem and always writes oscillator strengths
   and a Lorentz-broadened spectrum.

Obtain these standard ABACUS atomic-data files and place or symlink them beside
`STRU` (or set `ABACUS_ATOMIC_DATA_DIR` to a directory containing them):

- `Si_ONCV_PBE-1.0.upf`
- `Si_gga_8au_100Ry_3s3p2d.orb`
- `Si_3s3p2d1f1g_pca1e-6.abfs`

Then run:

```sh
export ABACUS_EXE=/path/to/abacus
export LIBRPA_EXE=/path/to/chi0_main.exe
export LIBBSE_EXE=/path/to/LibBSE
export ABACUS_ATOMIC_DATA_DIR=/path/to/abacus/atomic-data
export NPROCS=4 OMP_NUM_THREADS=1
./run.sh
```

`INPUT.nscf` and `KPT.nscf` document the optional 3x3x3 fine-grid follow-up;
they are not part of the coarse-grid `run.sh`. For production results, converge
the orbital/auxiliary bases, k grids, empty bands, minimax frequency grid, BSE
window, and spectral broadening.
