# FHI-aims 3x3x3 silicon example

This directory contains the complete three-stage input flow:

1. FHI-aims writes DFT/RI/Coulomb/wavefunction data plus `mommat.h5` into
   `aims_export`; patched builds may additionally write packed momentum files.
2. `tools/aims_mommat_to_velocity.py` creates LibBSE's canonical optical
   matrix and corrects the Fortran HDF5 k-grid axis order.
   Standalone LibRPA uses the other FHI-aims exports, runs G0W0, and writes
   `energy_qp` plus the lowest-frequency
   `librpa.d/Wc_Mu_*_ifreq_0.mtx` blocks. The velocity matrix is not a GW
   input.
3. LibBSE reads the original FHI-aims data through its compatibility view, but
   reads QP energies and screening only from the standalone LibRPA products.
   It writes excitation energies, oscillator strengths, and broadened spectra.

The `qpe_calc gw_expt` line remains because the currently patched FHI-aims
`output librpa bse` parser requires it. FHI-aims QP output is not consumed by
LibBSE. `compute_momentummatrix 0 100 0` is the periodic all-k extension used
by the validated FHI-aims build. Adjust the energy window if the selected BSE
bands lie outside 0--100 eV relative to the FHI-aims internal zero.

The LibRPA input uses the FHI-aims macroscopic dielectric function
(`option_dielect_func=0`, `replace_w_head=t`) for the q=0 head correction.
`control.in` uses `periodic_gw_optimize_kgrid_symmetry inverse`; LibRPA
recovers q/-q partners by conjugation from `bz_sampling_out`, so point-group
symmetry restoration remains off. If FHI-aims uses `all` instead, enable both
`use_symmetry_gw=t` and `use_symmetry_exx=t`. A `none` export also leaves both
options off.
`i_state_low=10`, `i_state_high=18` is LibRPA's zero-based, half-open interval
covering the BSE bands plus one guard band. Keep LibRPA's `nfreq` equal to
FHI-aims' `frequency_points` when using `option_dielect_func=0` (both are 16
here). This validation input uses the perturbative/linearized QP solver
(`option_qpe_solver=2`) to select the root connected to the KS state and a
`1e-8` Coulomb square-root cutoff for the large auxiliary basis.

Run with, for example:

```sh
export AIMS_EXE=/path/to/aims.x
export LIBRPA_EXE=/path/to/chi0_main.exe
export LIBBSE_EXE=/path/to/LibBSE
export MOMMAT_CONVERTER=/path/to/LibBSE/tools/aims_mommat_to_velocity.py
export NPROCS=4 OMP_NUM_THREADS=1
./run.sh
```

The script stops unless `energy_qp`, a static Wc block,
`oscillator_strength_singlet_tda.dat`, and `spectrum_singlet_tda.dat` are all
nonempty. The validation BSE window intentionally keeps only the lowest three
empty QP bands; higher roots from this compact GW setup are not a converged
production data set. Converge the k grid, basis, empty
bands, frequency grid, BSE window, and spectral broadening for production work.
