# Eight Si workflows for LibBSE

These standalone templates extend `../1_LibBSE_test/9_aims_abacus_template`. Native W, chi0 and dynamical settings follow the local `11_LibBSE_interpolate`, `12_LibBSE_chi0`, `13_LibBSE_dyn` and `16_non_interpolate` benchmarks and current readers. Each directory includes inputs, `env.sh`, `run.sh` and its own instructions; ABACUS examples include the original Si pseudopotential and basis files.

| Example | DFT | GW/QP | Screening | Coarse -> BSE grid | BSE |
| --- | --- | --- | --- | --- | --- |
| [01_abacus_librpa_libbse](01_abacus_librpa_libbse/README.md) | ABACUS | LibRPA | `librpa_wc` | 2x2x2 | static |
| [02_fhiaims_librpa_libbse](02_fhiaims_librpa_libbse/README.md) | FHI-aims | LibRPA | `librpa_wc` | 2x2x2 | static |
| [03_fhiaims_fhiaims_libbse_Wmatrix](03_fhiaims_fhiaims_libbse_Wmatrix/README.md) | FHI-aims | FHI-aims | `fhi_aims_w` | 2x2x2 | static |
| [04_fhiaims_fhiaims_libbse_chi0](04_fhiaims_fhiaims_libbse_chi0/README.md) | FHI-aims | FHI-aims | `fhi_aims_chi0` | 2x2x2 | static |
| [05_fhiaims_fhiaims_libbse_dynamics](05_fhiaims_fhiaims_libbse_dynamics/README.md) | FHI-aims | FHI-aims | `fhi_aims_chi0` | 2x2x2 | dynamical |
| [06_abacus_librpa_libbse_dynamics](06_abacus_librpa_libbse_dynamics/README.md) | ABACUS | LibRPA | `librpa_chi0` | 2x2x2 | dynamical |
| [07_abacus_librpa_libbse_interpolate_333_444](07_abacus_librpa_libbse_interpolate_333_444/README.md) | ABACUS | LibRPA | `librpa_wc` | 3x3x3 -> 4x4x4 | static |
| [08_fhiaims_fhiaims_libbse_interpolate_333_444](08_fhiaims_fhiaims_libbse_interpolate_333_444/README.md) | FHI-aims | FHI-aims | `fhi_aims_w` | 3x3x3 -> 4x4x4 | static |

Enter an example directory, adjust `env.sh` and Slurm resources if needed, run `./run.sh --check`, then `sbatch run.sh`. The default local executables contain interface extensions that may not be available in an unmodified upstream build. The workflow creates its own intermediate directories and does not reuse old benchmark outputs.

Examples 01--06 use a 2x2x2 single grid. Examples 07--08 use 3x3x3 SCF/RI/screening and 4x4x4 fine KS/QP/BSE data with `bse_use_fine_kgrid 1`. Their expected singlet TDA dimensions are 1024 (ABACUS, 4 occupied x 4 empty x 64 k points) and 768 (FHI-aims, 4 occupied x 3 empty x 64). The fine grid evaluates the existing coarse real-space representation; it is not a new fine-grid screening calculation.

FHI-aims native GW workflows separate `control.in` data export from `control_gw.in` GW because native W/chi0 export cannot be combined with `output librpa bse`. The HDF5 export uses one MPI rank on node-local scratch and copies its output back. `AIMS_EXPORT_EXE` and `AIMS_GW_EXE` can select separate compatible builds. Keep export/GW geometry, basis and coarse grids identical. The native single-grid examples also set `output k_eigenvalue 8`; without it this build writes only the Gamma QP block.

The native W reader consumes full W. The LibRPA Wc reader adds bare V itself. Chi0 workflows reconstruct screening using full Coulomb and LibRPA analytic head/wing correction; their Gamma treatment differs from native FHI-aims W and exact equality is not expected. FHI-aims chi0 uses the lowest finite imaginary frequency; LibRPA dynamics exports an additional exactly-zero-frequency chi0 with further LibRPA compression disabled. For ABACUS this uses a consistent small-basis view of the producer RI/full-V/cut-V exports; the static Wc workflows retain the producer shrink transform and use a single LibRPA rank to avoid this build's compressed-Wc MPI stall.

Dynamical examples enable `bse_plasma_energy_ev 15` with chi0, TDA and ELPA. They first determine the static binding energy, then solve a one-shot effective kernel. Standard spectra contain the effective result; separate files preserve the static/effective comparison. These are not real-time dynamics inputs.

The small meshes and basis sets demonstrate workflows, not production convergence. Independent end-to-end runs are stored in [../3_LibBSE_templatetest](../3_LibBSE_templatetest/README.md); validation checks job completion, finite outputs, matrix dimensions, fine-grid coverage and dynamical consistency. See its validation report for measured results and any remaining limitations.

The FHI-aims interpolation example inherits a three-empty-band window that cuts some degenerate conduction groups. Its independently regenerated spectrum can differ slightly from the old benchmark; the test report includes controlled old/new-data reruns that isolate this window effect. Use complete degenerate groups when converging a production band window.

Validation completed on 2026-09-23 (UTC): all eight workflows passed the numerical/file checks and their final Slurm jobs completed with exit code 0:0. The ABACUS tests reused their own completed DFT exports when retrying corrected post-processing. See the [validation report](../3_LibBSE_templatetest/validation_report.md) for job IDs, excitation energies and the FHI-aims interpolation comparison limitation.
