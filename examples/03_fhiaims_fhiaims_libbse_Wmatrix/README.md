# FHI-aims -> FHI-aims -> LibBSE: static TDA, 2x2x2

`control.in` and `geometry.in` export KS/RI/Coulomb data and `mommat.h5`; the included Python converter writes `aims_export/velocity_matrix`. NumPy and h5py are required. The exporter runs with one MPI rank and four OpenMP threads on node-local scratch, then copies all outputs back, avoiding the shared-filesystem HDF5 stall recorded in the existing benchmark. `AIMS_EXPORT_THREADS` can override the export thread count within your allocated CPUs.

`control_gw.in` performs native FHI-aims GW in `aims_gw`. It uses the same structure, species/basis, coarse k grid and 16 frequency points as the export stage. Native W/chi0 output is incompatible with `output librpa bse`, so the two aims runs must remain separate. No standalone LibRPA job is used. `output k_eigenvalue 8` is required in addition to `read_write_qpe w` to write all eight regular-grid QP blocks.

`libbse.in` selects `screened_format fhi_aims_w`, singlet TDA, ELPA, velocity-gauge spectra and all excitation states. The window is four occupied and three empty bands (FHI-aims bands 11--17; GW also includes band 18).

LibBSE directly reads full W from `periodic_gw_w_q_*_rank_*.dat`, selecting the lowest absolute imaginary frequency as a static approximation. Do not add bare V again or convert these files to Wc.

Run from this directory after adjusting executable paths in `env.sh` and Slurm resources in `run.sh`:

```bash
./run.sh --check
sbatch run.sh
# Or run ./run.sh within an existing compute allocation.
```

The default paths target the local patched builds. Export environment variables to override executable paths, `NPROCS`, or `OMP_NUM_THREADS`; use `LOAD_MODULES=0` with a prepared environment. `--check` verifies files, executables and Python dependencies without starting a calculation. The workflow checks required stage outputs and refuses to overwrite existing generated data; use a fresh copy to repeat a full run.

These small Si inputs demonstrate the interface and are not a basis/k-grid/frequency convergence study. The eight end-to-end test copies and their validation report are kept in `../../3_LibBSE_templatetest`.
