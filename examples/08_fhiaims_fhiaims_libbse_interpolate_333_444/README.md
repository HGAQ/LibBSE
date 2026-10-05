# FHI-aims -> FHI-aims -> LibBSE: static TDA, 3x3x3 -> 4x4x4

`control.in` and `geometry.in` export KS/RI/Coulomb data and `mommat.h5`; the included Python converter writes `aims_export/velocity_matrix`. NumPy and h5py are required. The exporter runs with one MPI rank and four OpenMP threads on node-local scratch, then copies all outputs back, avoiding the shared-filesystem HDF5 stall recorded in the existing benchmark. `AIMS_EXPORT_THREADS` can override the export thread count within your allocated CPUs.

`control_gw.in` performs native FHI-aims GW in `aims_gw`. It uses the same structure, species/basis, coarse k grid and 16 frequency points as the export stage. Native W/chi0 output is incompatible with `output librpa bse`, so the two aims runs must remain separate. No standalone LibRPA job is used.

The 16 `output band` segments in both control files sample all 64 points of the unshifted 4x4x4 grid. `bands_444.in` records those lines. `band_to_energy_qp.py` validates and converts `GW_band1001.out` through `GW_band1016.out` to `aims_gw/energy_qp`, following the existing `11_LibBSE_interpolate` benchmark. The common chemical-potential energy shift cancels from transitions; its e_gs column copies QP energies as a placeholder, while LibBSE obtains KS data from the exporter.

`libbse.in` selects `screened_format fhi_aims_w`, singlet TDA, ELPA, velocity-gauge spectra and all excitation states. The window is four occupied and three empty bands (FHI-aims bands 11--17; GW also includes band 18).

`bse_use_fine_kgrid 1` keeps the RI/screening period at 3x3x3 and uses 4x4x4 KS/QP points for BSE. The expected matrix dimension is 768. This interpolates the coarse real-space interaction; it does not run screening on a 4x4x4 mesh.

LibBSE directly reads full W from `periodic_gw_w_q_*_rank_*.dat`, selecting the lowest absolute imaginary frequency as a static approximation. Do not add bare V again or convert these files to Wc.

Run from this directory after adjusting executable paths in `env.sh` and Slurm resources in `run.sh`:

```bash
./run.sh --check
sbatch run.sh
# Or run ./run.sh within an existing compute allocation.
```

The default paths target the local patched builds. Export environment variables to override executable paths, `NPROCS`, or `OMP_NUM_THREADS`; use `LOAD_MODULES=0` with a prepared environment. `--check` verifies files, executables and Python dependencies without starting a calculation. The workflow checks required stage outputs and refuses to overwrite existing generated data; use a fresh copy to repeat a full run.

These small Si inputs demonstrate the interface and are not a basis/k-grid/frequency convergence study. The eight end-to-end test copies and their validation report are kept in `../../3_LibBSE_templatetest`.

The inherited three-empty-band window cuts a degenerate conduction subspace at some fine k points (bands 17/18). Independently regenerated KS eigenvectors can therefore change the truncated BSE subspace even when KS/QP energies agree. This run differs from the old benchmark by 0.277189 meV for the first state and at most 7.791300 meV across 768 states. A current-binary rerun with the old electronic data reproduces the saved spectrum at output precision; using new KS data with old QP/W reproduces the new spectrum within 0.000014 meV. This is a band-window limitation, not an exact independent-run regression target. Include complete degenerate groups and converge the band window for production work.
