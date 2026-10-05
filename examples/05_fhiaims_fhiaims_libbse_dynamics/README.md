# FHI-aims -> FHI-aims -> LibBSE: effective dynamical TDA, 2x2x2

`control.in` and `geometry.in` export KS/RI/Coulomb data and `mommat.h5`; the included Python converter writes `aims_export/velocity_matrix`. NumPy and h5py are required. The exporter runs with one MPI rank and four OpenMP threads on node-local scratch, then copies all outputs back, avoiding the shared-filesystem HDF5 stall recorded in the existing benchmark. `AIMS_EXPORT_THREADS` can override the export thread count within your allocated CPUs.

`control_gw.in` performs native FHI-aims GW in `aims_gw`. It uses the same structure, species/basis, coarse k grid and 16 frequency points as the export stage. Native W/chi0 output is incompatible with `output librpa bse`, so the two aims runs must remain separate. No standalone LibRPA job is used. `output k_eigenvalue 8` is required in addition to `read_write_qpe w` to write all eight regular-grid QP blocks.

`libbse.in` selects `screened_format fhi_aims_chi0`, singlet TDA, ELPA, velocity-gauge spectra and all excitation states. The window is four occupied and three empty bands (FHI-aims bands 11--17; GW also includes band 18).

`chi0_coulomb_metric full` and `chi0_headwing true` reconstruct screening with full V inside the dielectric matrix, cut V outside, and LibRPA analytic head/wing correction. FHI-aims chi0 is selected at the lowest finite imaginary-frequency node, not exactly zero.

`bse_plasma_energy_ev 15` enables the one-shot effective dynamical kernel. LibBSE first solves static TDA, obtains Eb from the minimum direct QP gap minus the lowest static exciton, and solves once more with the effective kernel. This is not real-time propagation. The model requires chi0, TDA, ELPA, a bound static lowest state and a physically admissible dielectric spectrum.

`libbse.d/dynamical_singlet.dat` reports static/effective energies and shifts in eV. `static_excitation_singlet.dat` preserves static energies in Ry. The standard excitation and optical files contain effective dynamical results.

Run from this directory after adjusting executable paths in `env.sh` and Slurm resources in `run.sh`:

```bash
./run.sh --check
sbatch run.sh
# Or run ./run.sh within an existing compute allocation.
```

The default paths target the local patched builds. Export environment variables to override executable paths, `NPROCS`, or `OMP_NUM_THREADS`; use `LOAD_MODULES=0` with a prepared environment. `--check` verifies files, executables and Python dependencies without starting a calculation. The workflow checks required stage outputs and refuses to overwrite existing generated data; use a fresh copy to repeat a full run.

These small Si inputs demonstrate the interface and are not a basis/k-grid/frequency convergence study. The eight end-to-end test copies and their validation report are kept in `../../3_LibBSE_templatetest`.
