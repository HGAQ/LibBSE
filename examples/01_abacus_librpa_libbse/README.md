# ABACUS -> LibRPA -> LibBSE: static TDA, 2x2x2

`STRU`, the Si pseudopotential, orbital file and auxiliary basis come from `9_aims_abacus_template/abacus_template`. `INPUT` and `KPT` run SCF and export `OUT.librpa` on the same 2x2x2 mesh used for GW and BSE. No NSCF step is needed.

`fn_vxc_scf = ../OUT.scf/vxc_out.dat` is resolved relative to `OUT.librpa`. The coarse velocity matrix and uncompressed Coulomb export are required by LibBSE.

`librpa.in` runs standalone G0W0. Both the QP energies and screening used by LibBSE come from that calculation. QP energies are written to `energy_qp` in the example directory.

`libbse.in` selects `screened_format librpa_wc`, singlet TDA, ELPA, velocity-gauge spectra and all excitation states. The window is four occupied and four empty bands (pseudopotential bands 1--8).

LibBSE reads the lowest-frequency `librpa.d/Wc_Mu_*_ifreq_0.mtx` correlation blocks and adds the bare cut Coulomb interaction in its reader.

Run from this directory after adjusting executable paths in `env.sh` and Slurm resources in `run.sh`:

```bash
./run.sh --check
sbatch run.sh
# Or run ./run.sh within an existing compute allocation.
```

The default paths target the local patched builds. Export environment variables to override executable paths, `NPROCS`, or `OMP_NUM_THREADS`; use `LOAD_MODULES=0` with a prepared environment. `--check` verifies files, executables and Python dependencies without starting a calculation. The workflow checks required stage outputs and refuses to overwrite existing generated data; use a fresh copy to repeat a full run.

These small Si inputs demonstrate the interface and are not a basis/k-grid/frequency convergence study. The eight end-to-end test copies and their validation report are kept in `../../3_LibBSE_templatetest`.

The ABACUS producer exports small-basis full Coulomb together with large-basis RI data and a shrink transform, so `use_shrink_abfs = t` is required here. The tested compressed Wc export stalls with multiple LibRPA ranks; `run.sh` therefore uses one MPI rank with the allocated cores as OpenMP threads for LibRPA, while ABACUS and LibBSE use `NPROCS`.
