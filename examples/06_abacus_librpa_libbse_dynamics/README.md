# ABACUS -> LibRPA -> LibBSE: effective dynamical TDA, 2x2x2

`STRU`, the Si pseudopotential, orbital file and auxiliary basis come from `9_aims_abacus_template/abacus_template`. `INPUT` and `KPT` run SCF and export `OUT.librpa` on the same 2x2x2 mesh used for GW and BSE. No NSCF step is needed.

`fn_vxc_scf = ../OUT.scf/vxc_out.dat` is resolved relative to the selected `OUT.librpa_chi0` view. The coarse velocity matrix and uncompressed Coulomb export are required by LibBSE.

`librpa.in` runs standalone G0W0. Both the QP energies and screening used by LibBSE come from that calculation. QP energies are written to `energy_qp` in the example directory.

`libbse.in` selects `screened_format librpa_chi0`, singlet TDA, ELPA, velocity-gauge spectra and all excitation states. The window is four occupied and four empty bands (pseudopotential bands 1--8).

`chi0_coulomb_metric full` and `chi0_headwing true` reconstruct screening with full V inside the dielectric matrix, cut V outside, and LibRPA analytic head/wing correction. `output_chi0_static = t` exports the exactly-zero-frequency response in `librpa.d/chi0_static`. `use_shrink_abfs = f` disables any additional LibRPA compression in the selected producer basis. This extra static sample does not replace the GW quadrature.

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

ABACUS exports full Coulomb and `Cs_shrinked_data_*` in its small auxiliary basis, but `Cs_data_*` and `coulomb_unshrinked_cut_*` in its large basis. Mixing them gives invalid screening/QP data. `prepare_chi0_input.py` creates `OUT.librpa_chi0` with relative links to the consistent **small-basis** RI/full-V/cut-V files and unchanged KS/velocity data. The cut files are also aliased to the name required by LibBSE. Both LibRPA and LibBSE consume this view with no further compression; no matrix values are modified. `input_view.json` records the mapping and auxiliary dimension.
