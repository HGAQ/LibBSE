# LibBSE

```text
                ██╗     ██╗██╗     ██████╗ ███████╗███████╗
                ██║     ╚═╝██║     ██╔══██╗██╔════╝██╔════╝
                ██║     ██║██████╗ ██████╔╝███████╗█████╗
                ██║     ██║██╔══██╗██╔══██╗╚════██║██╔══╝
                ███████╗██║██████╔╝██████╔╝███████║███████╗
                ╚══════╝╚═╝╚═════╝ ╚═════╝ ╚══════╝╚══════╝
```

[中文说明](README_zh.md)

LibBSE solves the periodic Bethe–Salpeter equation from FHI-aims/ABACUS/LibRPA output.
The migrated calculation path reads data through LibRPA, constructs the BSE
matrix elements with the external LibRI checkout, and diagonalizes both the
Tamm–Dancoff (TDA) and full BSE problems with ELPA. Optical analysis is
velocity-gauge only. The producer-specific derivation and data mapping are
documented in [`docs/fhi_aims_io.md`](docs/fhi_aims_io.md).

## How to build LibBSE

- CMake 3.16 or newer and a C++17 compiler
- MPI, OpenMP, BLAS, LAPACK, and ScaLAPACK
- Python 3 with NumPy and h5py for the FHI-aims momentum converter
- an ELPA installation with the skew-symmetric eigensolver enabled
- a sibling LibRPA source checkout; LibRI defaults to `thirdparty/LibRI`

The dependency header paths are configured consistently through
`CEREAL_INCLUDE_DIR`, `LIBRPA_INCLUDE_DIR`, `LIBRI_INCLUDE_DIR`, and
`LIBCOMM_INCLUDE_DIR`. `LIBRPA_INCLUDE_DIR` must be the `include` directory of
a LibRPA source checkout because LibBSE builds the LibRPA BSE reader API as a
subproject. The ELPA prefix is configured separately with `EXTERNAL_ELPA_DIR`.

Compile command example:

```sh
cd LibBSE
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DEXTERNAL_ELPA_DIR=/work/users/l/s/lsr/b_BSE/elpa-2024.05.001/build/install \
  -DCEREAL_INCLUDE_DIR=../LibRPA/thirdparty/cereal-1.3.0/include \
  -DLIBRPA_INCLUDE_DIR=../LibRPA/include \
  -DLIBRI_INCLUDE_DIR=thirdparty/LibRI/include \
  -DLIBCOMM_INCLUDE_DIR=../LibRPA/thirdparty/LibComm/include
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

## Required input

LibBSE reads `libbse.in` from the current working directory. The `input_dir`
parameter points to the `OUT.librpa` directory produced by the
LibRPA workflow. The working path consumes:

- LibRPA structure, basis, k-grid, eigenvalue, eigenvector, velocity, Cs, and truncated-Coulomb files under `OUT.librpa`;
- for `bse_use_fine_kgrid=0`, coarse-grid quasiparticle energies from
  `energy_qp`;
- for `bse_use_fine_kgrid=1`, `GW_band_spin_1.dat`, fine-grid
  `band_kpath_info`, and band eigenvalues/eigenvectors;
- lowest-imaginary-frequency `Wc_Mu_*_Nu_*_iR_*_ifreq_0.mtx` files in the
  sibling `librpa.d`, used as the static-limit approximation.

For direct FHI-aims exports, use `input_format=fhi_aims`. First run
`tools/aims_mommat_to_velocity.py AIMS_EXPORT_DIR`; it converts `mommat.h5`
to the canonical `velocity_matrix` while correcting the Fortran HDF5 k-grid
axis order. LibBSE aliases that file and the FHI-aims `coulomb_cut_*` files in
a non-destructive compatibility view. A velocity matrix is required for every
calculation. Quasiparticle energies must come from the standalone LibRPA step
(`energy_qp`); FHI-aims `GW_band*.out` files are not a LibBSE input.

The coarse `velocity_matrix` contains `velocity_mo` on the SCF k-grid.  If the BSE grid is identical, LibBSE uses those MO matrix elements directly.  For a double-grid calculation, such as the 5x5x5 to 6x6x6 Si example, it transforms the operator through a localized AO/real-space representation and then forms `velocity_mo` with the fine-grid KS wavefunctions.  The final spectrum contraction always uses `velocity_mo`.

Set `bse_use_fine_kgrid=0` to use the SCF k points and eigenvectors together
with quasiparticle energies from `energy_qp`. In this mode `band_kpath_info` and
`band_KS_{eigenvalue,eigenvector}_k_*.txt` are not required. Mode `1` retains
the separate fine-grid/double-grid path.

## Running

Create `libbse.in` in the calculation directory. A complete template as

```text
input_dir               OUT.librpa
output_dir              libbse.d
input_format            auto
qp_data                  OUT.librpa
qp_format                auto
screened_dir             librpa.d
bse_nstates             -1
nocc                    4
nvirt                   4
bse_solver              elpa
bse_spin_types          singlet triplet
bse_continue            0
bse_tda                 both
bse_ri_hartree          1
bse_use_fine_kgrid      1
bse_q_approx_mode       0
out_bse_ab              0
abs_gauge               velocity
wavefunction_gauge       auto
spectrum_broadening_ev   0.10
spectrum_energy_step_ev  0.01
spectrum_energy_min_ev   0.0
spectrum_energy_max_ev  -1
```
Then run
```sh
OMP_NUM_THREADS=8 mpirun -n 4 /PATH/build/LibBSE
```
Both whitespace-separated `key value` and `key = value` assignments are accepted.
Comments start with `#` or `!`, and the last occurrence of a key wins.

Complete runnable workflow templates for both producers are indexed in
[`examples/README.md`](examples/README.md).

Parameter names and enumerated values are case-insensitive. Paths retain their
case and are resolved relative to the directory containing `libbse.in`.
Boolean parameters accept `1`, `true`, `t`, `.true.`, or `yes` for true and
`0`, `false`, `f`, `.false.`, or `no` for false.

## Complete parameter reference

### `input_dir`

- **Required; no default.** This is the common mean-field/RI dataset directory.
- For a LibRPA dataset it normally points to `OUT.librpa` and is passed directly
  to the LibRPA reader.
- For an FHI-aims dataset it must contain an `aims` producer marker in
  `basis_out`, along with the structure, k-grid, band, wavefunction, RI, and
  Coulomb files required by the selected calculation.
- Relative paths are interpreted relative to `libbse.in`, not to the executable.

### `output_dir`

- **Default:** `libbse.d` next to `libbse.in`.
- Stores excitation energies, rank-local amplitudes, optical-analysis files,
  LibRI logs, and the FHI-aims compatibility view when applicable.
- `bse_solver spectrum` reads previously computed energies and amplitudes from
  this directory; normal calculations create the directory if necessary.

### `input_format`

- **Default:** `auto`.
- `auto`: inspect the producer field in `basis_out`; select `fhi_aims` when it
  is `aims`, otherwise select `librpa`.
- `librpa`: consume the directory with the standard LibRPA file conventions and
  do not build an FHI-aims compatibility view.
- `fhi_aims`: require the `aims` marker, alias `coulomb_cut_*` and the
  pre-converted canonical `velocity_matrix` in
  `output_dir/fhi_aims_reader_view`.

### `qp_data`

- **Default:** the resolved value of `input_dir`.
- Specifies the LibRPA quasiparticle-energy file or directory independently of
  the mean-field/RI dataset. This is useful because standalone LibRPA commonly
  writes `energy_qp` in the three-stage calculation directory.
- With `qp_format energy_qp`, a directory means `<qp_data>/energy_qp`; a file
  is used directly.
- On the fine-grid path, a directory means
  `<qp_data>/GW_band_spin_1.dat`; a file is used directly.

### `qp_format`

- **Default:** `auto`.
- `auto`: select `energy_qp` if that file exists at `qp_data`; otherwise use
  `GW_band_spin_1.dat` when `bse_use_fine_kgrid=1`. No FHI-aims QP fallback is
  performed.
- `energy_qp`: read LibRPA coarse-grid blocks containing occupations, KS
  energies, and QP energies in Hartree.
- `fine_band`: read the historical `GW_band_spin_1.dat` fine-grid table.

### `screened_dir`

- **Default:** `librpa.d` beside `input_dir`, i.e.
  `<parent-of-input_dir>/librpa.d`.
- Must contain the real-space correlation-interaction files
  `Wc_Mu_*_Nu_*_iR_*_ifreq_0.mtx` written by LibRPA.
- LibBSE constructs `W(R,iw0) = V(R) + Wc(R,iw0)`. The directory is required
  by `singlet` and `triplet` channels but not by pure `rpa` or `ipa` channels.

### `bse_nstates`

- **Default:** `-1`.
- `-1`: compute or read all states in the electron-hole pair space.
- Any positive integer: retain that many lowest states.
- The BSE dimension is `Nk * nocc * nvirt`; `0`, values below `-1`, and values
  larger than this dimension are rejected.

### `nocc`

- **Default:** `4`; any positive integer is accepted.
- Number of occupied bands included at every k point. LibBSE selects the last
  `nocc` occupied QP entries, so this is the active valence window rather than
  the total number of occupied all-electron states.
- Together with `nvirt` and the active k grid, it sets the BSE matrix dimension
  and the band rows used in RI and optical contractions.

### `nvirt`

- **Default:** `4`; any positive integer is accepted.
- Number of empty bands included at every k point. LibBSE selects the first
  `nvirt` entries following the occupied QP window.
- Every mapped QP record and the KS wavefunction dataset must cover the full
  requested occupied-plus-virtual window.

### `bse_solver`

- **Default:** `elpa`.
- `elpa`: construct the requested IPA/BSE matrices, solve for eigenstates, write
  them to `output_dir`, and perform optical analysis.
- `spectrum`: skip kernel construction and ELPA, reload the requested energy
  and rank-local amplitude files from `output_dir`, and run only the optical
  analysis. The restart should use the same MPI process count as the
  calculation that wrote the amplitude files.

### `bse_spin_types`

- **Default:** `singlet triplet`.
- Accepts a whitespace- or comma-separated list. Values must be unique, and at
  least one value is required.
- `singlet`: `A = gap + 2V - W`, `B = 2V - W`; requires both bare and screened
  interactions.
- `triplet`: `A = gap - W`, `B = -W`; omits the Hartree/exchange contribution
  and requires the screened interaction.
- `rpa`: `A = gap + 2V`, `B = 2V`; uses the bare interaction but not Wc.
- `ipa`: `A = gap`, `B = 0`. A pure IPA list uses a direct sorted-transition
  path without LibRI or ELPA and requires `bse_tda=tda`. IPA may also appear in
  a mixed list, in which case it follows the common solver path.

### `bse_continue`

- **Default and only supported value:** `0`.
- `0`: start the requested kernel/eigensolver workflow normally.
- Any nonzero integer is rejected. This legacy switch is not the spectrum
  restart mechanism; use `bse_solver=spectrum` to reuse eigenstates.

### `bse_tda`

- **Default:** `both`.
- `tda`: solve only the resonant Hermitian A matrix. Output names use
  `Excitation_Energy_<type>.dat` and `Excitation_Amplitude_<type>_<rank>.dat`.
- `full`: solve the coupled A/B problem only and write `full` energy plus X/Y
  amplitude files.
- `both`: run both calculations for every requested spin type.
- A pure IPA calculation supports only `tda`.

### `bse_ri_hartree`

- **Default:** true.
- True: allow construction of the bare-interaction Hartree/exchange term. This
  is mandatory when `bse_spin_types` contains `singlet` or `rpa`.
- False: accepted only when no requested channel needs the Hartree term, such
  as triplet-only or IPA-only calculations. It does not disable screened W.

### `bse_use_fine_kgrid`

- **Default:** `1`.
- `0`: use the SCF k points and regular-grid KS eigenvectors. Separate
  `band_kpath_info` and band-path eigenvector files are not read. QP rows must
  map to the SCF grid; this is the mode used by the direct FHI-aims 3x3x3 path.
- `1`: use the separate fine/band k grid and its eigenvalues/eigenvectors. For
  optical analysis, velocity is transformed through the localized AO/real-space
  representation when the coarse and fine grids differ.
- Other integers are rejected.

### `bse_q_approx_mode`

- **Default and only supported value:** `0`.
- `0`: use the implemented explicit LibRI momentum-transfer/real-space phase
  path without selecting another q-approximation branch.
- Nonzero legacy modes are not implemented and are rejected.

### `out_bse_ab`

- **Default and only supported value:** false.
- False: write eigenvalues, distributed amplitudes, and requested spectrum
  analysis, but do not dump raw distributed A/B matrices.
- True is currently unimplemented and is rejected instead of being ignored.

### `abs_gauge`

- **Default and only supported value:** `velocity`.
- `velocity`: calculate transition dipoles and oscillator strengths from the
  momentum/velocity matrix and KS transition gaps.
- Length gauge and all other values are currently unsupported and rejected.

### `spectrum_broadening_ev`, `spectrum_energy_step_ev`,
`spectrum_energy_min_ev`, `spectrum_energy_max_ev`

- Control the Lorentzian half width and energy grid of every generated
  `spectrum_*.dat` file. Defaults are 0.10 eV, 0.01 eV, and 0 eV.
- A negative maximum (default `-1`) selects the largest excitation energy plus
  five half widths. Width and step must be positive.
- Optical analysis is unconditional. The removed `bse_compute_spectrum`
  switch is rejected, and an input without velocity/momentum data is an error.

### `wavefunction_gauge`

- **Default:** `auto`.
- `auto`: resolve to `native` for FHI-aims and `first_k` for other LibRPA
  datasets.
- `native`: use KS and RI coefficients in the producer's original mutually
  consistent band gauge. This is the physically appropriate FHI-aims choice,
  including at degeneracies.
- `first_k`: apply the historical scalar-phase alignment of each band to its
  overlap with the same band at the first k point. This remains available for
  datasets generated with that convention, but is unsafe as a general gauge
  fixing inside degenerate subspaces.

After initialization, rank 0 prints the MPI process count, requested/provided
MPI thread level, and OpenMP thread configuration. At shutdown it also prints
a hierarchical LibBSE timing profile with call counts, CPU time, and wall time
for file reading, interaction preparation, LibRI matrix construction, ELPA
solves, velocity preparation, and spectrum analysis. Major completed stages
also emit `DONE(elapsed SEC) : description` markers. After each
A matrix is assembled, LibBSE checks `||A-A^H||_F`; after each B matrix is
assembled, it checks `||B-B^T||_F`. These checks operate on the existing 2D
block-cyclic distribution with ScaLAPACK and MPI reductions and report a
warning when the `1e-6` threshold is exceeded. LibRI k-space
blocks are sent directly to their owning two-dimensional matrix blocks using the
`transform_k_2dlocal` algorithm; a complete BSE matrix is never assembled on
each rank. After ELPA, eigenvectors are redistributed without a root gather so
that every rank stores all requested states for only its contiguous
electron-hole-pair interval. `FineVelocityMo` stores only the three velocity
components and KS gap for the same local pair interval; double-grid values are
sent directly from the fine-wavefunction owner to the amplitude owner.
Velocity-gauge contractions and k-point weights consume these local data
directly and are reduced to rank 0. Only the small final transition tables are
written serially. Wc input is likewise limited to the local LibRI
`list_I x list_J` atom pairs.

The channel matrices use the coefficients as:

- singlet: `A = gap + 2V - W`, `B = 2V - W`;
- triplet: `A = gap - W`, `B = -W`;
- RPA: `A = gap + 2V`, `B = 2V`;
- IPA: `A = gap`, `B = 0`.

A pure IPA calculation requires `bse_tda=tda` and directly constructs the
sorted independent-particle states without LibRI or ELPA.
Other channels use LibRI and ELPA. Each requested `<type>` writes:

- `Excitation_Energy_<type>.dat`
- `Excitation_Amplitude_<type>_<rank>.dat`
- `Excitation_Energy_full_<type>.dat`
- `Excitation_Amplitude_full_{X,Y}_<type>_<rank>.dat`
- `trans_dipole_<type>_{tda,full}.dat`
- `oscillator_strength_<type>_{tda,full}.dat`
- `spectrum_<type>_{tda,full}.dat`
- `trans_analysis_<type>_{tda,full}.dat`
- `trans_kweight_<type>_{tda,full}.dat`

The spectrum file contains the Lorentz-broadened directional and isotropic
oscillator-strength density. Oscillator strengths are dimensionless and the
spectrum density has units of eV^-1. The reported strength is intensive per
primitive cell: the coherent transition dipole is divided by the k-grid
normalization through a `1/Nk` factor in the strength.
Triplet files are written consistently with the other channels but contain
zero dipoles and strengths, as required by the spin selection rule for an
electric-dipole transition from a singlet ground state.
Amplitude files are written and read independently by each MPI rank. Spectrum
restart calculations are expected to use the same number of MPI ranks as the
ELPA calculation that produced these files.

The unit-test inventory is documented in [`tests/README.md`](tests/README.md).
