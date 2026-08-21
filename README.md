# LibBSE

LibBSE solves the periodic Bethe–Salpeter equation from FHI-aims/ABACUS/LibRPA output.
The migrated calculation path reads data through LibRPA, constructs the BSE
matrix elements with the external LibRI checkout, and diagonalizes both the
Tamm–Dancoff (TDA) and full BSE problems with ELPA.  Optical analysis is
velocity-gauge only.

## How to build LibBSE

- CMake 3.16 or newer and a C++17 compiler
- MPI, OpenMP, BLAS, LAPACK, and ScaLAPACK
- an ELPA installation with the skew-symmetric eigensolver enabled
- sibling source checkouts of LibRPA and external LibRI by default

The dependency header paths are configured consistently through
`CEREAL_INCLUDE_DIR`, `LIBRPA_INCLUDE_DIR`, `LIBRI_INCLUDE_DIR`, and
`LIBCOMM_INCLUDE_DIR`. `LIBRPA_INCLUDE_DIR` must be the `include` directory of
a LibRPA source checkout because LibBSE builds the LibRPA BSE reader API as a
subproject. The ELPA prefix is configured separately with `EXTERNAL_ELPA_DIR`.

Compile command example:

```sh
cd LibBSE
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=mpiicpx \
  -DEXTERNAL_ELPA_DIR=/opt/elpa_2025.06.001-install \
  -DCEREAL_INCLUDE_DIR=../LibRPA/thirdparty/cereal-1.3.0/include \
  -DLIBRPA_INCLUDE_DIR=../LibRPA/include \
  -DLIBRI_INCLUDE_DIR=../LibRI/include \
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
- static `Wc_Mu_*_Nu_*_iR_*_ifreq_0.mtx` files in the sibling `librpa.d`.

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
```
Then run
```sh
OMP_NUM_THREADS=8 mpirun -n 4 /PATH/build/LibBSE
```
Both whitespace-separated `key value` and `key = value` assignments are accepted.
Comments start with `#` or `!`, and the last occurrence of a key wins.
`output_dir` defaults to `libbse.d` when omitted.
`bse_spin_types` accepts any non-repeated list drawn from `singlet`, `triplet`,
`rpa`, and `ipa`; its default is `singlet triplet`.
`bse_solver=spectrum` reuses eigenstates from `output_dir` and runs only the
velocity-gauge optical analysis. Unsupported modes are rejected explicitly
instead of being silently ignored.

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
- `trans_dipole_<type>_{tda,full}.dat`, except for triplet
- `trans_analysis_<type>_{tda,full}.dat`
- `trans_kweight_<type>_{tda,full}.dat`

The `trans_dipole` data can be Lorentz broadened to obtain the plotted imaginary dielectric function.
Amplitude files are written and read independently by each MPI rank. Spectrum
restart calculations are expected to use the same number of MPI ranks as the
ELPA calculation that produced these files.

The unit-test inventory is documented in [`tests/README.md`](tests/README.md).
