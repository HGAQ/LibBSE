# FHI-aims to LibRPA to LibBSE data path

[中文说明](fhi_aims_io_zh.md)

This interface keeps the many-body algebra in LibRPA/LibRI and adds only the
producer-specific translation needed for FHI-aims files. It is not an
independent BSE implementation.

## RI convention and the screened interaction

FHI-aims exports the localized auxiliary-basis expansion

```text
phi_i*(k+q,r) phi_j(k,r)
  = sum_mu Ctilde_ij^mu(k+q,k) P_mu^{q*}(r),

C_mn^mu(k+q,k)
  = sum_ij c_i,m*(k+q) c_j,n(k) Ctilde_ij^mu(k+q,k).
```

LibRPA constructs the independent-particle response in that basis,

```text
chi0_mu,nu(q,iw) = sum_(s,m,n,k) w_k
  C_mn,s^mu(k+q,k) C_nm,s^nu(k,k+q)
  / (epsilon_m,s(k+q) - epsilon_n,s(k) - iw),
```

and uses the symmetrized dielectric matrix

```text
epsilon(q,0) = I - V^(1/2)(q) chi0(q,0) V^(1/2)(q),
W(q,0)       = V^(1/2)(q) epsilon^(-1)(q,0) V^(1/2)(q).
```

The standalone LibRPA calculation writes the correlation part
`Wc(q,iw) = W(q,iw) - V(q)`. In the exact static limit, Fourier linearity
gives

```text
W_mu,nu(R,0) = V_mu,nu(R) + Wc_mu,nu(R,0),
W_mu,nu(R,0) = (1/Nq) sum_q exp(-i q.R) W_mu,nu(q,0).
```

LibBSE therefore reads the real-space `Wc_Mu_*` blocks, adds the real-space
bare interaction produced by `FT_Vq`, remaps both to the same nearest
Born--von Karman cells, and passes the resulting static-limit `W(R,iw_0)` to
LibRI. The direct BSE kernel is
then a contraction of two transition densities with this static `W`; the
exchange/Hartree kernel uses the corresponding bare `V`. The common `1/Nk`
Fourier normalization and Hartree-to-Rydberg conversion are applied once when
the LibRI k blocks enter the distributed BSE matrix.

This route avoids exporting a dense four-index screened electron-hole tensor.
Local auxiliary blocks of `W(R,iw_0)` are retained instead.

In the transition basis `(i k -> a k)`, the two auxiliary-basis contractions
implemented by the LibRI kernel correspond to

```text
<i k, a k | V | j k', b k'>
  = sum_(mu,nu) C_ia^mu(k,k)* V_mu,nu(q=0) C_jb^nu(k',k'),

<i k, j k' | W | a k, b k'>
  = sum_(mu,nu) C_ij^mu(k,k')* W_mu,nu(k'-k) C_ab^nu(k,k').
```

The first contraction is the repulsive exchange/Hartree part (with the
channel-dependent spin prefactor); the second enters the resonant BSE block
with the attractive minus sign. In real space the momentum transfer is
recovered by LibRI's phase sum over `R`, so FHI-aims only needs to export the
localized vertices and LibRPA only needs to export auxiliary blocks of `W`.
No dense `(i,a,j,b,k,k')` object is written or reconstructed by the IO layer.

The symmetry setting must match the FHI-aims export. The supplied Si input
uses `periodic_gw_optimize_kgrid_symmetry inverse`; q/-q partners are then
recovered by conjugation through `bz_sampling_out`, and the LibRPA point-group
symmetry switches stay off. With the FHI-aims `all` setting, LibRPA must use
both `use_symmetry_gw = t` and `use_symmetry_exx = t` to rotate the
correlation and exchange data back onto the full Born--von Karman grid. A
`none` export also leaves both switches off. Do not infer this choice only
from the number of SCF k points. Restrict `i_state_low` and
`i_state_high` to the BSE band window
(the bounds are zero-based and half-open), because remote QP roots are both
unnecessary and less stable. LibBSE reports A-Hermiticity and B-symmetry
diagnostics before diagonalization; an even k grid can legitimately leave B
non-Hermitian/non-symmetric under the stored finite-grid convention.

The current LibRPA MatrixMarket writer samples the minimax imaginary-frequency
grid, which does not contain exactly zero. Consequently `ifreq_0` is the
lowest positive node and LibBSE uses `W(R,iw_0)` as the static-limit
approximation. The actual node is recorded in every MatrixMarket header.
Keep LibRPA's `nfreq` equal to the FHI-aims `frequency_points` value when
`option_dielect_func = 0`; increase both together and check convergence when
this approximation matters. The interface does not silently label that node
as an exact zero-frequency value.

## Band gauge

For a band phase change
`c_i,n(k) -> exp(i theta_n(k)) c_i,n(k)`, the MO RI vertex acquires the
opposite bra/ket phases. Those phases cancel against the electron-hole basis
transformation in the BSE kernel. FHI-aims data are consequently consumed in
their native gauge. Aligning each band independently to an overlap with the
same band at k=0 is especially unsafe at degeneracies, where an allowed gauge
change is a unitary rotation rather than a scalar phase.

`wavefunction_gauge auto` selects `native` for FHI-aims and preserves the
historical `first_k` convention for other LibRPA datasets. The choice can be
made explicit for diagnostics.

## FHI-aims files and compatibility view

Use `output librpa bse` in the patched FHI-aims interface. The preset produces
regular-grid KS eigenvectors, folded RI coefficients, Coulomb matrices, and
packed `mommat_ks_kpt_*.dat`; the LibRPA reader auto-detects the supported
legacy-binary or v1 containers. The standard FHI-aims command
`compute_momentummatrix Emin Emax 0` can additionally produce `mommat.h5` for
all regular-grid k points. The value `0` in the example is the all-k extension
used by the supplied interface; for an unmodified cluster calculation use k
point 1 as documented by FHI-aims.

`tools/aims_mommat_to_velocity.py` converts the Cartesian gradient matrix
elements in `mommat.h5` from bohr^-1 using `v = p = -i nabla`. It reverses the
three h5py grid axes from the Fortran HDF5 view `(kz,ky,kx)` to
`(kx,ky,kz)`, reconstructs the Hermitian matrix from the packed band upper
triangle, and writes LibRPA's binary-v1 `velocity_matrix` with the required
eV*angstrom scaling. Run the converter before LibBSE. The compatibility view
only aliases an existing canonical `velocity_matrix`; it does not implement
another conversion path. Missing velocity data are fatal. The velocity matrix
is an optical/BSE input and is not an input to the GW calculation itself.

FHI-aims names truncated matrices `coulomb_cut_*`; the published LibRPA reader
requests the older `coulomb_unshrinked_cut_*` prefix. LibBSE creates symlinks
with the older names in `output_dir/fhi_aims_reader_view`. It never renames or
modifies the source export.

## Quasiparticle energies

FHI-aims is the DFT/RI/momentum producer in this workflow; it is not the QP
source consumed by LibBSE. Standalone LibRPA reads the FHI-aims export, runs
G0W0, and writes both `energy_qp` and the static screened-interaction blocks.
LibBSE reads that `energy_qp` with `qp_format energy_qp`. Direct parsing of
FHI-aims `GW_band*.out` has been removed, which keeps one band ordering, one
unit convention, and one GW result shared by the LibRPA and LibBSE stages.

The all-electron KS file can contain core states. The occupations in
`energy_qp` determine the occupied window; LibBSE selects its last `nocc`
occupied and first `nvirt` empty entries and applies the corresponding offset
to the KS eigenvectors and velocity matrix.

## Three-stage run

1. Run FHI-aims with a 3x3x3 `k_grid`, `output librpa bse`, and a momentum
   export. This stage supplies DFT, RI, Coulomb, wavefunction, and velocity
   data only.
2. Convert `mommat.h5` to LibBSE's canonical `velocity_matrix`, then
   run standalone LibRPA with `output_energy_qp = t`, `output_wc_rf = t`, and
   `ifreq_output_wc_end = 1`. Use `option_dielect_func = 0` and
   `replace_w_head = t` for the FHI-aims dielectric-head correction; match
   `nfreq` to the FHI-aims frequency grid, and restrict the QP state interval
   to the BSE window. Leave LibRPA point-group symmetry off for the example's
   `inverse` export; if FHI-aims uses `all`, enable both `use_symmetry_gw` and
   `use_symmetry_exx`. For the large auxiliary basis in this example,
   `sqrt_coulomb_threshold = 1e-8` removes numerically null Coulomb channels
   and `option_qpe_solver = 2` uses the perturbative root connected to the KS
   state instead of a remote satellite root.
3. Run LibBSE with `input_format fhi_aims`, `qp_format energy_qp`, `qp_data`
   pointing to LibRPA's `energy_qp`, and `screened_dir` pointing to LibRPA's
   `Wc_Mu_*` output.

Portable templates are under `examples/fhi_aims_si_333`.
