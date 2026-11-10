# Full spatial symmetry for imported χ₀

The reader can reconstruct bare χ₀ on the original coarse q mesh from representative matrices and explicitly exported auxiliary-basis transformations. The same reconstruction kernel is shared by aims and LibRPA; their file adapters remain separate.

For each target q,

\[
\chi_0(q)=D(q)\,\mathcal C^t[\chi_0(q_\mathrm{rep})]D(q)^\dagger,
\]

where `t=1` requests complex conjugation. `D` includes atom permutation, angular rotations and Bloch/folding phases in the producer's auxiliary-basis convention. A spatial rotation generally cannot be replaced with conjugation alone.

## Screening order

Restore the entire auxiliary χ₀ matrix first, then construct

\[
\varepsilon(q)=I-\sqrt{V_\mathrm{full}(q)}\chi_0(q)\sqrt{V_\mathrm{full}(q)},\qquad
W(q)=\sqrt{V_\mathrm{cut}(q)}\varepsilon(q)^{-1}\sqrt{V_\mathrm{cut}(q)}.
\]

With `chi0_headwing true`, the existing analytic Γ correction is evaluated at the imported physical frequency. After screening, local MPI atom blocks are extracted, and the complete q mesh is Fourier transformed once. Dynamic-screening caches also contain full target-q channels.

The RI/KS dataset must match the response's original auxiliary basis and coarse grid. Coulomb input currently needs complete or q/−q coverage; this feature does not add general-star Coulomb reconstruction. Spinor/magnetic symmetry, compressed auxiliary bases and arbitrary basis conversion are not covered.

## aims

Use an aims executable containing the Δ exporter with:

```text
periodic_gw_optimize_kgrid_symmetry all
periodic_gw_output_chi0 .true.
```

The native producer writes its usual `periodic_gw_chi0_q_*_rank_*.dat` representative matrices plus `aims_symmetry_chi0.dat` and `aims_delta_q_*.dat`. The physical frequency is read from the response file; lowest positive imaginary frequency is not relabelled as zero.

```text
input_format fhi_aims
screened_format fhi_aims_chi0
screened_dir path/to/aims_chi0_export
chi0_coulomb_metric full
chi0_headwing true
```

Without a manifest the previous full/q−q reader remains available. An unrelated `aims_symmetry_all.dat` full-W manifest is not reused as a χ₀ manifest.

## LibRPA

Enable full crystal symmetry in LibRPA and select one or both response exports:

```text
use_symmetry_rpa = t
use_shrink_abfs = f
output_chi0_rf = t
output_chi0_static = t
```

The first output uses the lowest available imaginary frequency; the second builds a dedicated strictly zero-frequency response. For a full-crystal q view, each output directory contains:

- `chi0_q.info`: physical frequency, auxiliary dimension and coarse grid.
- `chi0_q_<representative_id>.dat`: bare representative χ₀ matrices.
- `rpa_symmetry_chi0.dat`: all target-q mappings.
- `rpa_delta_q_<target_id>.dat`: effective transformations from the actual stored representative, including time-reversal parity.

The producer composes target and inverse-source transformations because the stored representative need not equal LibRPA's canonical star source. It uses LibRPA's own operator-transform convention, not aims rotation matrices.

With symmetry disabled or only time reversal used, LibRPA retains its existing `chi0_rf.info` and atom-block χ₀(R) output. Use fresh output directories when switching formats.

```text
screened_format librpa_chi0
screened_dir path/to/librpa.d/chi0_static
chi0_coulomb_metric full
chi0_headwing true
```

LibBSE detects `chi0_q.info` to select the new representative-q reader. The same `screened_format` also reads the previous real-space format.

## Version-1 interchange records

The manifest begins with:

```text
AIMS_SYMMETRY_ALL 1
naux nq_full
BARE_CHI0 D_CHI0_DAGGER
target_id representative_id operation_id conjugate_flag target_q[3] representative_q[3] delta_filename
...
```

LibRPA substitutes `LIBRPA_SYMMETRY_ALL`. Coordinates are fractional reciprocal coordinates; IDs are positive producer labels, not consumer array indices. Each complete target q must appear exactly once. Δ files start with `AIMS_DELTA 1` or `LIBRPA_DELTA 1`, followed by `naux target_id representative_id` and all `naux²` entries `mu nu real imag`, with 1-based target rows and representative columns.

LibRPA representative χ₀ files use the same dense explicit-index layout with header `LIBRPA_CHI0_Q 1` and `naux representative_id representative_id`. `chi0_q.info` starts with `LIBRPA_CHI0_Q_GRID 1`, followed by `omega_Ha naux nx ny nz`.

Readers reject incomplete/duplicate entries, inconsistent grids/dimensions and nonfinite values. Δ is checked against `max(abs(D D†−I)) <= 1e−6`, allowing the native aims rotation precision established in the W test. No orthogonalization or adjustment of Δ is performed. W and bare χ₀ conventions remain distinct.

## Code

- [Shared mapping, Δ validation and matrix reconstruction](../src/io/symmetry_matrix.cpp)
- [aims response reader](../src/io/aims_screened.cpp)
- [LibRPA response reader](../src/io/rpa_chi0.cpp)
- [Screening and Γ corrections](../src/io/chi0_screening.cpp)
- [Reader regression with a complex Δ and noncommuting Coulomb](../regression_test/general_testcase/test_chi0_screening.cpp)

## Validation

A Si 3×3×3 test with 694 auxiliary functions reconstructs all 27 q points from 4 representatives. All three paths passed: native aims lowest imaginary frequency, LibRPA lowest imaginary frequency, and LibRPA strictly zero frequency. With common KS/QP inputs and Γ head/wing enabled, the maximum relative χ₀ errors were 8.73e−10, 3.71e−10 and 3.71e−10; the corresponding dielectric-spectrum relative errors were 4.42e−10, 2.25e−9 and 2.19e−9. Five serial/MPI reader and symmetry regressions also passed. The nonmagnetic scalar Si test does not establish arbitrary material, spinor or magnetic symmetry coverage.
