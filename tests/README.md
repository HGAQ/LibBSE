# LibBSE unit tests

The tests are intentionally small and deterministic. Run all of them with:

```sh
ctest --test-dir build-bse --output-on-failure
```

## `test_parameter.cpp`

- `require()` turns a failed Boolean test condition into a descriptive test
  failure.
- `test_whitespace_separated_parameters()` checks whitespace-separated input,
  all core BSE parameter names, the multi-value `bse_spin_types` list,
  relative `input_dir`/`output_dir` resolution, and `bse_tda=full` dispatch.
- `test_librpa_style_assignments_and_last_value_wins()` checks optional
  `key = value` syntax, inline comments, Boolean parsing, and the LibRPA/ABACUS
  rule that the last occurrence of a parameter wins, plus the default
  `output_dir=libbse.d`.
- `test_coarse_kgrid_mode()` checks that `bse_use_fine_kgrid=0` selects the
  SCF coarse-grid calculation path.
- `test_invalid_or_unsupported_parameters_are_rejected()` checks rejection of
  length gauge, unsupported or duplicate spin types, invalid pure-IPA and
  k-grid modes, unknown or obsolete keywords, and malformed integer values.
- `main()` runs the parameter tests and reports exceptions as a nonzero
  test result.

## `test_velocity_gauge.cpp`

- `close()` compares complex transition moments within a strict tolerance.
- `velocity_index()` reproduces the direction-major storage index used for a
  minimal locally owned electron-hole pair.
- `main()` checks the TDA `vX/gap` contraction, the full-BSE
  `-conj(v)Y/gap` term, complex conjugation for imaginary velocity elements,
  and rejection of a zero Kohn-Sham gap.

## `test_profiler.cpp`

- `require()` reports failed profiler invariants without another test-framework
  dependency.
- `exercise_exception_path()` checks that `ScopedTimer` stops its timer while
  unwinding an exception.
- `main()` checks nested entries, accumulated call counts, CPU/wall-time
  accessors, exception-safe scope cleanup, and the timing-table headings and
  indentation.

## `test_spectrum_mpi.cpp`

- `velocity_index()` reproduces the compact, direction-major local-pair
  `velocity_mo` storage layout.
- `main()` places the only nonzero electron-hole contribution in rank 1's
  local pair block, checks that each rank stores only its own velocity/gap
  data, and verifies that the reduced transition dipole reaches rank 0. It
  also checks independent rank-specific amplitude file write/read.

## `test_progress.cpp`

- `main()` checks the `DONE(elapsed SEC) : description` format
  and verifies that only communicator rank 0 writes the completion marker.

## `test_bse_files.cpp`

- `test_coarse_qp_reader()` checks that `bse_use_fine_kgrid = 0` reads
  `energy_qp`, converts Hartree energies to Rydberg, skips core states, and
  computes direct and indirect gaps on the coarse grid.
- `write_wc()` creates one minimal MatrixMarket Wc block for a selected atom
  pair.
- `main()` supplies a complete bare-Coulomb map but creates only the Wc file
  required by the local LibRI `list_I x list_J` pair. It checks that nonlocal
  files are not accessed and that the returned local screened interaction is
  `W = V + Wc`.

## `test_elpa_solver.cpp`

- `require()` and the two `require_close()` overloads provide Boolean, real,
  and complex assertions without adding another test-framework dependency.
- `make_descriptor()` creates the block-cyclic descriptor used by the serial
  and two-rank ELPA tests.
- `localize()` maps a deterministic dense matrix onto the local block-cyclic
  storage owned by each MPI rank.
- `globalize()` is a test-only MPI reduction that constructs dense references;
  production spectrum code never globalizes amplitudes.
- `test_skew_solver_reference()` ports the fixed 2x2 A/B case from
  `hamilt_bse_solver_test.cpp` and checks the two positive full-BSE energies.
- `test_distributed_matrix_checks()` checks the ScaLAPACK/MPI Hermitian test
  for A and symmetric test for B with both valid and deliberately broken
  distributed matrices.
- `test_tda_solver_residual()` solves a
  deterministic Hermitian matrix with ELPA and checks `A v = omega v` for
  every eigenpair, then checks redistribution into the spectrum pair layout.
- `test_full_solver_residual_and_metric()` checks the residual of
  `[[A,B],[-conj(B),-conj(A)]]`, plus the full-BSE symplectic normalization
  `X^H X - Y^H Y = I` and redistribution of both X/Y components.
- `deterministic_hermitian()`, `deterministic_symmetric()`, and
  `full_bse_matrix()` construct reproducible test matrices without random
  seeds.
- `main()` initializes MPI, LibRPA/ELPA, and BLACS in the required order, runs
  the three solver tests in both one- and two-rank CTest entries, combines
  rank-local status, and finalizes cleanly.

## `test_molecular_lri_comm.cpp`

- `block_value()` defines deterministic complex LibRI k-block values.
- `main()` distributes source k blocks between two ranks, calls the migrated
  `transform_k_2dlocal()` over two communication batches, and checks every
  locally owned 2D block-cyclic matrix element against the dense formula.
