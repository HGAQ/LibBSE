# Si 222 local validation

Reference: clean LibBSE 2b727c0, fixed inputs and numerical baseline; not an independent physical reference.

| Case | Pass | Max energy difference / eV | Amplitude subspace error | W replay |
|---|---|---:|---:|---|
| boundary | True | 0 | 2.98e-08 | not applicable |
| chi0_headwing | True | 0 | 3.33e-08 | True |
| chi0_headwing_mpi2 | True | 0 | 2.58e-08 | True |
| dynamic | True | 0 | 2.11e-08 | True |
| native_w | True | 0 | 2.98e-08 | not applicable |
| native_w_1v1c | True | 0 | 2.98e-08 | not applicable |
| native_w_mpi2 | True | 0 | 2.98e-08 | not applicable |
| native_w_unbatched | True | 0 | 2.98e-08 | not applicable |

All 8 cases passed. The registered CTest real_si section also passed 8/8 (65.54 s). General tests after reorganization: 24/24 passed. GitHub-hosted validation of the new physical section is pending.

Trial auxiliary basis sizes: 428 (unmodified light), 116 (rejected: negative excitation), 218 (selected). Input archive: 18.9 MiB; reference archive: 18.4 MiB.

Numerical tolerances were not loosened for this campaign. No production source files were changed.
