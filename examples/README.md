# End-to-end examples

- `abacus_si_222`: ABACUS SCF export -> standalone LibRPA G0W0 -> LibBSE.
- `fhi_aims_si_333`: FHI-aims DFT/RI/momentum export -> momentum conversion ->
  standalone LibRPA G0W0 -> LibBSE.

Both workflows enforce a real velocity matrix and finish by checking the
oscillator-strength and broadened-spectrum files. Read each subdirectory's
README before running; the examples are compact interface validations, not
converged production calculations.
