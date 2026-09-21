#pragma once
#include "bse/bse_types.h"
namespace librpa_int { class Dataset; }
namespace libbse {
// Read full W(q,iw) in the original aims auxiliary basis, select the lowest
// available |omega|, and return atom-pair W(R) on the coarse GW BvK cell.
TensorMap<Complex> read_aims_screened_interaction(
    const InputParameters &options, librpa_int::Dataset &dataset,
    const std::vector<int> &local_i_atoms, const std::vector<int> &local_j_atoms);
}
