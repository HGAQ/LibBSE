#pragma once

#include "bse/bse_types.h"

#include <vector>

namespace librpa_int
{
class Dataset;
}

namespace libbse
{

// Read full static W on the complete coarse q mesh and return the requested
// atom-pair W(R) blocks in the original auxiliary basis. No bare V is added.
TensorMap<Complex> read_full_w_binary(
    const InputParameters &w_options, librpa_int::Dataset &w_dataset,
    const std::vector<int> &w_i_atoms, const std::vector<int> &w_j_atoms);

} // namespace libbse
