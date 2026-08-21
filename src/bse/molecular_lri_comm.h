#pragma once

#include "bse_types.h"

#include <librpa_file_reader.hpp>

#include <RI/global/Tensor.h>

#include <map>
#include <vector>

namespace libbse
{

using KMatrixMap = std::map<int, std::map<int, RI::Tensor<Complex>>>;

// Each LibRI k block is sent only to the ranks that
// own its destination blocks; no process materializes the dense global matrix.
void transform_k_2dlocal(
    std::vector<Complex> &matrix,
    const KMatrixMap &blocks,
    const librpa_int::ArrayDesc &descriptor,
    int nk, int pair_dimension, double coefficient);

} // namespace libbse
