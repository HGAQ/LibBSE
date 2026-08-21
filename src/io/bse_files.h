#pragma once

#include "bse/bse_types.h"

namespace librpa_int
{
class Dataset;
}

namespace libbse
{

QuasiparticleBands read_qp_bands(const InputParameters &options,
                                 const librpa_int::Dataset &dataset);

TensorMap<Complex> read_screened_interaction(
    const InputParameters &options,
    const TensorMap<Complex> &bare_coulomb,
    std::size_t cell_count,
    const std::vector<int> &local_i_atoms,
    const std::vector<int> &local_j_atoms);

TensorMap<Complex> convert_lri_coefficients(librpa_int::Dataset &dataset);

void remap_to_nearest_bvk_cell(TensorMap<Complex> &tensors,
                               const librpa_int::Dataset &dataset);

} // namespace libbse
