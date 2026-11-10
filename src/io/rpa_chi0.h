#pragma once
#include "bse/bse_types.h"
namespace librpa_int
{
class Dataset;
}
namespace libbse
{
class Chi0Screening;
// Read LibRPA chi0(R) exports, then screen on the original complete q mesh.
TensorMap<Complex> read_librpa_chi0(const InputParameters &, librpa_int::Dataset &,
                                    const std::vector<int> &, const std::vector<int> &,
                                    Chi0Screening *screening = nullptr);
} // namespace libbse
