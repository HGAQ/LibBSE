#pragma once
#include "bse/bse_types.h"
namespace librpa_int { class Dataset; }
namespace libbse {
class Chi0Screening;
// Read W(q,iw) or bare chi0(q,iw) in the original aims auxiliary basis.
// Select the lowest |omega|, screen chi0 with LibRPA dielectric routines when
// requested, and return full atom-pair W(R) on the coarse GW BvK cell.
TensorMap<Complex> read_aims_screened_interaction(
    const InputParameters &options, librpa_int::Dataset &dataset,
    const std::vector<int> &local_i_atoms, const std::vector<int> &local_j_atoms,
    Chi0Screening *screening = nullptr);
}
