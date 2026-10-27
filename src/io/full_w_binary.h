#pragma once
#include "aims_screened.h"
namespace libbse {
TensorMap<Complex> read_full_w_binary(const InputParameters &, librpa_int::Dataset &,
    const std::vector<int> &, const std::vector<int> &);
}
