#pragma once

#include "bse_types.h"

#include <memory>

namespace librpa_int
{
class Dataset;
}

namespace libbse
{

void run_bse(const InputParameters &options,
             const std::shared_ptr<librpa_int::Dataset> &dataset);

} // namespace libbse
