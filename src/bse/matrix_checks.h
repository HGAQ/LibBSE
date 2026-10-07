#pragma once

#include "bse_types.h"

#include <librpa_file_reader.hpp>

#include <vector>

namespace libbse
{

struct MatrixCheckSolution
{
    bool passed = false;
    double difference_norm = 0.0;
    double sum_norm = 0.0;
    double relative_error = 0.0;
};

MatrixCheckSolution check_hermitian(
    const std::vector<Complex> &matrix,
    const librpa_int::ArrayDesc &descriptor, double threshold);

MatrixCheckSolution check_symmetric(
    const std::vector<Complex> &matrix,
    const librpa_int::ArrayDesc &descriptor, double threshold);

} // namespace libbse
