#pragma once

#include "bse_types.h"

#include <librpa_file_reader.hpp>

#include <vector>

namespace libbse
{

EigenSolution solve_tda_elpa(std::vector<Complex> &matrix,
                             librpa_int::ArrayDesc &descriptor,
                             int nstates);

EigenSolution solve_full_elpa(std::vector<Complex> &matrix_a,
                              std::vector<Complex> &matrix_b,
                              const librpa_int::ArrayDesc &pair_descriptor,
                              librpa_int::ArrayDesc &full_descriptor,
                              int nstates);

} // namespace libbse
