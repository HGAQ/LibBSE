#pragma once
#include "symmetry_matrix.h"
namespace libbse
{
using AimsSymmetryQMapping = SymmetryQMapping;
using AimsSymmetryAll = SymmetryMatrices;
AimsSymmetryAll read_aims_symmetry_all(const std::filesystem::path &manifest,
                                       std::size_t expected_naux, std::size_t expected_nq,
                                       const std::string &quantity = "FULL_W");
std::vector<std::complex<double>> read_aims_symmetry_delta(const AimsSymmetryQMapping &q_mapping,
                                                           std::size_t naux);
std::vector<std::complex<double>>
reconstruct_aims_symmetry_w(const std::vector<std::complex<double>> &representative_w,
                            const std::vector<std::complex<double>> &symmetry_delta,
                            std::size_t naux, bool conjugate_representative = false);
} // namespace libbse
