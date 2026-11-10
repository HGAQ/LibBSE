#pragma once

#include <array>
#include <complex>
#include <filesystem>
#include <vector>
#include <string>

namespace libbse
{

// Manifest IDs belong to the producer; coordinates define the consumer grid mapping.
struct SymmetryQMapping
{
    int target_id = 0;
    int representative_id = 0;
    int operation_id = 0;
    bool conjugate_representative = false;
    std::array<double, 3> target_q{};
    std::array<double, 3> representative_q{};
    std::filesystem::path delta_file;
    std::string producer;
};

struct SymmetryMatrices
{
    std::size_t naux = 0;
    std::vector<SymmetryQMapping> q_mappings;
};

// Version 1 stores an explicitly typed operator and target-row/source-column Delta.
SymmetryMatrices read_symmetry_matrices(const std::filesystem::path &manifest,
                                        std::size_t expected_naux, std::size_t expected_nq,
                                        const std::string &producer, const std::string &quantity);
std::vector<std::complex<double>> read_symmetry_delta(const SymmetryQMapping &q_mapping,
                                                      std::size_t naux);

// All dense buffers are row-major. No Coulomb addition or unit conversion here.
std::vector<std::complex<double>>
reconstruct_symmetry_operator(const std::vector<std::complex<double>> &representative_operator,
                              const std::vector<std::complex<double>> &symmetry_delta,
                              std::size_t naux, bool conjugate_representative = false);

} // namespace libbse
