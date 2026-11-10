#include "aims_symmetry_all.h"
namespace libbse
{
AimsSymmetryAll read_aims_symmetry_all(const std::filesystem::path &manifest,
                                       std::size_t expected_naux, std::size_t expected_nq,
                                       const std::string &quantity)
{
    return read_symmetry_matrices(manifest, expected_naux, expected_nq, "AIMS", quantity);
}
std::vector<std::complex<double>> read_aims_symmetry_delta(const AimsSymmetryQMapping &q_mapping,
                                                           std::size_t naux)
{
    auto aims_q_mapping = q_mapping;
    aims_q_mapping.producer = "AIMS";
    return read_symmetry_delta(aims_q_mapping, naux);
}
std::vector<std::complex<double>>
reconstruct_aims_symmetry_w(const std::vector<std::complex<double>> &representative_w,
                            const std::vector<std::complex<double>> &symmetry_delta,
                            std::size_t naux, bool conjugate_representative)
{
    return reconstruct_symmetry_operator(representative_w, symmetry_delta, naux,
                                         conjugate_representative);
}
} // namespace libbse
