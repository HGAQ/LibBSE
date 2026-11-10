#include "symmetry_matrix.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace libbse
{
namespace
{
bool finite_complex(const std::complex<double> &entry)
{
    return std::isfinite(entry.real()) && std::isfinite(entry.imag());
}
void check_square_size(std::size_t naux, std::size_t entries)
{
    if (naux == 0 || naux > std::numeric_limits<std::size_t>::max() / naux ||
        entries != naux * naux)
    {
        throw std::runtime_error("invalid auxiliary symmetry matrix dimensions");
    }
}
} // namespace

SymmetryMatrices read_symmetry_matrices(const std::filesystem::path &manifest,
                                        std::size_t expected_naux, std::size_t expected_nq,
                                        const std::string &producer, const std::string &quantity)
{
    std::ifstream symmetry_input(manifest);
    std::string magic, convention, delta_convention;
    int version = 0;
    std::size_t nq = 0;
    SymmetryMatrices symmetry_metadata;
    if (!(symmetry_input >> magic >> version >> symmetry_metadata.naux >> nq >> convention >>
          delta_convention) ||
        magic != producer + "_SYMMETRY_ALL" || version != 1 ||
        symmetry_metadata.naux != expected_naux || nq != expected_nq || convention != quantity ||
        delta_convention != (quantity == "FULL_W" ? "D_W_DAGGER" : "D_CHI0_DAGGER"))
    {
        throw std::runtime_error(
            "invalid auxiliary symmetry manifest dimensions/version/convention: " +
            manifest.string());
    }
    std::set<int> target_ids;
    for (std::size_t iq = 0; iq < nq; ++iq)
    {
        SymmetryQMapping q_mapping;
        q_mapping.producer = producer;
        int conjugate_flag = -1;
        std::string delta_filename;
        if (!(symmetry_input >> q_mapping.target_id >> q_mapping.representative_id >>
              q_mapping.operation_id >> conjugate_flag >> q_mapping.target_q[0] >>
              q_mapping.target_q[1] >> q_mapping.target_q[2] >> q_mapping.representative_q[0] >>
              q_mapping.representative_q[1] >> q_mapping.representative_q[2] >> delta_filename) ||
            q_mapping.target_id < 1 || q_mapping.representative_id < 1 ||
            q_mapping.operation_id < 1 || (conjugate_flag != 0 && conjugate_flag != 1) ||
            !target_ids.insert(q_mapping.target_id).second)
        {
            throw std::runtime_error("invalid or duplicate auxiliary symmetry q mapping");
        }
        for (int axis = 0; axis < 3; ++axis)
        {
            if (!std::isfinite(q_mapping.target_q[axis]) ||
                !std::isfinite(q_mapping.representative_q[axis]))
            {
                throw std::runtime_error("non-finite auxiliary symmetry q coordinate");
            }
        }
        const std::filesystem::path delta_relative(delta_filename);
        if (delta_relative.is_absolute() || delta_relative.has_parent_path())
        {
            throw std::runtime_error("symmetry Delta must be a file in the manifest directory");
        }
        q_mapping.delta_file = manifest.parent_path() / delta_relative;
        q_mapping.conjugate_representative = conjugate_flag == 1;
        symmetry_metadata.q_mappings.push_back(q_mapping);
    }
    std::string trailing_token;
    if (symmetry_input >> trailing_token)
    {
        throw std::runtime_error("extra auxiliary symmetry mapping records");
    }
    return symmetry_metadata;
}

std::vector<std::complex<double>> read_symmetry_delta(const SymmetryQMapping &q_mapping,
                                                      std::size_t naux)
{
    if (naux == 0 || naux > std::numeric_limits<std::size_t>::max() / naux)
    {
        throw std::runtime_error("invalid symmetry Delta dimension");
    }
    std::ifstream delta_input(q_mapping.delta_file);
    std::string magic;
    int version = 0, target_id = 0, representative_id = 0;
    std::size_t file_naux = 0;
    if (!(delta_input >> magic >> version >> file_naux >> target_id >> representative_id) ||
        magic != q_mapping.producer + "_DELTA" || version != 1 || file_naux != naux ||
        target_id != q_mapping.target_id || representative_id != q_mapping.representative_id)
    {
        throw std::runtime_error("invalid symmetry Delta header: " + q_mapping.delta_file.string());
    }
    std::vector<std::complex<double>> symmetry_delta(naux * naux);
    std::vector<bool> delta_entry_seen(naux * naux, false);
    for (std::size_t entry = 0; entry < naux * naux; ++entry)
    {
        std::size_t mu = 0, nu = 0;
        double delta_real = 0., delta_imag = 0.;
        if (!(delta_input >> mu >> nu >> delta_real >> delta_imag) || mu < 1 || nu < 1 ||
            mu > naux || nu > naux || !std::isfinite(delta_real) || !std::isfinite(delta_imag))
        {
            throw std::runtime_error("missing or invalid symmetry Delta entry");
        }
        const auto delta_index = (mu - 1) * naux + nu - 1;
        if (delta_entry_seen[delta_index])
        {
            throw std::runtime_error("duplicate symmetry Delta entry");
        }
        delta_entry_seen[delta_index] = true;
        symmetry_delta[delta_index] = {delta_real, delta_imag};
    }
    std::string trailing_token;
    if (delta_input >> trailing_token)
    {
        throw std::runtime_error("extra symmetry Delta entries");
    }
    // Atom permutations and spherical-harmonic rotations must preserve the basis norm.
    // Accumulate D D^H column by column, skipping only exact structural zeros.
    std::vector<std::complex<double>> delta_gram(naux * naux, 0.);
    for (std::size_t k = 0; k < naux; ++k)
    {
        for (std::size_t i = 0; i < naux; ++i)
        {
            const auto delta_ik = symmetry_delta[i * naux + k];
            if (delta_ik == std::complex<double>{})
            {
                continue;
            }
            for (std::size_t j = 0; j < naux; ++j)
            {
                delta_gram[i * naux + j] += delta_ik * std::conj(symmetry_delta[j * naux + k]);
            }
        }
    }
    // Native aims rotations can accumulate ~1e-7 errors from intermediate precision.
    // Allow that numerical noise while still rejecting malformed transformations.
    constexpr double symmetry_delta_unitarity_tolerance = 1.e-6;
    for (std::size_t i = 0; i < naux; ++i)
    {
        for (std::size_t j = 0; j < naux; ++j)
        {
            if (std::abs(delta_gram[i * naux + j] - (i == j ? 1. : 0.)) >
                symmetry_delta_unitarity_tolerance)
            {
                throw std::runtime_error(
                    "symmetry Delta is not unitary in the exported auxiliary basis");
            }
        }
    }
    return symmetry_delta;
}

std::vector<std::complex<double>>
reconstruct_symmetry_operator(const std::vector<std::complex<double>> &representative_operator,
                              const std::vector<std::complex<double>> &symmetry_delta,
                              std::size_t naux, bool conjugate_representative)
{
    check_square_size(naux, representative_operator.size());
    check_square_size(naux, symmetry_delta.size());
    for (const auto &entry : representative_operator)
    {
        if (!finite_complex(entry))
        {
            throw std::runtime_error("non-finite representative operator");
        }
    }
    for (const auto &entry : symmetry_delta)
    {
        if (!finite_complex(entry))
        {
            throw std::runtime_error("non-finite symmetry Delta");
        }
    }

    // Rotate the bra index first: Delta * O (or Delta * O* for time reversal).
    std::vector<std::complex<double>> delta_times_operator(naux * naux, 0.);
    std::vector<std::complex<double>> reconstructed_operator(naux * naux, 0.);
    for (std::size_t i = 0; i < naux; ++i)
    {
        for (std::size_t k = 0; k < naux; ++k)
        {
            const auto delta_ik = symmetry_delta[i * naux + k];
            if (delta_ik == std::complex<double>{})
            {
                continue;
            }
            for (std::size_t j = 0; j < naux; ++j)
            {
                const auto operator_kj = representative_operator[k * naux + j];
                delta_times_operator[i * naux + j] +=
                    delta_ik * (conjugate_representative ? std::conj(operator_kj) : operator_kj);
            }
        }
    }

    // Rotate the ket index with the adjoint: (Delta * O) * Delta^dagger.
    for (std::size_t j = 0; j < naux; ++j)
    {
        for (std::size_t k = 0; k < naux; ++k)
        {
            const auto delta_jk_conjugate = std::conj(symmetry_delta[j * naux + k]);
            if (delta_jk_conjugate == std::complex<double>{})
            {
                continue;
            }
            for (std::size_t i = 0; i < naux; ++i)
            {
                reconstructed_operator[i * naux + j] +=
                    delta_times_operator[i * naux + k] * delta_jk_conjugate;
            }
        }
    }
    return reconstructed_operator;
}
} // namespace libbse
