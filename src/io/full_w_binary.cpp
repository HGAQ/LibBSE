#include "full_w_binary.h"

#include "interface/librpa_api.h"
#include "utils/memory_views.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace libbse
{

TensorMap<Complex> read_full_w_binary(
    const InputParameters &w_options, librpa_int::Dataset &w_dataset,
    const std::vector<int> &w_i_atoms, const std::vector<int> &w_j_atoms)
{
    // V1 header: 16-byte magic, four int32 dimensions, four doubles (q, omega).
    constexpr std::size_t w_header_bytes = 64;
    constexpr double w_static_tolerance = 1.e-14;
    constexpr double w_q_tolerance = 1.e-8;
    constexpr double w_hermitian_tolerance = 1.e-7;

    const auto w_grid = librpa_int::build_uniform_kmesh_frac(w_dataset.pbc.period);
    const int w_naux = w_dataset.basis_aux.nb_total;
    const std::uint64_t w_expected_bytes = w_header_bytes
        + static_cast<std::uint64_t>(w_naux) * w_naux * sizeof(Complex);
    std::map<std::size_t, std::filesystem::path> w_files;

    for (const auto &w_entry : std::filesystem::directory_iterator(w_options.screened_dir))
    {
        const auto w_path = w_entry.path();
        const auto w_name = w_path.filename().string();
        if (!w_entry.is_regular_file()
            || w_name.rfind("full_w_q_", 0) != 0
            || w_path.extension() != ".bin")
            continue;

        std::ifstream w_input(w_path, std::ios::binary);
        char w_magic[16]{};
        std::int32_t w_dimensions[4]{};
        double w_q_frequency[4]{};
        w_input.read(w_magic, sizeof(w_magic));
        w_input.read(reinterpret_cast<char *>(w_dimensions), sizeof(w_dimensions));
        w_input.read(reinterpret_cast<char *>(w_q_frequency), sizeof(w_q_frequency));
        if (!w_input
            || std::string(w_magic, 16) != std::string("LIBBSE_FULL_W_V1", 16)
            || w_dimensions[0] != w_naux
            || w_dimensions[1] != w_dataset.pbc.period.x
            || w_dimensions[2] != w_dataset.pbc.period.y
            || w_dimensions[3] != w_dataset.pbc.period.z)
        {
            throw std::runtime_error("incompatible full W binary header: "
                                     + w_path.string());
        }

        for (double w_component : w_q_frequency)
        {
            if (!std::isfinite(w_component))
                throw std::runtime_error("non-finite W q/frequency");
        }
        // A lowest nonzero imaginary-frequency sample is not static W(0).
        if (std::abs(w_q_frequency[3]) > w_static_tolerance)
            throw std::runtime_error("full W binary cache requires actual static omega=0");

        // Match coordinates modulo reciprocal vectors, not file numbering.
        std::size_t w_iq = 0;
        for (; w_iq < w_grid.size(); ++w_iq)
        {
            const double w_delta[3] = {
                w_q_frequency[0] - w_grid[w_iq].x,
                w_q_frequency[1] - w_grid[w_iq].y,
                w_q_frequency[2] - w_grid[w_iq].z
            };
            bool w_match = true;
            for (double w_component : w_delta)
            {
                w_match = w_match
                          && std::abs(w_component - std::round(w_component))
                                 < w_q_tolerance;
            }
            if (w_match) break;
        }
        if (w_iq == w_grid.size() || !w_files.emplace(w_iq, w_path).second)
            throw std::runtime_error("off-grid or duplicate full W binary q point");
        if (std::filesystem::file_size(w_path) != w_expected_bytes)
            throw std::runtime_error("truncated or oversized full W binary payload");
    }

    // The cache has no symmetry metadata, so every coarse q point is required.
    if (w_files.size() != w_grid.size())
        throw std::runtime_error("missing full W binary q points");
    auto w_pbc = w_dataset.pbc;
    w_pbc.map_irk_ks.clear();

    LibRPA_API::ScreenedQBlocks w_q_blocks;
    auto w_q_blocks_memory = watch_memory("full_w_binary.w_q_blocks", w_q_blocks);
    for (const auto &[w_iq, w_path] : w_files)
    {
        std::vector<Complex> w_auxiliary_q(static_cast<std::size_t>(w_naux) * w_naux);
        auto w_auxiliary_q_memory = watch_memory(
            "full_w_binary.w_auxiliary_q", w_auxiliary_q);
        std::ifstream w_input(w_path, std::ios::binary);
        w_input.seekg(w_header_bytes);
        w_input.read(reinterpret_cast<char *>(w_auxiliary_q.data()),
                     w_auxiliary_q.size() * sizeof(Complex));
        if (!w_input)
            throw std::runtime_error("cannot read full W binary payload");

        // Validate the full row-major matrix before extracting atom-pair blocks.
        double w_norm_squared = 0.0;
        double w_hermitian_error_squared = 0.0;
        for (int w_mu = 0; w_mu < w_naux; ++w_mu)
        {
            for (int w_nu = 0; w_nu < w_naux; ++w_nu)
            {
                const auto w_element = w_auxiliary_q[
                    static_cast<std::size_t>(w_mu) * w_naux + w_nu];
                if (!std::isfinite(w_element.real()) || !std::isfinite(w_element.imag()))
                    throw std::runtime_error("non-finite full W binary element");
                const auto w_transposed_element = w_auxiliary_q[
                    static_cast<std::size_t>(w_nu) * w_naux + w_mu];
                w_norm_squared += std::norm(w_element);
                w_hermitian_error_squared += std::norm(
                    w_element - std::conj(w_transposed_element));
            }
        }
        if (std::sqrt(w_hermitian_error_squared)
            > w_hermitian_tolerance * std::max(1.0, std::sqrt(w_norm_squared)))
            throw std::runtime_error("full W binary matrix is not Hermitian");

        const auto w_q = w_grid[w_iq] * w_pbc.G;
        w_pbc.map_irk_ks[w_q].push_back(w_q);
        for (int w_iatom : w_i_atoms)
        {
            for (int w_jatom : w_j_atoms)
            {
                librpa_int::Matz w_atom_pair(
                    w_dataset.basis_aux[w_iatom], w_dataset.basis_aux[w_jatom],
                    librpa_int::MAJOR::ROW);
                for (std::size_t w_mu = 0; w_mu < w_atom_pair.nr(); ++w_mu)
                {
                    const auto w_global_mu = w_dataset.basis_aux.get_global_index(
                        w_iatom, w_mu);
                    for (std::size_t w_nu = 0; w_nu < w_atom_pair.nc(); ++w_nu)
                    {
                        const auto w_global_nu = w_dataset.basis_aux.get_global_index(
                            w_jatom, w_nu);
                        w_atom_pair(w_mu, w_nu) = w_auxiliary_q[
                            w_global_mu * w_naux + w_global_nu];
                    }
                }
                w_q_blocks[w_iatom][w_jatom][w_q] = std::move(w_atom_pair);
            }
        }
    }

    if (w_dataset.comm_h.myid == 0)
    {
        std::cout << "Read full W binary cache: " << w_files.size()
                  << " q points, omega=0 Ha, original auxiliary basis, full W (no added V)\n";
    }
    // Reuse the text reader's Fourier phase and normalization. This is full W;
    // adding bare V here would count that interaction twice.
    return LibRPA_API::transform_screened_q_to_r(w_dataset, w_pbc, w_q_blocks);
}

} // namespace libbse
