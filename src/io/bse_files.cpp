#include "bse_files.h"

#include <librpa_file_reader.hpp>

#include <RI/global/Global_Func-2.h>
#include <RI/ri/Cell_Nearest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;

namespace libbse
{
namespace
{

void move_tensor(TensorMap<Complex> &map, int iat, int jat,
                 const Cell &old_cell, const Cell &new_cell)
{
    auto outer = map.find(iat);
    if (outer == map.end()) return;
    auto &blocks = outer->second;
    auto old_iter = blocks.find({jat, old_cell});
    if (old_iter == blocks.end()) return;
    const AtomCell new_key{jat, new_cell};
    if (blocks.count(new_key) != 0)
        throw std::runtime_error("nearest-cell remap produced a duplicate tensor key");
    blocks.emplace(new_key, std::move(old_iter->second));
    blocks.erase(old_iter);
}

void store_requested_bands(QuasiparticleBands &result,
                           int ik, const std::vector<double> &energies_ry)
{
    if (static_cast<int>(energies_ry.size()) < result.nbands)
        throw std::runtime_error("not enough occupied/virtual quasiparticle bands at k-point "
                                 + std::to_string(ik + 1));

    const int ncore_here = static_cast<int>(energies_ry.size()) - result.nbands;
    if (ik == 0) result.ncore = ncore_here;
    if (ncore_here != result.ncore)
        throw std::runtime_error("inconsistent core-band count in quasiparticle data");

    for (int ib = 0; ib != result.nbands; ++ib)
    {
        const std::size_t dst = static_cast<std::size_t>(ik) * result.nbands + ib;
        result.energies_ry[dst]
            = energies_ry[static_cast<std::size_t>(result.ncore + ib)];
    }
}

QuasiparticleBands read_fine_qp_bands(const InputParameters &options,
                                      const librpa_int::Dataset &dataset)
{
    const fs::path file = fs::path(options.input_dir) / "GW_band_spin_1.dat";
    std::ifstream input(file);
    if (!input) throw std::runtime_error("cannot open GW band file: " + file.string());

    QuasiparticleBands result;
    result.nk = dataset.mf_band.get_n_kpoints();
    result.nbands = options.nocc + options.nvirt;
    result.energies_ry.resize(static_cast<std::size_t>(result.nk) * result.nbands);

    std::string line;
    for (int ik = 0; ik != result.nk; ++ik)
    {
        if (!std::getline(input, line))
            throw std::runtime_error("GW band file ended before k-point " + std::to_string(ik + 1));
        if (line.empty())
        {
            --ik;
            continue;
        }
        std::istringstream parser(line);
        int index = 0;
        KPoint k{};
        if (!(parser >> index >> k[0] >> k[1] >> k[2]) || index != ik + 1)
            throw std::runtime_error("invalid k-point header in " + file.string());
        const auto &expected = dataset.kfrac_band_list.at(static_cast<std::size_t>(ik));
        if (std::abs(k[0] - expected.x) > PARAM.constants.band_file_kpoint_tolerance
            || std::abs(k[1] - expected.y) > PARAM.constants.band_file_kpoint_tolerance
            || std::abs(k[2] - expected.z) > PARAM.constants.band_file_kpoint_tolerance)
            throw std::runtime_error("GW band k-point does not match the fine BSE grid at index "
                                     + std::to_string(ik + 1));

        std::vector<double> energies;
        double occupation = 0.0;
        double energy_ev = 0.0;
        int virtual_count = 0;
        while (parser >> occupation >> energy_ev)
        {
            energies.push_back(energy_ev / PARAM.constants.ry_to_ev);
            if (occupation * result.nk < PARAM.constants.occupation_tolerance)
                ++virtual_count;
            if (virtual_count == options.nvirt) break;
        }
        if (virtual_count != options.nvirt)
            throw std::runtime_error("not enough virtual GW bands at k-point "
                                     + std::to_string(ik + 1));
        store_requested_bands(result, ik, energies);
    }
    return result;
}

QuasiparticleBands read_coarse_qp_bands(const InputParameters &options,
                                        const librpa_int::Dataset &dataset)
{
    const fs::path file = fs::path(options.input_dir) / "energy_qp";
    std::ifstream input(file);
    if (!input)
        throw std::runtime_error("cannot open coarse-grid quasiparticle file: "
                                 + file.string());

    QuasiparticleBands result;
    result.nk = dataset.mf_band.get_n_kpoints();
    result.nbands = options.nocc + options.nvirt;
    result.energies_ry.resize(static_cast<std::size_t>(result.nk) * result.nbands);

    std::string line;
    int ik = 0;
    while (ik != result.nk && std::getline(input, line))
    {
        if (line.find("K_point") == std::string::npos) continue;

        std::istringstream header(line);
        std::string label;
        char colon = '\0';
        int index = 0;
        KPoint k{};
        if (!(header >> label >> index >> colon >> k[0] >> k[1] >> k[2])
            || label != "K_point" || colon != ':' || index != ik + 1)
            throw std::runtime_error("invalid k-point header in " + file.string());

        const auto &expected = dataset.kfrac_band_list.at(static_cast<std::size_t>(ik));
        if (std::abs(k[0] - expected.x) > PARAM.constants.energy_qp_kpoint_tolerance
            || std::abs(k[1] - expected.y) > PARAM.constants.energy_qp_kpoint_tolerance
            || std::abs(k[2] - expected.z) > PARAM.constants.energy_qp_kpoint_tolerance)
            throw std::runtime_error("energy_qp k-point does not match the coarse BSE grid at index "
                                     + std::to_string(ik + 1));

        std::vector<double> energies;
        int virtual_count = 0;
        bool saw_state = false;
        while (std::getline(input, line))
        {
            const auto first = line.find_first_not_of(" \t\r");
            if (first == std::string::npos) continue;
            if (line[first] == '-')
            {
                if (saw_state) break;
                continue;
            }

            std::istringstream state_line(line);
            int state = 0;
            double occupation = 0.0;
            double ks_energy_ha = 0.0;
            double qp_energy_ha = 0.0;
            if (!(state_line >> state >> occupation >> ks_energy_ha >> qp_energy_ha))
                continue;
            saw_state = true;
            energies.push_back(qp_energy_ha * PARAM.constants.ha_to_ry);
            if (occupation < PARAM.constants.occupation_tolerance) ++virtual_count;
            if (virtual_count == options.nvirt) break;
        }
        if (virtual_count != options.nvirt)
            throw std::runtime_error("not enough virtual quasiparticle bands at k-point "
                                     + std::to_string(ik + 1));
        store_requested_bands(result, ik, energies);
        ++ik;
    }
    if (ik != result.nk)
        throw std::runtime_error("energy_qp ended before k-point " + std::to_string(ik + 1));
    return result;
}

void calculate_gaps(QuasiparticleBands &result, const InputParameters &options)
{
    double cbm = std::numeric_limits<double>::max();
    double vbm = -std::numeric_limits<double>::max();
    result.direct_gap_ry = std::numeric_limits<double>::max();
    for (int ik = 0; ik != result.nk; ++ik)
    {
        const auto offset = static_cast<std::size_t>(ik) * result.nbands;
        double vk = result.energies_ry[offset];
        double ck = result.energies_ry[offset + options.nocc];
        for (int ib = 0; ib != options.nocc; ++ib)
            vk = std::max(vk, result.energies_ry[offset + ib]);
        for (int ib = options.nocc; ib != result.nbands; ++ib)
            ck = std::min(ck, result.energies_ry[offset + ib]);
        vbm = std::max(vbm, vk);
        cbm = std::min(cbm, ck);
        result.direct_gap_ry = std::min(result.direct_gap_ry, ck - vk);
    }
    result.indirect_gap_ry = cbm - vbm;
}

} // namespace

QuasiparticleBands read_qp_bands(const InputParameters &options,
                                 const librpa_int::Dataset &dataset)
{
    QuasiparticleBands result = options.bse_use_fine_kgrid == 0
        ? read_coarse_qp_bands(options, dataset)
        : read_fine_qp_bands(options, dataset);
    calculate_gaps(result, options);
    return result;
}

TensorMap<Complex> convert_lri_coefficients(librpa_int::Dataset &dataset)
{
    TensorMap<Complex> result;
    for (const auto &[iat, blocks] : dataset.cs_data.data_libri)
        for (const auto &[key, tensor] : blocks)
            result[static_cast<int>(iat)][{static_cast<int>(key.first), key.second}]
                = RI::Global_Func::convert<Complex>(tensor);
    dataset.cs_data.clear();
    return result;
}

TensorMap<Complex> read_screened_interaction(
    const InputParameters &options,
    const TensorMap<Complex> &bare_coulomb,
    std::size_t cell_count,
    const std::vector<int> &local_i_atoms,
    const std::vector<int> &local_j_atoms)
{
    TensorMap<Complex> screened;
    const fs::path wc_dir = fs::path(options.input_dir).parent_path() / "librpa.d";
    for (const int iat : local_i_atoms)
    {
        for (const int jat : local_j_atoms)
        {
            for (std::size_t ir = 0; ir != cell_count; ++ir)
            {
                std::ostringstream name;
                name << "Wc_Mu_" << iat << "_Nu_" << jat << "_iR_" << ir
                     << "_ifreq_0.mtx";
                const fs::path file = wc_dir / name.str();
                std::ifstream input(file);
                if (!input) throw std::runtime_error("cannot open screened interaction: "
                                                     + file.string());
                std::string line;
                std::getline(input, line);
                Cell r{};
                bool found_r = false;
                while (std::getline(input, line) && !line.empty() && line.front() == '%')
                {
                    const auto left = line.find('(');
                    const auto right = line.find(')', left);
                    if (left != std::string::npos && right != std::string::npos)
                    {
                        std::istringstream rs(line.substr(left + 1, right - left - 1));
                        if (rs >> r[0] >> r[1] >> r[2]) found_r = true;
                    }
                }
                if (!found_r) throw std::runtime_error("missing R vector in " + file.string());

                std::size_t nrow = 0, ncol = 0, nnz = 0;
                {
                    std::istringstream dims(line);
                    if (!(dims >> nrow >> ncol >> nnz))
                        throw std::runtime_error("invalid MatrixMarket dimensions in " + file.string());
                }
                const auto bare_iter = bare_coulomb.at(iat).find({jat, r});
                if (bare_iter == bare_coulomb.at(iat).end())
                    throw std::runtime_error("Wc R vector is absent from bare Coulomb map");
                if (bare_iter->second.shape.size() != 2
                    || bare_iter->second.shape[0] != nrow
                    || bare_iter->second.shape[1] != ncol)
                    throw std::runtime_error("Wc and bare Coulomb dimensions differ in " + file.string());

                RI::Tensor<Complex> tensor({nrow, ncol});
                std::size_t row = 0, col = 0;
                double re = 0.0, im = 0.0;
                for (std::size_t inz = 0; inz != nnz; ++inz)
                {
                    if (!(input >> row >> col >> re >> im) || row == 0 || col == 0
                        || row > nrow || col > ncol)
                        throw std::runtime_error("invalid MatrixMarket entry in " + file.string());
                    tensor(row - 1, col - 1) = Complex(re, im);
                }
                tensor += bare_iter->second;
                screened[iat][{jat, r}] = std::move(tensor);
            }
        }
    }
    return screened;
}

void remap_to_nearest_bvk_cell(TensorMap<Complex> &tensors,
                               const librpa_int::Dataset &dataset)
{
    std::map<int, std::array<double, 3>> positions;
    for (const auto &[iat, position] : dataset.atoms.coords)
        positions[static_cast<int>(iat)] = {position.x, position.y, position.z};

    RI::Cell_Nearest<int, int, 3, double, 3> nearest;
    nearest.init(positions, dataset.pbc.latvec_array, dataset.pbc.period_array);
    for (int iat = 0; iat != static_cast<int>(dataset.atoms.size()); ++iat)
    {
        for (int jat = 0; jat != static_cast<int>(dataset.atoms.size()); ++jat)
        {
            for (const auto &rv : dataset.pbc.Rlist)
            {
                const Cell old_cell{rv.x, rv.y, rv.z};
                double distance = 0.0;
                const Cell new_cell = nearest.cell_nearest_direction(
                    iat, jat, old_cell, distance);
                if (new_cell != old_cell)
                    move_tensor(tensors, iat, jat, old_cell, new_cell);
            }
        }
    }
}

} // namespace libbse
