#include "utils/memory_views.h"
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

struct QpRecord
{
    KPoint k{};
    std::vector<double> occupations;
    std::vector<double> energies_ry;
    std::string source;
    double occupation_scale = 1.0;
};

double periodic_coordinate_distance(double left, double right)
{
    const double difference = left - right;
    return std::abs(difference - std::round(difference));
}

int match_kpoint(const KPoint &k,
                 const std::vector<librpa_int::Vector3_Order<double>> &grid,
                 double tolerance, const std::string &source)
{
    int match = -1;
    for (int ik = 0; ik != static_cast<int>(grid.size()); ++ik)
    {
        const auto &candidate = grid[static_cast<std::size_t>(ik)];
        if (periodic_coordinate_distance(k[0], candidate.x) <= tolerance
            && periodic_coordinate_distance(k[1], candidate.y) <= tolerance
            && periodic_coordinate_distance(k[2], candidate.z) <= tolerance)
        {
            if (match >= 0)
                throw std::runtime_error("ambiguous periodic k-point in " + source);
            match = ik;
        }
    }
    if (match < 0)
        throw std::runtime_error("QP k-point is absent from the BSE grid: " + source);
    return match;
}

int wavefunction_core_offset(const librpa_int::Dataset &dataset,
                             int ik, int nocc, int record_offset)
{
    // LibRPA QP files may contain only the calculated valence/conduction
    // window, while band_out and KS eigenvectors also contain core states.
    // Mean-field occupations therefore determine the wavefunction row offset.
    // Synthetic inputs without occupations retain the QP-record offset.
    int highest_occupied = -1;
    const double stored_weight_tolerance
        = PARAM.constants.occupation_tolerance
          / dataset.mf_band.get_n_kpoints();
    const auto &weights = dataset.mf_band.get_weight().at(0);
    for (int ib = 0; ib != dataset.mf_band.get_n_bands(); ++ib)
        if (weights(ik, ib) > stored_weight_tolerance)
            highest_occupied = ib;
    if (highest_occupied < 0) return record_offset;
    const int offset = highest_occupied + 1 - nocc;
    if (offset < 0)
        throw std::runtime_error("mean-field data contain fewer occupied bands than nocc");
    return offset;
}

void store_requested_bands(QuasiparticleBands &result, int ik,
                           const QpRecord &record,
                           const InputParameters &options,
                           const librpa_int::Dataset &dataset)
{
    if (record.occupations.size() != record.energies_ry.size())
        throw std::runtime_error("inconsistent QP record columns in " + record.source);
    const auto first_virtual = std::find_if(
        record.occupations.begin(), record.occupations.end(),
        [&record](double occupation)
        {
            return occupation * record.occupation_scale
                   < PARAM.constants.occupation_tolerance;
        });
    const int occupied_count = static_cast<int>(
        std::distance(record.occupations.begin(), first_virtual));
    if (occupied_count < options.nocc
        || static_cast<int>(record.energies_ry.size())
               < occupied_count + options.nvirt)
        throw std::runtime_error("not enough occupied/virtual QP bands in "
                                 + record.source);
    if (std::any_of(first_virtual, record.occupations.end(),
                    [&record](double occupation)
                    {
                        return occupation * record.occupation_scale
                               >= PARAM.constants.occupation_tolerance;
                    }))
        throw std::runtime_error("QP occupations are not an occupied-then-virtual window in "
                                 + record.source);

    const int record_offset = occupied_count - options.nocc;
    const int ncore_here = wavefunction_core_offset(
        dataset, ik, options.nocc, record_offset);
    if (result.ncore < 0) result.ncore = ncore_here;
    if (ncore_here != result.ncore)
        throw std::runtime_error("inconsistent wavefunction core-band offset in QP data");

    for (int ib = 0; ib != result.nbands; ++ib)
    {
        const std::size_t dst = static_cast<std::size_t>(ik) * result.nbands + ib;
        result.energies_ry[dst]
            = record.energies_ry[static_cast<std::size_t>(record_offset + ib)];
    }
}

QuasiparticleBands assign_qp_records(
    const std::vector<QpRecord> &records,
    const InputParameters &options,
    const librpa_int::Dataset &dataset,
    double tolerance)
{
    QuasiparticleBands result;
    result.ncore = -1;
    result.nk = dataset.mf_band.get_n_kpoints();
    result.nbands = options.nocc + options.nvirt;
    result.energies_ry.resize(static_cast<std::size_t>(result.nk) * result.nbands);
    std::vector<bool> assigned(static_cast<std::size_t>(result.nk), false);
    for (const QpRecord &record : records)
    {
        const int ik = match_kpoint(record.k, dataset.kfrac_band_list,
                                    tolerance, record.source);
        if (assigned[static_cast<std::size_t>(ik)])
            throw std::runtime_error("duplicate QP data for BSE k-point "
                                     + std::to_string(ik + 1));
        store_requested_bands(result, ik, record, options, dataset);
        assigned[static_cast<std::size_t>(ik)] = true;
    }
    const auto missing = std::find(assigned.begin(), assigned.end(), false);
    if (missing != assigned.end())
        throw std::runtime_error("missing QP data for BSE k-point "
                                 + std::to_string(std::distance(assigned.begin(), missing) + 1));
    return result;
}

fs::path data_file(const InputParameters &options, const char *default_name)
{
    const fs::path path(options.qp_data.empty() ? options.input_dir
                                                : options.qp_data);
    return fs::is_directory(path) ? path / default_name : path;
}

QuasiparticleBands read_fine_qp_bands(const InputParameters &options,
                                      const librpa_int::Dataset &dataset)
{
    const fs::path file = data_file(options, "GW_band_spin_1.dat");
    std::ifstream input(file);
    if (!input) throw std::runtime_error("cannot open GW band file: " + file.string());

    std::vector<QpRecord> records;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line))
    {
        ++line_number;
        if (line.empty()) continue;
        std::istringstream parser(line);
        int index = 0;
        QpRecord record;
        record.source = file.string() + ":" + std::to_string(line_number);
        // The historical GW_band_spin file stores occupations including the
        // uniform k weight.  Undo that convention only for this format.
        record.occupation_scale = dataset.mf_band.get_n_kpoints();
        if (!(parser >> index >> record.k[0] >> record.k[1] >> record.k[2]))
            throw std::runtime_error("invalid k-point header in " + file.string());
        double occupation = 0.0;
        double energy_ev = 0.0;
        while (parser >> occupation >> energy_ev)
        {
            record.occupations.push_back(occupation);
            record.energies_ry.push_back(energy_ev / PARAM.constants.ry_to_ev);
        }
        records.push_back(std::move(record));
    }
    return assign_qp_records(records, options, dataset,
                             PARAM.constants.band_file_kpoint_tolerance);
}

QuasiparticleBands read_coarse_qp_bands(const InputParameters &options,
                                        const librpa_int::Dataset &dataset)
{
    const fs::path file = data_file(options, "energy_qp");
    std::ifstream input(file);
    if (!input)
        throw std::runtime_error("cannot open coarse-grid quasiparticle file: "
                                 + file.string());

    std::vector<QpRecord> records;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line))
    {
        ++line_number;
        if (line.find("K_point") == std::string::npos) continue;

        std::istringstream header(line);
        std::string label;
        char colon = '\0';
        int index = 0;
        QpRecord record;
        record.source = file.string() + ":" + std::to_string(line_number);
        if (!(header >> label >> index >> colon
                     >> record.k[0] >> record.k[1] >> record.k[2])
            || label != "K_point" || colon != ':')
            throw std::runtime_error("invalid k-point header in " + file.string());
        bool saw_state = false;
        while (std::getline(input, line))
        {
            ++line_number;
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
            record.occupations.push_back(occupation);
            record.energies_ry.push_back(
                qp_energy_ha * PARAM.constants.ha_to_ry);
        }
        if (!saw_state)
            throw std::runtime_error("empty QP block in " + record.source);
        records.push_back(std::move(record));
    }
    return assign_qp_records(records, options, dataset,
                             PARAM.constants.energy_qp_kpoint_tolerance);
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
    std::string format = options.qp_format;
    if (format == "auto")
    {
        const fs::path path(options.qp_data.empty() ? options.input_dir
                                                    : options.qp_data);
        if (fs::is_regular_file(path / "energy_qp")
            || (fs::is_regular_file(path) && path.filename() == "energy_qp"))
            format = "energy_qp";
        else if (options.bse_use_fine_kgrid == 1)
            format = "fine_band";
        else
            throw std::runtime_error(
                "cannot find LibRPA energy_qp; FHI-aims quasiparticle files "
                "are intentionally unsupported");
    }
    QuasiparticleBands result;
    if (format == "energy_qp")
        result = read_coarse_qp_bands(options, dataset);
    else if (format == "fine_band")
        result = read_fine_qp_bands(options, dataset);
    else
        throw std::invalid_argument("unsupported quasiparticle format: "
                                    + format);
    calculate_gaps(result, options);
    return result;
}

TensorMap<Complex> convert_lri_coefficients(librpa_int::Dataset &dataset)
{
    TensorMap<Complex> result;
    auto result_memory = watch_memory("RI.coefficients_complex", result);
    for (const auto &[iat, blocks] : dataset.cs_data.data_libri)
        for (const auto &[key, tensor] : blocks)
            result[static_cast<int>(iat)][{static_cast<int>(key.first), key.second}]
                = RI::Global_Func::convert<Complex>(tensor);
    MemoryTracker::instance().checkpoint();
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
    auto screened_memory = watch_memory("screening.W_R", screened);
    const fs::path wc_dir = options.screened_dir.empty()
        ? fs::path(options.input_dir).parent_path() / "librpa.d"
        : fs::path(options.screened_dir);
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
                // LibRPA writes the correlation part Wc = W - V.  At the
                // lowest minimax imaginary-frequency node used here, Fourier
                // linearity gives W(R,iw0) = V(R) + Wc(R,iw0).  This is the
                // static-limit real-space interaction contracted by LibRI in
                // the direct electron-hole kernel.
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
