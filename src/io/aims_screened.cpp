#include "aims_screened.h"
#include "interface/librpa_api.h"
#include "chi0_screening.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace libbse {
namespace {
namespace fs = std::filesystem;
constexpr double q_tolerance = 1.e-8;
struct Header {
    int naux = 0, nfreq = 0, iq = 0;
    KPoint q{};
};
Header read_header(const fs::path &file, const std::string &expected) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot open aims W file: " + file.string());
    Header h;
    bool quantity = false, dimensions = false, coordinates = false, units = false;
    std::string line;
    while (std::getline(in, line) && !line.empty() && line[0] == '#') {
        if (line.rfind("# quantity:", 0) == 0) {
            std::istringstream s(line.substr(11));
            std::string value; s >> value;
            quantity = value == expected;
        } else if (line.rfind("# n_basbas n_freq q_index:", 0) == 0) {
            std::istringstream s(line.substr(line.find(':') + 1));
            dimensions = bool(s >> h.naux >> h.nfreq >> h.iq);
        } else if (line.rfind("# q_fractional:", 0) == 0) {
            std::istringstream s(line.substr(line.find(':') + 1));
            coordinates = bool(s >> h.q[0] >> h.q[1] >> h.q[2]);
        } else if (line.find("Original RI auxiliary basis; atomic units;") != std::string::npos) {
            units = true;
        }
    }
    if (!quantity || !dimensions || !coordinates || !units || h.naux <= 0 || h.nfreq <= 0)
        throw std::runtime_error("invalid response header (expected aims RI basis and atomic units): " + file.string());
    return h;
}
double distance(double a, double b) {
    const double x = a - b;
    return std::abs(x - std::round(x));
}
int grid_index(const KPoint &q, const std::vector<librpa_int::Vector3_Order<double>> &grid) {
    for (std::size_t i = 0; i < grid.size(); ++i)
        if (distance(q[0], grid[i].x) < q_tolerance && distance(q[1], grid[i].y) < q_tolerance
            && distance(q[2], grid[i].z) < q_tolerance) return static_cast<int>(i);
    throw std::runtime_error("aims W q point does not belong to the coarse GW/RI mesh");
}
// The file frequency index is not a physical frequency. Select by |omega|,
// even when the producer's quadrature nodes are not sorted by index.
std::pair<int, double> lowest_frequency(const fs::path &file, int nfreq) {
    std::ifstream in(file);
    std::map<int, double> frequencies;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream s(line);
        int iw; double omega;
        if (!(s >> iw >> omega) || iw < 1 || iw > nfreq || !std::isfinite(omega))
            throw std::runtime_error("invalid frequency in " + file.string());
        const auto [it, inserted] = frequencies.emplace(iw, omega);
        if (!inserted && it->second != omega)
            throw std::runtime_error("inconsistent frequency index in " + file.string());
    }
    if (frequencies.size() != static_cast<std::size_t>(nfreq))
        throw std::runtime_error("missing frequency nodes in " + file.string());
    return *std::min_element(frequencies.begin(), frequencies.end(),
        [](const auto &a, const auto &b) { return std::abs(a.second) < std::abs(b.second); });
}
} // namespace

TensorMap<Complex> read_aims_screened_interaction(
    const InputParameters &options, librpa_int::Dataset &dataset,
    const std::vector<int> &local_i_atoms, const std::vector<int> &local_j_atoms, Chi0Screening *screening) {
    Chi0Screening owned(options,dataset); if(!screening) screening=&owned;
    if (options.input_format != "fhi_aims")
        throw std::runtime_error("screened_format fhi_aims_w requires matching FHI-aims RI/KS input");
    const std::string quantity = options.screened_format == "fhi_aims_chi0" ? "chi0" : "w";
    const auto grid = librpa_int::build_uniform_kmesh_frac(dataset.pbc.period);
    std::map<int, std::vector<fs::path>> files;
    std::map<int, Header> headers;
    const std::regex pattern("periodic_gw_" + quantity + "_q_([0-9]+)_rank_([0-9]+)\\.dat");
    for (const auto &entry : fs::directory_iterator(options.screened_dir)) {
        std::smatch match;
        const auto name = entry.path().filename().string();
        if (!entry.is_regular_file() || !std::regex_match(name, match, pattern)) continue;
        const auto h = read_header(entry.path(), quantity);
        if (h.naux != static_cast<int>(dataset.basis_aux.nb_total) || h.iq != std::stoi(match[1]))
            throw std::runtime_error("aims W auxiliary dimension/q index mismatch: " + entry.path().string());
        const int iq = grid_index(h.q, grid);
        const auto [it, inserted] = headers.emplace(iq, h);
        if (!inserted && (it->second.iq != h.iq || it->second.nfreq != h.nfreq))
            throw std::runtime_error("inconsistent aims W rank-file headers");
        files[iq].push_back(entry.path());
    }
    if (files.empty()) throw std::runtime_error("no matching periodic_gw matrix rank files in " + options.screened_dir);

    // Construct explicit q stars from exported coordinates. Only time reversal
    // W(-q,iw)=conj(W(q,iw)) can be inferred without auxiliary-basis rotations.
    // Never treat a general space-group star as a q/-q pair: request full-BZ
    // (symmetry none) or inversion-only aims output if another q is missing.
    auto pbc = dataset.pbc;
    pbc.map_irk_ks.clear();
    int restored = 0;
    for (int iq = 0; iq < static_cast<int>(grid.size()); ++iq) {
        int ir = iq;
        if (files.count(ir) == 0) {
            ir = grid_index({-grid[iq].x, -grid[iq].y, -grid[iq].z}, grid);
            if (files.count(ir) == 0)
                throw std::runtime_error("incomplete aims W q mesh: use periodic_gw_optimize_kgrid_symmetry none or inverse");
            ++restored;
        }
        const auto q = grid[ir] * pbc.G;
        const auto q_full = grid[iq] * pbc.G;
        pbc.map_irk_ks[q].push_back(q_full);
    }

    LibRPA_API::ScreenedQBlocks wq;
    double selected_omega = std::numeric_limits<double>::quiet_NaN();
    const std::size_t naux = dataset.basis_aux.nb_total;
    for (auto &[iq, paths] : files) {
        std::sort(paths.begin(), paths.end());
        // Empty BLACS rank blocks can have headers but no rows. Find a nonempty
        // block to identify the frequency grid; completeness is checked below.
        std::pair<int, double> node{};
        bool found = false;
        for (const auto &path : paths) {
            std::ifstream in(path); std::string line;
            while (std::getline(in, line))
                if (!line.empty() && line[0] != '#') { found = true; break; }
            if (found) { node = lowest_frequency(path, headers.at(iq).nfreq); break; }
        }
        if (!found) throw std::runtime_error("aims W q point has no matrix entries");
        if (std::isfinite(selected_omega) && std::abs(selected_omega - node.second) > 1.e-12)
            throw std::runtime_error("aims W lowest frequencies differ between q points");
        selected_omega = node.second;
        std::vector<Complex> matrix(naux * naux);
        std::vector<unsigned char> seen(naux * naux, 0);
        std::size_t count = 0;
        for (const auto &path : paths) {
            std::ifstream in(path); std::string line;
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream s(line);
                int iw; double omega;
                if (!(s >> iw >> omega)) throw std::runtime_error("invalid aims W row in " + path.string());
                if (iw != node.first) continue;
                std::size_t mu, nu; double re, im;
                if (!(s >> mu >> nu >> re >> im) || mu == 0 || nu == 0 || mu > naux || nu > naux
                    || !std::isfinite(re) || !std::isfinite(im) || std::abs(omega - selected_omega) > 1.e-12)
                    throw std::runtime_error("invalid selected-frequency W entry in " + path.string());
                const auto index = (mu - 1) * naux + nu - 1;
                if (seen[index]++) throw std::runtime_error("duplicate aims W matrix entry: " + path.string());
                matrix[index] = {re, im}; ++count;
            }
        }
        if (count != naux * naux) throw std::runtime_error("missing aims W rank block/entries at q " + std::to_string(headers.at(iq).iq));
        if (quantity == "chi0") {
            librpa_int::Matz chi(naux, naux, librpa_int::MAJOR::ROW);
            std::copy(matrix.begin(), matrix.end(), chi.ptr());
            const auto screened = screening->screen(chi, grid[iq] * pbc.G, selected_omega);
            std::copy_n(screened.ptr(), matrix.size(), matrix.begin());
        }
        // Keep only the atom pairs needed on this MPI rank. Global aims indices
        // are one-based; the shared basis_out fixes their atom/local ordering.
        const auto q = grid[iq] * pbc.G;
        for (int iat : local_i_atoms) for (int jat : local_j_atoms) {
            const auto ni = dataset.basis_aux[iat], nj = dataset.basis_aux[jat];
            librpa_int::Matz block(ni, nj, librpa_int::MAJOR::ROW);
            for (std::size_t i = 0; i < ni; ++i) for (std::size_t j = 0; j < nj; ++j)
                block(i,j) = matrix[dataset.basis_aux.get_global_index(iat,i) * naux
                                   + dataset.basis_aux.get_global_index(jat,j)];
            wq[iat][jat][q] = std::move(block);
        }
    }
    if (dataset.comm_h.myid == 0)
        std::cout << "FHI-aims " << quantity << " -> full W: omega = " << selected_omega << " Ha (lowest |omega|; static approximation), "
                  << files.size() << " stored q, " << restored << " time-reversal partners; coarse mesh "
                  << pbc.period.x << 'x' << pbc.period.y << 'x' << pbc.period.z
                  << ", BSE k points " << dataset.mf_band.get_n_kpoints() << '\n';
    // Do not add V: the aims writer exports W = Wc + v_cut, in Hartree.
    // W(R,iw0) = (1/Nq) sum_q exp(-2*pi*i*q.R) W(q,iw0).
    // LibBSE's existing nearest-cell remap and LibRI contraction subsequently
    // evaluate sum_R exp(+2*pi*i*q_fine.R) W(R,iw0). This is spatial Fourier
    // interpolation on the coarse BvK cell, not frequency interpolation or a
    // new dielectric calculation on the fine BSE mesh. Hartree->Ry conversion
    // remains in the existing BSE contraction and must not be applied here.
    screening->pbc=pbc;
    return LibRPA_API::transform_screened_q_to_r(dataset, pbc, wq);
}
} // namespace libbse
