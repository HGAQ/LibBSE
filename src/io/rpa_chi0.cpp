#include "rpa_chi0.h"
#include "chi0_screening.h"
#include "symmetry_matrix.h"
#include <algorithm>
#include <map>
#include "utils/memory_views.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace libbse
{
using librpa_int::Matz;
namespace
{
int rpa_grid_index(const std::array<double, 3> &rpa_q,
                   const std::vector<librpa_int::Vector3_Order<double>> &rpa_grid)
{
    for (std::size_t iq = 0; iq < rpa_grid.size(); ++iq)
    {
        const double rpa_difference[3] = {rpa_q[0] - rpa_grid[iq].x, rpa_q[1] - rpa_grid[iq].y,
                                          rpa_q[2] - rpa_grid[iq].z};
        bool matches = true;
        for (double component : rpa_difference)
        {
            if (std::abs(component - std::round(component)) > 1.e-8)
            {
                matches = false;
            }
        }
        if (matches)
        {
            return iq;
        }
    }
    throw std::runtime_error("LibRPA chi0 symmetry q outside consumer grid");
}

std::vector<Complex> read_rpa_representative_chi0(const std::filesystem::path &path, int naux,
                                                  int representative_id)
{
    std::ifstream rpa_input(path);
    std::string magic;
    int version = 0, file_naux = 0, target_id = 0, source_id = 0;
    if (!(rpa_input >> magic >> version >> file_naux >> target_id >> source_id) ||
        magic != "LIBRPA_CHI0_Q" || version != 1 || file_naux != naux ||
        target_id != representative_id || source_id != representative_id)
    {
        throw std::runtime_error("invalid LibRPA representative chi0 header: " + path.string());
    }
    std::vector<Complex> rpa_chi0_operator(std::size_t(naux) * naux);
    std::vector<bool> rpa_chi0_seen(rpa_chi0_operator.size(), false);
    for (std::size_t entry = 0; entry < rpa_chi0_operator.size(); ++entry)
    {
        int mu = 0, nu = 0;
        double real = 0, imag = 0;
        if (!(rpa_input >> mu >> nu >> real >> imag) || mu < 1 || nu < 1 || mu > naux ||
            nu > naux || !std::isfinite(real) || !std::isfinite(imag))
        {
            throw std::runtime_error("missing/invalid LibRPA representative chi0 entry");
        }
        const auto index = std::size_t(mu - 1) * naux + nu - 1;
        if (rpa_chi0_seen[index])
        {
            throw std::runtime_error("duplicate LibRPA representative chi0 entry");
        }
        rpa_chi0_seen[index] = true;
        rpa_chi0_operator[index] = {real, imag};
    }
    std::string extra;
    if (rpa_input >> extra)
    {
        throw std::runtime_error("extra LibRPA representative chi0 entries");
    }
    return rpa_chi0_operator;
}

TensorMap<Complex> read_rpa_symmetry_chi0(const InputParameters &opts, librpa_int::Dataset &ds,
                                          const std::vector<int> &rpa_i_atoms,
                                          const std::vector<int> &rpa_j_atoms,
                                          Chi0Screening &screening)
{
    const std::filesystem::path rpa_directory(opts.screened_dir);
    std::ifstream rpa_info(rpa_directory / "chi0_q.info");
    std::string magic;
    int version = 0, naux = 0, px = 0, py = 0, pz = 0;
    double omega = 0;
    if (!(rpa_info >> magic >> version >> omega >> naux >> px >> py >> pz) ||
        magic != "LIBRPA_CHI0_Q_GRID" || version != 1 || naux != int(ds.basis_aux.nb_total) ||
        px != ds.pbc.period.x || py != ds.pbc.period.y || pz != ds.pbc.period.z ||
        !std::isfinite(omega) || omega < 0)
    {
        throw std::runtime_error("incompatible LibRPA chi0_q.info");
    }

    // Validate the mapping against the consumer grid before loading large operators.
    const auto rpa_q_grid = librpa_int::build_uniform_kmesh_frac(ds.pbc.period);
    const auto rpa_symmetry = read_symmetry_matrices(rpa_directory / "rpa_symmetry_chi0.dat", naux,
                                                     rpa_q_grid.size(), "LIBRPA", "BARE_CHI0");
    std::vector<bool> rpa_q_seen(rpa_q_grid.size(), false);
    std::map<int, int> rpa_representative_indices;
    for (const auto &rpa_mapping : rpa_symmetry.q_mappings)
    {
        const int iq = rpa_grid_index(rpa_mapping.target_q, rpa_q_grid);
        const int irep = rpa_grid_index(rpa_mapping.representative_q, rpa_q_grid);
        if (rpa_q_seen[iq])
        {
            throw std::runtime_error("duplicate LibRPA chi0 target coordinate");
        }
        rpa_q_seen[iq] = true;
        const auto rpa_insertion =
            rpa_representative_indices.emplace(rpa_mapping.representative_id, irep);
        if (!rpa_insertion.second && rpa_insertion.first->second != irep)
        {
            throw std::runtime_error("inconsistent LibRPA chi0 representative coordinate");
        }
    }

    // Load each representative once; several target q points can share it.
    std::map<int, std::vector<Complex>> rpa_representative_chi0;
    auto rpa_representative_chi0_memory =
        watch_memory("rpa_representative_chi0", rpa_representative_chi0);
    for (const auto &[id, iq] : rpa_representative_indices)
    {
        rpa_representative_chi0[id] = read_rpa_representative_chi0(
            rpa_directory / ("chi0_q_" + std::to_string(id) + ".dat"), naux, id);
    }

    // Restore every target q explicitly, so the later Fourier transform needs no star expansion.
    auto rpa_pbc = ds.pbc;
    rpa_pbc.map_irk_ks.clear();
    LibRPA_API::ScreenedQBlocks rpa_w_q_blocks;
    auto rpa_w_q_blocks_memory = watch_memory("rpa_w_q_blocks", rpa_w_q_blocks);
    for (const auto &rpa_mapping : rpa_symmetry.q_mappings)
    {
        const auto rpa_delta = read_symmetry_delta(rpa_mapping, naux);
        auto rpa_delta_memory = watch_memory("rpa_delta", rpa_delta);
        const auto rpa_restored_chi0 =
            reconstruct_symmetry_operator(rpa_representative_chi0.at(rpa_mapping.representative_id),
                                          rpa_delta, naux, rpa_mapping.conjugate_representative);
        auto rpa_restored_chi0_memory = watch_memory("rpa_restored_chi0", rpa_restored_chi0);
        Matz rpa_chi0_q(naux, naux, librpa_int::MAJOR::ROW);
        auto rpa_chi0_q_memory = watch_memory("rpa_chi0_q", rpa_chi0_q);
        std::copy(rpa_restored_chi0.begin(), rpa_restored_chi0.end(), rpa_chi0_q.ptr());
        const auto rpa_q = rpa_q_grid[rpa_grid_index(rpa_mapping.target_q, rpa_q_grid)] * rpa_pbc.G;
        // Bare response is restored before the nonlinear dielectric inverse.
        // The target q determines Coulomb matrices and Gamma head/wing handling.
        const auto rpa_full_w = screening.screen(rpa_chi0_q, rpa_q, omega);
        auto rpa_full_w_memory = watch_memory("rpa_full_w", rpa_full_w);
        rpa_pbc.map_irk_ks[rpa_q].push_back(rpa_q);

        // Retain only the atom pairs needed by this MPI rank.
        for (int i : rpa_i_atoms)
        {
            for (int j : rpa_j_atoms)
            {
                Matz rpa_w_atom_pair(ds.basis_aux[i], ds.basis_aux[j], librpa_int::MAJOR::ROW);
                for (std::size_t a = 0; a < rpa_w_atom_pair.nr(); ++a)
                {
                    for (std::size_t b = 0; b < rpa_w_atom_pair.nc(); ++b)
                    {
                        rpa_w_atom_pair(a, b) = rpa_full_w(ds.basis_aux.get_global_index(i, a),
                                                           ds.basis_aux.get_global_index(j, b));
                    }
                }
                rpa_w_q_blocks[i][j][rpa_q] = std::move(rpa_w_atom_pair);
            }
        }
    }
    screening.pbc = rpa_pbc;
    if (ds.comm_h.is_root())
    {
        std::cout << "LibRPA representative chi0 + Delta -> full chi0 -> W: "
                  << rpa_representative_indices.size() << " representatives, " << rpa_q_grid.size()
                  << " q points; omega = " << omega << " Ha\n";
    }
    return LibRPA_API::transform_screened_q_to_r(ds, rpa_pbc, rpa_w_q_blocks);
}
} // namespace

// LibRPA's real-space file format is separate from the producer-neutral
// chi0 -> W calculation shared with the aims reader.
TensorMap<Complex> read_librpa_chi0(const InputParameters &opts, librpa_int::Dataset &ds,
                                    const std::vector<int> &rpa_i_atoms,
                                    const std::vector<int> &rpa_j_atoms, Chi0Screening *screening)
{
    Chi0Screening owned(opts, ds);
    if (!screening)
    {
        screening = &owned;
    }
    namespace fs = std::filesystem;
    if (fs::exists(fs::path(opts.screened_dir) / "chi0_q.info"))
    {
        return read_rpa_symmetry_chi0(opts, ds, rpa_i_atoms, rpa_j_atoms, *screening);
    }
    std::ifstream metadata(fs::path(opts.screened_dir) / "chi0_rf.info");
    std::string magic;
    int version, n, px, py, pz;
    double omega;
    if (!(metadata >> magic >> version >> omega >> n >> px >> py >> pz) ||
        magic != "LIBRPA_CHI0_RF" || version != 1 || n != int(ds.basis_aux.nb_total) ||
        px != ds.pbc.period.x || py != ds.pbc.period.y || pz != ds.pbc.period.z ||
        !std::isfinite(omega))
    {
        throw std::runtime_error("invalid/incompatible chi0_rf.info");
    }

    // Legacy input: assemble the complete auxiliary-basis response for each lattice cell.
    std::vector<Matz> rpa_chi0_r;
    auto rpa_chi0_r_memory = watch_memory("rpa_chi0_r", rpa_chi0_r);
    for (std::size_t ir = 0; ir < ds.pbc.Rlist.size(); ++ir)
    {
        Matz rpa_chi0_r_matrix(n, n, librpa_int::MAJOR::ROW);
        auto rpa_chi0_r_matrix_memory = watch_memory("rpa_chi0_r_matrix", rpa_chi0_r_matrix);
        const auto rpa_lattice_cell = ds.pbc.Rlist[ir];
        for (int i = 0; i < ds.basis_aux.n_atoms; ++i)
        {
            for (int j = 0; j < ds.basis_aux.n_atoms; ++j)
            {
                const auto path = fs::path(opts.screened_dir) /
                                  ("Chi0_Mu_" + std::to_string(i) + "_Nu_" + std::to_string(j) +
                                   "_iR_" + std::to_string(ir) + "_ifreq_0.mtx");
                std::ifstream in(path);
                std::string line;
                if (!std::getline(in, line) ||
                    line != "%%MatrixMarket matrix coordinate complex general")
                {
                    throw std::runtime_error("invalid chi0 MatrixMarket file: " + path.string());
                }
                bool found = false;
                int x = 0, y = 0, z = 0;
                while (std::getline(in, line) && !line.empty() && line[0] == '%')
                {
                    const auto pos = line.find('(');
                    if (pos != std::string::npos)
                    {
                        std::istringstream s(line.substr(pos + 1));
                        if (s >> x >> y >> z)
                        {
                            found = true;
                        }
                    }
                }
                int nr, nc, nz;
                std::istringstream dims(line);
                if (!found || x != rpa_lattice_cell.x || y != rpa_lattice_cell.y ||
                    z != rpa_lattice_cell.z || !(dims >> nr >> nc >> nz) ||
                    nr != int(ds.basis_aux[i]) || nc != int(ds.basis_aux[j]) || nz < 0 ||
                    nz > nr * nc)
                {
                    throw std::runtime_error("invalid chi0 block dimensions/R: " + path.string());
                }
                std::vector<bool> rpa_chi0_entry_seen(nr * nc, false);
                for (int k = 0; k < nz; ++k)
                {
                    int a, b;
                    double re, im;
                    if (!(in >> a >> b >> re >> im) || a < 1 || a > nr || b < 1 || b > nc ||
                        !std::isfinite(re) || !std::isfinite(im) ||
                        rpa_chi0_entry_seen[(a - 1) * nc + b - 1])
                    {
                        throw std::runtime_error("invalid/duplicate chi0 entry: " + path.string());
                    }
                    rpa_chi0_entry_seen[(a - 1) * nc + b - 1] = true;
                    rpa_chi0_r_matrix(ds.basis_aux.get_global_index(i, a - 1),
                                      ds.basis_aux.get_global_index(j, b - 1)) = {re, im};
                }
            }
        }
        rpa_chi0_r.push_back(std::move(rpa_chi0_r_matrix));
    }

    // Screen on the original q mesh before transforming W to real space.
    auto pbc = ds.pbc;
    pbc.map_irk_ks.clear();
    LibRPA_API::ScreenedQBlocks rpa_w_q_blocks;
    auto rpa_w_q_blocks_memory = watch_memory("rpa_w_q_blocks", rpa_w_q_blocks);
    const auto rpa_q_grid = librpa_int::build_uniform_kmesh_frac(pbc.period);
    for (const auto &rpa_q_fractional : rpa_q_grid)
    {
        const auto rpa_q_cartesian = rpa_q_fractional * pbc.G;
        pbc.map_irk_ks[rpa_q_cartesian].push_back(rpa_q_cartesian);
        Matz rpa_chi0_q(n, n, librpa_int::MAJOR::ROW);
        auto rpa_chi0_q_memory = watch_memory("rpa_chi0_q", rpa_chi0_q);
        // chi0(q)=sum_R exp(+2*pi*i*q.R) chi0(R), on the ORIGINAL coarse mesh.
        // Only W, after the nonlinear screening step, is interpolated to BSE k.
        for (std::size_t ir = 0; ir < rpa_chi0_r.size(); ++ir)
        {
            const auto rpa_lattice_cell = pbc.Rlist[ir];
            rpa_chi0_q += rpa_chi0_r[ir] *
                          std::exp(Complex(0, 2 * PARAM.constants.pi *
                                                  (rpa_q_fractional.x * rpa_lattice_cell.x +
                                                   rpa_q_fractional.y * rpa_lattice_cell.y +
                                                   rpa_q_fractional.z * rpa_lattice_cell.z)));
        }
        const auto rpa_full_w_q = screening->screen(rpa_chi0_q, rpa_q_cartesian, omega);
        auto rpa_full_w_q_memory = watch_memory("rpa_full_w_q", rpa_full_w_q);

        // Retain only the atom pairs needed by this MPI rank.
        for (int i : rpa_i_atoms)
        {
            for (int j : rpa_j_atoms)
            {
                Matz rpa_w_atom_pair(ds.basis_aux[i], ds.basis_aux[j], librpa_int::MAJOR::ROW);
                for (std::size_t a = 0; a < rpa_w_atom_pair.nr(); ++a)
                {
                    for (std::size_t b = 0; b < rpa_w_atom_pair.nc(); ++b)
                    {
                        rpa_w_atom_pair(a, b) = rpa_full_w_q(ds.basis_aux.get_global_index(i, a),
                                                             ds.basis_aux.get_global_index(j, b));
                    }
                }
                rpa_w_q_blocks[i][j][rpa_q_cartesian] = std::move(rpa_w_atom_pair);
            }
        }
    }
    if (ds.comm_h.myid == 0)
    {
        std::cout << "LibRPA chi0(R) -> chi0(q) -> W(q): omega = " << omega
                  << " Ha; LibRPA dielectric screening\n";
    }
    screening->pbc = pbc;
    return LibRPA_API::transform_screened_q_to_r(ds, pbc, rpa_w_q_blocks);
}
} // namespace libbse
