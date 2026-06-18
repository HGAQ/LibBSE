#include "Chi_0.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <valarray>


namespace LibBSE{
    constexpr double threshold_C = 0.0;
    constexpr double threshold_G = 0.0;

    struct BasisInfo{
        std::vector<int> atoms;
        std::map<int, int> n_basis;
        std::map<int, int> n_aux_basis;
        std::map<int, int> first_basis;
    };

    FileAssignment assign_chi0_file_to_rank(int file_order,
                                            int file_count,
                                            const MpiComm& Comm)
    {
        FileAssignment assignment;
        const int mpi_size = Comm.LibBSE_MPI_size();
        const int mpi_rank = Comm.LibBSE_MPI_rank();

        if (file_count <= 0 || mpi_size <= 0) {
            return assignment;
        }

        if (mpi_size >= file_count) {
            // Same policy as Read_aims_output.cpp: if there are more ranks
            // than files, a contiguous rank group shares one large file by
            // splitting its inner blocks.
            const int base_ranks = mpi_size / file_count;
            const int extra_ranks = mpi_size % file_count;
            assignment.rank_count = base_ranks + (file_order < extra_ranks ? 1 : 0);
            assignment.first_rank = file_order * base_ranks
                                  + (file_order < extra_ranks ? file_order : extra_ranks);
            assignment.local_rank = mpi_rank - assignment.first_rank;
            assignment.read_file = assignment.local_rank >= 0
                                && assignment.local_rank < assignment.rank_count;
        }
        else {
            // If there are more files than ranks, each file has one owner rank
            // and later files are assigned round-robin.
            assignment.first_rank = file_order % mpi_size;
            assignment.rank_count = 1;
            assignment.local_rank = 0;
            assignment.read_file = mpi_rank == assignment.first_rank;
        }
        return assignment;
    }

    void record_count(std::vector<int>& count_min,
                      std::vector<int>& count_max,
                      int atom,
                      int count,
                      const char* name)
    {
        if (atom < 0 || atom >= static_cast<int>(count_min.size())) {
            throw std::runtime_error(std::string("LibBSE[Chi0]: atom index is outside geometry.in while reading ")
                                   + name);
        }
        if (count <= 0) {
            throw std::runtime_error(std::string("LibBSE[Chi0]: invalid ") + name);
        }

        if (count_max[static_cast<std::size_t>(atom)] != 0
            && count_max[static_cast<std::size_t>(atom)] != count) {
            throw std::runtime_error(std::string("LibBSE[Chi0]: inconsistent local ") + name);
        }

        count_min[static_cast<std::size_t>(atom)] =
            std::min(count_min[static_cast<std::size_t>(atom)], count);
        count_max[static_cast<std::size_t>(atom)] =
            std::max(count_max[static_cast<std::size_t>(atom)], count);
    }

    void finalize_basis_info(const Enviroment& Enviro,
                             const MpiComm& Comm,
                             const std::vector<int>& local_basis_min,
                             const std::vector<int>& local_basis_max,
                             const std::vector<int>& local_aux_min,
                             const std::vector<int>& local_aux_max,
                             BasisInfo& info)
    {
        const int n_atom = Enviro.n_atom;
        std::vector<int> global_basis_min(static_cast<std::size_t>(n_atom), 0);
        std::vector<int> global_basis_max(static_cast<std::size_t>(n_atom), 0);
        std::vector<int> global_aux_min(static_cast<std::size_t>(n_atom), 0);
        std::vector<int> global_aux_max(static_cast<std::size_t>(n_atom), 0);

        MPI_Allreduce(local_basis_min.data(), global_basis_min.data(),
                      n_atom, MPI_INT, MPI_MIN, Comm.LibBSE_MPI_raw());
        MPI_Allreduce(local_basis_max.data(), global_basis_max.data(),
                      n_atom, MPI_INT, MPI_MAX, Comm.LibBSE_MPI_raw());
        MPI_Allreduce(local_aux_min.data(), global_aux_min.data(),
                      n_atom, MPI_INT, MPI_MIN, Comm.LibBSE_MPI_raw());
        MPI_Allreduce(local_aux_max.data(), global_aux_max.data(),
                      n_atom, MPI_INT, MPI_MAX, Comm.LibBSE_MPI_raw());

        info.atoms.clear();
        info.n_basis.clear();
        info.n_aux_basis.clear();
        info.first_basis.clear();

        int offset = 0;
        for (int atom = 0; atom < n_atom; ++atom) {
            if (global_basis_max[static_cast<std::size_t>(atom)] <= 0
                || global_basis_min[static_cast<std::size_t>(atom)] == std::numeric_limits<int>::max()) {
                throw std::runtime_error("LibBSE[Chi0]: missing AO basis count in Cs_data");
            }
            if (global_basis_min[static_cast<std::size_t>(atom)]
                != global_basis_max[static_cast<std::size_t>(atom)]) {
                throw std::runtime_error("LibBSE[Chi0]: inconsistent AO basis count across Cs_data files");
            }

            // n_aux_basis is used for diagnostics and sanity checks; Cs_data
            // gives it on the first atom of each block, so it should also be
            // present for ordinary all-pair LRI output.
            if (global_aux_max[static_cast<std::size_t>(atom)] > 0
                && global_aux_min[static_cast<std::size_t>(atom)]
                   != global_aux_max[static_cast<std::size_t>(atom)]) {
                throw std::runtime_error("LibBSE[Chi0]: inconsistent aux basis count across Cs_data files");
            }

            info.atoms.push_back(atom);
            info.n_basis[atom] = global_basis_max[static_cast<std::size_t>(atom)];
            info.n_aux_basis[atom] = global_aux_max[static_cast<std::size_t>(atom)];
            info.first_basis[atom] = offset;
            offset += info.n_basis.at(atom);
        }

        if (Enviro.n_basis > 0 && offset != Enviro.n_basis) {
            throw std::runtime_error("LibBSE[Chi0]: atom-resolved AO basis count does not match n_basis");
        }
    }

    std::array<std::array<double, 3>, 3> make_lattice(const Enviroment& Enviro){
        if (Enviro.lattice_vect.size() != 3) {
            throw std::runtime_error("LibBSE[Chi0]: lattice vectors are missing");
        }
        std::array<std::array<double, 3>, 3> lat{};
        for (int i = 0; i < 3; ++i) {
            if (Enviro.lattice_vect[i].size() != 3) {
                throw std::runtime_error("LibBSE[Chi0]: lattice vector dimension is invalid");
            }
            for (int j = 0; j < 3; ++j) {
                lat[i][j] = Enviro.lattice_vect[i][j];
            }
        }
        return lat;
    }
    
    Chi0Cell make_period(const Enviroment& Enviro){
        if (Enviro.k_point_dim.size() != 3) {
            throw std::runtime_error("LibBSE[Chi0]: k-point grid dimensions are missing");
        }
        Chi0Cell period{};
        for (int i = 0; i < 3; ++i) {
            if (Enviro.k_point_dim[i] <= 0) {
                throw std::runtime_error("LibBSE[Chi0]: k-point grid dimension must be positive");
            }
            period[i] = Enviro.k_point_dim[i];
        }
        return period;
    }
    std::vector<Chi0Cell> make_period_cells(const Chi0Cell& period){
        std::vector<Chi0Cell> cells;
        cells.reserve(static_cast<std::size_t>(period[0] * period[1] * period[2]));
        // LibRI will fold keys by period.  Using the compact 0..N-1 cell
        // list gives one Green block for each periodic image in the k mesh.
        for (int n1 = 0; n1 < period[0]; ++n1) {
            for (int n2 = 0; n2 < period[1]; ++n2) {
                for (int n3 = 0; n3 < period[2]; ++n3) {
                    cells.push_back({n1, n2, n3});
                }
            }
        }
        return cells;
    }
    std::map<int, std::array<double, 3>> make_atom_positions(const Enviroment& Enviro,
                                                             const BasisInfo& basis){
        std::map<int, std::array<double, 3>> atoms_pos;
        for (const int atom : basis.atoms) {
            if (atom < 0 || atom >= static_cast<int>(Enviro.atoms_pos.size())) {
                throw std::runtime_error("LibBSE[Chi0]: geometry.in atom positions are missing");
            }
            const std::vector<double>& pos = Enviro.atoms_pos[static_cast<std::size_t>(atom)].second;
            if (pos.size() != 3) {
                throw std::runtime_error("LibBSE[Chi0]: geometry.in atom position dimension is invalid");
            }
            // LibRI expects positions in the same Cartesian unit as latvec.
            // read_geometry_in() has already converted atom_frac / atom lines
            // into the stru_out lattice-vector unit before storing atoms_pos.
            atoms_pos[atom] = {pos[0], pos[1], pos[2]};
        }
        return atoms_pos;
    }

    Chi0TensorMap read_Cs_data_libri(const Enviroment& Enviro,
                                     const MpiComm& Comm,
                                     BasisInfo& basis)
    {
        if (Enviro.dataset_dir.empty()) {
            throw std::runtime_error("LibBSE[Chi0]: dataset_dir is empty; cannot read Cs_data");
        }
        if (Enviro.n_atom <= 0) {
            throw std::runtime_error("LibBSE[Chi0]: n_atom is not initialized before reading Cs_data");
        }

        const fs::path directory(Enviro.dataset_dir);
        const std::vector<IndexedFile> files = find_indexed_files(directory, "Cs_data_", ".txt");
        if (files.empty()) {
            throw std::runtime_error("LibBSE[Chi0]: no Cs_data_*.txt files found");
        }

        Chi0TensorMap Cs_libri;
        const int unset_count = std::numeric_limits<int>::max();
        std::vector<int> local_basis_min(static_cast<std::size_t>(Enviro.n_atom), unset_count);
        std::vector<int> local_basis_max(static_cast<std::size_t>(Enviro.n_atom), 0);
        std::vector<int> local_aux_min(static_cast<std::size_t>(Enviro.n_atom), unset_count);
        std::vector<int> local_aux_max(static_cast<std::size_t>(Enviro.n_atom), 0);
        std::size_t local_blocks = 0;

        for (int ifile = 0; ifile < static_cast<int>(files.size()); ++ifile) {
            const IndexedFile& file = files[static_cast<std::size_t>(ifile)];
            const FileAssignment assignment =
                assign_chi0_file_to_rank(ifile, static_cast<int>(files.size()), Comm);
            if (!assignment.read_file || fs::file_size(file.path) == 0) {
                continue;
            }

            std::ifstream input(file.path);
            if (!input) {
                throw std::runtime_error("LibBSE[Chi0]: failed to open " + file.path.string());
            }

            int n_atom_file = 0;
            int n_cell = 0;
            input >> n_atom_file >> n_cell;
            if (!input) {
                throw std::runtime_error("LibBSE[Chi0]: failed to read Cs_data header " + file.path.string());
            }
            if (n_atom_file != Enviro.n_atom) {
                throw std::runtime_error("LibBSE[Chi0]: Cs_data n_atom does not match geometry.in");
            }

            int iblock = 0;
            while (true) {
                int i_atom_raw = 0;
                int j_atom_raw = 0;
                int n1 = 0;
                int n2 = 0;
                int n3 = 0;
                int n_basis_i = 0;
                int n_basis_j = 0;
                int n_aux_basis_i = 0;

                if (!(input >> i_atom_raw >> j_atom_raw
                            >> n1 >> n2 >> n3
                            >> n_basis_i >> n_basis_j >> n_aux_basis_i)) {
                    if (input.eof()) {
                        break;
                    }
                    throw std::runtime_error("LibBSE[Chi0]: failed to read Cs_data block header "
                                           + file.path.string());
                }

                // FHI-aims Cs_data atom ids are 1-based.  LibRI uses the 0-based
                // atom ids used by geometry.in order inside this LibBSE bridge.
                const int I = i_atom_raw - 1;
                const int J = j_atom_raw - 1;
                if (I < 0 || J < 0 || I >= Enviro.n_atom || J >= Enviro.n_atom) {
                    throw std::runtime_error("LibBSE[Chi0]: Cs_data atom index is outside geometry.in");
                }

                const std::size_t ni = static_cast<std::size_t>(n_basis_i);
                const std::size_t nj = static_cast<std::size_t>(n_basis_j);
                const std::size_t naux = static_cast<std::size_t>(n_aux_basis_i);
                const std::size_t nij = ni * nj;
                const std::size_t block_size = nij * naux;
                if (block_size == 0) {
                    throw std::runtime_error("LibBSE[Chi0]: invalid Cs_data block size");
                }

                // Only one rank stores a block before calling rpa.set_Cs().
                // LibRI/LibComm will redistribute the local blocks to the
                // atom/period ownership required by the RPA loops.
                const bool keep_block =
                    (iblock % assignment.rank_count == assignment.local_rank);

                std::shared_ptr<std::valarray<Chi0Data>> data;
                if (keep_block) {
                    record_count(local_basis_min, local_basis_max, I, n_basis_i, "AO basis count");
                    record_count(local_basis_min, local_basis_max, J, n_basis_j, "AO basis count");
                    record_count(local_aux_min, local_aux_max, I, n_aux_basis_i, "aux basis count");
                    data = std::make_shared<std::valarray<Chi0Data>>(block_size);
                }

                // Cs_data is written as value[(i*nj+j)*naux + mu].
                // LibRI wants tensor(mu,i,j), so write the transposed position
                // immediately while reading and avoid keeping an RIBlock copy.
                // For blocks owned by another rank we still consume the values
                // from the file stream but do not store them in memory.
                for (std::size_t ij = 0; ij < nij; ++ij) {
                    for (std::size_t mu = 0; mu < naux; ++mu) {
                        double value = 0.0;
                        input >> value;
                        if (!input) {
                            throw std::runtime_error("LibBSE[Chi0]: failed to read Cs_data value "
                                                   + file.path.string());
                        }
                        if (keep_block) {
                            (*data)[mu * nij + ij] = value;
                        }
                    }
                }

                if (keep_block) {
                    Cs_libri[I][{J, Chi0Cell{n1, n2, n3}}] =
                        RI::Tensor<Chi0Data>({naux, ni, nj}, data);
                    ++local_blocks;
                }
                ++iblock;
            }
        }

        finalize_basis_info(Enviro, Comm,
                            local_basis_min, local_basis_max,
                            local_aux_min, local_aux_max,
                            basis);

        LibBSE_printf_root(Comm, "Read Cs_data into LibRI tensors: blocks per rank=%zu\n", local_blocks);
        return Cs_libri;
    }


    const BandEntry* find_band_entry(const Enviroment& Enviro,
                                     const int k_point,
                                     const int spin,
                                     const int state){
        for (const BandVect& bands : Enviro.KS_Band) {
            if (bands.i_k_point == k_point && bands.i_band_spin == spin) {
                const int index = state - 1;
                if (index < 0 || index >= static_cast<int>(bands.Band.size())) {
                    return nullptr;
                }
                return &bands.Band[static_cast<std::size_t>(index)];
            }
        }
        return nullptr;
    }


    double phase_angle(const Enviroment& Enviro, const int k_point, const Chi0Cell& R){
        if (k_point <= 0 || k_point > static_cast<int>(Enviro.k_point_list.size())) {
            throw std::runtime_error("LibBSE[Chi0]: k point index is outside k_point_list");
        }
        std::array<double, 3> R_cart{0.0, 0.0, 0.0};
        for (int a = 0; a < 3; ++a) {
            for (int x = 0; x < 3; ++x) {
                R_cart[x] += static_cast<double>(R[a]) * Enviro.lattice_vect[a][x];
            }
        }
        // stru_out stores k vectors in reciprocal-space Cartesian units.
        // Therefore k.R already contains the 2*pi factor from the reciprocal
        // lattice; no extra TwoPi is needed here.
        double angle = 0.0;
        for (int x = 0; x < 3; ++x) {
            angle += Enviro.k_point_list[static_cast<std::size_t>(k_point - 1)][x] * R_cart[x];
        }
        return -angle;
    }


    double occupation_unit(const Enviroment& Enviro){
        // FHI-aims spin-unpolarized output uses occupation 0..2, while
        // spin-polarized channels usually use 0..1.  Normalize to f in [0,1]
        // before selecting occupied/unoccupied Green-function branches.
        return Enviro.n_band_spin == 1 ? 2.0 : 1.0;
    }


    void add_ks_block_to_green(const Enviroment& Enviro,
                               const BasisInfo& basis,
                               const std::vector<Chi0Cell>& cells,
                               const KSBlock& ks,
                               const bool positive_branch,
                               const double tau,
                               Chi0TensorMap& G_libri){
        if (ks.value.size() != static_cast<std::size_t>(Enviro.n_basis)) {
            throw std::runtime_error("LibBSE[Chi0]: KS eigenvector length does not match n_basis");
        }
        const BandEntry* band = find_band_entry(Enviro, ks.i_k_point, ks.i_band_spin, ks.i_state);
        if (band == nullptr) {
            throw std::runtime_error("LibBSE[Chi0]: cannot find band entry for KS eigenvector");
        }
        const double occ_unit = occupation_unit(Enviro);
        const double occ = std::clamp(band->band_occ / occ_unit, 0.0, 1.0);
        const double branch_weight = positive_branch ? (1.0 - occ) : occ;
        if (branch_weight == 0.0) {
            return;
        }
        // This is the same branch split as LibRPA's space-time chi0 path:
        // positive tau uses the unoccupied part, negative tau the occupied
        // part.  The small clamp avoids numerical blow-up for the wrong
        // branch if a state sits on the other side of E_Fermi.
        const double signed_tau = positive_branch ? tau : -tau;
        double exponent = -signed_tau * (band->E_band - Enviro.E_Fermi);
        if (exponent > 0.0) {
            exponent = 0.0;
        }
        double scale = std::exp(exponent) * branch_weight
                     / static_cast<double>(std::max(1, Enviro.n_k_point));
        if (!positive_branch) {
            scale *= -1.0;
        }
        for (const Chi0Cell& R : cells) {
            const double angle = phase_angle(Enviro, ks.i_k_point, R);
            const std::complex<double> phase(std::cos(angle), std::sin(angle));
            for (const int I : basis.atoms) {
                const int nI = basis.n_basis.at(I);
                const int firstI = basis.first_basis.at(I);
                for (const int J : basis.atoms) {
                    const int nJ = basis.n_basis.at(J);
                    const int firstJ = basis.first_basis.at(J);
                    RI::Tensor<Chi0Data>& block = G_libri[I][{J, R}];
                    if (block.empty()) {
                        auto data = std::make_shared<std::valarray<Chi0Data>>(
                            static_cast<std::size_t>(nI * nJ));
                        block = RI::Tensor<Chi0Data>(
                            {static_cast<std::size_t>(nI), static_cast<std::size_t>(nJ)},
                            data);
                    }
                    for (int i = 0; i < nI; ++i) {
                        const std::complex<double> ci = ks.value[static_cast<std::size_t>(firstI + i)];
                        for (int j = 0; j < nJ; ++j) {
                            const std::complex<double> cj = ks.value[static_cast<std::size_t>(firstJ + j)];
                            (*block.data)[static_cast<std::size_t>(i * nJ + j)]
                                += (phase * scale * ci * std::conj(cj)).real();
                        }
                    }
                }
            }
        }
    }
    Chi0TensorMap make_G_libri(const Enviroment& Enviro,
                               const BasisInfo& basis,
                               const Chi0Cell& period,
                               const bool positive_branch,
                               const double tau){
        Chi0TensorMap G_libri;
        const std::vector<Chi0Cell> cells = make_period_cells(period);
        for (const KSBlock& ks : Enviro.local_KS_eigenvector) {
            add_ks_block_to_green(Enviro, basis, cells, ks, positive_branch, tau, G_libri);
        }
        return G_libri;
    }
    

    int calculate_chi0(const Enviroment& Enviro, MpiComm Comm){
        LibBSE_printf_root(Comm, "Start calculating chi0...\n");

        try{
            BasisInfo basis;
            Chi0TensorMap Cs_libri = read_Cs_data_libri(Enviro, Comm, basis);
            const auto atoms_pos = make_atom_positions(Enviro, basis);
            const auto latvec = make_lattice(Enviro);
            const Chi0Cell period = make_period(Enviro);

            // Step 1: create the LibRI RPA object and pass parallel/lattice data.
            // This mirrors LibRPA's rpa.set_parallel(comm, atoms_pos, latvec, period).
            RI::RPA<int, int, 3, Chi0Data> rpa;
            rpa.set_parallel(Comm.LibBSE_MPI_raw(), atoms_pos, latvec, period);

            // Step 2: hand Cs_data blocks to LibRI.  This is the official
            // wrapper used by LibRPA; inside LibRI it also performs the
            // periodic folding, MPI redistribution and threshold filtering.
            rpa.set_Cs(Cs_libri, threshold_C);

            // Step 3/4: build the Green tensors and register both branches.
            // For a full chi0(omega) implementation this block should be placed
            // inside a loop over imaginary-time grid points.  tau=0.0 keeps this
            // first LibRI bridge compact while wiring the exact set_Gs_pos and
            // set_Gs_neg interfaces used by LibRPA.
            const double tau = 0.0;
            Chi0TensorMap Gpos = make_G_libri(Enviro, basis, period, true, tau);
            Chi0TensorMap Gneg = make_G_libri(Enviro, basis, period, false, tau);
            rpa.set_Gs_pos(Gpos, threshold_G);
            rpa.set_Gs_neg(Gneg, threshold_G);

            LibBSE_printf_root(Comm,
                "LibRI RPA setup finished: Cs atom blocks=%zu, Gpos atom blocks=%zu, Gneg atom blocks=%zu\n",
                Cs_libri.size(), Gpos.size(), Gneg.size());
        }
        catch (const std::exception& e){
            LibBSE_printf_root(Comm, "%s\n", e.what());
            return 1;
        }
        return 0;
    }
}
