#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "Read_aims_output.h"

namespace fs = std::filesystem;

namespace LibBSE
{
    
    static FileAssignment assign_file_to_rank(int file_order, int file_count, const MpiComm& comm)
    {
        FileAssignment assignment;
        const int mpi_size = comm.LibBSE_MPI_size();
        const int mpi_rank = comm.LibBSE_MPI_rank();

        if (file_count <= 0 || mpi_size <= 0) {
            return assignment;
        }

        if (mpi_size >= file_count) {
            // More ranks than files:
            // split ranks into contiguous groups first, then split each file by blocks.
            // Example: 14 ranks, 10 files -> first 4 files get 2 ranks, rest get 1 rank.
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
            // More files than ranks:
            // each file has one owner rank; ranks read later files again round-robin.
            // Example: 10 ranks, 14 files -> rank 0 reads file 0 and file 10.
            assignment.first_rank = file_order % mpi_size;
            assignment.rank_count = 1;
            assignment.local_rank = 0;
            assignment.read_file = mpi_rank == assignment.first_rank;
        }
        return assignment;
    }

    static void bcast_vector_int(std::vector<int>& values, const MpiComm& comm)
    {
        int count = comm.LibBSE_MPI_is_root() ? static_cast<int>(values.size()) : 0;
        MPI_Bcast(&count, 1, MPI_INT, 0, comm.LibBSE_MPI_raw());
        if (!comm.LibBSE_MPI_is_root()) {
            values.resize(static_cast<std::size_t>(count));
        }
        if (count > 0) {
            MPI_Bcast(values.data(), count, MPI_INT, 0, comm.LibBSE_MPI_raw());
        }
    }

    static void bcast_vector_atompos(std::vector<AtomPos>& values, const MpiComm& comm)
    {
        int count = comm.LibBSE_MPI_is_root() ? static_cast<int>(values.size()) : 0;
        MPI_Bcast(&count, 1, MPI_INT, 0, comm.LibBSE_MPI_raw());
        if (!comm.LibBSE_MPI_is_root()) {
            values.resize(static_cast<std::size_t>(count));
        }

        for (int i = 0; i < count; ++i) {
            // Atom symbols are tiny, but still need an explicit length before
            // broadcasting the string payload.
            int name_size = comm.LibBSE_MPI_is_root()
                          ? static_cast<int>(values[static_cast<std::size_t>(i)].first.size())
                          : 0;
            MPI_Bcast(&name_size, 1, MPI_INT, 0, comm.LibBSE_MPI_raw());

            std::string name;
            if (comm.LibBSE_MPI_is_root()) {
                name = values[static_cast<std::size_t>(i)].first;
            }
            else {
                name.resize(static_cast<std::size_t>(name_size));
            }
            if (name_size > 0) {
                MPI_Bcast(&name[0], name_size, MPI_CHAR, 0, comm.LibBSE_MPI_raw());
            }

            int pos_size = comm.LibBSE_MPI_is_root()
                         ? static_cast<int>(values[static_cast<std::size_t>(i)].second.size())
                         : 0;
            MPI_Bcast(&pos_size, 1, MPI_INT, 0, comm.LibBSE_MPI_raw());
            if (!comm.LibBSE_MPI_is_root()) {
                values[static_cast<std::size_t>(i)].second.resize(static_cast<std::size_t>(pos_size));
            }
            if (pos_size > 0) {
                MPI_Bcast(values[static_cast<std::size_t>(i)].second.data(),
                          pos_size, MPI_DOUBLE, 0, comm.LibBSE_MPI_raw());
            }

            if (!comm.LibBSE_MPI_is_root()) {
                values[static_cast<std::size_t>(i)].first = std::move(name);
            }
        }
    }


    static void bcast_matrix_double(std::vector<std::vector<double>>& matrix,
                                    int rows,
                                    int cols,
                                    const MpiComm& comm)
    {
        std::vector<double> flat(static_cast<std::size_t>(rows * cols), 0.0);

        if (comm.LibBSE_MPI_is_root()) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    flat[static_cast<std::size_t>(i * cols + j)] = matrix[i][j];
                }
            }
        }

        if (!flat.empty()) {
            MPI_Bcast(flat.data(), rows * cols, MPI_DOUBLE, 0, comm.LibBSE_MPI_raw());
        }

        if (!comm.LibBSE_MPI_is_root()) {
            matrix.assign(static_cast<std::size_t>(rows),
                          std::vector<double>(static_cast<std::size_t>(cols), 0.0));
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    matrix[i][j] = flat[static_cast<std::size_t>(i * cols + j)];
                }
            }
        }
    }

 

    static void bcast_KS_band(Enviroment& Envir, const MpiComm& comm)
    {
        const int block_count = Envir.n_k_point * Envir.n_band_spin;
        std::vector<int> band_meta(static_cast<std::size_t>(2 * block_count), 0);
        std::vector<double> band_data(
            static_cast<std::size_t>(2 * block_count * Envir.n_band_state), 0.0);

        if (comm.LibBSE_MPI_is_root()) {
            for (int iblock = 0; iblock < block_count; ++iblock) {
                const BandVect& bands = Envir.KS_Band[static_cast<std::size_t>(iblock)];
                band_meta[static_cast<std::size_t>(2 * iblock)] = bands.i_k_point;
                band_meta[static_cast<std::size_t>(2 * iblock + 1)] = bands.i_band_spin;
                for (int istate = 0; istate < Envir.n_band_state; ++istate) {
                    const std::size_t index =
                        static_cast<std::size_t>(2 * (iblock * Envir.n_band_state + istate));
                    band_data[index] = bands.Band[static_cast<std::size_t>(istate)].band_occ;
                    band_data[index + 1] = bands.Band[static_cast<std::size_t>(istate)].E_band;
                }
            }
        }

        if (!band_meta.empty()) {
            MPI_Bcast(band_meta.data(), static_cast<int>(band_meta.size()),
                      MPI_INT, 0, comm.LibBSE_MPI_raw());
        }
        if (!band_data.empty()) {
            MPI_Bcast(band_data.data(), static_cast<int>(band_data.size()),
                      MPI_DOUBLE, 0, comm.LibBSE_MPI_raw());
        }

        if (!comm.LibBSE_MPI_is_root()) {
            Envir.KS_Band.clear();
            Envir.KS_Band.reserve(static_cast<std::size_t>(block_count));
            for (int iblock = 0; iblock < block_count; ++iblock) {
                BandVect bands;
                bands.i_k_point = band_meta[static_cast<std::size_t>(2 * iblock)];
                bands.i_band_spin = band_meta[static_cast<std::size_t>(2 * iblock + 1)];
                bands.Band.resize(static_cast<std::size_t>(Envir.n_band_state));
                for (int istate = 0; istate < Envir.n_band_state; ++istate) {
                    const std::size_t index =
                        static_cast<std::size_t>(2 * (iblock * Envir.n_band_state + istate));
                    bands.Band[static_cast<std::size_t>(istate)].band_occ = band_data[index];
                    bands.Band[static_cast<std::size_t>(istate)].E_band = band_data[index + 1];
                }
                Envir.KS_Band.push_back(std::move(bands));
            }
        }
    }

    int read_coulomb_group(const fs::path directory,
                                         std::vector<CoulombBlock>& local_coulomb_blocks,
                                         const MpiComm& comm,
                                         const std::string& prefix){
        const std::vector<IndexedFile> files = find_indexed_files(directory, prefix, ".txt");

        int n_aux_basis = 0;
        for (int ifile = 0; ifile < files.size(); ++ifile) {
            const IndexedFile& file = files[ifile];
            const FileAssignment assignment = assign_file_to_rank(ifile, files.size(), comm);
            if (!assignment.read_file || fs::file_size(file.path) == 0) {
                continue;
            }
            std::ifstream input(file.path);
            if (!input) {
                std::cerr << "LibBSE[Read]: failed to open Coulomb file " << file.path.string() << "\n";
                return -1;
            }

            //Start read file!
            int ir_k_point = 0;
            input >> ir_k_point;
            if (!input) {
                std::cerr << "LibBSE[Read]: failed to read Coulomb block count " << file.path.string() << "\n";
                return -1;
            }

            for (int iblock = 0; iblock < ir_k_point; ++iblock) {
                CoulombBlock block;
                block.file_index = file.index;
                //block.global_block_index = file.index * expected_blocks + iblock;

                input >> n_aux_basis >> block.row_first >> block.row_last
                      >> block.col_first >> block.col_last;
                input >> block.i_k_point >> block.weight_k_point;

                const int rows = block.row_last - block.row_first + 1;
                const int cols = block.col_last - block.col_first + 1;
                if (rows <= 0 || cols <= 0 || !input) {
                    std::cerr << "LibBSE[Read]: invalid Coulomb block " << file.path.string() << "\n";
                    return -1;
                }

                // Split blocks only inside the rank group assigned to this file.
                // If the file has one owner rank, that rank keeps all blocks.
                const bool keep_block =
                    (iblock % assignment.rank_count == assignment.local_rank);
                if (keep_block) {
                    block.value.reserve(static_cast<std::size_t>(rows * cols));
                }
                for (int i = 0; i < rows * cols; ++i) {
                    double real = 0.0;
                    double imag = 0.0;
                    input >> real >> imag;
                    if (!input) {
                        std::cerr << "LibBSE[Read]: failed to read Coulomb value " << file.path.string() << "\n";
                        return -1;
                    }
                    if (keep_block) {
                        block.value.emplace_back(real, imag);
                    }
                }
                if (keep_block) {
                    local_coulomb_blocks.push_back(std::move(block));
                }
            }
        }
        return n_aux_basis;
    }

    

    int read_KS_eigenvector_group(const fs::path directory,
                                  std::vector<KSBlock>& local_KS_eigenvector,
                                  const Enviroment& Envir,
                                  const MpiComm& comm,
                                  const std::string& prefix){
        const std::vector<IndexedFile> files = find_indexed_files(directory, prefix, ".txt");

        for (int ifile = 0; ifile < files.size(); ++ifile) {
            const IndexedFile& file = files[ifile];
            const FileAssignment assignment = assign_file_to_rank(ifile, files.size(), comm);
            if (!assignment.read_file || fs::file_size(file.path) == 0) {
                continue;
            }
            std::ifstream input(file.path);
            if (!input) {
                std::cerr << "LibBSE[Read]: failed to open KS eigenvector file " << file.path.string() << "\n";
                return 1;
            }

            int i_k_point = 0;
            input >> i_k_point;
            if (!input) {
                std::cerr << "LibBSE[Read]: failed to read KS eigenvector header " << file.path.string() << "\n";
                return 1;
            }

            // Treat each (spin,state) vector as one block of n_basis complex values.
            // This keeps file ownership local but still lets multiple ranks share a large k-point file.
            const int expected_blocks = Envir.n_band_spin * Envir.n_band_state;
            for (int iblock = 0; iblock < expected_blocks; ++iblock) {
                KSBlock block;
                block.file_index = file.index;
                block.i_k_point = i_k_point;
                block.i_band_spin = iblock / Envir.n_band_state + 1;
                block.i_state = iblock % Envir.n_band_state + 1;

                const bool keep_block =
                    (iblock % assignment.rank_count == assignment.local_rank);
                if (keep_block) {
                    block.value.reserve(static_cast<std::size_t>(Envir.n_basis));
                }
                for (int ibasis = 0; ibasis < Envir.n_basis; ++ibasis) {
                    double real = 0.0;
                    double imag = 0.0;
                    input >> real >> imag;
                    if (!input) {
                        std::cerr << "LibBSE[Read]: failed to read KS eigenvector value " << file.path.string() << "\n";
                        return 1;
                    }
                    if (keep_block) {
                        block.value.emplace_back(real, imag);
                    }
                }
                if (keep_block) {
                    local_KS_eigenvector.push_back(std::move(block));
                }
            }
        }
        return 0;
    }

    int read_RI_coeff_group(const fs::path directory,
                            std::vector<RIBlock>& local_RI_coeff,
                            const MpiComm& comm,
                            const std::string& prefix){
        const std::vector<IndexedFile> files = find_indexed_files(directory, prefix, ".txt");

        for (int ifile = 0; ifile < files.size(); ++ifile) {
            const IndexedFile& file = files[ifile];
            const FileAssignment assignment = assign_file_to_rank(ifile, files.size(), comm);
            if (!assignment.read_file || fs::file_size(file.path) == 0) {
                continue;
            }
            std::ifstream input(file.path);
            if (!input) {
                std::cerr << "LibBSE[Read]: failed to open RI coeff file " << file.path.string() << "\n";
                return 1;
            }

            //Start read file!
            int n_atom = 0;
            int n_cell = 0;
            input >> n_atom >> n_cell;
            if (!input) {
                std::cerr << "LibBSE[Read]: failed to read RI coeff header " << file.path.string() << "\n";
                return 1;
            }

            // Cs_data starts with n_atom and n_cell, but the file stores the actual
            // (i_atom,j_atom,R) blocks sequentially.  Some datasets do not contain
            // all n_atom*n_atom*n_cell combinations, so read block headers until EOF.
            int iblock = 0;
            while (true) {
                RIBlock block;
                block.file_index = file.index;

                if (!(input >> block.i_atom >> block.j_atom
                            >> block.n_1 >> block.n_2 >> block.n_3
                            >> block.n_basis_i >> block.n_basis_j >> block.n_aux_basis_i)) {
                    if (input.eof()) {
                        break;
                    }
                    std::cerr << "LibBSE[Read]: failed to read RI coeff block header "
                              << file.path.string() << "\n";
                    return 1;
                }

                const int block_size =
                    block.n_basis_i * block.n_basis_j * block.n_aux_basis_i;
                if (block_size <= 0 || !input) {
                    std::cerr << "LibBSE[Read]: invalid RI coeff block " << file.path.string() << "\n";
                    return 1;
                }

                // Split blocks only inside the rank group assigned to this file.
                // If the file has one owner rank, that rank keeps all blocks.
                const bool keep_block =
                    (iblock % assignment.rank_count == assignment.local_rank);
                if (keep_block) {
                    // Store RI_coeff as a real 3D tensor on the owning rank.
                    // Cs_data is written in the order:
                    //   basis_i -> basis_j -> aux_basis_i
                    // Enviroment keeps it as:
                    //   value(aux_basis_i, basis_j, basis_i)
                    // so later code can access the physical dimensions directly.
                    block.value = tensor<double>(block.n_aux_basis_i,
                                                 block.n_basis_j,
                                                 block.n_basis_i);
                }
                for (int ibasis_i = 0; ibasis_i < block.n_basis_i; ++ibasis_i) {
                    for (int ibasis_j = 0; ibasis_j < block.n_basis_j; ++ibasis_j) {
                        for (int iaux = 0; iaux < block.n_aux_basis_i; ++iaux) {
                            double value = 0.0;
                            input >> value;
                            if (!input) {
                                std::cerr << "LibBSE[Read]: failed to read RI coeff value "
                                          << file.path.string() << "\n";
                                return 1;
                            }
                            if (keep_block) {
                                block.value(iaux, ibasis_j, ibasis_i) = value;
                            }
                        }
                    }
                }
                if (keep_block) {
                    local_RI_coeff.push_back(std::move(block));
                }
                ++iblock;
            }
        }
        return 0;
    }

    int read_aims_output(const std::string& directory, Enviroment &Envir, const MpiComm& Comm){
        const fs::path root(directory);
        Envir.dataset_dir = root.string();
        int read_status = 0;
        if (Comm.LibBSE_MPI_is_root()) {
            read_status = read_band_out(root / "band_out", Envir);
            if (read_status == 0) {
                read_status = read_struct_out(root / "stru_out", Envir);
            }
            if (read_status == 0) {
                read_status = read_vxc_out(root / "vxc_out", Envir);
            }
            if (read_status == 0) {
                read_status = read_geometry_in(root / "geometry.in", Envir);
            }
        }
        MPI_Bcast(&read_status, 1, MPI_INT, 0, Comm.LibBSE_MPI_raw());
        if (read_status != 0) {
            return read_status;
        }

        // Root reads the small global metadata from band_out/stru_out/geometry.in.
        // Every rank later calls LibRI RPA setup, so every rank needs the same
        // lattice, k mesh, band data, and atom positions.
        MPI_Bcast(&Envir.n_k_point, 1, MPI_INT, 0, Comm.LibBSE_MPI_raw());
        MPI_Bcast(&Envir.n_band_spin, 1, MPI_INT, 0, Comm.LibBSE_MPI_raw());
        MPI_Bcast(&Envir.n_band_state, 1, MPI_INT, 0, Comm.LibBSE_MPI_raw());
        MPI_Bcast(&Envir.n_basis, 1, MPI_INT, 0, Comm.LibBSE_MPI_raw());
        MPI_Bcast(&Envir.E_Fermi, 1, MPI_DOUBLE, 0, Comm.LibBSE_MPI_raw());
        MPI_Bcast(&Envir.ir_k_point, 1, MPI_INT, 0, Comm.LibBSE_MPI_raw());
        MPI_Bcast(&Envir.n_atom, 1, MPI_INT, 0, Comm.LibBSE_MPI_raw());
        bcast_matrix_double(Envir.lattice_vect, 3, 3, Comm);
        bcast_matrix_double(Envir.reciprocal_vect, 3, 3, Comm);
        bcast_vector_int(Envir.k_point_dim, Comm);
        bcast_vector_atompos(Envir.atoms_pos, Comm);
        bcast_matrix_double(Envir.k_point_list, Envir.n_k_point, 3, Comm);
        bcast_vector_int(Envir.map_from_FullBZ_to_IBZ, Comm);
        bcast_KS_band(Envir, Comm);
        
        // Each rank stores only the Coulomb blocks assigned to it.
        Envir.local_coulomb_cut.clear();
        Envir.local_coulomb_mat.clear();
        Envir.local_KS_eigenvector.clear();
        Envir.local_RI_coeff.clear();
        read_status = read_coulomb_cut(root, Envir, Comm);
        if (read_status != 0) {
            return read_status;
        }
        read_status = read_coulomb_mat(root, Envir, Comm);
        if (read_status != 0) {
            return read_status;
        }
        MPI_Bcast(&Envir.n_aux_basis, 1, MPI_INT, 0, Comm.LibBSE_MPI_raw());
        read_status = read_Cs_data(root, Envir, Comm);
        if (read_status != 0) {
            return read_status;
        }
        read_status = read_KS_eigenvector(root, Envir, Comm);
        if (read_status != 0) {
            return read_status;
        }

        LibBSE_printf_all(Comm, "stored cut_blocks=%zu"
                                ", mat_blocks=%zu"
                                ", RI_blocks=%zu"
                                ", KS_blocks=%zu"
                                ", ir_k_point=%d\n",
                                Envir.local_coulomb_cut.size(),
                                Envir.local_coulomb_mat.size(),
                                Envir.local_RI_coeff.size(),
                                Envir.local_KS_eigenvector.size(),
                                Envir.ir_k_point);
        LibBSE_printf_root(Comm, "Read band_out: nkpoint=%d, spin=%d, states=%d, basis=%d, E_Fermi=%f\n",
                                Envir.n_k_point, 
                                Envir.n_band_spin, 
                                Envir.n_band_state, 
                                Envir.n_basis, 
                                Envir.E_Fermi);
        LibBSE_printf_root(Comm, "Read struct_out: irkp=%d\n",
                                Envir.ir_k_point);
        LibBSE_printf_root(Comm, "Read coulumb_mat: n_aux_basis=%d\n",
                                Envir.n_aux_basis);
        return 0;
    }

    int read_band_out(const fs::path file, Enviroment &Envir){
        if (!fs::exists(file)) {
            std::cerr << "LibBSE[Read]: band_out does not exist: " << file << "\n";
            return 1;
        }
        std::ifstream input(file);
        if (!input) {
            std::cerr << "LibBSE[Read]: failed to open file " << file << "\n";
            return 1;
        }

        input >> Envir.n_k_point >> Envir.n_band_spin >> Envir.n_band_state >> Envir.n_basis >>
            Envir.E_Fermi;
        if (!input) {
            std::cerr << "LibBSE[Read]: failed to read band header " << file << "\n";
            return 1;
        }

        Envir.KS_Band.clear();
        Envir.KS_Band.reserve(static_cast<std::size_t>(Envir.n_k_point * Envir.n_band_spin));

        for (int ik = 0; ik < Envir.n_k_point * Envir.n_band_spin; ++ik) {
            BandVect Band_curr_kpoint;
            if (!(input >> Band_curr_kpoint.i_k_point >> Band_curr_kpoint.i_band_spin)) {
                std::cerr << "LibBSE[Read]: failed to read k-point header " << file << "\n";
                return 1;
            }
            Band_curr_kpoint.Band.reserve(static_cast<std::size_t>(Envir.n_band_state));
            for (int istate = 0; istate < Envir.n_band_state; ++istate) {
                BandEntry entry;
                double _;
                if (!(input >> _ >> entry.band_occ >> entry.E_band >> _)) {
                    std::cerr << "LibBSE[Read]: failed to read Band Entry " << file << "\n";
                    return 1;
                }
                Band_curr_kpoint.Band.push_back(entry);
            }
            Envir.KS_Band.push_back(std::move(Band_curr_kpoint));
        }
        return 0;
    }

    int read_struct_out(const fs::path file, Enviroment &Envir){
        if (!fs::exists(file)) {
            std::cerr << "LibBSE[Read]: stru_out does not exist: " << file << "\n";
            return 1;
        }
        std::ifstream input(file);
        if (!input) {
            std::cerr << "LibBSE[Read]: failed to open file " << file << "\n";
            return 1;
        }

        Envir.lattice_vect.assign(3, std::vector<double>(3, 0.0));
        Envir.reciprocal_vect.assign(3, std::vector<double>(3, 0.0));
        Envir.k_point_dim.assign(3, 0);
        Envir.k_point_list.clear();
        Envir.map_from_FullBZ_to_IBZ.clear();

        // stru_out stores 3 lattice vectors followed by 3 reciprocal vectors.
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                input >> Envir.lattice_vect[i][j];
            }
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                input >> Envir.reciprocal_vect[i][j];
            }
        }
        for (int i = 0; i < 3; ++i) {
            input >> Envir.k_point_dim[i];
        }
        if (!input) {
            std::cerr << "LibBSE[Read]: failed to read stru_out header " << file << "\n";
            return 1;
        }

        // Full BZ k-point list length is the product of k-grid dimensions.
        Envir.k_point_list.reserve(static_cast<std::size_t>(Envir.n_k_point));
        for (int ik = 0; ik < Envir.n_k_point; ++ik) {
            std::vector<double> k_point(3, 0.0);
            for (int j = 0; j < 3; ++j) {
                input >> k_point[j];
            }
            if (!input) {
                std::cerr << "LibBSE[Read]: failed to read full BZ k-point list " << file << "\n";
                return 1;
            }
            Envir.k_point_list.push_back(std::move(k_point));
        }

        // Remaining integers map each full-BZ k point to an irreducible-BZ index.
        int map_value = 0;
        while (input >> map_value) {
            Envir.map_from_FullBZ_to_IBZ.push_back(map_value);
        }
        std::unordered_set<int> unique_set(Envir.map_from_FullBZ_to_IBZ.begin(), Envir.map_from_FullBZ_to_IBZ.end());
        Envir.ir_k_point = unique_set.size();
        
        size_t unique_count = unique_set.size();
        if (!input.eof()) {
            std::cerr << "LibBSE[Read]: failed to read full BZ to IBZ map " << file << "\n";
            return 1;
        }
        return 0;
    }

    int read_vxc_out(const fs::path file, Enviroment &Envir){
        if (!fs::exists(file)) {
            std::cerr << "LibBSE[Read]: vxc_out does not exist: " << file << "\n";
            return 1;
        }
        std::ifstream input(file);
        if (!input) {
            std::cerr << "LibBSE[Read]: failed to open file " << file << "\n";
            return 1;
        }

        int n_k_point = 0;
        int n_spin = 0;
        int n_state = 0;
        input >> n_k_point >> n_spin >> n_state;
        if (!input) {
            std::cerr << "LibBSE[Read]: failed to read vxc_out header " << file << "\n";
            return 1;
        }

        Envir.vxc.assign(static_cast<std::size_t>(n_k_point),
                         std::vector<std::vector<double>>(
                             static_cast<std::size_t>(n_spin),
                             std::vector<double>(static_cast<std::size_t>(n_state), 0.0)));

        for (int ik = 0; ik < n_k_point; ++ik) {
            for (int ispin = 0; ispin < n_spin; ++ispin) {
                for (int istate = 0; istate < n_state; ++istate) {
                    double value_Ha = 0.0;
                    double value_eV = 0.0;
                    input >> value_Ha >> value_eV;
                    if (!input) {
                        std::cerr << "LibBSE[Read]: failed to read vxc value " << file << "\n";
                        return 1;
                    }
                    // Keep the Hartree value; the second column is the same value in eV.
                    Envir.vxc[ik][ispin][istate] = value_Ha;
                }
            }
        }
        return 0;
    }


    int read_geometry_in(const fs::path file, Enviroment &Envir){
        if (!fs::exists(file)) {
            std::cerr << "LibBSE[Read]: geometry.in does not exist: " << file << "\n";
            return 1;
        }
        std::ifstream input(file);
        if (!input) {
            std::cerr << "LibBSE[Read]: failed to open file " << file << "\n";
            return 1;
        }
        std::string line;
        Envir.atoms_pos.clear();
        Envir.n_atom = 0;

        while(std::getline(input, line)){
            if(line.empty()||line[0] == '#'){
                continue;
            }

            std::istringstream iss(line);
            std::string keyword;
            iss >> keyword;
            if (!iss) {
                continue;
            }

            if (keyword != "atom_frac" && keyword != "atom") {
                continue;
            }

            std::string atom;
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            if (!(iss >> x >> y >> z >> atom)) {
                std::cerr << "LibBSE[Read]: failed to read geometry.in atom line " << file << "\n";
                return 1;
            }

            std::vector<double> pos(3, 0.0);
            if (keyword == "atom_frac") {
                if (Envir.lattice_vect.size() != 3) {
                    std::cerr << "LibBSE[Read]: lattice vectors are needed before atom_frac positions "
                              << file << "\n";
                    return 1;
                }
                // geometry.in atom_frac gives fractional coordinates.  Convert
                // with stru_out lattice vectors so atom positions use the same
                // unit system as the latvec passed to LibRI.
                const double frac[3] = {x, y, z};
                for (int ia = 0; ia < 3; ++ia) {
                    for (int ix = 0; ix < 3; ++ix) {
                        pos[ix] += frac[ia] * Envir.lattice_vect[ia][ix];
                    }
                }
            }
            else {
                constexpr double AngstromToBohr = 1.8897259886;
                // FHI-aims Cartesian atom lines are in Angstrom.  stru_out
                // lattice vectors are in Bohr, so convert before LibRI sees it.
                pos[0] = x * AngstromToBohr;
                pos[1] = y * AngstromToBohr;
                pos[2] = z * AngstromToBohr;
            }

            Envir.atoms_pos.push_back(AtomPos{atom, pos});
            ++Envir.n_atom;
        }
        return 0;
    }

    int read_coulomb_cut(const fs::path directory, Enviroment& Envir, const MpiComm& comm){
        Envir.n_aux_basis = read_coulomb_group(directory, Envir.local_coulomb_cut, comm, "coulomb_cut_");
        if(Envir.n_aux_basis > 0){
            return 0;
        }
        else{
            return 1;
        }
    }

    int read_coulomb_mat(const fs::path directory, Enviroment& Envir, const MpiComm& comm){
        int ret = read_coulomb_group(directory, Envir.local_coulomb_mat, comm, "coulomb_mat_");
        if(ret > 0){
            return 0;
        }
        else{
            return 1;
        }
    }

    int read_Cs_data(const fs::path directory, Enviroment& Envir, const MpiComm& comm){
        return read_RI_coeff_group(directory, Envir.local_RI_coeff, comm, "Cs_data_");
    }

    int read_KS_eigenvector(const fs::path directory, Enviroment& Envir, const MpiComm& comm){
        return read_KS_eigenvector_group(directory, Envir.local_KS_eigenvector, Envir, comm, "KS_eigenvector_");
    }

    //Function to find Files named as perfix_xxx.suffix
    std::vector<IndexedFile> find_indexed_files(const fs::path& directory,
                                                const std::string& prefix,
                                                const std::string& suffix){
        std::vector<IndexedFile> files;
        if (!fs::exists(directory)) {
            std::cerr << "LibBSE[Read]: Dataset directory does not exist: " << directory.string() << "\n";
        }

        for (const auto& entry : fs::directory_iterator(directory)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (name.rfind(prefix, 0) != 0 || name.size() < prefix.size() + suffix.size()) {
                continue;
            }
            if (name.substr(name.size() - suffix.size()) != suffix) {
                continue;
            }
            const std::string index_text =
                name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
            try {
                files.push_back({std::stoi(index_text), entry.path()});
            } 
            catch (const std::exception&) {
                continue;
            }
        }
        std::sort(files.begin(), files.end(),
                  [](const IndexedFile& lhs, const IndexedFile& rhs) { return lhs.index < rhs.index; });
        return files;
    }

    bool assigned_to_rank(int block_index, const MpiComm& comm)
    {
        return comm.LibBSE_MPI_size() <= 1 || block_index % comm.LibBSE_MPI_size() == comm.LibBSE_MPI_rank();
    }

} // namespace LibBSE
