/*
    LibBSE_Chi_0.cpp: 
    This is the part trying to use our own routines to calculate 
    chi_0 from Cs_data and KS_eigenvector data. 
*/
#include "LibBSE_Chi_0.h"
#include "../math/Tensor.h"
#include "../interface/Blas_Interface.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <valarray>

namespace LibBSE{

    void reduce_sum_RI_row(const matrix<double>& local_row,
                           matrix<double>* root_matrix,
                           const int local_ik,
                           const int expected_count,
                           const int root_rank,
                           const MpiComm& Comm,
                           std::string& local_error){
        if(expected_count <= 0){
            return;
        }

        // MPI_Reduce needs the same count on every rank.  If this rank has no
        // local contribution for the current atom-pair/k-point, send zeros.
        matrix<double> send_row(1, expected_count);
        zeros(send_row);
        if(local_row.size != 0){
            if(local_row.size != expected_count){
                if(local_error.empty()){
                    local_error = "LibBSE[Chi0_BSE]: local RI row size mismatch";
                }
            }
            else{
                for(int i = 0; i < expected_count; ++i){
                    send_row(0, i) = local_row.matrix_ptr[i];
                }
            }
        }

        // root_rank is the rank that recorded this k point in KS_eigenvector.
        // That rank receives the summed row directly into its local
        // RI_coeff_k_real/imag matrix.  Other ranks only contribute send_row.
        double* recv_row = nullptr;
        if(Comm.LibBSE_MPI_rank() == root_rank && root_matrix != nullptr){
            recv_row = root_matrix->matrix_ptr + local_ik * root_matrix->col;
        }
        MPI_Reduce(send_row.matrix_ptr, recv_row,
                   expected_count, MPI_DOUBLE, MPI_SUM,
                   root_rank, Comm.LibBSE_MPI_raw());
    }


    void write_RI_coeff_k_real_debug(const Enviroment& Envir,
                                     const MpiComm& Comm,
                                     const std::vector<matrix<double>>& RI_coeff_k_real){
        const std::filesystem::path debug_dir =
            std::filesystem::current_path() / "RI_coeff_k_debug";
        std::filesystem::create_directories(debug_dir);
        const int mpi_rank = Comm.LibBSE_MPI_rank();

        for(int i_atom = 0; i_atom < Envir.n_atom; ++i_atom){
            for(int j_atom = 0; j_atom < Envir.n_atom; ++j_atom){
                const int block_index = i_atom * Envir.n_atom + j_atom;
                const matrix<double>& block =
                    RI_coeff_k_real[static_cast<std::size_t>(block_index)];
                if(block.size == 0){
                    continue;
                }

                const std::filesystem::path file =
                    debug_dir / ("RI_coeff_k_real_rank_"
                               + std::to_string(mpi_rank)
                               + "_atom_"
                               + std::to_string(i_atom) + "_"
                               + std::to_string(j_atom) + ".txt");
                std::ofstream output(file);
                output << "# RI_coeff_k_real\n";
                output << "# rank " << mpi_rank << "\n";
                output << "# i_atom " << i_atom << " j_atom " << j_atom << "\n";
                output << "# rows: k_point, columns: aux_i * basis_j * basis_i\n";
                output << "# local_n_kpoint " << block.row << " n_value " << block.col << "\n";
                for(int local_ik = 0; local_ik < block.row; ++local_ik){
                    // The matrix row is local to recorded_k_points, not the
                    // global 0..n_kpoint-1 row anymore.
                    output << "i_kpoint = "
                           << Envir.recorded_k_points[static_cast<std::size_t>(local_ik)]
                           << "\n";
                    for(int icol = 0; icol < block.col; ++icol){
                        output << std::setw(12) << std::fixed <<  std::setprecision(7)
                               << block(local_ik, icol);
                        if(icol % 10 == 9){
                            output << "\n";
                        }
                    }
                    output << "\n";
                }
            }
        }
    }


    int get_RI_k_coeff(Enviroment& Envir, MpiComm Comm, 
                               std::vector<matrix<double>>& RI_coeff_k_real, 
                               std::vector<matrix<double>>& RI_coeff_k_imag){
        std::string local_error;
        int n_RI_coeff_k = 0;
        //in the future we may need larger k-point mesh
        try{
            if(Envir.lattice_vect.row != 3 || Envir.lattice_vect.col != 3){
                throw std::runtime_error("LibBSE[Chi0_BSE]: lattice vectors are missing");
            }
            if(Envir.k_point_list.row <= 0 || Envir.k_point_list.col != 3){
                throw std::runtime_error("LibBSE[Chi0_BSE]: k-point list is missing");
            }
            // First do A_times_k(a,k) = A_a . k
            matrix<double> A_times_k(3, Envir.k_point_list.row);
            A_times_k = transpose_times(Envir.lattice_vect, Envir.k_point_list, 'N', 'T');
            //print_matrix(Comm, "A_times_k", A_times_k);
            //Get the phase vector ready for each Local block
            for(auto& curr_RI_block: Envir.local_RI_coeff){
                if(curr_RI_block.n_basis_i <= 0 || curr_RI_block.n_basis_j <= 0
                || curr_RI_block.n_aux_basis_i <= 0){
                    throw std::runtime_error("LibBSE[Chi0_BSE]: invalid RI block size");
                }

                const std::size_t block_size =
                  curr_RI_block.n_basis_i* curr_RI_block.n_basis_j* curr_RI_block.n_aux_basis_i;
                if(curr_RI_block.value.size != block_size){
                    throw std::runtime_error("LibBSE[Chi0_BSE]: RI block value size mismatch");
                }

                matrix<double> n_vect(1,3);
                n_vect(0,0) = curr_RI_block.n_1;
                n_vect(0,1) = curr_RI_block.n_2;
                n_vect(0,2) = curr_RI_block.n_3;
                curr_RI_block.angle_vect = n_vect * A_times_k;
            }

            //std::vector<matrix<double>> RI_coeff_k_real; 
            //std::vector<matrix<double>> RI_coeff_k_imag; 

            const int n_atom_pair = Envir.n_atom * Envir.n_atom;
            const int n_kpoint = Envir.k_point_list.row;
            const int local_n_kpoint =
                static_cast<int>(Envir.recorded_k_points.size());

            
            if(static_cast<int>(Envir.n_basis_atom.size()) != Envir.n_atom
            || static_cast<int>(Envir.n_aux_basis_atom.size()) != Envir.n_atom){
                throw std::runtime_error("LibBSE[Chi0_BSE]: atom basis metadata is missing");
            }

            // Each rank now keeps only the k points that it owns in
            // KS_eigenvector.  Therefore each atom-pair block has
            // local_n_kpoint rows, and row local_ik maps to
            // Envir.recorded_k_points[local_ik].
            RI_coeff_k_real.resize(static_cast<std::size_t>(n_atom_pair));
            RI_coeff_k_imag.resize(static_cast<std::size_t>(n_atom_pair));
            n_RI_coeff_k = n_atom_pair;
            for(int i_atom = 0; i_atom < Envir.n_atom; ++i_atom){
                for(int j_atom = 0; j_atom < Envir.n_atom; ++j_atom){
                    const int block_index = i_atom * Envir.n_atom + j_atom;
                    const int n_value =
                        Envir.n_aux_basis_atom[static_cast<std::size_t>(i_atom)]
                      * Envir.n_basis_atom[static_cast<std::size_t>(j_atom)]
                      * Envir.n_basis_atom[static_cast<std::size_t>(i_atom)];

                    // The column layout still follows read_Cs_data():
                    //   aux_i * (basis_j * basis_i) + basis_j * basis_i + basis_i.
                    RI_coeff_k_real[static_cast<std::size_t>(block_index)] =
                        matrix<double>(local_n_kpoint, n_value);
                    RI_coeff_k_imag[static_cast<std::size_t>(block_index)] =
                        matrix<double>(local_n_kpoint, n_value);
                    zeros(RI_coeff_k_real[static_cast<std::size_t>(block_index)]);
                    zeros(RI_coeff_k_imag[static_cast<std::size_t>(block_index)]);
                }
            }

            // Build one common reduce schedule for every rank:
            //   rank r receives RI_coeff_k for every kpoint in rank r's
            //   recorded_k_points.
            // All ranks must follow this exact same schedule, otherwise MPI_Reduce
            // collectives would be called in different orders.
            const int mpi_size = Comm.LibBSE_MPI_size();
            const int mpi_rank = Comm.LibBSE_MPI_rank();
            const int local_recorded_count = local_n_kpoint;
            std::vector<int> recorded_counts(static_cast<std::size_t>(mpi_size), 0);
            MPI_Allgather(&local_recorded_count, 1, MPI_INT,
                          recorded_counts.data(), 1, MPI_INT,
                          Comm.LibBSE_MPI_raw());

            std::vector<int> recorded_displs(static_cast<std::size_t>(mpi_size), 0);
            int total_recorded_count = 0;
            for(int irank = 0; irank < mpi_size; ++irank){
                recorded_displs[static_cast<std::size_t>(irank)] = total_recorded_count;
                total_recorded_count += recorded_counts[static_cast<std::size_t>(irank)];
            }

            std::vector<int> all_recorded_k_points(
                static_cast<std::size_t>(total_recorded_count), 0);
            MPI_Allgatherv(local_recorded_count ? Envir.recorded_k_points.data() : nullptr,
                           local_recorded_count, MPI_INT,
                           total_recorded_count ? all_recorded_k_points.data() : nullptr,
                           recorded_counts.data(),
                           recorded_displs.data(),
                           MPI_INT,
                           Comm.LibBSE_MPI_raw());

            std::vector<matrix<double>> curr_RI_coeff_k_addon_real;
            std::vector<matrix<double>> curr_RI_coeff_k_addon_imag;
            curr_RI_coeff_k_addon_real.resize(static_cast<std::size_t>(n_atom_pair));
            curr_RI_coeff_k_addon_imag.resize(static_cast<std::size_t>(n_atom_pair));
            std::string local_reduce_error;

            for(int target_rank = 0; target_rank < mpi_size; ++target_rank){
                const int target_count =
                    recorded_counts[static_cast<std::size_t>(target_rank)];
                const int target_displ =
                    recorded_displs[static_cast<std::size_t>(target_rank)];

                for(int target_local_ik = 0; target_local_ik < target_count; ++target_local_ik){
                    const int i_kpoint =
                        all_recorded_k_points[static_cast<std::size_t>(target_displ + target_local_ik)];
                    const int ik = i_kpoint - 1;
                    if(ik < 0 || ik >= n_kpoint){
                        throw std::runtime_error("LibBSE[Chi0_BSE]: recorded k point is outside k-point list");
                    }

                    // Build local contribution for this global k point.  Every
                    // rank contributes its local R blocks; target_rank receives
                    // the sum into its target_local_ik row.
                    for(const auto& curr_RI_block: Envir.local_RI_coeff){
                        int block_index = curr_RI_block.i_atom * Envir.n_atom + curr_RI_block.j_atom;
                        if(block_index < 0 || block_index >= n_atom_pair){
                            throw std::runtime_error("LibBSE[Chi0_BSE]: RI block atom index is invalid");
                        }
                        const double phase_real = std::cos(curr_RI_block.angle_vect(0,ik));
                        const double phase_imag = std::sin(curr_RI_block.angle_vect(0,ik));
                        if(curr_RI_coeff_k_addon_real[static_cast<std::size_t>(block_index)].is_empty()){
                            curr_RI_coeff_k_addon_real[static_cast<std::size_t>(block_index)] =
                                phase_real * curr_RI_block.value;
                            curr_RI_coeff_k_addon_imag[static_cast<std::size_t>(block_index)] =
                                phase_imag * curr_RI_block.value;
                        }
                        else{
                            curr_RI_coeff_k_addon_real[static_cast<std::size_t>(block_index)] +=
                                phase_real * curr_RI_block.value;
                            curr_RI_coeff_k_addon_imag[static_cast<std::size_t>(block_index)] +=
                                phase_imag * curr_RI_block.value;
                        }
                    }

                    for(int iatom = 0; iatom < n_atom_pair; iatom ++){
                        const int i_atom = iatom / Envir.n_atom;
                        const int j_atom = iatom % Envir.n_atom;
                        const int expected_count =
                            Envir.n_aux_basis_atom[static_cast<std::size_t>(i_atom)]
                          * Envir.n_basis_atom[static_cast<std::size_t>(j_atom)]
                          * Envir.n_basis_atom[static_cast<std::size_t>(i_atom)];

                        matrix<double>* target_real = (mpi_rank == target_rank)
                            ? &RI_coeff_k_real[static_cast<std::size_t>(iatom)] : nullptr;
                        matrix<double>* target_imag = (mpi_rank == target_rank)
                            ? &RI_coeff_k_imag[static_cast<std::size_t>(iatom)] : nullptr;

                        reduce_sum_RI_row(curr_RI_coeff_k_addon_real[static_cast<std::size_t>(iatom)],
                                          target_real, target_local_ik, expected_count,
                                          target_rank, Comm, local_reduce_error);
                        reduce_sum_RI_row(curr_RI_coeff_k_addon_imag[static_cast<std::size_t>(iatom)],
                                          target_imag, target_local_ik, expected_count,
                                          target_rank, Comm, local_reduce_error);

                        zeros(curr_RI_coeff_k_addon_real[static_cast<std::size_t>(iatom)]);
                        zeros(curr_RI_coeff_k_addon_imag[static_cast<std::size_t>(iatom)]);
                    }
                }
            }
            if(!local_reduce_error.empty()){
                throw std::runtime_error(local_reduce_error);
            }
            if(local_n_kpoint > 0){
                write_RI_coeff_k_real_debug(Envir, Comm, RI_coeff_k_real);
            }
        }
        catch(const std::exception& e){
            local_error = e.what();
        }

        // Find out if any error.
        const int local_failed = local_error.empty() ? 0 : 1;
        int any_failed = 0;
        MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX,
                      Comm.LibBSE_MPI_raw());
        if(any_failed){
            LibBSE_printf_all(Comm, "RI-k local build status: %s\n",
                              local_error.empty() ? "ok" : local_error.c_str());
            throw std::runtime_error("LibBSE[Chi0_BSE]: failed to build local k-space RI coeff");
        }

        return n_RI_coeff_k;
    }

    int get_M_mat(Enviroment& Envir, MpiComm Comm,
                  const std::vector<matrix<double>>& RI_coeff_k_real,
                  const std::vector<matrix<double>>& RI_coeff_k_imag,
                  std::vector<matrix<double>>& M_mat_real,
                  std::vector<matrix<double>>& M_mat_imag){
        std::string local_error;
        int n_M_mat = 0;
        try{
            const int n_atom = Envir.n_atom;
            const int n_atom_pair = n_atom * n_atom;
            const int local_n_kpoint =
                static_cast<int>(Envir.recorded_k_points.size());
            const int n_KS_col = Envir.n_band_spin * Envir.n_band_state;
            const int n_M_row = local_n_kpoint * n_KS_col;

            if(static_cast<int>(Envir.n_basis_atom.size()) != n_atom
            || static_cast<int>(Envir.n_aux_basis_atom.size()) != n_atom){
                throw std::runtime_error("LibBSE[Chi0_BSE]: atom basis metadata is missing while building M");
            }
            if(static_cast<int>(RI_coeff_k_real.size()) != n_atom_pair
            || static_cast<int>(RI_coeff_k_imag.size()) != n_atom_pair){
                throw std::runtime_error("LibBSE[Chi0_BSE]: RI_coeff_k atom-pair count mismatch");
            }

            // Build atom basis offsets inside the full KS eigenvector:
            // atom 0 occupies [0, n_basis_atom[0]), atom 1 follows it, etc.
            std::vector<int> first_basis_atom(static_cast<std::size_t>(n_atom + 1), 0);
            for(int i_atom = 0; i_atom < n_atom; ++i_atom){
                first_basis_atom[static_cast<std::size_t>(i_atom + 1)] =
                    first_basis_atom[static_cast<std::size_t>(i_atom)]
                  + Envir.n_basis_atom[static_cast<std::size_t>(i_atom)];
            }
            if(first_basis_atom[static_cast<std::size_t>(n_atom)] != Envir.n_basis){
                throw std::runtime_error("LibBSE[Chi0_BSE]: atom basis count does not match KS basis count");
            }

            // Map global k-point id to the local row of RI_coeff_k_*.
            // RI_coeff_k_* rows follow Envir.recorded_k_points exactly.
            std::map<int, int> local_kpoint_row;
            for(int local_ik = 0; local_ik < local_n_kpoint; ++local_ik){
                local_kpoint_row[Envir.recorded_k_points[static_cast<std::size_t>(local_ik)]]
                    = local_ik;
            }

            // One KSBlock now stores the full eigenvector matrix for one k point.
            // Build a small lookup so the M loop can find the matching KS matrix.
            std::map<int, const KSBlock*> KS_block_of_kpoint;
            for(const KSBlock& KS_block: Envir.local_KS_eigenvector){
                if(KS_block.value.row != Envir.n_basis
                || KS_block.value.col != n_KS_col){
                    throw std::runtime_error("LibBSE[Chi0_BSE]: KS eigenvector matrix size mismatch");
                }
                if(local_kpoint_row.find(KS_block.i_k_point) == local_kpoint_row.end()){
                    throw std::runtime_error("LibBSE[Chi0_BSE]: KS eigenvector k point was not recorded");
                }
                KS_block_of_kpoint[KS_block.i_k_point] = &KS_block;
            }
            for(const auto& k_row: local_kpoint_row){
                if(KS_block_of_kpoint.find(k_row.first) == KS_block_of_kpoint.end()){
                    throw std::runtime_error("LibBSE[Chi0_BSE]: recorded k point has no KS eigenvector matrix");
                }
            }

            // M_mat_real/imag:
            //   vector index: i_atom
            //   row:          local_kpoint * (spin*state) + i_state_spin
            //   column:       i_aux_basis * i_basis + i_basis
            M_mat_real.resize(static_cast<std::size_t>(n_atom));
            M_mat_imag.resize(static_cast<std::size_t>(n_atom));
            n_M_mat = n_atom;
            for(int i_atom = 0; i_atom < n_atom; ++i_atom){
                const int n_basis_i =
                    Envir.n_basis_atom[static_cast<std::size_t>(i_atom)];
                const int n_aux_basis_i =
                    Envir.n_aux_basis_atom[static_cast<std::size_t>(i_atom)];
                const int n_value = n_aux_basis_i * n_basis_i;
                M_mat_real[static_cast<std::size_t>(i_atom)] =
                    matrix<double>(n_M_row, n_value);
                M_mat_imag[static_cast<std::size_t>(i_atom)] =
                    matrix<double>(n_M_row, n_value);
                zeros(M_mat_real[static_cast<std::size_t>(i_atom)]);
                zeros(M_mat_imag[static_cast<std::size_t>(i_atom)]);
            }

            for(int local_ik = 0; local_ik < local_n_kpoint; ++local_ik){
                const int i_kpoint =
                    Envir.recorded_k_points[static_cast<std::size_t>(local_ik)];
                const KSBlock& KS_block =
                    *KS_block_of_kpoint[static_cast<int>(i_kpoint)];

                for(int i_atom = 0; i_atom < n_atom; ++i_atom){
                    const int n_basis_i =
                        Envir.n_basis_atom[static_cast<std::size_t>(i_atom)];
                    const int n_aux_basis_i =
                        Envir.n_aux_basis_atom[static_cast<std::size_t>(i_atom)];
                    const int n_M_col = n_aux_basis_i * n_basis_i;
                    matrix<double>& curr_M_real =
                        M_mat_real[static_cast<std::size_t>(i_atom)];
                    matrix<double>& curr_M_imag =
                        M_mat_imag[static_cast<std::size_t>(i_atom)];

                    for(int j_atom = 0; j_atom < n_atom; ++j_atom){
                        const int n_basis_j =
                            Envir.n_basis_atom[static_cast<std::size_t>(j_atom)];
                        const int block_index = i_atom * n_atom + j_atom;
                        const matrix<double>& curr_RI_real =
                            RI_coeff_k_real[static_cast<std::size_t>(block_index)];
                        const matrix<double>& curr_RI_imag =
                            RI_coeff_k_imag[static_cast<std::size_t>(block_index)];
                        const int expected_RI_col =
                            n_aux_basis_i * n_basis_j * n_basis_i;
                        if(curr_RI_real.row != local_n_kpoint
                        || curr_RI_imag.row != local_n_kpoint
                        || curr_RI_real.col != expected_RI_col
                        || curr_RI_imag.col != expected_RI_col){
                            throw std::runtime_error("LibBSE[Chi0_BSE]: RI_coeff_k matrix size mismatch while building M");
                        }

                        // C_ij is reshaped from one RI_coeff_k row:
                        //   row    -> j_basis
                        //   column -> i_aux_basis * i_basis + i_basis
                        // This turns c_j,n(k) C_i,j^mu(k) into one GEMM for
                        // all local states at this k point.
                        matrix<std::complex<double>> C_mat(n_basis_j, n_M_col);
                        for(int ibasis_j = 0; ibasis_j < n_basis_j; ++ibasis_j){
                            for(int iaux = 0; iaux < n_aux_basis_i; ++iaux){
                                for(int ibasis_i = 0; ibasis_i < n_basis_i; ++ibasis_i){
                                    const int RI_col =
                                        iaux * n_basis_i * n_basis_j
                                      + ibasis_j * n_basis_i
                                      + ibasis_i;
                                    const int C_col =
                                        iaux * n_basis_i + ibasis_i;
                                    C_mat(ibasis_j, C_col) =
                                        std::complex<double>(curr_RI_real(local_ik, RI_col),
                                                             curr_RI_imag(local_ik, RI_col));
                                }
                            }
                        }

                        // KS_j holds atom-j rows of c(i_basis,n,spin).
                        // Transposing it gives c_j,n(k) in the formula:
                        //   M_i,n^mu(k) = c_j,n(k) C_i,j^mu(k).
                        matrix<std::complex<double>> KS_mat(
                            n_basis_j, n_KS_col);
                        const int first_basis_j =
                            first_basis_atom[static_cast<std::size_t>(j_atom)];
                        for(int ibasis_j = 0; ibasis_j < n_basis_j; ++ibasis_j){
                            for(int iKS_col = 0; iKS_col < n_KS_col; ++iKS_col){
                                KS_mat(ibasis_j, iKS_col) =
                                    KS_block.value(first_basis_j + ibasis_j, iKS_col);
                            }
                        }

                        matrix<std::complex<double>> M_addon =
                            transpose_times(KS_mat, C_mat, 'T', 'N');
                        for(int iKS_col = 0; iKS_col < M_addon.row; ++iKS_col){
                            const int M_row = local_ik * n_KS_col + iKS_col;
                            for(int icol = 0; icol < M_addon.col; ++icol){
                                curr_M_real(M_row, icol) += M_addon(iKS_col, icol).real();
                                curr_M_imag(M_row, icol) += M_addon(iKS_col, icol).imag();
                            }
                        }
                    }
                }
            }
        }
        catch(const std::exception& e){
            local_error = e.what();
        }

        const int local_failed = local_error.empty() ? 0 : 1;
        int any_failed = 0;
        MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX,
                      Comm.LibBSE_MPI_raw());
        if(any_failed){
            LibBSE_printf_all(Comm, "M local build status: %s\n",
                              local_error.empty() ? "ok" : local_error.c_str());
            throw std::runtime_error("LibBSE[Chi0_BSE]: failed to build M matrix");
        }
        return n_M_mat;
    }

    int calculate_chi0_BSE(Enviroment& Envir, MpiComm Comm){
        LibBSE_printf_root(Comm, "Start calculating chi0...\n");
        try{
            //1. convert the LRI into RI-coeff.
            // C^μ(0)_i(k),j(0) = \sum_R e^{ikR}  \tilde{C}^μ(0)_i(R),j(0)
            // Try to implement a naive mult since this is not the bottleneck.
            // TODO: implement a more efficient way like matrix flattening
            // TODO: use irkp to save memory and time.
            
            // vector <i_atom * j_atom> 
            // matrix <n_kpoint, i_basis * j_basis * i_aux_basis>
            std::vector<matrix<double>> RI_coeff_k_real; 
            std::vector<matrix<double>> RI_coeff_k_imag; 
            const int n_RI_coeff_k = get_RI_k_coeff(Envir, Comm, RI_coeff_k_real, RI_coeff_k_imag);            
            //2. \tilde{M}_i,n^\mu(k) = c_j,n(k) \tilde{C}_i(0),j(k)^\mu(0)
            std::vector<matrix<double>> M_mat_real; 
            std::vector<matrix<double>> M_mat_imag; 
            get_M_mat(Envir, Comm, RI_coeff_k_real, RI_coeff_k_imag, M_mat_real, M_mat_imag);

            //3. C_m,n,\sigma^\mu(k+q,k) = 
            


            //

            return 0;
            //
        }
        catch (const std::exception& e){
            LibBSE_printf_root(Comm, "LibBSE[Chi0_BSE]: %s\n", e.what());
            return 1;
        }
        return 0;
    }
}
