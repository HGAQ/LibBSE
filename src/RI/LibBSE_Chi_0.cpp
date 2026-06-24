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
        int type = 1;// 0 is style that easy to understand by human, 1 is for compare w/ FHI-aims
        const std::filesystem::path debug_dir =
            std::filesystem::current_path() / "RI_coeff_k_debug";
        std::filesystem::create_directories(debug_dir);
        const int mpi_rank = Comm.LibBSE_MPI_rank();

        switch (type){
        case 0:
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
                    output << "# rows: k_point, columns: basis_i * aux_i * basis_j\n";
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
            break;
        case 1:
            {
                // This output keeps the same numeric layout as the FHI-aims
                // debug write:
                //   (4I5,2F15.10) k_point, aux, basis_i, basis_j, real, imag.
                // write_RI_coeff_k_real_debug only receives the real part, so the
                // second floating column is kept as 0.0 for a format-compatible
                // file.
                if(static_cast<int>(RI_coeff_k_real.size()) != Envir.n_atom * Envir.n_atom){
                    throw std::runtime_error("LibBSE[Chi0_BSE]: RI_coeff_k atom-pair count mismatch in debug output");
                }
                if(static_cast<int>(Envir.n_basis_atom.size()) != Envir.n_atom
                || static_cast<int>(Envir.n_aux_basis_atom.size()) != Envir.n_atom){
                    throw std::runtime_error("LibBSE[Chi0_BSE]: atom basis metadata is missing in debug output");
                }

                // Convert atom-local basis/aux indices to the 1-based global
                // indices printed by the reference Fortran debug code.
                std::vector<int> first_basis_atom(static_cast<std::size_t>(Envir.n_atom), 0);
                std::vector<int> first_aux_atom(static_cast<std::size_t>(Envir.n_atom), 0);
                for(int i_atom = 1; i_atom < Envir.n_atom; ++i_atom){
                    first_basis_atom[static_cast<std::size_t>(i_atom)] =
                        first_basis_atom[static_cast<std::size_t>(i_atom - 1)]
                      + Envir.n_basis_atom[static_cast<std::size_t>(i_atom - 1)];
                    first_aux_atom[static_cast<std::size_t>(i_atom)] =
                        first_aux_atom[static_cast<std::size_t>(i_atom - 1)]
                      + Envir.n_aux_basis_atom[static_cast<std::size_t>(i_atom - 1)];
                }

                const std::filesystem::path file =
                    debug_dir / ("RI_coeff_k_real_rank_"
                               + std::to_string(mpi_rank)
                               + "_fhi_aims.txt");
                std::ofstream output(file);
                output << std::fixed << std::setprecision(10);

                for(int local_ik = 0; local_ik < static_cast<int>(Envir.recorded_k_points.size()); ++local_ik){
                    const int i_k_point =
                        Envir.recorded_k_points[static_cast<std::size_t>(local_ik)];
                    for(int i_atom = 0; i_atom < Envir.n_atom; ++i_atom){
                        const int n_basis_i =
                            Envir.n_basis_atom[static_cast<std::size_t>(i_atom)];
                        const int n_aux_basis_i =
                            Envir.n_aux_basis_atom[static_cast<std::size_t>(i_atom)];
                        for(int j_atom = 0; j_atom < Envir.n_atom; ++j_atom){
                            const int n_basis_j =
                                Envir.n_basis_atom[static_cast<std::size_t>(j_atom)];
                            const int block_index = i_atom * Envir.n_atom + j_atom;
                            const matrix<double>& block =
                                RI_coeff_k_real[static_cast<std::size_t>(block_index)];
                            const int expected_col =
                                n_basis_i * n_aux_basis_i * n_basis_j;
                            if(block.row != static_cast<int>(Envir.recorded_k_points.size())
                            || block.col != expected_col){
                                throw std::runtime_error("LibBSE[Chi0_BSE]: RI_coeff_k matrix size mismatch in debug output");
                            }

                            // RI_coeff_k is stored as [i_basis][i_aux][j_basis].
                            // The loop order below mirrors the Fortran output:
                            // aux row first, then two NAO basis indices.
                            for(int iaux = 0; iaux < n_aux_basis_i; ++iaux){
                                const int i_aux_global =
                                    first_aux_atom[static_cast<std::size_t>(i_atom)] + iaux + 1;
                                for(int ibasis_i = 0; ibasis_i < n_basis_i; ++ibasis_i){
                                    const int i_basis_global =
                                        first_basis_atom[static_cast<std::size_t>(i_atom)] + ibasis_i + 1;
                                    for(int ibasis_j = 0; ibasis_j < n_basis_j; ++ibasis_j){
                                        const int j_basis_global =
                                            first_basis_atom[static_cast<std::size_t>(j_atom)] + ibasis_j + 1;
                                        const int RI_col =
                                            ibasis_i * n_aux_basis_i * n_basis_j
                                          + iaux * n_basis_j
                                          + ibasis_j;
                                        output << std::setw(5) << i_k_point
                                               << std::setw(5) << i_aux_global
                                               << std::setw(5) << i_basis_global
                                               << std::setw(5) << j_basis_global
                                               << std::setw(15) << block(local_ik, RI_col)
                                               << "\n";
                                    }
                                }
                            }
                        }
                    }
                }
            }
            break;
        }
        
    }


    int get_RI_k_coeff(Enviroment& Envir, MpiComm Comm, 
                               std::vector<matrix<double>>& RI_coeff_k_real, 
                               std::vector<matrix<double>>& RI_coeff_k_imag){
        std::string local_error;

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

            for(int i_atom = 0; i_atom < Envir.n_atom; ++i_atom){
                for(int j_atom = 0; j_atom < Envir.n_atom; ++j_atom){
                    const int block_index = i_atom * Envir.n_atom + j_atom;
                    const int n_value =
                        Envir.n_aux_basis_atom[static_cast<std::size_t>(i_atom)]
                      * Envir.n_basis_atom[static_cast<std::size_t>(j_atom)]
                      * Envir.n_basis_atom[static_cast<std::size_t>(i_atom)];

                    // The column layout follows read_Cs_data():
                    //   basis_i * (aux_i * basis_j) + aux_i * basis_j + basis_j.
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
            LibBSE_printf_all(Comm, "LibBSE[Chi0_BSE]: RI-k local build status: %s\n",
                              local_error.empty() ? "ok" : local_error.c_str());
            throw std::runtime_error("LibBSE[Chi0_BSE]: failed to build local k-space RI coeff");
        }

        return 0;
    }

    int get_M_mat(Enviroment& Envir, MpiComm Comm,
                  const std::vector<matrix<double>>& RI_coeff_k_real,
                  const std::vector<matrix<double>>& RI_coeff_k_imag,
                  std::vector<matrix<double>>& M_mat_real,
                  std::vector<matrix<double>>& M_mat_imag){
        std::string local_error;
        try{
            const int n_atom = Envir.n_atom;
            const int n_atom_pair = n_atom * n_atom;
            const int local_n_kpoint =
                static_cast<int>(Envir.recorded_k_points.size());
            const int n_KS_col = Envir.n_band_spin * Envir.n_band_state;

            if(static_cast<int>(Envir.n_basis_atom.size()) != n_atom
            || static_cast<int>(Envir.n_aux_basis_atom.size()) != n_atom){
                throw std::runtime_error("LibBSE[Chi0_BSE]: atom basis metadata is missing while building M");
            }
            if(static_cast<int>(RI_coeff_k_real.size()) != n_atom_pair
            || static_cast<int>(RI_coeff_k_imag.size()) != n_atom_pair){
                throw std::runtime_error("LibBSE[Chi0_BSE]: RI_coeff_k atom-pair count mismatch");
            }
            if(static_cast<int>(Envir.local_KS_eigenvector.size()) != local_n_kpoint){
                throw std::runtime_error("LibBSE[Chi0_BSE]: local KS eigenvector count does not match recorded k points");
            }

            // Build atom basis offsets inside the full KS eigenvector:
            // atom 0 occupies [0, n_basis_atom[0]), atom 1 follows it, etc.
            std::vector<int> first_basis_atom(static_cast<std::size_t>(n_atom), 0);
            for(int i_atom = 0; i_atom < n_atom - 1; ++i_atom){
                first_basis_atom[static_cast<std::size_t>(i_atom + 1)] =
                    first_basis_atom[static_cast<std::size_t>(i_atom)]
                  + Envir.n_basis_atom[static_cast<std::size_t>(i_atom)];
            }
            if(first_basis_atom[static_cast<std::size_t>(n_atom-1)] 
             + Envir.n_basis_atom[static_cast<std::size_t>(n_atom-1)] != Envir.n_basis){
                throw std::runtime_error("LibBSE[Chi0_BSE]: atom basis count does not match KS basis count");
            }
            // M_mat_real/imag:
            //   vector : <i_atom * j_atom> 
            //   matrix : <local_kpoint, n_state_spin * i_basis * i_aux_basis>
            // Each rank now keeps only the k points that it owns in
            // KS_eigenvector.  Therefore each atom-pair block has
            // local_n_kpoint rows, and row local_ik maps to
            // Envir.recorded_k_points[local_ik].
            M_mat_real.resize(static_cast<std::size_t>(n_atom_pair));
            M_mat_imag.resize(static_cast<std::size_t>(n_atom_pair));
            
            for(int i_atom = 0; i_atom < Envir.n_atom; ++i_atom){
                for(int j_atom = 0; j_atom < Envir.n_atom; ++j_atom){
                    const int block_index = i_atom * Envir.n_atom + j_atom;
                    const int n_value =
                        Envir.n_aux_basis_atom[static_cast<std::size_t>(i_atom)]
                      * Envir.n_basis_atom[static_cast<std::size_t>(i_atom)]
                      * n_KS_col;

                    // One row stores the row-major result of
                    //   (j_basis x n_state_spin).T
                    //     @ ((i_basis * i_aux_basis) x j_basis).T
                    // so the flattened column order is:
                    //   state_spin * (i_basis * i_aux_basis)
                    //     + i_basis * n_aux_basis_i + i_aux_basis.
                    M_mat_real[static_cast<std::size_t>(block_index)] =
                        matrix<double>(local_n_kpoint, n_value);
                    M_mat_imag[static_cast<std::size_t>(block_index)] =
                        matrix<double>(local_n_kpoint, n_value);
                    zeros(M_mat_real[static_cast<std::size_t>(block_index)]);
                    zeros(M_mat_imag[static_cast<std::size_t>(block_index)]);
                }
            }
            // \tilde{M}_i,n^\mu(k) = c_j,n(k) \tilde{C}_i(0),j(k)^\mu(0)
            for(int local_ik = 0; local_ik < local_n_kpoint; ++local_ik){
                auto& curr_eigenvector = Envir.local_KS_eigenvector[local_ik].value;
                if(curr_eigenvector.row != Envir.n_basis || curr_eigenvector.col != n_KS_col){
                    throw std::runtime_error("LibBSE[Chi0_BSE]: KS eigenvector matrix size mismatch while building M");
                }
                auto curr_eigenvector_split_ptr = split(curr_eigenvector, first_basis_atom);
                for(int j_atom = 0; j_atom < Envir.n_atom; ++j_atom){
                    auto curr_ptr = curr_eigenvector_split_ptr[j_atom];
                    const int n_basis_j =
                        Envir.n_basis_atom[static_cast<std::size_t>(j_atom)];
                    for(int i_atom = 0; i_atom < Envir.n_atom; ++i_atom){
                        const int n_basis_i =
                            Envir.n_basis_atom[static_cast<std::size_t>(i_atom)];
                        const int n_aux_basis_i =
                            Envir.n_aux_basis_atom[static_cast<std::size_t>(i_atom)];
                        const int block_index = i_atom * Envir.n_atom + j_atom;
                        const int RI_col =
                            n_basis_i * n_aux_basis_i * n_basis_j;
                        const int M_col =
                            n_KS_col * n_basis_i * n_aux_basis_i;
                        const matrix<double>& curr_RI_real =
                            RI_coeff_k_real[static_cast<std::size_t>(block_index)];
                        const matrix<double>& curr_RI_imag =
                            RI_coeff_k_imag[static_cast<std::size_t>(block_index)];
                        matrix<double>& curr_M_real =
                            M_mat_real[static_cast<std::size_t>(block_index)];
                        matrix<double>& curr_M_imag =
                            M_mat_imag[static_cast<std::size_t>(block_index)];

                        if(curr_RI_real.row != local_n_kpoint || curr_RI_imag.row != local_n_kpoint
                        || curr_RI_real.col != RI_col || curr_RI_imag.col != RI_col
                        || curr_M_real.col != M_col || curr_M_imag.col != M_col){
                            throw std::runtime_error("LibBSE[Chi0_BSE]: matrix size mismatch while building M");
                        }

                        // Take one k-row of C_i,j^mu(k), combine the stored real
                        // and imaginary parts, then reshape without moving data.
                        // The original layout is [i_basis][i_aux][j_basis], so
                        // this view is ((i_basis * i_aux_basis) x j_basis).
                        matrix<std::complex<double>> curr_RI_coeff(1, RI_col);
                        for(int icol = 0; icol < RI_col; ++icol){
                            curr_RI_coeff(0, icol) =
                                std::complex<double>(curr_RI_real(local_ik, icol),
                                                     curr_RI_imag(local_ik, icol));
                        }
                        if(!curr_RI_coeff.reshape(n_basis_i * n_aux_basis_i, n_basis_j)){
                            throw std::runtime_error("LibBSE[Chi0_BSE]: failed to reshape RI_coeff_k row while building M");
                        }

                        // curr_ptr is the atom-j slice of the KS eigenvector:
                        //   j_basis x n_state_spin.
                        // The product below is:
                        //   (j_basis x n_state_spin).T
                        //     @ ((i_basis * i_aux_basis) x j_basis).T
                        // giving n_state_spin x (i_basis * i_aux_basis).
                        matrix<std::complex<double>> curr_M =
                            pointer_times(curr_ptr, curr_RI_coeff, 'T', 'T',
                                          n_KS_col, n_basis_j);

                        if(curr_M.size != M_col){
                            throw std::runtime_error("LibBSE[Chi0_BSE]: M product size mismatch");
                        }
                        for(int icol = 0; icol < M_col; ++icol){
                            curr_M_real(local_ik, icol) = curr_M.matrix_ptr[icol].real();
                            curr_M_imag(local_ik, icol) = curr_M.matrix_ptr[icol].imag();
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
        return 0;
    }

    int calculate_chi0_BSE(Enviroment& Envir, MpiComm Comm){
        LibBSE_printf_root(Comm, "Start calculating chi0...\n");
        try{
            //1. convert the LRI into RI-coeff.
            // C^μ(0)_i(k),j(0) = \sum_R e^{ikR}  \tilde{C}^μ(0)_i(R),j(0)
            // Try to implement a naive mult since this is not the bottleneck.
            // TODO: implement a more efficient way like matrix flattening
            // TODO: use irkp to save memory and time.
            
            LibBSE_printf_root(Comm, "Calculating LRI coeff in k space...\n");
            // vector <i_atom * j_atom> 
            // matrix <local_kpoint, i_basis * i_aux_basis * j_basis>
            std::vector<matrix<double>> RI_coeff_k_real; 
            std::vector<matrix<double>> RI_coeff_k_imag; 
            const int ret = get_RI_k_coeff(Envir, Comm, RI_coeff_k_real, RI_coeff_k_imag);   
            LibBSE_printf_root(Comm, "Finished LRI coeff in k space!\n");

            //2. \tilde{M}_i,n^\mu(k) = c_j,n(k) \tilde{C}_i(0),j(k)^\mu(0)
            LibBSE_printf_root(Comm, "Calculating RI coeff in k space...\n");
            //   vector : <i_atom * j_atom> 
            //   matrix : <local_kpoint * n_spin, i_basis * i_aux_basis * n_state>
            // store in real & imag seperately is useful for avoid repeat matrix mult in conjuates.
            std::vector<matrix<double>> M_mat_real; 
            std::vector<matrix<double>> M_mat_imag; 
            get_M_mat(Envir, Comm, RI_coeff_k_real, RI_coeff_k_imag, M_mat_real, M_mat_imag);
            
            //Now, we have 2 ways: A) calculate brute force, store a 5-indice mat. B) calculate by q index and store 4-indice mat.
            int chi0_route = 0; // 0: A), 1: B)
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
