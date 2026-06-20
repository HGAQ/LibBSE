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
                           const int ik,
                           const int expected_count,
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

        // Root receives the summed row directly into RI_coeff_k_real/imag.
        // Non-root recvbuf is ignored by MPI_Reduce.
        double* recv_row = nullptr;
        if(Comm.LibBSE_MPI_is_root() && root_matrix != nullptr){
            recv_row = root_matrix->matrix_ptr + ik * root_matrix->col;
        }
        MPI_Reduce(send_row.matrix_ptr, recv_row,
                   expected_count, MPI_DOUBLE, MPI_SUM,
                   0, Comm.LibBSE_MPI_raw());
    }


    void write_RI_coeff_k_real_debug(const Enviroment& Envir,
                                     const std::vector<matrix<double>>& RI_coeff_k_real){
        const std::filesystem::path debug_dir =
            std::filesystem::current_path() / "RI_coeff_k_debug";
        std::filesystem::create_directories(debug_dir);

        for(int i_atom = 0; i_atom < Envir.n_atom; ++i_atom){
            for(int j_atom = 0; j_atom < Envir.n_atom; ++j_atom){
                const int block_index = i_atom * Envir.n_atom + j_atom;
                const matrix<double>& block =
                    RI_coeff_k_real[static_cast<std::size_t>(block_index)];
                if(block.size == 0){
                    continue;
                }

                const std::filesystem::path file =
                    debug_dir / ("RI_coeff_k_real_atom_"
                               + std::to_string(i_atom) + "_"
                               + std::to_string(j_atom) + ".txt");
                std::ofstream output(file);
                output << "# RI_coeff_k_real\n";
                output << "# i_atom " << i_atom << " j_atom " << j_atom << "\n";
                output << "# rows: k_point, columns: aux_i * basis_j * basis_i\n";
                output << "# n_kpoint " << block.row << " n_value " << block.col << "\n";
                for(int ik = 0; ik < block.row; ++ik){
                    output << "i_kpoint = " << ik << "\n";
                    for(int icol = 0; icol < block.col; ++icol){
                        output << std::setw(12) << std::fixed <<  std::setprecision(7) << block(ik, icol);
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

            
            if(static_cast<int>(Envir.n_basis_atom.size()) != Envir.n_atom
            || static_cast<int>(Envir.n_aux_basis_atom.size()) != Envir.n_atom){
                throw std::runtime_error("LibBSE[Chi0_BSE]: atom basis metadata is missing");
            }

            if(Comm.LibBSE_MPI_is_root()){
                RI_coeff_k_real.resize(static_cast<std::size_t>(n_atom_pair));
                RI_coeff_k_imag.resize(static_cast<std::size_t>(n_atom_pair));
                for(int i_atom = 0; i_atom < Envir.n_atom; ++i_atom){
                    for(int j_atom = 0; j_atom < Envir.n_atom; ++j_atom){
                        const int block_index = i_atom * Envir.n_atom + j_atom;
                        const int n_value =
                            Envir.n_aux_basis_atom[static_cast<std::size_t>(i_atom)]
                          * Envir.n_basis_atom[static_cast<std::size_t>(j_atom)]
                          * Envir.n_basis_atom[static_cast<std::size_t>(i_atom)];

                        // Each atom pair owns one dense block over all k points.
                        // The column layout follows read_Cs_data():
                        //   aux_i * (basis_j * basis_i) + basis_j * basis_i + basis_i.
                        RI_coeff_k_real[static_cast<std::size_t>(block_index)] =
                            matrix<double>(n_kpoint, n_value);
                        RI_coeff_k_imag[static_cast<std::size_t>(block_index)] =
                            matrix<double>(n_kpoint, n_value);
                        zeros(RI_coeff_k_real[static_cast<std::size_t>(block_index)]);
                        zeros(RI_coeff_k_imag[static_cast<std::size_t>(block_index)]);
                    }
                }
            }

            std::vector<matrix<double>> curr_RI_coeff_k_addon_real;
            std::vector<matrix<double>> curr_RI_coeff_k_addon_imag;
            curr_RI_coeff_k_addon_real.resize(static_cast<std::size_t>(n_atom_pair));
            curr_RI_coeff_k_addon_imag.resize(static_cast<std::size_t>(n_atom_pair));
            std::string local_reduce_error;

            //mult the phase vector for each Local block
            for(int ik = 0; ik < n_kpoint; ++ik){
                for(const auto& curr_RI_block: Envir.local_RI_coeff){
                    //constuct the local matrix to store the data.
                    //since we are not sure it is in which atom 
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

                    matrix<double>* root_real = Comm.LibBSE_MPI_is_root()
                        ? &RI_coeff_k_real[static_cast<std::size_t>(iatom)] : nullptr;
                    matrix<double>* root_imag = Comm.LibBSE_MPI_is_root()
                        ? &RI_coeff_k_imag[static_cast<std::size_t>(iatom)] : nullptr;

                    reduce_sum_RI_row(curr_RI_coeff_k_addon_real[static_cast<std::size_t>(iatom)],
                                      root_real, ik, expected_count, Comm, local_reduce_error);
                    reduce_sum_RI_row(curr_RI_coeff_k_addon_imag[static_cast<std::size_t>(iatom)],
                                      root_imag, ik, expected_count, Comm, local_reduce_error);

                    zeros(curr_RI_coeff_k_addon_real[static_cast<std::size_t>(iatom)]);
                    zeros(curr_RI_coeff_k_addon_imag[static_cast<std::size_t>(iatom)]);
                }
            }
            if(!local_reduce_error.empty()){
                throw std::runtime_error(local_reduce_error);
            }
            if(Comm.LibBSE_MPI_is_root()){
                write_RI_coeff_k_real_debug(Envir, RI_coeff_k_real);
                n_RI_coeff_k = RI_coeff_k_real.size();
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

        if(!Comm.LibBSE_MPI_is_root()){
            return 0;
        }
        return n_RI_coeff_k;
    }

    int get_M_mat(Enviroment& Envir, MpiComm Comm, 
                  std::vector<matrix<double>>& RI_coeff_k_real, 
                  std::vector<matrix<double>>& RI_coeff_k_imag){
        
    }

    int calculate_chi0_BSE(Enviroment& Envir, MpiComm Comm){
        LibBSE_printf_root(Comm, "Start calculating chi0...\n");
        try{
            //1. convert the LRI into RI-coeff.
            // C^μ(0)_i(k),j(0) = \sum_R e^{ikR}  \tilde{C}^μ(0)_i(R),j(0)
            // Try to implement a naive mult since this is not the bottleneck.
            // TODO: implement a more efficient way like matrix flattening
            
            // vector <i_atom * j_atom> 
            // matrix <n_kpoint, i_basis * j_basis * i_aux_basis>
            std::vector<matrix<double>> RI_coeff_k_real; 
            std::vector<matrix<double>> RI_coeff_k_imag; 
            const int n_RI_coeff_k = get_RI_k_coeff(Envir, Comm, RI_coeff_k_real, RI_coeff_k_imag);            
            //2. \tilde{M}_i,n^\mu(k) = c_j,n(k) \tilde{C}_i(0),j(k)^\mu(0)
            get_M_mat(Envir, Comm, RI_coeff_k_real, RI_coeff_k_imag);

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
