/*
    LibBSE_RI_M.cpp: 
    This is the part trying to use our own routines to calculate 
    M_mat from Cs_data and KS_eigenvector data. 
*/
#include "LibBSE_RI_M.h"
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <valarray>

namespace LibBSE{

    // atom_pair_row_count: count how many local kpoint needed for each atom pair
    // task_atom_pair_row: this is the ? th task for this atom pair, if -1, no task.
    void build_task_atom_pair_rows(const std::vector<task>& recorded_task,
                                   const int n_atom_pair,
                                   std::vector<std::vector<int>>& task_atom_pair_row,
                                   std::vector<int>& atom_pair_row_count){
        task_atom_pair_row.assign(recorded_task.size(), std::vector<int>(n_atom_pair, -1));
        atom_pair_row_count.assign(n_atom_pair, 0);

        // A row number is local to one atom-pair matrix.  
        // If this rank owns task 0: k=3 pair [2,4) and task 1: k=5 pair [3,4), 
        // then pair 3 has two rows while pair 2 only has one row. 
        for(int itask = 0; itask < static_cast<int>(recorded_task.size()); ++itask){
            const task& curr_task = recorded_task[itask];
            for(int iatom = curr_task.atom_pair_lb; iatom < curr_task.atom_pair_ub; iatom ++){
                task_atom_pair_row[itask][iatom] = atom_pair_row_count[iatom];
                atom_pair_row_count[iatom] ++;
            }
        }
    }

    void write_RI_coeff_k_real_debug(const Enviroment& Envir,
                                     const MpiComm& Comm,
                                     const std::vector<task>& recorded_task,
                                     const std::vector<matrix<double>>& RI_coeff_k_real){
        int type = 1;// 0 is style that easy to understand by human, 1 is for compare w/ FHI-aims
        const std::filesystem::path debug_dir =
            std::filesystem::current_path() / "RI_coeff_k_debug";
        std::filesystem::create_directories(debug_dir);
        const int mpi_rank = Comm.LibBSE_MPI_rank();

        const int n_atom_pair = Envir.n_atom * Envir.n_atom;
        std::vector<std::vector<int>> task_atom_pair_row;
        std::vector<int> atom_pair_row_count;
        build_task_atom_pair_rows(recorded_task, n_atom_pair,
                                  task_atom_pair_row, atom_pair_row_count);

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
                    for(int itask = 0; itask < static_cast<int>(recorded_task.size()); ++itask){
                        const int local_row = task_atom_pair_row[itask][block_index];
                        if(local_row < 0){
                            continue;
                        }
                        output << "i_kpoint = "
                               << recorded_task[static_cast<std::size_t>(itask)].task_k_point
                               << "\n";
                        for(int icol = 0; icol < block.col; ++icol){
                            output << std::setw(12) << std::fixed <<  std::setprecision(7)
                                   << block(local_row, icol);
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
                std::vector<int> first_basis_atom(Envir.n_atom, 0);
                std::vector<int> first_aux_atom(Envir.n_atom, 0);
                for(int i_atom = 1; i_atom < Envir.n_atom; ++i_atom){
                    first_basis_atom[i_atom] = first_basis_atom[i_atom - 1] + Envir.n_basis_atom[i_atom - 1];
                    first_aux_atom[i_atom] = first_aux_atom[i_atom - 1] + Envir.n_aux_basis_atom[i_atom - 1];
                }

                const std::filesystem::path file =
                    debug_dir / ("RI_coeff_k_real_rank_"
                               + std::to_string(mpi_rank)
                               + "_fhi_aims.txt");
                std::ofstream output(file);
                output << std::fixed << std::setprecision(10);

                for(int itask = 0; itask < static_cast<int>(recorded_task.size()); ++itask){
                    const task& curr_task = recorded_task[itask];
                    const int i_k_point = curr_task.task_k_point;
                    for(int block_index = curr_task.atom_pair_lb;
                        block_index < curr_task.atom_pair_ub; block_index ++){
                        const int i_atom = block_index / Envir.n_atom;
                        const int j_atom = block_index % Envir.n_atom;
                        const int n_basis_i = Envir.n_basis_atom[i_atom];
                        const int n_aux_basis_i = Envir.n_aux_basis_atom[i_atom];
                        const int n_basis_j = Envir.n_basis_atom[j_atom];
                        const matrix<double>& block = RI_coeff_k_real[block_index];
                        const int expected_col = n_basis_i * n_aux_basis_i * n_basis_j;
                        const int local_row = task_atom_pair_row[itask][block_index];
                        if(local_row < 0 || local_row >= block.row || block.col != expected_col){
                            throw std::runtime_error("LibBSE[Chi0_BSE]: RI_coeff_k matrix size mismatch in debug output");
                        }

                        // RI_coeff_k is stored as [i_basis][i_aux][j_basis].
                        // The loop order below mirrors the Fortran output:
                        // aux row first, then two NAO basis indices.
                        for(int iaux = 0; iaux < n_aux_basis_i; ++iaux){
                            const int i_aux_global = first_aux_atom[i_atom] + iaux + 1;
                            for(int ibasis_i = 0; ibasis_i < n_basis_i; ++ibasis_i){
                                const int i_basis_global = first_basis_atom[i_atom] + ibasis_i + 1;
                                for(int ibasis_j = 0; ibasis_j < n_basis_j; ++ibasis_j){
                                    const int j_basis_global = first_basis_atom[j_atom] + ibasis_j + 1;
                                    const int RI_col =
                                        ibasis_i * n_aux_basis_i * n_basis_j
                                      + iaux * n_basis_j
                                      + ibasis_j;
                                    output << std::setw(5) << i_k_point
                                           << std::setw(5) << i_aux_global
                                           << std::setw(5) << i_basis_global
                                           << std::setw(5) << j_basis_global
                                           << std::setw(15) << block(local_row, RI_col)
                                           << "\n";
                                }
                            }
                        }
                    }
                }
            }
            break;
        }
        
    }


    void allocate_task_for_rank(const Enviroment& Envir,
                                const int mpi_size,
                                const int rank,
                                std::vector<task>& recorded_task){
        recorded_task.clear();
        const int n_atom_pair = Envir.n_atom * Envir.n_atom;
        if(Envir.n_k_point <= 0 || n_atom_pair <= 0 || mpi_size <= 0){
            return;
        }

        const int total_task = Envir.n_k_point * n_atom_pair;

        // Case 1: fewer MPI ranks than k points.  Keep a whole atom-pair
        // block set together and only split by k point.
        if(mpi_size <= Envir.n_k_point){
            const int k_lb = Envir.n_k_point * rank / mpi_size;
            const int k_ub = Envir.n_k_point * (rank + 1) / mpi_size;
            for(int ik = k_lb; ik < k_ub; ++ik){
                task curr_task;
                curr_task.task_k_point = ik + 1;
                curr_task.total_rank_num = mpi_size;
                curr_task.atom_pair_lb = 0;
                curr_task.atom_pair_ub = n_atom_pair;
                recorded_task.push_back(curr_task);
            }
            return;
        }

        // Case 2/3: more MPI ranks than k points.  Split the linearized
        // (k_point, atom_pair) task space.  If there are even more ranks than
        // tasks, only the first total_task ranks receive work.
        const int active_rank_num = std::min<int>(mpi_size, total_task);
        if(rank >= active_rank_num){
            return;
        }

        int task_lb = total_task * rank / active_rank_num;
        const int task_ub = total_task * (rank + 1) / active_rank_num;
        while(task_lb < task_ub){
            const int ik = task_lb / n_atom_pair;
            const int atom_pair_lb = task_lb % n_atom_pair;
            const int next_k_task_ub = std::min<int>(task_ub, ik + 1 * n_atom_pair);
            const int atom_pair_ub = next_k_task_ub - ik * n_atom_pair;

            task curr_task;
            curr_task.task_k_point = ik + 1;
            curr_task.total_rank_num = active_rank_num;
            curr_task.atom_pair_lb = atom_pair_lb;
            curr_task.atom_pair_ub = atom_pair_ub;
            recorded_task.push_back(curr_task);

            task_lb = next_k_task_ub;
        }
    }

    int allocate_task(const Enviroment& Envir, MpiComm Comm, std::vector<task>& recorded_task){
        allocate_task_for_rank(Envir,
                               Comm.LibBSE_MPI_size(),
                               Comm.LibBSE_MPI_rank(),
                               recorded_task);

        std::ostringstream output;
        output << "RI task allocation:";
        if(recorded_task.empty()){
            output << " idle";
        }
        else{
            for(const task& curr_task: recorded_task){
                output << " k=" << curr_task.task_k_point
                       << " atom_pair=["
                       << curr_task.atom_pair_lb << ","
                       << curr_task.atom_pair_ub << ")";
            }
        }
        output << "\n";
        LibBSE_printf_all(Comm, "%s", output.str().c_str());
        return 0;
    }


    int get_LRI_k_coeff(Enviroment& Envir, MpiComm Comm,
                               const std::vector<task>& recorded_task,
                               std::vector<matrix<double>>& RI_coeff_k_real, 
                               std::vector<matrix<double>>& RI_coeff_k_imag){
        
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
        const int n_kpoint = Envir.n_k_point;

        if(Envir.n_basis_atom.size() != Envir.n_atom || Envir.n_aux_basis_atom.size() != Envir.n_atom){
            throw std::runtime_error("LibBSE[Chi0_BSE]: atom basis metadata is missing");
        }
        
        //temp vectors:
        //this is the ? th task for this atom pair, if -1, no task.
        std::vector<std::vector<int>> task_atom_pair_row;
        //each atom pair, how many k point i need to allocate?
        std::vector<int> atom_pair_row_count;
        //record the block size for each atom pair
        std::vector<int> atom_pair_count(n_atom_pair, 0);
        //cumulate the block size for each atom pair atom_pair_offset = \sum_i atom_pair_count(i)
        std::vector<int> atom_pair_offset(n_atom_pair + 1, 0);


        build_task_atom_pair_rows(recorded_task, n_atom_pair,
                                  task_atom_pair_row, atom_pair_row_count);

        RI_coeff_k_real.resize(n_atom_pair);
        RI_coeff_k_imag.resize(n_atom_pair);

        // allocate `RI_coeff_k_real/imag` base on `atom_pair_row_count`
        for(int i_atom = 0; i_atom < Envir.n_atom; ++i_atom){
            for(int j_atom = 0; j_atom < Envir.n_atom; ++j_atom){
                const int block_index = i_atom * Envir.n_atom + j_atom;
                const int block_size = Envir.n_aux_basis_atom[i_atom] * Envir.n_basis_atom[j_atom] 
                                                                      * Envir.n_basis_atom[i_atom];

                atom_pair_count[block_index] = block_size;
                atom_pair_offset[block_index + 1] = atom_pair_offset[block_index] + atom_pair_count[block_index];

                // The column layout follows read_Cs_data():
                //   basis_i * (aux_i * basis_j) + aux_i * basis_j + basis_j.
                RI_coeff_k_real[block_index] =
                    matrix<double>(atom_pair_row_count[block_index], block_size);
                RI_coeff_k_imag[block_index] =
                    matrix<double>(atom_pair_row_count[block_index], block_size);
                zeros(RI_coeff_k_real[block_index]);
                zeros(RI_coeff_k_imag[block_index]);
            }
        }

        const int mpi_size = Comm.LibBSE_MPI_size();
        const int mpi_rank = Comm.LibBSE_MPI_rank();

        // loacl function to cpy data from MPI_reduce
        auto copy_task_to_output =
            [&](const int itask, const task& curr_task, 
                const std::vector<double>& task_real, const std::vector<double>& task_imag)
            {
                const int task_offset =
                    atom_pair_offset[curr_task.atom_pair_lb];
                for(int iatom = curr_task.atom_pair_lb; iatom < curr_task.atom_pair_ub; iatom ++){
                    const int count = atom_pair_count[iatom];
                    const int local_row = task_atom_pair_row[itask][iatom];
                    const int local_offset_ll = atom_pair_offset[iatom] - task_offset;
                    const int local_offset = static_cast<int>(local_offset_ll);
                    double* target_real = RI_coeff_k_real[iatom].matrix_ptr + local_row * count;
                    double* target_imag = RI_coeff_k_imag[iatom].matrix_ptr + local_row * count;
                    for(int icol = 0; icol < count; ++icol){
                        target_real[icol] = task_real[local_offset + icol];
                        target_imag[icol] = task_imag[local_offset + icol];
                    }
                }
            };

        LibBSE_printf_root(Comm, "RI-k uses MPI tasks on k_point * atom_pair ranges.\n");

        // All ranks must follow this exact recorded_task/task order, otherwise
        // MPI_Reduce collectives would be called in different orders.  Only
        // task_rank stores the reduced task row.
        for(int task_rank = 0; task_rank < mpi_size; ++task_rank){
            for(int itask = 0; itask < recorded_task.size(); ++itask){
                const task& curr_task = recorded_task[itask];
                const int ik = curr_task.task_k_point - 1;
                if(ik < 0 || ik >= n_kpoint){
                    throw std::runtime_error("LibBSE[Chi0_BSE]: task k point is outside k-point list");
                }
                if(curr_task.atom_pair_lb < 0
                || curr_task.atom_pair_ub > n_atom_pair
                || curr_task.atom_pair_lb >= curr_task.atom_pair_ub){
                    throw std::runtime_error("LibBSE[Chi0_BSE]: invalid atom-pair task range");
                }

                const int task_offset = atom_pair_offset[curr_task.atom_pair_lb];
                const int task_count_ll = atom_pair_offset[curr_task.atom_pair_ub] - task_offset;
                if(task_count_ll <= 0
                || task_count_ll > std::numeric_limits<int>::max()){
                    throw std::runtime_error("LibBSE[Chi0_BSE]: task size is invalid for MPI_Reduce");
                }
                const int task_count = static_cast<int>(task_count_ll);

                std::vector<double> send_real(task_count, 0.0);
                std::vector<double> send_imag(task_count, 0.0);
                for(int iatom = curr_task.atom_pair_lb; iatom < curr_task.atom_pair_ub; iatom ++){
                    const int count = atom_pair_count[iatom];
                    const int local_offset = atom_pair_offset[iatom] - task_offset;
                    double* send_real_ptr = send_real.data() + local_offset;
                    double* send_imag_ptr = send_imag.data() + local_offset;

                    for(const RIBlock curr_RI_block: Envir.local_RI_coeff){
                        const double phase_real = std::cos(curr_RI_block.angle_vect(0,ik));
                        const double phase_imag = std::sin(curr_RI_block.angle_vect(0,ik));
                        const double* value_ptr = curr_RI_block.value.matrix_ptr;
                        for(int icol = 0; icol < count; ++icol){
                            send_real_ptr[icol] += phase_real * value_ptr[icol];
                            send_imag_ptr[icol] += phase_imag * value_ptr[icol];
                        }
                    }
                }

                std::vector<double> recv_real;
                std::vector<double> recv_imag;
                if(mpi_rank == task_rank){
                    recv_real.resize(task_count);
                    recv_imag.resize(task_count);
                }

                MPI_Reduce(send_real.data(),
                           mpi_rank == task_rank ? recv_real.data() : nullptr,
                           task_count, MPI_DOUBLE, MPI_SUM,
                           task_rank, Comm.LibBSE_MPI_raw());
                MPI_Reduce(send_imag.data(),
                           mpi_rank == task_rank ? recv_imag.data() : nullptr,
                           task_count, MPI_DOUBLE, MPI_SUM,
                           task_rank, Comm.LibBSE_MPI_raw());

                if(mpi_rank == task_rank){
                    copy_task_to_output(itask, curr_task, recv_real, recv_imag);
                }
            }
        }
        if(!recorded_task.empty()){
            write_RI_coeff_k_real_debug(Envir, Comm, recorded_task, RI_coeff_k_real);
        }
        
        return 0;
    }

    int get_M_mat(Enviroment& Envir, MpiComm Comm,
                  const std::vector<task>& recorded_task,
                  const std::vector<matrix<double>>& RI_coeff_k_real,
                  const std::vector<matrix<double>>& RI_coeff_k_imag,
                  std::vector<matrix<complex>>& M_mat){
        
        const int n_atom = Envir.n_atom;
        const int n_atom_pair = n_atom * n_atom;
        const int local_n_kpoint = static_cast<int>(Envir.recorded_k_points.size());
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

        //temp vectors:
        //this is the ? th task for this atom pair, if -1, no task.
        std::vector<std::vector<int>> task_atom_pair_row;
        //each atom pair, how many k point i need to allocate?
        std::vector<int> atom_pair_row_count;
        // Build atom basis offsets inside the full KS eigenvector:
        // atom 0 occupies [0, n_basis_atom[0]), atom 1 follows it, etc.
        std::vector<int> first_basis_atom(n_atom, 0);

        build_task_atom_pair_rows(recorded_task, n_atom_pair,
                                  task_atom_pair_row, atom_pair_row_count);
        for(int i_atom = 0; i_atom < n_atom - 1; ++i_atom){
            first_basis_atom[i_atom + 1] =
            first_basis_atom[i_atom] + Envir.n_basis_atom[i_atom];
        }
        if(first_basis_atom[n_atom-1] + Envir.n_basis_atom[n_atom-1] != Envir.n_basis){
            throw std::runtime_error("LibBSE[Chi0_BSE]: atom basis count does not match KS basis count");
        }

        // Find which MPI rank owns each KS eigenvector row.  The RI/M task
        // owner may be a different rank now, so get_M_mat broadcasts one
        // k-point eigenvector at a time from this owner to all task ranks.
        const int mpi_size = Comm.LibBSE_MPI_size();
        const int mpi_rank = Comm.LibBSE_MPI_rank();
        const int local_recorded_count = local_n_kpoint;
        std::vector<int> recorded_counts(mpi_size, 0);
        MPI_Allgather(&local_recorded_count, 1, MPI_INT,
                      recorded_counts.data(), 1, MPI_INT,
                      Comm.LibBSE_MPI_raw());

        std::vector<int> recorded_displs(mpi_size, 0);
        int total_recorded_count = 0;
        for(int irank = 0; irank < mpi_size; ++irank){
            recorded_displs[irank] = total_recorded_count;
            total_recorded_count += recorded_counts[irank];
        }

        std::vector<int> all_recorded_k_points(total_recorded_count, 0);
        MPI_Allgatherv(local_recorded_count ? Envir.recorded_k_points.data() : nullptr,
                       local_recorded_count, MPI_INT,
                       total_recorded_count ? all_recorded_k_points.data() : nullptr,
                       recorded_counts.data(),
                       recorded_displs.data(),
                       MPI_INT,
                       Comm.LibBSE_MPI_raw());

        //temp vectors:
        std::vector<int> k_owner_rank(Envir.n_k_point + 1, -1);
        std::vector<int> k_owner_local_ik(Envir.n_k_point + 1, -1);

        for(int target_rank = 0; target_rank < mpi_size; ++target_rank){
            const int target_count = recorded_counts[target_rank];
            const int target_displ = recorded_displs[target_rank];
            for(int target_local_ik = 0; target_local_ik < target_count; ++target_local_ik){
                const int i_kpoint = all_recorded_k_points[target_displ + target_local_ik];
                if(i_kpoint <= 0 || i_kpoint > Envir.n_k_point){
                    throw std::runtime_error("LibBSE[Chi0_BSE]: recorded k point is outside k-point list while building M");
                }
                k_owner_rank[i_kpoint] = target_rank;
                k_owner_local_ik[i_kpoint] = target_local_ik;
            }
        }

        // M_mat:
        //   vector : <i_atom * j_atom> 
        //   matrix : <local task rows for this atom pair,
        //             n_state_times_spin * i_basis * i_aux_basis>
        // Rows follow the same task/atom-pair mapping as RI_coeff_k.
        M_mat.resize(n_atom_pair);
        
        for(int i_atom = 0; i_atom < Envir.n_atom; ++i_atom){
            for(int j_atom = 0; j_atom < Envir.n_atom; ++j_atom){
                const int block_index = i_atom * Envir.n_atom + j_atom;
                const int n_value = Envir.n_aux_basis_atom[i_atom] * Envir.n_basis_atom[i_atom] * n_KS_col;
                // One row stores the row-major result of
                //   (j_basis x n_state_spin).T
                //     @ ((i_basis * i_aux_basis) x j_basis).T
                // so the flattened column order is:
                //   state_spin * (i_basis * i_aux_basis)
                //     + i_basis * n_aux_basis_i + i_aux_basis.
                M_mat[block_index] = matrix<complex>(atom_pair_row_count[block_index], n_value);
                zeros(M_mat[block_index]);
            }
        }

        // \tilde{M}_i,n^\mu(k) = c_j,n(k) \tilde{C}_i(0),j(k)^\mu(0)
        // temp mat
        matrix<complex> curr_eigenvector(Envir.n_basis, n_KS_col);

        for(int i_kpoint = 1; i_kpoint <= Envir.n_k_point; ++i_kpoint){
            const int owner_rank = k_owner_rank[i_kpoint];
            if(owner_rank < 0){
                throw std::runtime_error("LibBSE[Chi0_BSE]: missing KS owner rank while building M");
            }

            if(mpi_rank == owner_rank){
                const int owner_local_ik = k_owner_local_ik[i_kpoint];
                curr_eigenvector = Envir.local_KS_eigenvector[owner_local_ik].value;
            }

            // The task owner may not own the KS eigenvector.  Broadcast the
            // current k vector once, compute all local atom-pair tasks for
            // this k, then reuse the buffer for the next k.
            MPI_Bcast(reinterpret_cast<double*>(curr_eigenvector.matrix_ptr),
                      curr_eigenvector.size * 2,
                      MPI_DOUBLE, owner_rank, Comm.LibBSE_MPI_raw());

            auto curr_eigenvector_split_ptr = split(curr_eigenvector, first_basis_atom);
            for(int itask = 0; itask < static_cast<int>(recorded_task.size()); ++itask){
                const task& curr_task = recorded_task[itask];
                if(curr_task.task_k_point != i_kpoint){
                    continue;
                }
                for(int block_index = curr_task.atom_pair_lb;
                    block_index < curr_task.atom_pair_ub; block_index ++){
                    const int i_atom = block_index / Envir.n_atom;
                    const int j_atom = block_index % Envir.n_atom;

                    auto curr_ptr = curr_eigenvector_split_ptr[j_atom];
                    const int local_row = task_atom_pair_row[itask][block_index];

                    const int n_basis_j = Envir.n_basis_atom[j_atom];
                    const int n_basis_i = Envir.n_basis_atom[i_atom];
                    const int n_aux_basis_i = Envir.n_aux_basis_atom[i_atom];

                    const int RI_col = n_basis_i * n_aux_basis_i * n_basis_j;
                    const int M_col = n_KS_col * n_basis_i * n_aux_basis_i;
                    const matrix<double>& curr_RI_real = RI_coeff_k_real[block_index];
                    const matrix<double>& curr_RI_imag = RI_coeff_k_imag[block_index];
                    matrix<complex>& curr_M_block = M_mat[block_index];

                    if(local_row < 0 || local_row >= curr_RI_real.row
                    || local_row >= curr_RI_imag.row
                    || local_row >= curr_M_block.row
                    || curr_RI_real.col != RI_col || curr_RI_imag.col != RI_col
                    || curr_M_block.col != M_col){
                        throw std::runtime_error("LibBSE[Chi0_BSE]: matrix size mismatch while building M");
                    }

                    // Take one k-row of C_i,j^mu(k), combine the stored real
                    // and imaginary parts, then reshape without moving data.
                    // The original layout is [i_basis][i_aux][j_basis], so
                    // this view is ((i_basis * i_aux_basis) x j_basis).
                    matrix<complex> curr_RI_coeff(1, RI_col);
                    for(int icol = 0; icol < RI_col; ++icol){
                        curr_RI_coeff(0, icol) =
                            complex(curr_RI_real(local_row, icol),
                                                 curr_RI_imag(local_row, icol));
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
                    matrix<complex> curr_M =
                        pointer_times(curr_ptr, curr_RI_coeff, 'T', 'T',
                                      n_KS_col, n_basis_j);

                    if(curr_M.size != M_col){
                        throw std::runtime_error("LibBSE[Chi0_BSE]: M product size mismatch");
                    }
                    for(int icol = 0; icol < M_col; ++icol){
                        curr_M_block(local_row, icol) = curr_M.matrix_ptr[icol];
                    }
                }
            }
        }
        return 0;
    }


    int calculate_M_mat(Enviroment& Envir, MpiComm Comm){
        LibBSE_printf_root(Comm, "Start calculating chi0...\n");
        
        // TODO     : implement a more efficient way like matrix flattening
        // TODO     : use irkp to save memory and time.
        // TODO     : parellel in k is not efficient. use openmp?
        // TODO     : reduce communication?
        try{
            report_matrix_memory(Comm, "chi0 start");
            
            //used in RI task allocation
            //std::vector<task> recorded_task;

            allocate_task(Envir, Comm, Envir.recorded_task);
            //1. convert the LRI into RI-coeff.
            // C^μ(0)_i(k),j(0) = \sum_R e^{ikR}  \tilde{C}^μ(0)_i(R),j(0)
            // Try to implement a naive mult since this is not the bottleneck.
            
            LibBSE_printf_root(Comm, "Calculating LRI coeff in k space...\n");
            // vector <local_atom_pair> 
            // matrix <local_kpoint, i_basis * i_aux_basis * j_basis>
            std::vector<matrix<double>> RI_coeff_k_real; 
            std::vector<matrix<double>> RI_coeff_k_imag; 
            const int ret = get_LRI_k_coeff(Envir, Comm, Envir.recorded_task,
                                            RI_coeff_k_real, RI_coeff_k_imag);
            LibBSE_printf_root(Comm, "Finished LRI coeff in k space!\n");
            report_matrix_memory(Comm, "after RI_coeff_k");

            //2. \tilde{M}_i,n^\mu(k) = c_j,n(k) \tilde{C}_i(0),j(k)^\mu(0)
            LibBSE_printf_root(Comm, "Calculating RI coeff in k space...\n");
            //   vector : <local_atom_pair> 
            //   matrix : <local_kpoint, n_state_spin * i_basis * i_aux_basis>
            //std::vector<matrix<complex>> M_mat;
            get_M_mat(Envir, Comm, Envir.recorded_task,
                      RI_coeff_k_real, RI_coeff_k_imag, Envir.M_mat);
            // deallocate RI_coeff_k_real;
            RI_coeff_k_real.clear();
            RI_coeff_k_imag.clear();
            report_matrix_memory(Comm, "after M_mat");
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
