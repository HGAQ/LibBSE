#pragma once
//Global include
#include <cstdio>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <string>
#include <utility>
#include <algorithm>
#include <complex>
#include <vector>

//Local include
#include "../mpi/LibBSE_mpi.h"
#include "../math/Tensor.h"
#include "../math/Matrix.h"
#include "LibBSE_io.hpp"

namespace fs = std::filesystem;

namespace LibBSE{
    typedef std::pair<std::string, std::vector<double>> AtomPos;

    struct FileAssignment {
        bool read_file = false;
        int first_rank = 0;
        int rank_count = 1;
        int local_rank = 0;
    };

    struct BandEntry{
        public:
            double band_occ;
            double E_band;
    };

    struct BandVect{
        public:
            int i_k_point;
            int i_band_spin;
            std::vector<BandEntry> Band;
    };

    struct CoulombBlock {
        int file_index = -1;
        //int global_block_index = -1;
        //int global_aux_size = 0;
        int row_first = 0, row_last = 0;
        int col_first = 0, col_last = 0;
        int i_k_point = 0;
        double weight_k_point = 0.0;
        std::vector<std::complex<double>> value;
    };

    struct KSBlock {
        int file_index = -1;
        int i_k_point = 0;
        int i_band_spin = 0;
        int i_state = 0;
        std::vector<std::complex<double>> value;
    };

    struct RIBlock {
        int file_index = -1;
        int i_atom = 0, j_atom = 0;
        int n_1 = 0, n_2 = 0, n_3 = 0;
        int n_basis_i = 0, n_basis_j = 0, n_aux_basis_i = 0;
        matrix<double> value;
        matrix<double> angle_vect;
    };

    struct Enviroment{
        public:
            Enviroment(){}
            //dataset root
            std::string dataset_dir;
            //band_out inputs
            int n_k_point;
            int n_band_spin;
            int n_band_state;
            int n_basis;
            int n_aux_basis; //aux basis num
            double E_Fermi; //in Hartree
            //stru_out inputs
            matrix<double> lattice_vect;
            matrix<double> reciprocal_vect;
            std::vector<int> k_point_dim;
            matrix<double> k_point_list;
            std::vector<int> map_from_FullBZ_to_IBZ;
            std::vector<BandVect> KS_Band;
            int ir_k_point;
            //vxc_out inputs
            std::vector<std::vector<std::vector<double>>> vxc; // vxc[i_kpoint] [i_spin] [i_state]
            //geometry.in inputs
            std::vector<AtomPos> atoms_pos;
            int n_atom;
            //coulomb inputs
            std::vector<CoulombBlock> local_coulomb_cut; // Coulomb_cut [i_kpoint][n_aux_basis][n_aux_basis]
            std::vector<CoulombBlock> local_coulomb_mat; // Coulomb_mat [i_kpoint][n_aux_basis][n_aux_basis]
            //RI inputs
            std::vector<RIBlock> local_RI_coeff; // RI_coeff [ncell] [n_aux_basis_i] [n_basis_j] [n_basis_i]
            std::vector<int> n_basis_atom; // n_basis_atom[i_atom]
            std::vector<int> n_aux_basis_atom; // n_aux_basis_atom[i_atom]
            //KS_eigvect inputs
            std::vector<KSBlock> local_KS_eigenvector;
            std::vector<int> recorded_k_points;
    };
    
    struct IndexedFile {
        int index = -1;
        fs::path path;
    };

    std::vector<IndexedFile> find_indexed_files(const fs::path& directory, const std::string& prefix, const std::string& suffix);
     
    int read_aims_output(const std::string& directory, Enviroment& Envir, const MpiComm& Comm);
    int read_band_out(const fs::path directory, Enviroment& Envir);
    int read_struct_out(const fs::path directory, Enviroment& Envir);
    int read_vxc_out(const fs::path directory, Enviroment& Envir);
    int read_geometry_in(const fs::path directory, Enviroment& Envir);
    int read_coulomb_cut(const fs::path directory, Enviroment& Envir, const MpiComm& Comm);
    int read_coulomb_mat(const fs::path directory, Enviroment& Envir, const MpiComm& Comm);
    int read_Cs_data(const fs::path directory, Enviroment& Envir, const MpiComm& Comm);
    int read_KS_eigenvector(const fs::path directory, Enviroment& Envir, const MpiComm& Comm);
    bool assigned_to_rank(int block_index, const MpiComm& Comm);
}
