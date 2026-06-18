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
#include "LibBSE_io.hpp"

namespace fs = std::filesystem;

namespace LibBSE{
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

    struct RIBlock {
        int file_index = -1;
        int i_atom = 0, j_atom = 0;
        int n_1 = 0, n_2 = 0, n_3 = 0;
        int n_basis_i = 0, n_basis_j = 0, n_aux_basis_i = 0;
        std::vector<double> value;
    };

    struct KSBlock {
        int file_index = -1;
        int i_k_point = 0;
        int i_band_spin = 0;
        int i_state = 0;
        std::vector<std::complex<double>> value;
    };


    struct Enviroment{
        public:
            Enviroment(){}
            //band_out inputs
            int n_k_point;
            int n_band_spin;
            int n_band_state;
            int n_basis;
            double E_Fermi; //in Hartree
            //stru_out inputs
            std::vector<std::vector<double>> lattice_vect;
            std::vector<std::vector<double>> reciprocal_vect;
            std::vector<int> k_point_dim;
            std::vector<std::vector<double>> k_point_list;
            std::vector<int> map_from_FullBZ_to_IBZ;
            std::vector<BandVect> KS_Band;
            int ir_k_point;
            //vxc_out inputs
            std::vector<std::vector<std::vector<double>>> vxc;
            //coulomb inputs
            std::vector<CoulombBlock> local_coulomb_cut;
            std::vector<CoulombBlock> local_coulomb_mat;
            //RI inputs
            std::vector<RIBlock> local_RI_coeff;
            //KS_eigvect inputs
            std::vector<KSBlock> local_KS_eigenvector;
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
    int read_coulomb_cut(const fs::path directory, Enviroment& Envir, const MpiComm& Comm);
    int read_coulomb_mat(const fs::path directory, Enviroment& Envir, const MpiComm& Comm);
    int read_Cs_data(const fs::path directory, Enviroment& Envir, const MpiComm& Comm);
    int read_KS_eigenvector(const fs::path directory, Enviroment& Envir, const MpiComm& Comm);
    bool assigned_to_rank(int block_index, const MpiComm& Comm);
}
