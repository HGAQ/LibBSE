#pragma once
//Global include
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <utility>
#include <algorithm>
#include <vector>
#include <array>

//Local include
#include "../mpi/LibBSE_mpi.h"
#include "LibBSE_io.hpp"


namespace LibBSE{
    class BandEntry{
        public:
            double band_occ;
            double E_band;
    };

    class Enviroment{
        public:
            Enviroment(){}
            int n_k_point;
            int n_band;
            double E_Fermi; //in Hartree
            std::vector<BandEntry> KS_Band;
    };

    int read_aims_output(char* directory, Enviroment& Envir);
    int read_band_out(char* directory, Enviroment& Envir);
}


