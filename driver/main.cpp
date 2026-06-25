//Global include
#include <exception>
#include <iostream>
#include <mpi.h>

//Local include
#include "../src/io/LibBSE_io.hpp"
#include "../src/io/Read_aims_output.h"
#include "../src/mpi/LibBSE_mpi.h"
#include "../src/RI/LibBSE_RI_M.h"
#include "../src/RI/LibBSE_RI_coeff.h"

using namespace LibBSE;

int LibBSE_initialize(int argc, char **argv){
    const MpiComm LIBBSE_MPI_COMM;
    if (argc != 2) {
        std::cerr << "Usage: libbse_parallel_dataset_info <FHI-aims text dataset directory>\n";
        return 2;
    }
    else{    
        if(LibBSE_io_init(LIBBSE_MPI_COMM) != 0){
            std::cerr << "LibBSE[IO]: LibBSE io initialize error!\n";
            return 2;
        }
        else{
            LibBSE_printf_root(LIBBSE_MPI_COMM,
            "This is LibBSE, version 0.0.1 \n"
            "========================================\n"
            "Successfully initialized io! \n");
        }
        LibBSE_printf_all(LIBBSE_MPI_COMM,
            "Current work directory is set in: %s .\n", argv[1]);
        return 0;
    }
    return 0;
}

int LibBSE_finalized(){
    
    LibBSE_io_finalized();
    return 0;
}

int main(int argc, char** argv)
{
    MpiRuntime LibBSE_MPI(argc, argv);
    const MpiComm LIBBSE_MPI_COMM;

    if(LibBSE_initialize(argc, argv) != 0){
        return 1;
    }
    //start reading aims output;
    Enviroment Enviro;
    if (read_aims_output(argv[1], Enviro, LIBBSE_MPI_COMM) != 0) {
        return 1;
    }
    //start main loop of BSE;
    if (calculate_M_mat(Enviro, LIBBSE_MPI_COMM) != 0) {
        return 1;
    }

    if(LibBSE_finalized() != 0){
        return 1;
    }
    return 0;
}
