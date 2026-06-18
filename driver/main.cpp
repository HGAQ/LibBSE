//Global include
#include <exception>
#include <iostream>
#include <mpi.h>

//Local include
#include "../include/io/LibBSE_io.hpp"
#include "../include/mpi/LibBSE_mpi.h"
using namespace LibBSE;

int LibBSE_initialize(int argc, char **argv){
    const MpiComm LIBBSE_MPI_COMM;
    if (argc != 2) {
        return 2;
    }
    else{    
        if(LibBSE_io_init(LIBBSE_MPI_COMM) != 0){
            return 2;
        }
        else{
            LibBSE_printf_root(LIBBSE_MPI_COMM,
            "Successfully initialized io! \n");
        }
        LibBSE_printf_all(LIBBSE_MPI_COMM,
            "Test mpi, Current work directory is set in: %s .\n", argv[1]);
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
    if(LibBSE_initialize(argc, argv) != 0){
        return 1;
    }
    //
    if(LibBSE_finalized() != 0){
        return 1;
    }
    return 0;
}
