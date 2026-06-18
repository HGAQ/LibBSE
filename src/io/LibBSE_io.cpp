#include "../../include/io/LibBSE_io.hpp"

#include <fstream>
#include <mpi.h>

int LibBSE::LibBSE_io_init(MpiComm Mpi_COMM, bool redirect_stdout = false, const char *redirect_path = "LibBSE.out"){
    std::string s_fn(redirect_path);
    if (redirect_stdout && s_fn == "")
    {
        return 2;
    }
    if (redirect_stdout && s_fn != "stdout")
    {
        // Rank 0 touches the output file
        Mpi_COMM.LibBSE_MPI_barrier();
        if (Mpi_COMM.LibBSE_MPI_is_root() == 0)
        {
            std::ofstream ofs(redirect_path);
            ofs.close();
        }
        Mpi_COMM.LibBSE_MPI_barrier();
        // All process appending to the refreshed buffer
        ofs.open(redirect_path, std::ios::out | std::ios::app);
        if (!ofs.is_open()) {
            return 2;
        }
        cout_buf_old = std::cout.rdbuf();
        std::cout.rdbuf(ofs.rdbuf());
        // redirecting printf
        redirect_file = fopen(redirect_path, "a");
    }
    LibBSE_io_initialized = true;
}

void LibBSE::LibBSE_io_finalized(){
    // Recover original stdout behavior
    if (redirect_file != nullptr)
    {
        std::cout.rdbuf(cout_buf_old);
        fclose(redirect_file);
        redirect_file = nullptr;
    }
    LibBSE_io_initialized = false;
}