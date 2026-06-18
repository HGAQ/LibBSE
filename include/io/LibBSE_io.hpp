#pragma once
//Global include
#include <fstream>
#include <mpi.h>

//Local include
#include "../mpi/LibBSE_mpi.h"

namespace LibBSE{

    static std::ofstream ofs;
    static std::streambuf *cout_buf_old = nullptr;
    static bool LibBSE_io_initialized = false;
    FILE *redirect_file = nullptr;

    int LibBSE_io_init(MpiComm Mpi_COMM, bool redirect_stdout = false, const char *redirect_path = "LibBSE.out");
    void LibBSE_io_finalized();

    template <typename... Args>
    void LibBSE_printf(const char* s, Args&&... args) noexcept
    {
        if (redirect_file != nullptr)
        {
            std::fprintf(redirect_file, s, std::forward<Args>(args)...);
            std::fflush(redirect_file);
        }
        else
        {
            std::printf(s, std::forward<Args>(args)...);
        }
    }

    inline void LibBSE_printf(const char* s) noexcept
    {
        if (redirect_file)
        {
            std::fprintf(redirect_file, "%s", s);
            std::fflush(redirect_file);
        }
        else
        {
            std::printf("%s", s);
        }
    }

    template <typename... Args>
    void LibBSE_printf_root(LibBSE::MpiComm MPI_COMM, const char* s, Args&&... args) noexcept
    {
        if (MPI_COMM.LibBSE_MPI_is_root() == 0)
        {
            LibBSE_printf(s, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void LibBSE_printf_all(LibBSE::MpiComm MPI_COMM, const char* s, Args&&... args) noexcept
    {
        for (int i = 0; i < MPI_COMM.LibBSE_MPI_size(); i++)
        {
            if (MPI_COMM.LibBSE_MPI_rank() == i)
            {
                LibBSE_printf("Rank = ", i, ": ", s, std::forward<Args>(args)...);
            }
            MPI_Barrier(mpi_comm_global);
        }
    }
    
}