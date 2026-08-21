#include "utils/progress.h"

#include <mpi.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_status = 0;
    try
    {
        std::ostringstream capture;
        auto *original = std::cout.rdbuf(capture.rdbuf());
        libbse::done("test completion marker", MPI_COMM_WORLD);
        std::cout.rdbuf(original);

        const std::string output = capture.str();
        if (rank == 0)
        {
            if (output.find(" DONE(") == std::string::npos
                || output.find(" SEC) : test completion marker")
                       == std::string::npos)
                throw std::runtime_error("rank 0 DONE output has the wrong format");
        }
        else if (!output.empty())
            throw std::runtime_error("non-root rank printed a DONE marker");
    }
    catch (const std::exception &error)
    {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
        local_status = 1;
    }

    int global_status = 0;
    MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Finalize();
    return global_status;
}
