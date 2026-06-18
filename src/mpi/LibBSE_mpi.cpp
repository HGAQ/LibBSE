#include "../../include/mpi/LibBSE_mpi.h"

#include <stdexcept>

namespace LibBSE {

MpiRuntime::MpiRuntime(int& argc, char**& argv)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided = MPI_THREAD_SINGLE;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_initialized = true;
        LibBSE_mpi_initialized = true;
    }
}

MpiRuntime::~MpiRuntime()
{
    if (MPI_initialized) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
        MPI_initialized = false;
        LibBSE_mpi_initialized = false;
    }
}

MpiComm::MpiComm(MPI_Comm comm) : comm_(comm)
{
    if (comm_ == MPI_COMM_NULL) {
        throw std::invalid_argument("MpiComm cannot wrap MPI_COMM_NULL");
    }
}
int MpiComm::LibBSE_MPI_size() const
{
    int value = 1;
    MPI_Comm_size(comm_, &value);
    return value;
}

int MpiComm::LibBSE_MPI_rank() const
{
    int value = 0;
    MPI_Comm_rank(comm_, &value);
    return value;
}

void MpiComm::LibBSE_MPI_barrier() const
{
    MPI_Barrier(comm_);
}

} // namespace LibBSE

