#pragma once

#include <mpi.h>
#include <string>

namespace LibBSE {
extern bool LibBSE_mpi_initialized;

class MpiRuntime {
public:
    MpiRuntime(int& argc, char**& argv);
    MpiRuntime(const MpiRuntime&) = delete;
    MpiRuntime& operator=(const MpiRuntime&) = delete;
    ~MpiRuntime();
    inline bool Mpi_isinitalized(){return MPI_initialized;}

private:
    bool MPI_initialized = false;
};

class MpiComm {
public:
    explicit MpiComm(MPI_Comm comm = MPI_COMM_WORLD);
    MPI_Comm LibBSE_MPI_raw() const { return comm_; }
    int LibBSE_MPI_size() const;
    int LibBSE_MPI_rank() const;
    bool LibBSE_MPI_is_root() const { return LibBSE_MPI_rank() == 0; }
    void LibBSE_MPI_barrier() const;

private:
    MPI_Comm comm_ = MPI_COMM_WORLD;
};

} // namespace libbse
