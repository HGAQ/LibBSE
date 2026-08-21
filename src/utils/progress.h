#pragma once

#include <mpi.h>

#include <string>

namespace libbse
{

//! Print an cumulative completion marker on communicator rank 0.
void done(const std::string &description, MPI_Comm comm = MPI_COMM_WORLD);

} // namespace libbse
