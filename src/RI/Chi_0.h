#include "../io/LibBSE_io.hpp"
#include "../io/Read_aims_output.h"
#include "../mpi/LibBSE_mpi.h"
#include <RI/physics/RPA.h>
#include <array>
#include <complex>
#include <map>

namespace LibBSE{
    // LibRI stores real-space blocks as
    //   map[I][{J, R}] -> Tensor
    // where I,J are atom indices and R is a lattice-cell translation.
    // Cs_data is real, so keep the LibRI bridge in real tensors for now.
    using Chi0Data = double;
    using Chi0Cell = std::array<int, 3>;
    using Chi0AtomCell = std::pair<int, Chi0Cell>;
    using Chi0TensorMap = std::map<int, std::map<Chi0AtomCell, RI::Tensor<Chi0Data>>>;

    int calculate_chi0(const Enviroment& Enviro, MpiComm LIBBSE_MPI_COMM);
    
}// namespace LibBSE
