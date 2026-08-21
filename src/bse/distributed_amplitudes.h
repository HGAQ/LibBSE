#pragma once

#include "bse_types.h"

#include <librpa_file_reader.hpp>

#include <mpi.h>

#include <filesystem>
#include <vector>

namespace libbse
{

DistributedAmplitudes make_distributed_amplitudes(
    MPI_Comm comm, int dimension, int nstates);

// Redistribute a block of a two-dimensional block-cyclic matrix into the
// pair-block layout used by spectrum analysis and amplitude files. Offsets are
// zero based. No rank receives the complete eigenvector matrix.
DistributedAmplitudes redistribute_amplitudes(
    MPI_Comm comm,
    const std::vector<Complex> &source,
    const librpa_int::ArrayDesc &source_descriptor,
    int row_offset, int column_offset,
    int dimension, int nstates);

void write_distributed_amplitudes(
    const std::filesystem::path &file,
    const DistributedAmplitudes &amplitudes);

DistributedAmplitudes read_distributed_amplitudes(
    const std::filesystem::path &file,
    MPI_Comm comm, int dimension, int nstates);

} // namespace libbse
