#include "distributed_amplitudes.h"

#include "parameter/parameter.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace libbse
{
namespace
{

struct PairPartition
{
    int first = 0;
    int count = 0;
};

PairPartition pair_partition(int dimension, int mpi_size, int rank)
{
    const int base = dimension / mpi_size;
    const int remainder = dimension % mpi_size;
    return {rank * base + std::min(rank, remainder),
            base + (rank < remainder ? 1 : 0)};
}

int pair_owner(int pair, int dimension, int mpi_size)
{
    const int base = dimension / mpi_size;
    const int remainder = dimension % mpi_size;
    const int larger_end = (base + 1) * remainder;
    if (pair < larger_end) return pair / (base + 1);
    if (base == 0)
        throw std::logic_error("invalid owner for distributed amplitude pair");
    return remainder + (pair - larger_end) / base;
}

void validate_shape(const DistributedAmplitudes &amplitudes)
{
    if (amplitudes.dimension <= 0 || amplitudes.nstates <= 0
        || amplitudes.first_pair < 0 || amplitudes.local_pairs < 0
        || amplitudes.first_pair + amplitudes.local_pairs
               > amplitudes.dimension
        || amplitudes.values.size()
               != static_cast<std::size_t>(amplitudes.nstates)
                      * amplitudes.local_pairs)
        throw std::invalid_argument("invalid distributed excitation-amplitude shape");
}

} // namespace

DistributedAmplitudes make_distributed_amplitudes(
    MPI_Comm comm, int dimension, int nstates)
{
    if (dimension <= 0 || nstates <= 0)
        throw std::invalid_argument(
            "distributed excitation-amplitude dimensions must be positive");
    int rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpi_size);
    const PairPartition partition = pair_partition(dimension, mpi_size, rank);

    DistributedAmplitudes result;
    result.dimension = dimension;
    result.nstates = nstates;
    result.first_pair = partition.first;
    result.local_pairs = partition.count;
    result.values.assign(static_cast<std::size_t>(nstates) * partition.count,
                         Complex{});
    return result;
}

DistributedAmplitudes redistribute_amplitudes(
    MPI_Comm comm,
    const std::vector<Complex> &source,
    const librpa_int::ArrayDesc &source_descriptor,
    int row_offset, int column_offset,
    int dimension, int nstates)
{
    if (row_offset < 0 || column_offset < 0
        || row_offset + dimension > source_descriptor.m()
        || column_offset + nstates > source_descriptor.n()
        || source.size()
               != static_cast<std::size_t>(source_descriptor.lld())
                      * source_descriptor.n_loc())
        throw std::invalid_argument(
            "invalid block-cyclic eigenvector block for redistribution");

    int mpi_size = 1;
    MPI_Comm_size(comm, &mpi_size);
    auto result = make_distributed_amplitudes(comm, dimension, nstates);
    std::vector<std::vector<int>> indices(static_cast<std::size_t>(mpi_size));
    std::vector<std::vector<Complex>> values(static_cast<std::size_t>(mpi_size));

    for (int local_column = 0;
         local_column < source_descriptor.n_loc(); ++local_column)
    {
        const int global_column = source_descriptor.indx_l2g_c(local_column);
        if (global_column < column_offset
            || global_column >= column_offset + nstates)
            continue;
        const int state = global_column - column_offset;
        for (int local_row = 0; local_row < source_descriptor.m_loc();
             ++local_row)
        {
            const int global_row = source_descriptor.indx_l2g_r(local_row);
            if (global_row < row_offset
                || global_row >= row_offset + dimension)
                continue;
            const int pair = global_row - row_offset;
            const int destination = pair_owner(pair, dimension, mpi_size);
            const PairPartition destination_partition =
                pair_partition(dimension, mpi_size, destination);
            indices[destination].push_back(
                state * destination_partition.count
                + pair - destination_partition.first);
            values[destination].push_back(
                source[static_cast<std::size_t>(local_column)
                           * source_descriptor.lld()
                       + local_row]);
        }
    }

    std::vector<int> send_counts(static_cast<std::size_t>(mpi_size));
    for (int destination = 0; destination < mpi_size; ++destination)
    {
        if (indices[destination].size()
            > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error(
                "distributed amplitude message exceeds the MPI count limit");
        send_counts[destination] =
            static_cast<int>(indices[destination].size());
    }
    std::vector<int> receive_counts(static_cast<std::size_t>(mpi_size));
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 receive_counts.data(), 1, MPI_INT, comm);

    std::vector<int> send_offsets(static_cast<std::size_t>(mpi_size), 0);
    std::vector<int> receive_offsets(static_cast<std::size_t>(mpi_size), 0);
    for (int process = 1; process < mpi_size; ++process)
    {
        send_offsets[process] = send_offsets[process - 1]
                                + send_counts[process - 1];
        receive_offsets[process] = receive_offsets[process - 1]
                                   + receive_counts[process - 1];
    }
    const long long total_send_large = std::accumulate(
        send_counts.begin(), send_counts.end(), 0LL);
    const long long total_receive_large = std::accumulate(
        receive_counts.begin(), receive_counts.end(), 0LL);
    if (total_send_large > std::numeric_limits<int>::max()
        || total_receive_large > std::numeric_limits<int>::max())
        throw std::overflow_error(
            "distributed amplitude exchange exceeds the MPI count limit");
    const int total_send = static_cast<int>(total_send_large);
    const int total_receive = static_cast<int>(total_receive_large);
    if (static_cast<std::size_t>(total_receive) != result.values.size())
        throw std::runtime_error(
            "incomplete distributed eigenvector redistribution");

    std::vector<int> send_indices(static_cast<std::size_t>(total_send));
    std::vector<Complex> send_values(static_cast<std::size_t>(total_send));
    for (int destination = 0; destination < mpi_size; ++destination)
    {
        std::copy(indices[destination].begin(), indices[destination].end(),
                  send_indices.begin() + send_offsets[destination]);
        std::copy(values[destination].begin(), values[destination].end(),
                  send_values.begin() + send_offsets[destination]);
    }
    std::vector<int> receive_indices(static_cast<std::size_t>(total_receive));
    std::vector<Complex> receive_values(static_cast<std::size_t>(total_receive));
    MPI_Alltoallv(send_indices.data(), send_counts.data(), send_offsets.data(),
                  MPI_INT, receive_indices.data(), receive_counts.data(),
                  receive_offsets.data(), MPI_INT, comm);
    MPI_Alltoallv(send_values.data(), send_counts.data(), send_offsets.data(),
                  MPI_C_DOUBLE_COMPLEX, receive_values.data(),
                  receive_counts.data(), receive_offsets.data(),
                  MPI_C_DOUBLE_COMPLEX, comm);

    std::vector<unsigned char> assigned(result.values.size(), 0);
    for (int index = 0; index < total_receive; ++index)
    {
        const int local_index = receive_indices[index];
        if (local_index < 0
            || static_cast<std::size_t>(local_index) >= result.values.size()
            || assigned[local_index] != 0)
            throw std::runtime_error(
                "invalid distributed eigenvector ownership");
        result.values[local_index] = receive_values[index];
        assigned[local_index] = 1;
    }
    return result;
}

void write_distributed_amplitudes(
    const std::filesystem::path &file,
    const DistributedAmplitudes &amplitudes)
{
    validate_shape(amplitudes);
    std::ofstream output(file);
    if (!output) throw std::runtime_error("cannot write " + file.string());
    output << std::scientific << std::setprecision(8);
    for (int state = 0; state < amplitudes.nstates; ++state)
    {
        for (int local_pair = 0; local_pair < amplitudes.local_pairs;
             ++local_pair)
        {
            Complex value = amplitudes(state, local_pair);
            if (std::abs(value) <= PARAM.constants.output_zero_tolerance)
                value = Complex{};
            output << value << ' ';
        }
        output << '\n';
    }
    output << '\n';
}

DistributedAmplitudes read_distributed_amplitudes(
    const std::filesystem::path &file,
    MPI_Comm comm, int dimension, int nstates)
{
    auto result = make_distributed_amplitudes(comm, dimension, nstates);
    std::ifstream input(file);
    if (!input) throw std::runtime_error("cannot read " + file.string());
    for (Complex &value : result.values)
        if (!(input >> value))
            throw std::runtime_error(
                "truncated or MPI-incompatible excitation-amplitude file "
                + file.string());
    Complex extra;
    if (input >> extra)
        throw std::runtime_error(
            "oversized or MPI-incompatible excitation-amplitude file "
            + file.string());
    return result;
}

} // namespace libbse
