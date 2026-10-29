#include "distributed_amplitudes.h"

#include "parameter/parameter.h"
#include "utils/memory_views.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace libbse
{
namespace
{

struct PairPartition
{
    int first = 0;
    int count = 0;
};

// Assign contiguous pair ranges; the first remainder ranks get one extra pair.
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

void validate_amplitude_shape(const DistributedAmplitudes &amplitudes)
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

    // Store every requested state for this rank's pair range.
    DistributedAmplitudes local_amplitudes;
    auto local_amplitudes_memory = watch_memory(
        "distributed_amplitudes.local_amplitudes", local_amplitudes);
    local_amplitudes.dimension = dimension;
    local_amplitudes.nstates = nstates;
    local_amplitudes.first_pair = partition.first;
    local_amplitudes.local_pairs = partition.count;
    local_amplitudes.values.assign(
        static_cast<std::size_t>(nstates) * partition.count, Complex{});
    return local_amplitudes;
}

DistributedAmplitudes redistribute_amplitudes(
    MPI_Comm comm,
    const std::vector<Complex> &source_amplitudes,
    const librpa_int::ArrayDesc &source_descriptor,
    int row_offset, int column_offset,
    int dimension, int nstates, int max_batch_elements)
{
    static_assert(sizeof(std::size_t) >= sizeof(std::uint64_t),
                  "large eigenvector arrays require 64-bit addresses");
    int valid = dimension > 0 && nstates > 0
        && row_offset >= 0 && row_offset <= source_descriptor.m()
        && column_offset >= 0 && column_offset <= source_descriptor.n()
        && dimension <= source_descriptor.m() - row_offset
        && nstates <= source_descriptor.n() - column_offset
        && source_amplitudes.size() == static_cast<std::size_t>(source_descriptor.lld())
                                       * source_descriptor.n_loc();
    int all_valid = 0;
    MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, comm);
    if (!all_valid)
        throw std::invalid_argument(
            "invalid block-cyclic eigenvector block for redistribution");

    int rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpi_size);
    int min_budget = 0;
    int max_budget = 0;
    MPI_Allreduce(&max_batch_elements, &min_budget, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&max_batch_elements, &max_budget, 1, MPI_INT, MPI_MAX, comm);
    if (min_budget <= 0 || min_budget != max_budget)
        throw std::invalid_argument("redistribution batch budgets must agree and be positive");

    // Map local matrix rows to their destination rank and local pair index.
    struct AmplitudeRow
    {
        int source_row;
        int destination;
        int pair;
        int destination_pairs;
    };
    std::vector<AmplitudeRow> pair_rows;
    auto pair_rows_memory = libbse::watch_memory(
        "distributed_amplitudes.pair_rows", pair_rows);
    std::vector<int> pair_counts_per_destination(mpi_size, 0);
    auto pair_counts_per_destination_memory = libbse::watch_memory(
        "distributed_amplitudes.pair_counts_per_destination", pair_counts_per_destination);
    for (int row = 0; row < source_descriptor.m_loc(); ++row)
    {
        const int global_row = source_descriptor.indx_l2g_r(row);
        if (global_row < row_offset || global_row >= row_offset + dimension)
            continue;
        const int pair = global_row - row_offset;
        const int destination = pair_owner(pair, dimension, mpi_size);
        const auto partition = pair_partition(dimension, mpi_size, destination);
        pair_rows.push_back({row, destination, pair - partition.first, partition.count});
        ++pair_counts_per_destination[destination];
    }
    // The descriptor's local-to-global column mapping is monotone. Cache it
    // once; each local column is packed in exactly one batch.
    std::vector<std::pair<int, int>> state_columns;
    auto state_columns_memory = libbse::watch_memory(
        "distributed_amplitudes.state_columns", state_columns);
    for (int column = 0; column < source_descriptor.n_loc(); ++column)
    {
        const int state = source_descriptor.indx_l2g_c(column) - column_offset;
        if (state >= 0 && state < nstates) state_columns.emplace_back(state, column);
    }
    const auto partition = pair_partition(dimension, mpi_size, rank);
    const int local_rows = std::max(static_cast<int>(pair_rows.size()), partition.count);
    int global_rows = 0;
    MPI_Allreduce(&local_rows, &global_rows, 1, MPI_INT, MPI_MAX, comm);
    if (max_batch_elements < global_rows)
        throw std::invalid_argument("redistribution batch budget cannot hold one state");
    // Use one batch size on every rank, including ranks with no local data.
    // Bound both sends and receives so MPI counts and offsets fit signed int.
    const int batch_states = std::min(nstates, max_batch_elements / global_rows);
    if (rank == 0)
        std::cout << "Eigenvector redistribution: " << batch_states
                  << " states/batch, " << max_batch_elements
                  << " elements/rank exchange limit\n";
    auto local_amplitudes = make_distributed_amplitudes(comm, dimension, nstates);
    auto local_amplitudes_memory = watch_memory(
        "distributed_amplitudes.local_amplitudes", local_amplitudes);
    std::vector<int> send_counts(mpi_size);
    std::vector<int> receive_counts(mpi_size);
    auto send_counts_memory = libbse::watch_memory(
        "distributed_amplitudes.send_counts", send_counts);
    auto receive_counts_memory = libbse::watch_memory(
        "distributed_amplitudes.receive_counts", receive_counts);
    std::vector<int> send_offsets(mpi_size);
    std::vector<int> receive_offsets(mpi_size);
    std::vector<int> send_positions(mpi_size);
    auto send_offsets_memory = libbse::watch_memory(
        "distributed_amplitudes.send_offsets", send_offsets);
    auto receive_offsets_memory = libbse::watch_memory(
        "distributed_amplitudes.receive_offsets", receive_offsets);
    auto send_positions_memory = libbse::watch_memory(
        "distributed_amplitudes.send_positions", send_positions);
    // Reuse flat buffers across batches to limit temporary exchange storage.
    std::vector<std::uint64_t> send_amplitude_indices;
    std::vector<std::uint64_t> receive_amplitude_indices;
    auto send_amplitude_indices_memory = libbse::watch_memory(
        "distributed_amplitudes.send_amplitude_indices", send_amplitude_indices);
    auto receive_amplitude_indices_memory = libbse::watch_memory(
        "distributed_amplitudes.receive_amplitude_indices", receive_amplitude_indices);
    std::vector<Complex> send_amplitudes;
    std::vector<Complex> receive_amplitudes;
    auto send_amplitudes_memory = libbse::watch_memory(
        "distributed_amplitudes.send_amplitudes", send_amplitudes);
    auto receive_amplitudes_memory = libbse::watch_memory(
        "distributed_amplitudes.receive_amplitudes", receive_amplitudes);
    std::vector<unsigned char> received_amplitude_flags;
    auto received_amplitude_flags_memory = libbse::watch_memory(
        "distributed_amplitudes.received_amplitude_flags", received_amplitude_flags);
    // Reserve once to avoid capacity growth beyond the exchange budget.
    const auto max_send_elements = std::min(
        static_cast<std::size_t>(batch_states), state_columns.size()) * pair_rows.size();
    const auto max_receive_elements = static_cast<std::size_t>(batch_states)
                                      * partition.count;
    send_amplitude_indices.reserve(max_send_elements);
    send_amplitudes.reserve(max_send_elements);
    receive_amplitude_indices.reserve(max_receive_elements);
    receive_amplitudes.reserve(max_receive_elements);
    received_amplitude_flags.reserve(max_receive_elements);
    std::size_t column_begin = 0;
    for (int first = 0; first < nstates; )
    {
        const int count = std::min(batch_states, nstates - first);
        const int end = first + count;
        std::size_t column_end = column_begin;
        while (column_end < state_columns.size() && state_columns[column_end].first < end)
            ++column_end;
        for (int destination = 0; destination < mpi_size; ++destination)
            send_counts[destination] = static_cast<int>(
                (column_end - column_begin) * pair_counts_per_destination[destination]);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                     receive_counts.data(), 1, MPI_INT, comm);

        std::int64_t total_send = 0;
        std::int64_t total_receive = 0;
        valid = 1;
        for (int process = 0; process < mpi_size; ++process)
        {
            // Validate prefixes before converting, never after int addition.
            if (send_counts[process] < 0 || receive_counts[process] < 0
                || total_send > max_batch_elements || total_receive > max_batch_elements)
            {
                valid = 0;
                break;
            }
            send_offsets[process] = static_cast<int>(total_send);
            receive_offsets[process] = static_cast<int>(total_receive);
            total_send += send_counts[process];
            total_receive += receive_counts[process];
        }
        valid = valid && total_send <= max_batch_elements
            && total_receive <= max_batch_elements
            && static_cast<std::size_t>(total_receive)
                   == static_cast<std::size_t>(count) * local_amplitudes.local_pairs;
        MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, comm);
        if (!all_valid)
            throw std::runtime_error("invalid distributed eigenvector batch counts");

        send_amplitude_indices.resize(static_cast<std::size_t>(total_send));
        send_amplitudes.resize(static_cast<std::size_t>(total_send));
        receive_amplitude_indices.resize(static_cast<std::size_t>(total_receive));
        receive_amplitudes.resize(static_cast<std::size_t>(total_receive));
        send_positions = send_offsets;
        for (std::size_t c = column_begin; c < column_end; ++c)
        {
            const auto [state, column] = state_columns[c];
            for (const auto &row : pair_rows)
            {
                const int position = send_positions[row.destination]++;
                // Final array offsets may exceed INT_MAX; use amplitude_offset's
                // wide multiplication even though each MPI batch fits signed int.
                send_amplitude_indices[position] = amplitude_offset(
                    state, row.destination_pairs, row.pair);
                const auto source_amplitude_index = static_cast<std::size_t>(column)
                                                    * source_descriptor.lld()
                                                    + row.source_row;
                send_amplitudes[position] = source_amplitudes[source_amplitude_index];
            }
        }
        MPI_Alltoallv(send_amplitude_indices.data(), send_counts.data(),
                      send_offsets.data(), MPI_UINT64_T,
                      receive_amplitude_indices.data(), receive_counts.data(),
                      receive_offsets.data(), MPI_UINT64_T, comm);
        MPI_Alltoallv(send_amplitudes.data(), send_counts.data(), send_offsets.data(),
                      MPI_C_DOUBLE_COMPLEX, receive_amplitudes.data(),
                      receive_counts.data(), receive_offsets.data(),
                      MPI_C_DOUBLE_COMPLEX, comm);

        const auto batch_amplitude_offset = amplitude_offset(
            first, local_amplitudes.local_pairs, 0);
        received_amplitude_flags.assign(static_cast<std::size_t>(total_receive), 0);
        valid = 1;
        for (std::size_t i = 0; i < receive_amplitude_indices.size(); ++i)
        {
            const auto amplitude_index = receive_amplitude_indices[i];
            if (amplitude_index < batch_amplitude_offset
                || amplitude_index - batch_amplitude_offset >= received_amplitude_flags.size())
            {
                valid = 0;
                break;
            }
            const auto batch_index = static_cast<std::size_t>(
                amplitude_index - batch_amplitude_offset);
            if (received_amplitude_flags[batch_index] != 0)
            {
                valid = 0;
                break;
            }
            // Write directly into the final state-major array. The duplicate
            // check is batch-local, not another array spanning all eigenstates.
            local_amplitudes.values[static_cast<std::size_t>(amplitude_index)]
                = receive_amplitudes[i];
            received_amplitude_flags[batch_index] = 1;
        }
        MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, comm);
        if (!all_valid)
            throw std::runtime_error("invalid distributed eigenvector ownership");
        column_begin = column_end;
        first = end;
    }
    return local_amplitudes;
}

void write_distributed_amplitudes(
    const std::filesystem::path &file,
    const DistributedAmplitudes &amplitudes)
{
    // Each rank writes its own file, one state per line.
    validate_amplitude_shape(amplitudes);
    std::ofstream output(file);
    if (!output) throw std::runtime_error("cannot write " + file.string());
    output << std::scientific << std::setprecision(8);
    for (int state = 0; state < amplitudes.nstates; ++state)
    {
        for (int local_pair = 0; local_pair < amplitudes.local_pairs;
             ++local_pair)
        {
            Complex amplitude_value = amplitudes(state, local_pair);
            if (std::abs(amplitude_value) <= PARAM.constants.output_zero_tolerance)
                amplitude_value = Complex{};
            output << amplitude_value << ' ';
        }
        output << '\n';
    }
    output << '\n';
}

DistributedAmplitudes read_distributed_amplitudes(
    const std::filesystem::path &file,
    MPI_Comm comm, int dimension, int nstates)
{
    // Restart files must match the current state count and pair partition.
    auto local_amplitudes = make_distributed_amplitudes(comm, dimension, nstates);
    auto local_amplitudes_memory = watch_memory(
        "distributed_amplitudes.local_amplitudes", local_amplitudes);
    std::ifstream input(file);
    if (!input) throw std::runtime_error("cannot read " + file.string());
    for (Complex &amplitude_value : local_amplitudes.values)
    {
        if (!(input >> amplitude_value))
            throw std::runtime_error(
                "truncated or MPI-incompatible excitation-amplitude file "
                + file.string());
    }
    Complex extra_amplitude;
    if (input >> extra_amplitude)
        throw std::runtime_error(
            "oversized or MPI-incompatible excitation-amplitude file "
            + file.string());
    return local_amplitudes;
}

} // namespace libbse
