#include "utils/memory_views.h"
#include "distributed_amplitudes.h"

#include "parameter/parameter.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
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
    auto result_memory = watch_memory("distributed_amplitudes.result", result);
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
    int dimension, int nstates, int max_batch_elements)
{
    static_assert(sizeof(std::size_t) >= sizeof(std::uint64_t),
                  "large eigenvector arrays require 64-bit addresses");
    int valid = dimension > 0 && nstates > 0
        && row_offset >= 0 && row_offset <= source_descriptor.m()
        && column_offset >= 0 && column_offset <= source_descriptor.n()
        && dimension <= source_descriptor.m() - row_offset
        && nstates <= source_descriptor.n() - column_offset
        && source.size() == static_cast<std::size_t>(source_descriptor.lld())
                              * source_descriptor.n_loc();
    int all_valid = 0;
    MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, comm);
    if (!all_valid)
        throw std::invalid_argument(
            "invalid block-cyclic eigenvector block for redistribution");

    int rank = 0, mpi_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpi_size);
    int min_budget = 0, max_budget = 0;
    MPI_Allreduce(&max_batch_elements, &min_budget, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&max_batch_elements, &max_budget, 1, MPI_INT, MPI_MAX, comm);
    if (min_budget <= 0 || min_budget != max_budget)
        throw std::invalid_argument("redistribution batch budgets must agree and be positive");

    struct Row
    {
        int source_row, destination, pair, destination_pairs;
    };
    std::vector<Row> rows;
    auto rows_memory = libbse::watch_memory("distributed_amplitudes.rows", rows);
    std::vector<int> rows_per_destination(mpi_size, 0);
    auto rows_per_destination_memory = libbse::watch_memory("distributed_amplitudes.rows_per_destination", rows_per_destination);
    for (int row = 0; row < source_descriptor.m_loc(); ++row)
    {
        const int global_row = source_descriptor.indx_l2g_r(row);
        if (global_row < row_offset || global_row >= row_offset + dimension)
            continue;
        const int pair = global_row - row_offset;
        const int destination = pair_owner(pair, dimension, mpi_size);
        const auto partition = pair_partition(dimension, mpi_size, destination);
        rows.push_back({row, destination, pair - partition.first, partition.count});
        ++rows_per_destination[destination];
    }
    // The descriptor's local-to-global column mapping is monotone. Cache it
    // once; each local column is packed in exactly one batch.
    std::vector<std::pair<int, int>> columns;
    auto columns_memory = libbse::watch_memory("distributed_amplitudes.columns", columns);
    for (int column = 0; column < source_descriptor.n_loc(); ++column)
    {
        const int state = source_descriptor.indx_l2g_c(column) - column_offset;
        if (state >= 0 && state < nstates) columns.emplace_back(state, column);
    }
    const auto partition = pair_partition(dimension, mpi_size, rank);
    const int local_rows = std::max(static_cast<int>(rows.size()), partition.count);
    int global_rows = 0;
    MPI_Allreduce(&local_rows, &global_rows, 1, MPI_INT, MPI_MAX, comm);
    if (max_batch_elements < global_rows)
        throw std::invalid_argument("redistribution batch budget cannot hold one state");
    // Bound BOTH sends (block-cyclic rows) and receives (pair-block rows).
    // Every rank uses the same bound, including ranks with no local data, so
    // collective calls always cover the same state interval. Each batch's
    // total count and all prefix displacements fit MPI's signed int interface.
    const int batch_states = std::min(nstates, max_batch_elements / global_rows);
    if (rank == 0)
        std::cout << "Eigenvector redistribution: " << batch_states
                  << " states/batch, " << max_batch_elements
                  << " elements/rank exchange limit\n";
    auto result = make_distributed_amplitudes(comm, dimension, nstates);
    auto result_memory = watch_memory("distributed_amplitudes.result", result);
    std::vector<int> send_counts(mpi_size), receive_counts(mpi_size);
    auto send_counts_memory = libbse::watch_memory("distributed_amplitudes.send_counts", send_counts);
    auto receive_counts_memory = libbse::watch_memory("distributed_amplitudes.receive_counts", receive_counts);
    std::vector<int> send_offsets(mpi_size), receive_offsets(mpi_size), cursor(mpi_size);
    auto send_offsets_memory = libbse::watch_memory("distributed_amplitudes.send_offsets", send_offsets);
    auto receive_offsets_memory = libbse::watch_memory("distributed_amplitudes.receive_offsets", receive_offsets);
    auto cursor_memory = libbse::watch_memory("distributed_amplitudes.cursor", cursor);
    // Flat batch buffers avoid the old all-state per-destination vectors and
    // their extra flattened copies. Capacity is reused, bounded by the budget.
    std::vector<std::uint64_t> send_indices, receive_indices;
    auto send_indices_memory = libbse::watch_memory("distributed_amplitudes.send_indices", send_indices);
    auto receive_indices_memory = libbse::watch_memory("distributed_amplitudes.receive_indices", receive_indices);
    std::vector<Complex> send_values, receive_values;
    auto send_values_memory = libbse::watch_memory("distributed_amplitudes.send_values", send_values);
    auto receive_values_memory = libbse::watch_memory("distributed_amplitudes.receive_values", receive_values);
    std::vector<unsigned char> assigned;
    auto assigned_memory = libbse::watch_memory("distributed_amplitudes.assigned", assigned);
    // Reserve proven maxima once: vector growth must not double capacity past
    // the exchange budget when column ownership changes between batches.
    const auto max_send = std::min(static_cast<std::size_t>(batch_states), columns.size())
                          * rows.size();
    const auto max_receive = static_cast<std::size_t>(batch_states) * partition.count;
    send_indices.reserve(max_send);
    send_values.reserve(max_send);
    receive_indices.reserve(max_receive);
    receive_values.reserve(max_receive);
    assigned.reserve(max_receive);
    std::size_t column_begin = 0;
    for (int first = 0; first < nstates; )
    {
        const int count = std::min(batch_states, nstates - first);
        const int end = first + count;
        std::size_t column_end = column_begin;
        while (column_end < columns.size() && columns[column_end].first < end)
            ++column_end;
        for (int destination = 0; destination < mpi_size; ++destination)
            send_counts[destination] = static_cast<int>(
                (column_end - column_begin) * rows_per_destination[destination]);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                     receive_counts.data(), 1, MPI_INT, comm);

        std::int64_t total_send = 0, total_receive = 0;
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
                   == static_cast<std::size_t>(count) * result.local_pairs;
        MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, comm);
        if (!all_valid)
            throw std::runtime_error("invalid distributed eigenvector batch counts");

        send_indices.resize(static_cast<std::size_t>(total_send));
        send_values.resize(static_cast<std::size_t>(total_send));
        receive_indices.resize(static_cast<std::size_t>(total_receive));
        receive_values.resize(static_cast<std::size_t>(total_receive));
        cursor = send_offsets;
        for (std::size_t c = column_begin; c < column_end; ++c)
        {
            const auto [state, column] = columns[c];
            for (const auto &row : rows)
            {
                const int position = cursor[row.destination]++;
                // Widen BEFORE multiplying: final offsets may exceed INT_MAX
                // even though this batch's MPI counts and displacements do not.
                send_indices[position] = amplitude_offset(
                    state, row.destination_pairs, row.pair);
                send_values[position] = source[static_cast<std::size_t>(column)
                                                * source_descriptor.lld() + row.source_row];
            }
        }
        MPI_Alltoallv(send_indices.data(), send_counts.data(), send_offsets.data(),
                      MPI_UINT64_T, receive_indices.data(), receive_counts.data(),
                      receive_offsets.data(), MPI_UINT64_T, comm);
        MPI_Alltoallv(send_values.data(), send_counts.data(), send_offsets.data(),
                      MPI_C_DOUBLE_COMPLEX, receive_values.data(),
                      receive_counts.data(), receive_offsets.data(),
                      MPI_C_DOUBLE_COMPLEX, comm);

        const auto base = amplitude_offset(first, result.local_pairs, 0);
        assigned.assign(static_cast<std::size_t>(total_receive), 0);
        valid = 1;
        for (std::size_t i = 0; i < receive_indices.size(); ++i)
        {
            const auto index = receive_indices[i];
            if (index < base || index - base >= assigned.size()
                || assigned[static_cast<std::size_t>(index - base)] != 0)
            {
                valid = 0;
                break;
            }
            // Write directly into the final state-major array. The duplicate
            // check is batch-local, not another array spanning all eigenstates.
            result.values[static_cast<std::size_t>(index)] = receive_values[i];
            assigned[static_cast<std::size_t>(index - base)] = 1;
        }
        MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, comm);
        if (!all_valid)
            throw std::runtime_error("invalid distributed eigenvector ownership");
        column_begin = column_end;
        first = end;
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
    auto result_memory = watch_memory("distributed_amplitudes.result", result);
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
