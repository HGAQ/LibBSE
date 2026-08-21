#include "bse/distributed_amplitudes.h"
#include "bse/spectrum.h"

#include <mpi.h>

#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <unistd.h>

namespace
{

std::size_t velocity_index(int direction, int local_pairs)
{
    return static_cast<std::size_t>(direction) * local_pairs;
}

} // namespace

int main(int argc, char **argv)
{
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int failed = 0;
    try
    {
        if (provided < MPI_THREAD_FUNNELED || size != 2)
            throw std::runtime_error("test requires two FUNNELED-capable MPI ranks");

        libbse::InputParameters options;
        options.nocc = 1;
        options.nvirt = 1;

        libbse::FineVelocityMo velocity;
        velocity.nk = 2;
        velocity.nbands = 2;
        velocity.first_pair = rank;
        velocity.local_pairs = 1;
        velocity.values.assign(3, libbse::Complex{});
        velocity.gaps_ha = {rank == 0 ? 0.5 : 1.0};

        // Pair 1 belongs to rank 1 in the contiguous pair partition. A result on
        // rank 0 therefore proves that the MPI reduction includes rank 1.
        if (rank == 1)
            velocity.values[velocity_index(0, velocity.local_pairs)] = 1.0;
        auto amplitudes = libbse::make_distributed_amplitudes(
            MPI_COMM_WORLD, 2, 1);
        if (rank == 1) amplitudes(0, 0) = 2.0;
        const auto dipoles = libbse::velocity_gauge_transition_dipoles_mpi(
            MPI_COMM_WORLD, options, velocity, amplitudes, nullptr);
        if (rank == 0)
        {
            const libbse::Complex expected(0.0, 2.0 * std::sqrt(2.0));
            if (dipoles.size() != 1
                || std::abs(dipoles[0][0] - expected) > 1.0e-13
                || std::abs(dipoles[0][1]) > 1.0e-13
                || std::abs(dipoles[0][2]) > 1.0e-13)
                throw std::runtime_error(
                    "MPI velocity-gauge reduction is incorrect");
        }

        long long directory_id = rank == 0
                                     ? static_cast<long long>(::getpid())
                                     : 0;
        MPI_Bcast(&directory_id, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        const auto directory = std::filesystem::temp_directory_path()
                               / ("libbse_amplitudes_"
                                  + std::to_string(directory_id));
        if (rank == 0) std::filesystem::create_directories(directory);
        MPI_Barrier(MPI_COMM_WORLD);
        const auto file = directory
                          / ("Excitation_Amplitude_singlet_"
                             + std::to_string(rank) + ".dat");
        libbse::write_distributed_amplitudes(file, amplitudes);
        const auto reread = libbse::read_distributed_amplitudes(
            file, MPI_COMM_WORLD, 2, 1);
        if (reread.dimension != amplitudes.dimension
            || reread.nstates != amplitudes.nstates
            || reread.first_pair != amplitudes.first_pair
            || reread.local_pairs != amplitudes.local_pairs
            || reread.values != amplitudes.values)
            throw std::runtime_error(
                "distributed excitation-amplitude I/O is inconsistent");
        std::filesystem::remove(file);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::filesystem::remove(directory);
    }
    catch (const std::exception &error)
    {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
        failed = 1;
    }

    int any_failed = 0;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return any_failed;
}
