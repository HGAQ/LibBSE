#include "bse/molecular_lri_comm.h"
#include "interface/librpa_api.h"

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

libbse::Complex block_value(int k1, int k2, int i, int j)
{
    return {100.0 * k1 + 10.0 * k2 + 2.0 * j + i + 1.0,
            0.01 * (k1 - k2 + i - j)};
}

} // namespace

int main(int argc, char **argv)
{
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int failed = 0;
    bool librpa_initialized = false;
    try
    {
        if (provided < MPI_THREAD_FUNNELED || mpi_size != 2)
            throw std::runtime_error(
                "test requires two FUNNELED-capable MPI ranks");
        LibRPA_API::initialize();
        librpa_initialized = true;
        {
            constexpr int nk = 65;
            constexpr int pair_dimension = 2;
            constexpr int dimension = nk * pair_dimension;
            constexpr double coefficient = 0.75;
            const double factor = 2.0 * coefficient / nk;

            librpa_int::BlacsCtxtHandler blacs(MPI_COMM_WORLD);
            blacs.init();
            blacs.set_square_grid();
            librpa_int::ArrayDesc descriptor(blacs);
            if (descriptor.init(dimension, dimension, 3, 3, 0, 0) != 0)
                throw std::runtime_error(
                    "failed to initialize communication-test descriptor");

            libbse::KMatrixMap blocks;
            for (int k1 = rank; k1 < nk; k1 += mpi_size)
                for (int k2 = 0; k2 < nk; ++k2)
                {
                    RI::Tensor<libbse::Complex> tensor(
                        {pair_dimension, pair_dimension});
                    for (int j = 0; j < pair_dimension; ++j)
                        for (int i = 0; i < pair_dimension; ++i)
                            (*tensor.data)[i + j * pair_dimension]
                                = block_value(k1, k2, i, j);
                    blocks[k1][k2] = std::move(tensor);
                }

            std::vector<libbse::Complex> matrix(
                static_cast<std::size_t>(descriptor.lld())
                    * descriptor.n_loc(),
                libbse::Complex{});
            libbse::transform_k_2dlocal(
                matrix, blocks, descriptor,
                nk, pair_dimension, coefficient);

            for (int local_column = 0;
                 local_column < descriptor.n_loc(); ++local_column)
            {
                const int global_column =
                    descriptor.indx_l2g_c(local_column);
                const int k2 = global_column / pair_dimension;
                const int j = global_column % pair_dimension;
                for (int local_row = 0;
                     local_row < descriptor.m_loc(); ++local_row)
                {
                    const int global_row = descriptor.indx_l2g_r(local_row);
                    const int k1 = global_row / pair_dimension;
                    const int i = global_row % pair_dimension;
                    const auto actual = matrix[
                        static_cast<std::size_t>(local_column)
                            * descriptor.lld()
                        + local_row];
                    const auto expected = block_value(k1, k2, i, j) * factor;
                    if (std::abs(actual - expected) > 1.0e-13)
                        throw std::runtime_error(
                            "transform_k_2dlocal differs from the dense reference");
                }
            }
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
        failed = 1;
    }

    if (librpa_initialized) LibRPA_API::finalize();
    int any_failed = 0;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Finalize();
    return any_failed;
}
