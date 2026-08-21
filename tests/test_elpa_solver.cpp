#include "bse/distributed_amplitudes.h"
#include "bse/elpa_solver.h"
#include "bse/matrix_checks.h"
#include "interface/librpa_api.h"

#include <mpi.h>

#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using libbse::Complex;

void require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

void require_close(double actual, double expected, double tolerance,
                   const std::string &message)
{
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error(message + ": actual=" + std::to_string(actual)
                                 + ", expected=" + std::to_string(expected));
}

void require_close(Complex actual, Complex expected, double tolerance,
                   const std::string &message)
{
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error(message);
}

librpa_int::ArrayDesc make_descriptor(const librpa_int::BlacsCtxtHandler &blacs,
                                      int dimension)
{
    librpa_int::ArrayDesc descriptor(blacs);
    if (descriptor.init(dimension, dimension, 1, 1, 0, 0) != 0)
        throw std::runtime_error("failed to initialize test matrix descriptor");
    return descriptor;
}

std::vector<Complex> localize(const std::vector<Complex> &global,
                              const librpa_int::ArrayDesc &descriptor)
{
    require(global.size() == static_cast<std::size_t>(descriptor.m()) * descriptor.n(),
            "invalid dense matrix supplied to localize");
    std::vector<Complex> local(static_cast<std::size_t>(descriptor.lld())
                               * descriptor.n_loc());
    for (int local_column = 0; local_column < descriptor.n_loc(); ++local_column)
    {
        const int global_column = descriptor.indx_l2g_c(local_column);
        for (int local_row = 0; local_row < descriptor.m_loc(); ++local_row)
        {
            const int global_row = descriptor.indx_l2g_r(local_row);
            local[static_cast<std::size_t>(local_row)
                  + local_column * descriptor.lld()]
                = global[static_cast<std::size_t>(global_row)
                         + global_column * descriptor.m()];
        }
    }
    return local;
}

std::vector<Complex> globalize(const std::vector<Complex> &local,
                               const librpa_int::ArrayDesc &descriptor)
{
    require(local.size() == static_cast<std::size_t>(descriptor.lld())
                                * descriptor.n_loc(),
            "invalid local matrix supplied to globalize");
    std::vector<Complex> global(static_cast<std::size_t>(descriptor.m())
                                * descriptor.n());
    for (int local_column = 0; local_column < descriptor.n_loc(); ++local_column)
    {
        const int global_column = descriptor.indx_l2g_c(local_column);
        for (int local_row = 0; local_row < descriptor.m_loc(); ++local_row)
        {
            const int global_row = descriptor.indx_l2g_r(local_row);
            global[static_cast<std::size_t>(global_row)
                   + global_column * descriptor.m()]
                = local[static_cast<std::size_t>(local_row)
                        + local_column * descriptor.lld()];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, global.data(), static_cast<int>(global.size()),
                  MPI_C_DOUBLE_COMPLEX, MPI_SUM, descriptor.comm());
    return global;
}

std::vector<Complex> deterministic_hermitian(int dimension)
{
    std::vector<Complex> matrix(static_cast<std::size_t>(dimension) * dimension);
    for (int column = 0; column < dimension; ++column)
    {
        for (int row = column; row < dimension; ++row)
        {
            Complex value;
            if (row == column)
                value = Complex(20.0 + 0.2 * row, 0.0);
            else
                value = Complex(0.05 * (row + column + 1),
                                0.03 * (row - column));
            matrix[static_cast<std::size_t>(row) + column * dimension] = value;
            matrix[static_cast<std::size_t>(column) + row * dimension]
                = std::conj(value);
        }
    }
    return matrix;
}

std::vector<Complex> deterministic_symmetric(int dimension)
{
    std::vector<Complex> matrix(static_cast<std::size_t>(dimension) * dimension);
    for (int column = 0; column < dimension; ++column)
    {
        for (int row = column; row < dimension; ++row)
        {
            const Complex value(0.02 * (row + column + 1),
                                0.01 * (row + 2 * column + 1));
            matrix[static_cast<std::size_t>(row) + column * dimension] = value;
            matrix[static_cast<std::size_t>(column) + row * dimension] = value;
        }
    }
    return matrix;
}

std::vector<Complex> full_bse_matrix(const std::vector<Complex> &a,
                                     const std::vector<Complex> &b,
                                     int dimension)
{
    const int full_dimension = 2 * dimension;
    std::vector<Complex> result(static_cast<std::size_t>(full_dimension)
                                * full_dimension);
    for (int column = 0; column < dimension; ++column)
        for (int row = 0; row < dimension; ++row)
        {
            const Complex av = a[static_cast<std::size_t>(row) + column * dimension];
            const Complex bv = b[static_cast<std::size_t>(row) + column * dimension];
            result[static_cast<std::size_t>(row) + column * full_dimension] = av;
            result[static_cast<std::size_t>(row)
                   + (column + dimension) * full_dimension] = bv;
            result[static_cast<std::size_t>(row + dimension)
                   + column * full_dimension] = -std::conj(bv);
            result[static_cast<std::size_t>(row + dimension)
                   + (column + dimension) * full_dimension] = -std::conj(av);
        }
    return result;
}

void test_distributed_matrix_checks(
    const librpa_int::BlacsCtxtHandler &blacs)
{
    constexpr int dimension = 5;
    constexpr double threshold = 1.0e-12;
    const auto descriptor = make_descriptor(blacs, dimension);

    auto global_a = deterministic_hermitian(dimension);
    auto local_a = localize(global_a, descriptor);
    const auto hermitian = libbse::check_hermitian(
        local_a, descriptor, threshold);
    require(hermitian.passed, "distributed Hermitian check rejected A");
    require(hermitian.difference_norm < threshold,
            "Hermitian difference norm is too large");

    global_a[1] += Complex(0.1, 0.2);
    local_a = localize(global_a, descriptor);
    const auto nonhermitian = libbse::check_hermitian(
        local_a, descriptor, threshold);
    require(!nonhermitian.passed,
            "distributed Hermitian check accepted a non-Hermitian A");
    require(nonhermitian.difference_norm > threshold,
            "non-Hermitian difference norm is too small");

    auto global_b = deterministic_symmetric(dimension);
    auto local_b = localize(global_b, descriptor);
    const auto symmetric = libbse::check_symmetric(
        local_b, descriptor, threshold);
    require(symmetric.passed, "distributed symmetric check rejected B");
    require(symmetric.difference_norm < threshold,
            "symmetric difference norm is too large");

    global_b[1] += Complex(0.1, 0.2);
    local_b = localize(global_b, descriptor);
    const auto nonsymmetric = libbse::check_symmetric(
        local_b, descriptor, threshold);
    require(!nonsymmetric.passed,
            "distributed symmetric check accepted a nonsymmetric B");
    require(nonsymmetric.difference_norm > threshold,
            "nonsymmetric difference norm is too small");
}

void test_skew_solver_reference(const librpa_int::BlacsCtxtHandler &blacs)
{
    constexpr int dimension = 2;
    auto pair_descriptor = make_descriptor(blacs, dimension);
    auto full_descriptor = make_descriptor(blacs, 2 * dimension);
    const std::vector<Complex> global_a = {
        {3.0, 0.0}, {0.5, -1.0}, {0.5, 1.0}, {6.0, 0.0}};
    const std::vector<Complex> global_b = {
        {1.2, 0.6}, {0.4, 0.5}, {0.4, 0.5}, {1.4, 0.3}};
    auto a = localize(global_a, pair_descriptor);
    auto b = localize(global_b, pair_descriptor);

    const auto solution = libbse::solve_full_elpa(
        a, b, pair_descriptor, full_descriptor, dimension);
    require(solution.energies_ry.size() == dimension,
            "full solver returned the wrong number of positive eigenvalues");
    // Stores the negative branch first. LibBSE exposes the equivalent
    // positive branch used for excitation energies.
    require_close(solution.energies_ry[0], 2.299184312, 1.0e-8,
                  "first skew-solver reference eigenvalue differs");
    require_close(solution.energies_ry[1], 6.127295611, 1.0e-8,
                  "second skew-solver reference eigenvalue differs");
}

void test_tda_solver_residual(const librpa_int::BlacsCtxtHandler &blacs)
{
    constexpr int dimension = 5;
    auto descriptor = make_descriptor(blacs, dimension);
    const auto original = deterministic_hermitian(dimension);
    auto matrix = localize(original, descriptor);
    const auto solution = libbse::solve_tda_elpa(matrix, descriptor, dimension);
    const auto vectors = globalize(solution.vectors_local, descriptor);
    const auto distributed = libbse::redistribute_amplitudes(
        MPI_COMM_WORLD, solution.vectors_local, descriptor,
        0, 0, dimension, dimension);

    for (int state = 0; state < dimension; ++state)
        for (int local_pair = 0; local_pair < distributed.local_pairs;
             ++local_pair)
        {
            const int pair = distributed.first_pair + local_pair;
            require_close(
                distributed(state, local_pair),
                vectors[static_cast<std::size_t>(pair)
                        + state * dimension],
                1.0e-13,
                "distributed TDA amplitude differs from ELPA eigenvector");
        }

    for (int state = 0; state < dimension; ++state)
        for (int row = 0; row < dimension; ++row)
        {
            Complex hv{};
            for (int column = 0; column < dimension; ++column)
                hv += original[static_cast<std::size_t>(row) + column * dimension]
                      * vectors[static_cast<std::size_t>(column) + state * dimension];
            const Complex omega_v = solution.energies_ry[state]
                                    * vectors[static_cast<std::size_t>(row)
                                              + state * dimension];
            require_close(hv, omega_v, 1.0e-10, "TDA ELPA residual exceeds tolerance");
        }
}

void test_full_solver_residual_and_metric(
    const librpa_int::BlacsCtxtHandler &blacs)
{
    constexpr int dimension = 5;
    constexpr int full_dimension = 2 * dimension;
    auto pair_descriptor = make_descriptor(blacs, dimension);
    auto full_descriptor = make_descriptor(blacs, full_dimension);
    const auto original_a = deterministic_hermitian(dimension);
    const auto original_b = deterministic_symmetric(dimension);
    auto a = localize(original_a, pair_descriptor);
    auto b = localize(original_b, pair_descriptor);
    const auto solution = libbse::solve_full_elpa(
        a, b, pair_descriptor, full_descriptor, dimension);
    const auto hamiltonian = full_bse_matrix(original_a, original_b, dimension);
    const auto vectors = globalize(solution.vectors_local, full_descriptor);
    const auto distributed_x = libbse::redistribute_amplitudes(
        MPI_COMM_WORLD, solution.vectors_local, full_descriptor,
        0, dimension, dimension, dimension);
    const auto distributed_y = libbse::redistribute_amplitudes(
        MPI_COMM_WORLD, solution.vectors_local, full_descriptor,
        dimension, dimension, dimension, dimension);

    for (int state = 0; state < dimension; ++state)
        for (int local_pair = 0;
             local_pair < distributed_x.local_pairs; ++local_pair)
        {
            const int pair = distributed_x.first_pair + local_pair;
            const auto column_offset = static_cast<std::size_t>(dimension + state)
                                       * full_dimension;
            require_close(distributed_x(state, local_pair),
                          vectors[column_offset + pair], 1.0e-13,
                          "distributed full-BSE X amplitude differs");
            require_close(distributed_y(state, local_pair),
                          vectors[column_offset + dimension + pair], 1.0e-13,
                          "distributed full-BSE Y amplitude differs");
        }

    for (int state = 0; state < dimension; ++state)
    {
        const int vector_column = dimension + state;
        for (int row = 0; row < full_dimension; ++row)
        {
            Complex hv{};
            for (int column = 0; column < full_dimension; ++column)
                hv += hamiltonian[static_cast<std::size_t>(row)
                                  + column * full_dimension]
                      * vectors[static_cast<std::size_t>(column)
                                + vector_column * full_dimension];
            const Complex omega_v = solution.energies_ry[state]
                                    * vectors[static_cast<std::size_t>(row)
                                              + vector_column * full_dimension];
            require_close(hv, omega_v, 1.0e-9,
                          "full-BSE ELPA residual exceeds tolerance");
        }
    }

    for (int left = 0; left < dimension; ++left)
        for (int right = 0; right < dimension; ++right)
        {
            Complex metric{};
            for (int row = 0; row < dimension; ++row)
            {
                const Complex left_x = vectors[
                    static_cast<std::size_t>(row)
                    + (dimension + left) * full_dimension];
                const Complex right_x = vectors[
                    static_cast<std::size_t>(row)
                    + (dimension + right) * full_dimension];
                const Complex left_y = vectors[
                    static_cast<std::size_t>(row + dimension)
                    + (dimension + left) * full_dimension];
                const Complex right_y = vectors[
                    static_cast<std::size_t>(row + dimension)
                    + (dimension + right) * full_dimension];
                metric += std::conj(left_x) * right_x
                          - std::conj(left_y) * right_y;
            }
            require_close(metric, left == right ? Complex(1.0) : Complex{}, 1.0e-9,
                          "full-BSE symplectic eigenvector metric is incorrect");
        }
}

} // namespace

int main(int argc, char **argv)
{
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_status = 0;
    bool librpa_initialized = false;
    try
    {
        if (provided < MPI_THREAD_FUNNELED)
            throw std::runtime_error("MPI does not provide MPI_THREAD_FUNNELED");
        LibRPA_API::initialize();
        librpa_initialized = true;
        {
            librpa_int::BlacsCtxtHandler blacs(MPI_COMM_WORLD);
            blacs.init();
            blacs.set_square_grid();
            test_distributed_matrix_checks(blacs);
            test_skew_solver_reference(blacs);
            test_tda_solver_residual(blacs);
            test_full_solver_residual_and_metric(blacs);
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
        local_status = 1;
    }

    if (librpa_initialized) LibRPA_API::finalize();
    int global_status = 0;
    MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_status;
}
