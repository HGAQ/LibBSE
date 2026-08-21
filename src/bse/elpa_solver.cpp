#include "elpa_solver.h"
#include "interface/librpa_api.h"

#include <elpa/elpa.h>

#include <omp.h>

#include <cmath>
#include <stdexcept>

namespace libbse
{
namespace
{

void check_elpa(int status, const char *operation)
{
    if (status != ELPA_OK)
        throw std::runtime_error(std::string("ELPA failure in ") + operation
                                 + ": status " + std::to_string(status));
}

elpa_t make_elpa_handle(const librpa_int::ArrayDesc &descriptor, int nev)
{
    int status = ELPA_OK;
    const auto handle = elpa_allocate(&status);
    check_elpa(status, "allocation");
    elpa_set(handle, "na", descriptor.m(), &status);
    check_elpa(status, "setting matrix dimension");
    elpa_set(handle, "nev", nev, &status);
    check_elpa(status, "setting nev");
    elpa_set(handle, "local_nrows", descriptor.m_loc(), &status);
    check_elpa(status, "setting local rows");
    elpa_set(handle, "local_ncols", descriptor.n_loc(), &status);
    check_elpa(status, "setting local columns");
    elpa_set(handle, "nblk", descriptor.mb(), &status);
    check_elpa(status, "setting block size");
    elpa_set(handle, "mpi_comm_parent", MPI_Comm_c2f(descriptor.comm()), &status);
    check_elpa(status, "setting MPI communicator");
    elpa_set(handle, "process_row", descriptor.myprow(), &status);
    check_elpa(status, "setting process row");
    elpa_set(handle, "process_col", descriptor.mypcol(), &status);
    check_elpa(status, "setting process column");
    elpa_set(handle, "solver", ELPA_SOLVER_2STAGE, &status);
    check_elpa(status, "selecting two-stage solver");
    elpa_set(handle, "omp_threads", omp_get_max_threads(), &status);
    check_elpa(status, "setting OpenMP threads");
    check_elpa(elpa_setup(handle), "setup");
    return handle;
}

template <typename Function>
void fill_pair_block(const std::vector<Complex> &matrix_a,
                     const std::vector<Complex> &matrix_b,
                     const librpa_int::ArrayDesc &pair_descriptor,
                     std::vector<double> &temporary,
                     Function &&function)
{
#pragma omp parallel for schedule(static)
    for (int column = 0; column < pair_descriptor.n_loc(); ++column)
    {
        for (int row = 0; row < pair_descriptor.m_loc(); ++row)
        {
            const auto index = static_cast<std::size_t>(column)
                               * pair_descriptor.lld() + row;
            temporary[index] = function(matrix_a[index], matrix_b[index]);
        }
    }
}

void copy_pair_block(const std::vector<double> &source,
                     const librpa_int::ArrayDesc &pair_descriptor,
                     std::vector<double> &target,
                     const librpa_int::ArrayDesc &full_descriptor,
                     int target_row, int target_column)
{
    LibRPA_API::redistribute(
        pair_descriptor.m(), pair_descriptor.n(), source.data(), 1, 1,
        pair_descriptor, target.data(), target_row, target_column,
        full_descriptor);
}

} // namespace

EigenSolution solve_tda_elpa(std::vector<Complex> &matrix,
                             librpa_int::ArrayDesc &descriptor,
                             int nstates)
{
    const int dimension = descriptor.m();
    if (dimension != descriptor.n())
        throw std::invalid_argument("TDA matrix must be square");
    if (nstates < 0) nstates = dimension;
    if (nstates == 0 || nstates > dimension)
        throw std::invalid_argument("invalid number of requested TDA states");

    int status = ELPA_OK;
    const auto handle = make_elpa_handle(descriptor, nstates);

    EigenSolution result;
    result.energies_ry.resize(static_cast<std::size_t>(dimension));
    result.vectors_local.resize(static_cast<std::size_t>(descriptor.lld())
                                * descriptor.n_loc());
    elpa_eigenvectors(handle, matrix.data(), result.energies_ry.data(),
                      result.vectors_local.data(), &status);
    check_elpa(status, "TDA eigenvectors");
    elpa_deallocate(handle, &status);
    check_elpa(status, "deallocation");
    result.energies_ry.resize(static_cast<std::size_t>(nstates));
    return result;
}

EigenSolution solve_full_elpa(std::vector<Complex> &matrix_a,
                              std::vector<Complex> &matrix_b,
                              const librpa_int::ArrayDesc &pair_descriptor,
                              librpa_int::ArrayDesc &full_descriptor,
                              int nstates)
{
    const int pair_dimension = pair_descriptor.m();
    if (pair_dimension != pair_descriptor.n())
        throw std::invalid_argument("full-BSE A/B matrices must be square");
    if (nstates < 0) nstates = pair_dimension;
    if (nstates == 0 || nstates > pair_dimension)
        throw std::invalid_argument("invalid number of requested full-BSE states");

    const int full_dimension = 2 * pair_dimension;
    if (full_descriptor.m() != full_dimension
        || full_descriptor.n() != full_dimension)
        throw std::invalid_argument("invalid full-BSE matrix descriptor");
    const std::size_t pair_local_size = static_cast<std::size_t>(pair_descriptor.lld())
                                        * pair_descriptor.n_loc();
    if (matrix_a.size() != pair_local_size || matrix_b.size() != pair_local_size)
        throw std::invalid_argument("invalid distributed A/B matrix storage");

    // Real Hamiltonian representation:
    // M = {{Re(A+B), Im(A-B)}, {-Im(A+B), Re(A-B)}}.
    const std::size_t full_local_size = static_cast<std::size_t>(full_descriptor.lld())
                                        * full_descriptor.n_loc();
    std::vector<double> matrix_m(full_local_size, 0.0);
    std::vector<double> temporary(pair_local_size, 0.0);
    fill_pair_block(matrix_a, matrix_b, pair_descriptor, temporary,
                    [](Complex a, Complex b) { return a.real() + b.real(); });
    copy_pair_block(temporary, pair_descriptor, matrix_m, full_descriptor, 1, 1);
    fill_pair_block(matrix_a, matrix_b, pair_descriptor, temporary,
                    [](Complex a, Complex b) { return a.imag() - b.imag(); });
    copy_pair_block(temporary, pair_descriptor, matrix_m, full_descriptor,
                    1, pair_dimension + 1);
    fill_pair_block(matrix_a, matrix_b, pair_descriptor, temporary,
                    [](Complex a, Complex b) { return -a.imag() - b.imag(); });
    copy_pair_block(temporary, pair_descriptor, matrix_m, full_descriptor,
                    pair_dimension + 1, 1);
    fill_pair_block(matrix_a, matrix_b, pair_descriptor, temporary,
                    [](Complex a, Complex b) { return a.real() - b.real(); });
    copy_pair_block(temporary, pair_descriptor, matrix_m, full_descriptor,
                    pair_dimension + 1, pair_dimension + 1);
    matrix_a.clear();
    matrix_b.clear();
    matrix_a.shrink_to_fit();
    matrix_b.shrink_to_fit();
    temporary.clear();
    temporary.shrink_to_fit();

    // Symplectic metric J = {{0,I},{-I,0}}.
    std::vector<double> matrix_j(full_local_size, 0.0);
#pragma omp parallel for schedule(static)
    for (int column = 0; column < full_descriptor.n_loc(); ++column)
    {
        const int global_column = full_descriptor.indx_l2g_c(column);
        for (int row = 0; row < full_descriptor.m_loc(); ++row)
        {
            const int global_row = full_descriptor.indx_l2g_r(row);
            if (global_column - global_row == pair_dimension)
                matrix_j[static_cast<std::size_t>(column) * full_descriptor.lld() + row] = 1.0;
            else if (global_row - global_column == pair_dimension)
                matrix_j[static_cast<std::size_t>(column) * full_descriptor.lld() + row] = -1.0;
        }
    }

    int status = ELPA_OK;
    const auto handle = make_elpa_handle(full_descriptor, full_dimension);
    elpa_cholesky(handle, matrix_m.data(), &status);
    check_elpa(status, "full-BSE Cholesky factorization");

    // matrix_m is the upper-triangular Cholesky factor U. Form U J U^T.
    std::vector<double> uj(full_local_size, 0.0);
    LibRPA_API::multiply('N', 'N', full_dimension, full_dimension,
                         full_dimension, 1.0, matrix_m.data(), full_descriptor,
                         matrix_j.data(), full_descriptor, 0.0, uj.data(),
                         full_descriptor);
    LibRPA_API::multiply('N', 'T', full_dimension, full_dimension,
                         full_dimension, 1.0, uj.data(), full_descriptor,
                         matrix_m.data(), full_descriptor, 0.0, matrix_j.data(),
                         full_descriptor);
    uj.clear();
    uj.shrink_to_fit();

    std::vector<double> all_energies(static_cast<std::size_t>(full_dimension));
    // ELPA stores real and imaginary parts of skew eigenvectors in two planes.
    std::vector<double> skew_vectors(2 * full_local_size, 0.0);
    elpa_skew_eigenvectors(handle, matrix_j.data(), all_energies.data(),
                           skew_vectors.data(), &status);
    check_elpa(status, "full-BSE skew eigenvectors");
    matrix_j.clear();
    matrix_j.shrink_to_fit();

#pragma omp parallel for schedule(static)
    for (int column = 0; column < full_descriptor.n_loc(); ++column)
    {
        const int global_column = full_descriptor.indx_l2g_c(column);
        const double scale = std::sqrt(std::abs(all_energies[global_column]));
        if (scale == 0.0)
            continue;
        for (int row = 0; row < full_descriptor.m_loc(); ++row)
        {
            const std::size_t index = static_cast<std::size_t>(column)
                                      * full_descriptor.lld() + row;
            skew_vectors[index] /= scale;
            skew_vectors[index + full_local_size] /= scale;
        }
    }

    std::vector<double> lz_real(full_local_size, 0.0);
    std::vector<double> lz_imag(full_local_size, 0.0);
    LibRPA_API::multiply('T', 'N', full_dimension, full_dimension,
                         full_dimension, 1.0, matrix_m.data(), full_descriptor,
                         skew_vectors.data(), full_descriptor, 0.0,
                         lz_real.data(), full_descriptor);
    LibRPA_API::multiply('T', 'N', full_dimension, full_dimension,
                         full_dimension, 1.0, matrix_m.data(), full_descriptor,
                         skew_vectors.data() + full_local_size, full_descriptor,
                         0.0, lz_imag.data(), full_descriptor);
    matrix_m.clear();
    matrix_m.shrink_to_fit();
    skew_vectors.clear();
    skew_vectors.shrink_to_fit();

    std::vector<Complex> lz(full_local_size);
#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < full_local_size; ++index)
        lz[index] = Complex(lz_real[index], lz_imag[index]);
    lz_real.clear();
    lz_imag.clear();
    lz_real.shrink_to_fit();
    lz_imag.shrink_to_fit();

    // Q = {{I,-iI},{-I,-iI}} / sqrt(2), then v = Q L z |Omega|^-1/2.
    std::vector<Complex> q(full_local_size, Complex{});
    const double inv_sqrt_two = 1.0 / std::sqrt(2.0);
#pragma omp parallel for schedule(static)
    for (int column = 0; column < full_descriptor.n_loc(); ++column)
    {
        const int global_column = full_descriptor.indx_l2g_c(column);
        for (int row = 0; row < full_descriptor.m_loc(); ++row)
        {
            const int global_row = full_descriptor.indx_l2g_r(row);
            Complex value{};
            if (global_column < pair_dimension && global_row == global_column)
                value = inv_sqrt_two;
            else if (global_column >= pair_dimension && global_row == global_column)
                value = Complex(0.0, -inv_sqrt_two);
            else if (global_row - global_column == pair_dimension)
                value = -inv_sqrt_two;
            else if (global_column - global_row == pair_dimension)
                value = Complex(0.0, -inv_sqrt_two);
            q[static_cast<std::size_t>(column) * full_descriptor.lld() + row] = value;
        }
    }

    EigenSolution result;
    result.vectors_local.resize(full_local_size);
    LibRPA_API::multiply('N', 'N', full_dimension, full_dimension,
                         full_dimension, Complex(1.0), q.data(), full_descriptor,
                         lz.data(), full_descriptor, Complex(0.0),
                         result.vectors_local.data(), full_descriptor);

    result.energies_ry.assign(all_energies.begin() + pair_dimension,
                              all_energies.begin() + pair_dimension + nstates);
    elpa_deallocate(handle, &status);
    check_elpa(status, "deallocation");
    return result;
}

} // namespace libbse
