#include "matrix_checks.h"

#include "interface/librpa_api.h"

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace libbse
{
namespace
{

MatrixCheckResult check_matrix(const std::vector<Complex> &matrix,
                               const librpa_int::ArrayDesc &descriptor,
                               double threshold, bool conjugate)
{
    if (descriptor.m() != descriptor.n())
        throw std::invalid_argument("matrix symmetry check requires a square matrix");
    const std::size_t local_size = static_cast<std::size_t>(descriptor.lld())
                                   * descriptor.n_loc();
    if (matrix.size() != local_size)
        throw std::invalid_argument("invalid distributed matrix storage for symmetry check");
    if (threshold < 0.0)
        throw std::invalid_argument("matrix symmetry threshold must be nonnegative");

    int rank = 0;
    MPI_Comm_rank(descriptor.comm(), &rank);
    if (local_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        std::cerr << "Warning: distributed matrix on rank " << rank << " has "
                  << local_size << " local elements; use more MPI processes.\n";

    std::vector<Complex> transposed(local_size);
    if (conjugate)
        LibRPA_API::distributed_conjugate_transpose(
            descriptor.m(), descriptor.n(), matrix.data(), descriptor,
            transposed.data());
    else
        LibRPA_API::distributed_transpose(
            descriptor.m(), descriptor.n(), matrix.data(), descriptor,
            transposed.data());

    bool local_passed = true;
    double difference_squared = 0.0;
    double sum_squared = 0.0;
    for (int local_column = 0; local_column < descriptor.n_loc(); ++local_column)
    {
        for (int local_row = 0; local_row < descriptor.m_loc(); ++local_row)
        {
            const std::size_t index = static_cast<std::size_t>(local_column)
                                      * descriptor.lld() + local_row;
            const Complex difference = matrix[index] - transposed[index];
            if (std::abs(difference) > threshold) local_passed = false;
            difference_squared += std::norm(difference);
            sum_squared += std::norm(matrix[index] + transposed[index]);
        }
    }

    double norms[2]{difference_squared, sum_squared};
    MPI_Allreduce(MPI_IN_PLACE, norms, 2, MPI_DOUBLE, MPI_SUM,
                  descriptor.comm());
    int local_flag = local_passed ? 1 : 0;
    int global_flag = 0;
    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LAND,
                  descriptor.comm());

    MatrixCheckResult result;
    result.passed = global_flag != 0;
    result.difference_norm = std::sqrt(norms[0]);
    result.sum_norm = std::sqrt(norms[1]);
    if (result.sum_norm > 0.0)
        result.relative_error = result.difference_norm / result.sum_norm;
    else if (result.difference_norm > 0.0)
        result.relative_error = std::numeric_limits<double>::infinity();

    if (rank == 0)
    {
        const char *kind = conjugate ? "Hermitian" : "Symmetric";
        const char *symbol = conjugate ? "H" : "T";
        const char *matrix_name = conjugate ? "A" : "B";
        std::cout << "|  " << kind << " check: ||" << matrix_name << " - "
                  << matrix_name << '^' << symbol << "||_F = "
                  << result.difference_norm << ", ||" << matrix_name << " + "
                  << matrix_name << '^' << symbol << "||_F = "
                  << result.sum_norm << '\n'
                  << "|   ||" << matrix_name << " - " << matrix_name << '^'
                  << symbol << "||_F / ||" << matrix_name << " + "
                  << matrix_name << '^' << symbol << "||_F = "
                  << result.relative_error << '\n';
    }
    return result;
}

} // namespace

MatrixCheckResult check_hermitian(
    const std::vector<Complex> &matrix,
    const librpa_int::ArrayDesc &descriptor, double threshold)
{
    return check_matrix(matrix, descriptor, threshold, true);
}

MatrixCheckResult check_symmetric(
    const std::vector<Complex> &matrix,
    const librpa_int::ArrayDesc &descriptor, double threshold)
{
    return check_matrix(matrix, descriptor, threshold, false);
}

} // namespace libbse
