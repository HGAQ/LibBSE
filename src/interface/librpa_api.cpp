#include "librpa_api.h"

#include <stdexcept>
#include <vector>

namespace LibRPA_API
{

void initialize()
{
    librpa::set_output_level(LIBRPA_VERBOSE_INFO);
    librpa::init_global(LIBRPA_SWITCH_OFF);
}

void finalize()
{
    librpa::finalize_global();
}

std::shared_ptr<librpa_int::Dataset> read_dataset(
    MPI_Comm comm, const ReaderOptions &options)
{
    librpa::FileReaderOptions reader_options;
    reader_options.input_dir = options.input_dir;
    reader_options.cs_threshold = libbse::PARAM.constants.cs_threshold;
    reader_options.coulomb_threshold = libbse::PARAM.constants.coulomb_threshold;
    reader_options.read_ri = options.read_ri;
    reader_options.read_band_data = options.read_band_data;
    return librpa::read_dataset_from_files(comm, reader_options);
}

libbse::TensorMap<libbse::Complex> build_bare_coulomb(
    librpa_int::Dataset &dataset)
{
    auto real_space = librpa_int::FT_Vq(dataset.comm_h, dataset.basis_aux,
                                        dataset.symmetry_context, dataset.vq_cut,
                                        dataset.pbc, true, false);
    dataset.vq_cut.clear();

    libbse::TensorMap<libbse::Complex> result;
    for (const auto &[iat, atom_blocks] : real_space)
    {
        for (const auto &[jat, r_blocks] : atom_blocks)
        {
            for (const auto &[r, matrix_ptr] : r_blocks)
            {
                RI::Tensor<libbse::Complex> tensor(
                    {static_cast<std::size_t>(matrix_ptr->nr),
                     static_cast<std::size_t>(matrix_ptr->nc)});
                for (int i = 0; i != matrix_ptr->nr; ++i)
                    for (int j = 0; j != matrix_ptr->nc; ++j)
                        tensor(i, j) = (*matrix_ptr)(i, j);
                result[static_cast<int>(iat)]
                      [{static_cast<int>(jat), {r.x, r.y, r.z}}] = std::move(tensor);
            }
        }
    }
    return result;
}

librpa_int::ComplexMatrix inverse(const librpa_int::ComplexMatrix &matrix)
{
    if (matrix.nr != matrix.nc) throw std::invalid_argument("cannot invert a nonsquare matrix");
    librpa_int::ComplexMatrix result(matrix);
    std::vector<int> pivots(static_cast<std::size_t>(matrix.nr));
    std::vector<libbse::Complex> work(static_cast<std::size_t>(matrix.nr) * matrix.nr);
    int status = 0;
    librpa_int::LapackConnector::zgetrf(matrix.nr, matrix.nc, result,
                                        matrix.nr, pivots.data(), &status);
    if (status != 0) throw std::runtime_error("KS wavefunction matrix is singular");
    librpa_int::LapackConnector::zgetri(matrix.nr, result, matrix.nr,
                                        pivots.data(), work.data(),
                                        static_cast<int>(work.size()), &status);
    if (status != 0) throw std::runtime_error("failed to invert KS wavefunction matrix");
    return result;
}

librpa_int::ComplexMatrix conjugate(const librpa_int::ComplexMatrix &matrix)
{
    return librpa_int::conj(matrix);
}

librpa_int::ComplexMatrix transpose(const librpa_int::ComplexMatrix &matrix)
{
    return librpa_int::transpose(matrix, false);
}

void scale_accumulate(libbse::Complex factor,
                      const librpa_int::ComplexMatrix &source,
                      librpa_int::ComplexMatrix &target)
{
    librpa_int::scale_accumulate(factor, source, target);
}

void redistribute(int rows, int columns,
                  const libbse::Complex *source, int source_row, int source_column,
                  const librpa_int::ArrayDesc &source_descriptor,
                  libbse::Complex *target, int target_row, int target_column,
                  const librpa_int::ArrayDesc &target_descriptor)
{
    librpa_int::ScalapackConnector::pgemr2d_f(
        rows, columns, source, source_row, source_column, source_descriptor.desc,
        target, target_row, target_column, target_descriptor.desc,
        source_descriptor.ictxt());
}

void redistribute(int rows, int columns,
                  const double *source, int source_row, int source_column,
                  const librpa_int::ArrayDesc &source_descriptor,
                  double *target, int target_row, int target_column,
                  const librpa_int::ArrayDesc &target_descriptor)
{
    librpa_int::ScalapackConnector::pgemr2d_f(
        rows, columns, source, source_row, source_column, source_descriptor.desc,
        target, target_row, target_column, target_descriptor.desc,
        source_descriptor.ictxt());
}

void distributed_transpose(
    int rows, int columns, const libbse::Complex *source,
    const librpa_int::ArrayDesc &descriptor, libbse::Complex *target)
{
    librpa_int::ScalapackConnector::ptranu_f(
        rows, columns, libbse::Complex(1.0), source, 1, 1, descriptor.desc,
        libbse::Complex(0.0), target, 1, 1, descriptor.desc);
}

void distributed_conjugate_transpose(
    int rows, int columns, const libbse::Complex *source,
    const librpa_int::ArrayDesc &descriptor, libbse::Complex *target)
{
    librpa_int::ScalapackConnector::ptranc_f(
        rows, columns, libbse::Complex(1.0), source, 1, 1, descriptor.desc,
        libbse::Complex(0.0), target, 1, 1, descriptor.desc);
}

void multiply(char trans_a, char trans_b, int m, int n, int k,
              double alpha,
              const double *a, const librpa_int::ArrayDesc &a_descriptor,
              const double *b, const librpa_int::ArrayDesc &b_descriptor,
              double beta,
              double *c, const librpa_int::ArrayDesc &c_descriptor)
{
    librpa_int::ScalapackConnector::pgemm_f(
        trans_a, trans_b, m, n, k, alpha,
        a, 1, 1, a_descriptor.desc, b, 1, 1, b_descriptor.desc,
        beta, c, 1, 1, c_descriptor.desc);
}

void multiply(char trans_a, char trans_b, int m, int n, int k,
              libbse::Complex alpha,
              const libbse::Complex *a, const librpa_int::ArrayDesc &a_descriptor,
              const libbse::Complex *b, const librpa_int::ArrayDesc &b_descriptor,
              libbse::Complex beta,
              libbse::Complex *c, const librpa_int::ArrayDesc &c_descriptor)
{
    librpa_int::ScalapackConnector::pgemm_f(
        trans_a, trans_b, m, n, k, alpha,
        a, 1, 1, a_descriptor.desc, b, 1, 1, b_descriptor.desc,
        beta, c, 1, 1, c_descriptor.desc);
}

} // namespace LibRPA_API
