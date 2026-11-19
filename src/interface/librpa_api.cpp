#include "utils/memory_views.h"
#include "librpa_api.h"
#include "io/aims_adapter.h"
#include <../src/core/epsilon.h>
#include <../src/utils/constants.h>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
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
    std::string reader_input = options.input_dir;
    if (options.input_format == "fhi_aims")
    {
        libbse::InputParameters adapter_options;
        adapter_options.input_dir = options.input_dir;
        adapter_options.output_dir = options.output_dir;
        reader_input = libbse::prepare_fhi_aims_reader_view(
                           comm, adapter_options).string();
    }

    librpa::FileReaderOptions reader_options;
    reader_options.input_dir = reader_input;
    reader_options.cs_threshold = libbse::PARAM.constants.cs_threshold;
    reader_options.coulomb_threshold = libbse::PARAM.constants.coulomb_threshold;
    reader_options.read_ri = options.read_ri;
    reader_options.distribute_lri = options.read_ri
        && libbse::PARAM.inp.screened_format.find("chi0") != std::string::npos
        && libbse::PARAM.inp.chi0_headwing;
    reader_options.read_full_coulomb = options.read_ri
        && libbse::PARAM.inp.screened_format.find("chi0") != std::string::npos
        && libbse::PARAM.inp.chi0_coulomb_metric == "full";
    reader_options.read_band_data = options.read_band_data;
    // velocity_matrix is deliberately unconditional in the public LibRPA
    // reader.  Every LibBSE mode, including IPA and spectrum restart, produces
    // optical output and therefore follows the same data contract.
    return librpa::read_dataset_from_files(comm, reader_options);
}

libbse::TensorMap<libbse::Complex> build_bare_coulomb(
    librpa_int::Dataset &dataset)
{
    // V(R) = Re[(1/N_q) * sum_q exp(-2*pi*i*q.R) * V(q)] on the coarse BvK cell.
    // This is LibRPA's FT_Vq convention (inverse-symmetry stars only), evaluated
    // here with OpenMP over R and without complex temporaries: FT_Vq is serial
    // and allocates a scaled copy of V(q) for every (R, q) term.
    const auto &coulomb_q_blocks = dataset.vq_cut;
    const auto &coulomb_pbc = dataset.pbc;
    const auto &coulomb_r_list = coulomb_pbc.Rlist;
    const int coulomb_r_count = static_cast<int>(coulomb_r_list.size());
    const double coulomb_cell_count = double(coulomb_pbc.get_n_cells_bvk());

    libbse::TensorMap<libbse::Complex> coulomb_real_space_output;
    auto coulomb_real_space_output_memory =
        libbse::watch_memory("Coulomb.real_space_output", coulomb_real_space_output);
    for (const auto &[iat, coulomb_j_blocks] : coulomb_q_blocks)
    {
        for (const auto &[jat, coulomb_q_matrices] : coulomb_j_blocks)
        {
            const std::size_t n_mu = dataset.basis_aux[iat];
            const std::size_t n_nu = dataset.basis_aux[jat];
            // When only one ordering of an atom pair is stored, the reverse
            // block follows from V_ji(q) = V_ij(q)^H.
            const bool coulomb_needs_reverse_pair =
                iat != jat && (coulomb_q_blocks.count(jat) == 0 ||
                               coulomb_q_blocks.at(jat).count(iat) == 0);

            // List every Brillouin-zone term once: its V(q) block, the BZ point
            // of the star, and whether that point is the stored q or its inverse.
            std::vector<const librpa_int::ComplexMatrix *> coulomb_term_matrices;
            std::vector<librpa_int::Vector3_Order<double>> coulomb_term_q_points;
            std::vector<int> coulomb_term_is_stored_q;
            for (const auto &[q, coulomb_q_matrix] : coulomb_q_matrices)
            {
                if (static_cast<std::size_t>(coulomb_q_matrix->nr) != n_mu ||
                    static_cast<std::size_t>(coulomb_q_matrix->nc) != n_nu)
                {
                    throw std::runtime_error(
                        "cut Coulomb block size differs from the auxiliary basis");
                }
                for (const auto &q_bz : coulomb_pbc.map_irk_ks.at(q))
                {
                    coulomb_term_matrices.push_back(coulomb_q_matrix.get());
                    coulomb_term_q_points.push_back(q_bz);
                    coulomb_term_is_stored_q.push_back(q_bz == q ? 1 : 0);
                }
            }
            const int coulomb_term_count = static_cast<int>(coulomb_term_matrices.size());

            // Tensors are created serially; the parallel loop only fills them.
            std::vector<RI::Tensor<libbse::Complex>> coulomb_r_tensors(coulomb_r_count);
            std::vector<RI::Tensor<libbse::Complex>> coulomb_reverse_r_tensors(
                coulomb_needs_reverse_pair ? coulomb_r_count : 0);
            for (int ir = 0; ir < coulomb_r_count; ++ir)
            {
                coulomb_r_tensors[ir] = RI::Tensor<libbse::Complex>({n_mu, n_nu});
                if (coulomb_needs_reverse_pair)
                {
                    coulomb_reverse_r_tensors[ir] = RI::Tensor<libbse::Complex>({n_nu, n_mu});
                }
            }

#pragma omp parallel for schedule(static)
            for (int ir = 0; ir < coulomb_r_count; ++ir)
            {
                const auto &r = coulomb_r_list[ir];
                // Real parts of sum_q V(q)*phase and of sum_q conj(V(q))*phase.
                std::vector<double> coulomb_r_real(n_mu * n_nu, 0.);
                std::vector<double> coulomb_r_conjugate_real(
                    coulomb_needs_reverse_pair ? n_mu * n_nu : 0, 0.);
                for (int iterm = 0; iterm < coulomb_term_count; ++iterm)
                {
                    // phase = exp(-2*pi*i*q_bz.R) / N_q
                    const double coulomb_phase_angle =
                        -coulomb_term_q_points[iterm] * (r * coulomb_pbc.latvec) *
                        librpa_int::TWO_PI;
                    const double coulomb_phase_cos =
                        std::cos(coulomb_phase_angle) / coulomb_cell_count;
                    const double coulomb_phase_sin =
                        std::sin(coulomb_phase_angle) / coulomb_cell_count;
                    // An inverse star member contributes conj(V(q)) to V(-q).
                    const double coulomb_imag_sign = coulomb_term_is_stored_q[iterm] ? 1. : -1.;
                    const std::complex<double> *coulomb_q_values = coulomb_term_matrices[iterm]->c;
                    for (std::size_t k = 0; k < n_mu * n_nu; ++k)
                    {
                        const double coulomb_real_term =
                            coulomb_q_values[k].real() * coulomb_phase_cos;
                        const double coulomb_imag_term =
                            coulomb_imag_sign * coulomb_q_values[k].imag() * coulomb_phase_sin;
                        // Re[(a + i*b) * (cos + i*sin)] = a*cos - b*sin
                        coulomb_r_real[k] += coulomb_real_term - coulomb_imag_term;
                        if (coulomb_needs_reverse_pair)
                        {
                            // Re[(a - i*b) * (cos + i*sin)] = a*cos + b*sin
                            coulomb_r_conjugate_real[k] += coulomb_real_term + coulomb_imag_term;
                        }
                    }
                }
                for (std::size_t i = 0; i < n_mu; ++i)
                {
                    for (std::size_t j = 0; j < n_nu; ++j)
                    {
                        coulomb_r_tensors[ir](i, j) = coulomb_r_real[i * n_nu + j];
                        if (coulomb_needs_reverse_pair)
                        {
                            // V_ji(R) uses the conjugate transpose of V_ij(q).
                            coulomb_reverse_r_tensors[ir](j, i) =
                                coulomb_r_conjugate_real[i * n_nu + j];
                        }
                    }
                }
            }

            for (int ir = 0; ir < coulomb_r_count; ++ir)
            {
                const auto &r = coulomb_r_list[ir];
                coulomb_real_space_output[static_cast<int>(iat)]
                                         [{static_cast<int>(jat), {r.x, r.y, r.z}}] =
                    std::move(coulomb_r_tensors[ir]);
                if (coulomb_needs_reverse_pair)
                {
                    coulomb_real_space_output[static_cast<int>(jat)]
                                             [{static_cast<int>(iat), {r.x, r.y, r.z}}] =
                        std::move(coulomb_reverse_r_tensors[ir]);
                }
            }
        }
    }

    // Chi0 screening still needs V(q) after constructing the Hartree kernel.
    // Keep both the checkpoint and clear inside this condition: without braces
    // clear() also runs for chi0 input and destroys Coulomb data needed for W.
    if (libbse::PARAM.inp.screened_format.find("chi0") == std::string::npos)
    {
        libbse::MemoryTracker::instance().checkpoint();
        dataset.vq_cut.clear();
    }
    return coulomb_real_space_output;
}

libbse::TensorMap<libbse::Complex> transform_screened_q_to_r(
    librpa_int::Dataset &dataset, const librpa_int::PeriodicBoundaryData &pbc,
    const ScreenedQBlocks &wq)
{
    // W(R) = (1/N_q) * sum_q exp(-2*pi*i*q.R) * W(q), complex, on the coarse
    // BvK cell. Same convention and term order as LibRPA FT_Wc_q2R without the
    // symmetry context (inverse-symmetry stars only: a star member q_bz != q
    // contributes conj(W(q))). Linear spatial FT, so it applies unchanged to
    // full W. Evaluated here with OpenMP over (atom pair, R), phases computed
    // once and no complex temporaries: FT_Wc_q2R allocates W(q)*phase (and
    // conj(W(q))) for every (R, q) term.
    const auto &screened_r_list = pbc.Rlist;
    const int screened_r_count = static_cast<int>(screened_r_list.size());
    const double screened_cell_count = double(pbc.get_n_cells_bvk());

    // One task per stored atom pair, with its list of Brillouin-zone terms.
    struct ScreenedPairTask
    {
        int iat = 0;
        int jat = 0;
        std::size_t n_mu = 0;
        std::size_t n_nu = 0;
        std::vector<const libbse::Complex *> term_values;
        std::vector<librpa_int::Vector3_Order<double>> term_q_points;
        std::vector<int> term_is_stored_q;
    };
    std::vector<ScreenedPairTask> screened_pair_tasks;
    for (const auto &[iat, screened_j_blocks] : wq)
    {
        for (const auto &[jat, screened_q_matrices] : screened_j_blocks)
        {
            ScreenedPairTask screened_pair_task;
            screened_pair_task.iat = static_cast<int>(iat);
            screened_pair_task.jat = static_cast<int>(jat);
            screened_pair_task.n_mu = dataset.basis_aux[iat];
            screened_pair_task.n_nu = dataset.basis_aux[jat];
            for (const auto &[q, screened_q_matrix] : screened_q_matrices)
            {
                if (static_cast<std::size_t>(screened_q_matrix.nr()) != screened_pair_task.n_mu ||
                    static_cast<std::size_t>(screened_q_matrix.nc()) != screened_pair_task.n_nu)
                {
                    throw std::runtime_error("W(q) block size differs from the auxiliary basis");
                }
                if (!screened_q_matrix.is_row_major())
                {
                    throw std::runtime_error("W(q) block must be row major");
                }
                // Every Brillouin-zone member of the star of the stored q.
                for (const auto &q_bz : pbc.map_irk_ks.at(q))
                {
                    screened_pair_task.term_values.push_back(screened_q_matrix.ptr());
                    screened_pair_task.term_q_points.push_back(q_bz);
                    screened_pair_task.term_is_stored_q.push_back(q_bz == q ? 1 : 0);
                }
            }
            screened_pair_tasks.push_back(std::move(screened_pair_task));
        }
    }
    const int screened_pair_count = static_cast<int>(screened_pair_tasks.size());

    // Output tensors are created serially; the parallel loop only fills them.
    std::vector<std::vector<RI::Tensor<libbse::Complex>>> screened_r_tensors(
        static_cast<std::size_t>(screened_pair_count));
    for (int ipair = 0; ipair < screened_pair_count; ++ipair)
    {
        const auto &screened_pair_task = screened_pair_tasks[ipair];
        screened_r_tensors[ipair].resize(static_cast<std::size_t>(screened_r_count));
        for (int ir = 0; ir < screened_r_count; ++ir)
        {
            screened_r_tensors[ipair][ir] = RI::Tensor<libbse::Complex>(
                {screened_pair_task.n_mu, screened_pair_task.n_nu});
        }
    }

    // One (atom pair, R) element per iteration; all are independent.
    const long screened_task_count = static_cast<long>(screened_pair_count) * screened_r_count;
#pragma omp parallel for schedule(dynamic)
    for (long itask = 0; itask < screened_task_count; ++itask)
    {
        const int ipair = static_cast<int>(itask / screened_r_count);
        const int ir = static_cast<int>(itask % screened_r_count);
        const auto &screened_pair_task = screened_pair_tasks[ipair];
        const auto &r = screened_r_list[ir];
        const std::size_t screened_block_size = screened_pair_task.n_mu * screened_pair_task.n_nu;
        // Real and imaginary parts of sum_q W(q)*phase, accumulated separately.
        std::vector<double> screened_r_real(screened_block_size, 0.);
        std::vector<double> screened_r_imag(screened_block_size, 0.);
        const int screened_term_count = static_cast<int>(screened_pair_task.term_values.size());
        for (int iterm = 0; iterm < screened_term_count; ++iterm)
        {
            // phase = exp(-2*pi*i*q_bz.R) / N_q = cos + i*sin
            const double screened_phase_angle =
                -screened_pair_task.term_q_points[iterm] * (r * pbc.latvec) * librpa_int::TWO_PI;
            const double screened_phase_cos = std::cos(screened_phase_angle) / screened_cell_count;
            const double screened_phase_sin = std::sin(screened_phase_angle) / screened_cell_count;
            // An inverse star member contributes conj(W(q)) = a - i*b.
            const double screened_imag_sign = screened_pair_task.term_is_stored_q[iterm] ? 1. : -1.;
            const libbse::Complex *screened_q_values = screened_pair_task.term_values[iterm];
            for (std::size_t k = 0; k < screened_block_size; ++k)
            {
                const double screened_a = screened_q_values[k].real();
                const double screened_b = screened_imag_sign * screened_q_values[k].imag();
                // (a + i*b) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
                screened_r_real[k] += screened_a * screened_phase_cos - screened_b * screened_phase_sin;
                screened_r_imag[k] += screened_a * screened_phase_sin + screened_b * screened_phase_cos;
            }
        }
        // RI::Tensor is row major: element (i, j) at i*n_nu + j, as in W(q).
        libbse::Complex *screened_r_values = screened_r_tensors[ipair][ir].ptr();
        for (std::size_t k = 0; k < screened_block_size; ++k)
        {
            screened_r_values[k] = libbse::Complex(screened_r_real[k], screened_r_imag[k]);
        }
    }

    libbse::TensorMap<libbse::Complex> screened_real_space_output;
    auto screened_real_space_output_memory =
        libbse::watch_memory("Coulomb.real_space_output", screened_real_space_output);
    for (int ipair = 0; ipair < screened_pair_count; ++ipair)
    {
        const auto &screened_pair_task = screened_pair_tasks[ipair];
        for (int ir = 0; ir < screened_r_count; ++ir)
        {
            const auto &r = screened_r_list[ir];
            screened_real_space_output[screened_pair_task.iat]
                                      [{screened_pair_task.jat, {r.x, r.y, r.z}}] =
                std::move(screened_r_tensors[ipair][ir]);
        }
    }
    return screened_real_space_output;
}

librpa_int::ComplexMatrix inverse(const librpa_int::ComplexMatrix &matrix)
{
    if (matrix.nr != matrix.nc) throw std::invalid_argument("cannot invert a nonsquare matrix");
    librpa_int::ComplexMatrix result(matrix);
    auto result_memory = libbse::watch_memory("inverse.result", result);
    std::vector<int> pivots(static_cast<std::size_t>(matrix.nr));
    auto pivots_memory = libbse::watch_memory("librpa_api.pivots", pivots);
    std::vector<libbse::Complex> work(static_cast<std::size_t>(matrix.nr) * matrix.nr);
    auto work_memory = libbse::watch_memory("librpa_api.work", work);
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
