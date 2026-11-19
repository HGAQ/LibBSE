#include "utils/memory_views.h"
#include "chi0_screening.h"
#include "bse/elpa_solver.h"
#include <../src/api/dataset_helper.h>
#include <../src/math/utils_matrix_m_mpi.h>
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#include <algorithm>
#include <array>
#include <climits>
#include <set>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

extern "C"
{
// ScaLAPACK inverse from a Cholesky factor. LibRPA already declares pzpotrf_
// with this non-const Fortran-style signature, but not pzpotri_.
void pzpotri_(char *uplo, int *n, std::complex<double> *a, int *ia, int *ja, int *desca,
              int *info);
}

namespace libbse
{
using librpa_int::Matz;
namespace
{
// The FHI-aims BSE converter stores velocity(row,col)=<col|v|row>.
// LibRPA's analytic wing contracts C_vc=<v|P_mu|c> with <c|v|v>,
// and therefore needs the conventional <row|v|col> storage instead.
// IPA and the diagonal head use |v_cv|^2 and cannot detect this mismatch.
// The wing keeps the complex phase, so the wrong row/column convention changes
// the response even when an ordinary intensity check appears to agree.
// This is a transpose, not an adjoint: we swap the stored bra/ket positions.
// Adapt only while computing head/wing: subsequent BSE optics must continue
// to see the original storage convention, including after exceptions.
class AimsHeadwingVelocityScope
{
  public:
    AimsHeadwingVelocityScope(librpa_int::velocity_matrix_t &aims_velocity,
                              bool aims_needs_transpose)
        : aims_velocity_(aims_velocity), aims_needs_transpose_(aims_needs_transpose)
    {
        if (!aims_needs_transpose_)
        {
            return;
        }
        for (const auto &aims_spin_velocity : aims_velocity_)
        {
            for (const auto &aims_k_velocity : aims_spin_velocity)
            {
                for (const auto &aims_cartesian_velocity : aims_k_velocity)
                {
                    if (aims_cartesian_velocity.nr != aims_cartesian_velocity.nc)
                    {
                        throw std::runtime_error("FHI-aims head/wing velocity must be square");
                    }
                }
            }
        }
        transpose_aims_velocity();
    }
    ~AimsHeadwingVelocityScope()
    {
        if (aims_needs_transpose_)
        {
            transpose_aims_velocity();
        }
    }
    AimsHeadwingVelocityScope(const AimsHeadwingVelocityScope &) = delete;
    AimsHeadwingVelocityScope &operator=(const AimsHeadwingVelocityScope &) = delete;

  private:
    void transpose_aims_velocity() noexcept
    {
        for (auto &aims_spin_velocity : aims_velocity_)
        {
            for (auto &aims_k_velocity : aims_spin_velocity)
            {
                for (auto &aims_cartesian_velocity : aims_k_velocity)
                {
                    for (int aims_row = 0; aims_row < aims_cartesian_velocity.nr; ++aims_row)
                    {
                        for (int aims_col = aims_row + 1; aims_col < aims_cartesian_velocity.nc;
                             ++aims_col)
                        {
                            std::swap(aims_cartesian_velocity(aims_row, aims_col),
                                      aims_cartesian_velocity(aims_col, aims_row));
                        }
                    }
                }
            }
        }
    }
    librpa_int::velocity_matrix_t &aims_velocity_;
    bool aims_needs_transpose_;
};

// Each MPI rank copies only the entries owned by its BLACS descriptor.
Matz scatter_screening_operator(const Matz &screening_dense,
                                librpa_int::ArrayDesc &screening_descriptor)
{
    auto screening_local =
        librpa_int::init_local_mat<Complex>(screening_descriptor, librpa_int::MAJOR::COL);
    for (int i = 0; i < screening_descriptor.m_loc(); ++i)
    {
        for (int j = 0; j < screening_descriptor.n_loc(); ++j)
        {
            screening_local(i, j) = screening_dense(screening_descriptor.indx_l2g_r(i),
                                                    screening_descriptor.indx_l2g_c(j));
        }
    }
    return screening_local;
}

// Sum disjoint local entries to make the complete operator available on every rank.
Matz gather_screening_operator(const Matz &screening_local,
                               librpa_int::ArrayDesc &screening_descriptor, int n,
                               MPI_Comm communicator)
{
    Matz screening_dense(n, n, librpa_int::MAJOR::ROW);
    auto screening_dense_memory = watch_memory("screening_dense", screening_dense);
    for (int i = 0; i < screening_descriptor.m_loc(); ++i)
    {
        for (int j = 0; j < screening_descriptor.n_loc(); ++j)
        {
            screening_dense(screening_descriptor.indx_l2g_r(i),
                            screening_descriptor.indx_l2g_c(j)) = screening_local(i, j);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, screening_dense.ptr(), n * n, MPI_C_DOUBLE_COMPLEX, MPI_SUM,
                  communicator);
    return screening_dense;
}

bool same_screening_q(const librpa_int::Vector3_Order<double> &first_q,
                      const librpa_int::Vector3_Order<double> &second_q)
{
    return std::abs(first_q.x - second_q.x) + std::abs(first_q.y - second_q.y) +
               std::abs(first_q.z - second_q.z) <
           1.e-8;
}

Matz hermitian(const Matz &screening_operator)
{
    double error = 0, norm = 0;
    for (std::size_t i = 0; i < screening_operator.nr(); ++i)
    {
        for (std::size_t j = 0; j < screening_operator.nc(); ++j)
        {
            error += std::norm(screening_operator(i, j) - std::conj(screening_operator(j, i)));
            norm += std::norm(screening_operator(i, j));
        }
    }
    if (!std::isfinite(norm) || std::sqrt(error) > 1.e-7 * std::max(1., std::sqrt(norm)))
    {
        throw std::runtime_error("chi0/V is not Hermitian in the supplied auxiliary basis");
    }
    return (screening_operator + screening_operator.get_transpose(true)) * 0.5;
}

Matz spectral_power(Matz screening_operator, double power, bool clip)
{
    auto screening_operator_memory = watch_memory("screening_operator", screening_operator);
    // Reuse LibRPA's Hermitian matrix function, including its positive-Coulomb
    // projection. This is a matrix power, never an entry-wise power.
    return librpa_int::power_hemat(screening_operator, power, false, false, clip ? 0. : -1.e5);
}

// Check Hermiticity of a BLACS-distributed operator and replace it by
// (A + A^H) / 2. screening_adjoint_local is caller-owned scratch storage with
// the same descriptor; its previous content is overwritten.
void average_screening_hermitian_local(Matz &screening_local, Matz &screening_adjoint_local,
                                       librpa_int::ArrayDesc &screening_descriptor, int n,
                                       MPI_Comm communicator)
{
    // adjoint = A^H
    librpa_int::ScalapackConnector::ptranc_f(n, n, 1., screening_local.ptr(), 1, 1,
                                             screening_descriptor.desc, 0.,
                                             screening_adjoint_local.ptr(), 1, 1,
                                             screening_descriptor.desc);
    // error = ||A - A^H||_F^2 and norm = ||A||_F^2, summed over all ranks.
    double screening_hermitian_sums[2] = {0., 0.};
    for (int i = 0; i < screening_descriptor.m_loc(); ++i)
    {
        for (int j = 0; j < screening_descriptor.n_loc(); ++j)
        {
            screening_hermitian_sums[0] +=
                std::norm(screening_local(i, j) - screening_adjoint_local(i, j));
            screening_hermitian_sums[1] += std::norm(screening_local(i, j));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, screening_hermitian_sums, 2, MPI_DOUBLE, MPI_SUM, communicator);
    const double error = screening_hermitian_sums[0];
    const double norm = screening_hermitian_sums[1];
    if (!std::isfinite(norm) || std::sqrt(error) > 1.e-7 * std::max(1., std::sqrt(norm)))
    {
        throw std::runtime_error("chi0/V is not Hermitian in the supplied auxiliary basis");
    }
    for (int i = 0; i < screening_descriptor.m_loc(); ++i)
    {
        for (int j = 0; j < screening_descriptor.n_loc(); ++j)
        {
            screening_local(i, j) = (screening_local(i, j) + screening_adjoint_local(i, j)) * 0.5;
        }
    }
}

// Replace the distributed Hermitian positive-definite epsilon by epsilon^-1.
// Only the upper triangle of the input is referenced, as in the former
// eigensolver path. screening_adjoint_local is overwritten scratch storage.
void invert_screening_dielectric_cholesky(Matz &screening_dielectric_local,
                                          Matz &screening_adjoint_local,
                                          librpa_int::ArrayDesc &screening_descriptor, int n)
{
    // The Fortran interfaces take non-const pointers; these are inputs only.
    char uplo = 'U';
    int first_row = 1;
    int first_col = 1;
    int info = 0;
    // epsilon = U^H * U
    pzpotrf_(&uplo, &n, screening_dielectric_local.ptr(), &first_row, &first_col,
             screening_descriptor.desc, &info);
    if (info > 0)
    {
        // A failed leading minor means epsilon is not positive definite.
        throw std::runtime_error("non-positive dielectric eigenvalue");
    }
    if (info < 0)
    {
        throw std::runtime_error("dielectric Cholesky factorization received an invalid argument");
    }
    // epsilon^-1 = U^-1 * U^-H, returned in the upper triangle.
    pzpotri_(&uplo, &n, screening_dielectric_local.ptr(), &first_row, &first_col,
             screening_descriptor.desc, &info);
    if (info != 0)
    {
        throw std::runtime_error("dielectric Cholesky inversion failed");
    }
    // Fill the strictly lower triangle from the adjoint of the upper triangle.
    librpa_int::ScalapackConnector::ptranc_f(n, n, 1., screening_dielectric_local.ptr(), 1, 1,
                                             screening_descriptor.desc, 0.,
                                             screening_adjoint_local.ptr(), 1, 1,
                                             screening_descriptor.desc);
    for (int i = 0; i < screening_descriptor.m_loc(); ++i)
    {
        for (int j = 0; j < screening_descriptor.n_loc(); ++j)
        {
            const int screening_global_row = screening_descriptor.indx_l2g_r(i);
            const int screening_global_col = screening_descriptor.indx_l2g_c(j);
            if (screening_global_row > screening_global_col)
            {
                screening_dielectric_local(i, j) = screening_adjoint_local(i, j);
            }
            else if (screening_global_row == screening_global_col)
            {
                // The diagonal of a Hermitian inverse is real.
                screening_dielectric_local(i, j) = screening_dielectric_local(i, j).real();
            }
        }
    }
}

// MemAvailable of this node in GiB for the progress line; -1 if unreadable.
double screening_node_available_memory_gib()
{
    std::ifstream screening_meminfo("/proc/meminfo");
    std::string screening_meminfo_key;
    double screening_meminfo_kib = 0.;
    std::string screening_meminfo_unit;
    while (screening_meminfo >> screening_meminfo_key >> screening_meminfo_kib)
    {
        if (screening_meminfo_key == "MemAvailable:")
        {
            return screening_meminfo_kib / (1024. * 1024.);
        }
        std::getline(screening_meminfo, screening_meminfo_unit);
    }
    return -1.;
}

// Coulomb matrix power with the ELPA two-stage solver. Same conventions as
// LibRPA power_hemat_blacs_desc / power_hemat_blacs_real (pzheev / pdsyev),
// which this replaces for the chi0 screening path:
//   V = Z * diag(lambda) * Z^H, lambda in DESCENDING order (column 0 of Z is
//   the largest Coulomb eigenvector, needed by the Gamma head/wing step),
//   eigenvalues below coulomb_threshold are dropped (set to zero power),
//   V^p = Z * diag(lambda^p) * Z^H is written back into coulomb_local.
// coulomb_real_gamma: diagonalize only Re(V), as LibRPA does at q = Gamma.
void power_coulomb_elpa(Matz &coulomb_local, const librpa_int::ArrayDesc &coulomb_descriptor,
                        Matz &coulomb_eigenvectors_local, std::size_t &coulomb_n_filtered,
                        double *coulomb_eigenvalues, double coulomb_power,
                        double coulomb_threshold, bool coulomb_real_gamma)
{
    const int n = coulomb_descriptor.m();

    // ELPA returns ascending eigenvalues. Diagonalize -V so that the
    // ascending order of -lambda is the descending order of lambda.
    if (coulomb_real_gamma)
    {
        auto coulomb_real_local = coulomb_local.get_real();
        coulomb_real_local *= -1.0;
        auto coulomb_real_eigenvectors_local =
            librpa_int::init_local_mat<double>(coulomb_descriptor, librpa_int::MAJOR::COL);
        // (-Re V) * Z = Z * diag(-lambda)
        solve_symmetric_eigenvectors_elpa(coulomb_real_local.ptr(), coulomb_descriptor,
                                          coulomb_eigenvalues,
                                          coulomb_real_eigenvectors_local.ptr());
        coulomb_eigenvectors_local = coulomb_real_eigenvectors_local.to_complex();
    }
    else
    {
        // ELPA overwrites its input; coulomb_local is rebuilt below anyway.
        coulomb_local *= -1.0;
        // (-V) * Z = Z * diag(-lambda)
        solve_hermitian_eigenvectors_elpa(coulomb_local.ptr(), coulomb_descriptor,
                                          coulomb_eigenvalues,
                                          coulomb_eigenvectors_local.ptr());
    }
    // Restore the sign: lambda, now in descending order.
    for (int coulomb_index = 0; coulomb_index < n; ++coulomb_index)
    {
        coulomb_eigenvalues[coulomb_index] *= -1.0;
    }

    // Number of trailing (smallest) eigenvalues below the threshold.
    coulomb_n_filtered = static_cast<std::size_t>(n);
    for (int coulomb_index = 0; coulomb_index < n; ++coulomb_index)
    {
        if (coulomb_eigenvalues[n - 1 - coulomb_index] >= coulomb_threshold)
        {
            coulomb_n_filtered = static_cast<std::size_t>(coulomb_index);
            break;
        }
    }
    const std::size_t coulomb_n_kept = static_cast<std::size_t>(n) - coulomb_n_filtered;

    // lambda^p for kept eigenvalues, zero for filtered ones.
    std::vector<double> coulomb_powered_eigenvalues(static_cast<std::size_t>(n), 0.0);
    int coulomb_warning_count = 0;
    for (std::size_t coulomb_index = 0; coulomb_index < coulomb_n_kept; ++coulomb_index)
    {
        if (coulomb_eigenvalues[coulomb_index] < 0.0 && coulomb_power != std::floor(coulomb_power))
        {
            coulomb_warning_count += 1;
        }
        coulomb_powered_eigenvalues[coulomb_index] =
            std::pow(coulomb_eigenvalues[coulomb_index], coulomb_power);
    }
    if (coulomb_warning_count > 0 && coulomb_descriptor.myprow() == 0
        && coulomb_descriptor.mypcol() == 0)
    {
        std::cout << "Warning: " << coulomb_warning_count
                  << " unfiltered negative Coulomb eigenvalues with non-integer power "
                  << coulomb_power << '\n';
    }

    // Scaled eigenvectors: column j of Z times lambda_j^p.
    auto coulomb_scaled_eigenvectors_local = coulomb_eigenvectors_local.copy();
    for (int local_col = 0; local_col < coulomb_descriptor.n_loc(); ++local_col)
    {
        const int global_col = coulomb_descriptor.indx_l2g_c(local_col);
        const double coulomb_column_scale =
            coulomb_powered_eigenvalues[static_cast<std::size_t>(global_col)];
        for (int local_row = 0; local_row < coulomb_descriptor.m_loc(); ++local_row)
        {
            coulomb_scaled_eigenvectors_local(local_row, local_col) *= coulomb_column_scale;
        }
    }

    // V^p = Z * (Z * diag(lambda^p))^H
    librpa_int::ScalapackConnector::pgemm_f(
        'N', 'C', n, n, n, 1., coulomb_eigenvectors_local.ptr(), 1, 1, coulomb_descriptor.desc,
        coulomb_scaled_eigenvectors_local.ptr(), 1, 1, coulomb_descriptor.desc, 0.,
        coulomb_local.ptr(), 1, 1, coulomb_descriptor.desc);
}

// Coulomb square root dispatcher: "elpa" (default, two-stage, threaded) or
// "scalapack" (LibRPA pzheev / pdsyev, the former path, kept for comparison).
void sqrt_coulomb(const std::string &coulomb_eigensolver, Matz &coulomb_local,
                  librpa_int::ArrayDesc &coulomb_descriptor, Matz &coulomb_eigenvectors_local,
                  std::size_t &coulomb_n_filtered, double *coulomb_eigenvalues,
                  bool coulomb_real_gamma)
{
    // ELPA needs at least one local row and column on every rank. A tiny
    // auxiliary basis (n_aux smaller than the process grid times the block
    // size) leaves some ranks empty; use the ScaLAPACK path there.
    int coulomb_local_block_nonempty =
        (coulomb_descriptor.m_loc() > 0 && coulomb_descriptor.n_loc() > 0) ? 1 : 0;
    int coulomb_all_blocks_nonempty = 0;
    MPI_Allreduce(&coulomb_local_block_nonempty, &coulomb_all_blocks_nonempty, 1, MPI_INT,
                  MPI_MIN, coulomb_descriptor.comm());
    const bool coulomb_use_elpa =
        coulomb_eigensolver == "elpa" && coulomb_all_blocks_nonempty == 1;
    if (coulomb_descriptor.myprow() == 0 && coulomb_descriptor.mypcol() == 0)
    {
        std::cout << "Coulomb square root eigensolver: "
                  << (coulomb_use_elpa ? "elpa" : "scalapack")
                  << (coulomb_real_gamma ? " (real Gamma)" : "") << '\n';
    }

    // V^(1/2), dropping negative (numerically zero) eigenvalues: threshold 0.
    if (coulomb_use_elpa)
    {
        power_coulomb_elpa(coulomb_local, coulomb_descriptor, coulomb_eigenvectors_local,
                           coulomb_n_filtered, coulomb_eigenvalues, .5, 0., coulomb_real_gamma);
    }
    else if (coulomb_real_gamma)
    {
        librpa_int::power_hemat_blacs_real(coulomb_local, coulomb_descriptor,
                                           coulomb_eigenvectors_local, coulomb_descriptor,
                                           coulomb_n_filtered, coulomb_eigenvalues, .5, 0.);
    }
    else
    {
        librpa_int::power_hemat_blacs_desc(coulomb_local, coulomb_descriptor,
                                           coulomb_eigenvectors_local, coulomb_descriptor,
                                           coulomb_n_filtered, coulomb_eigenvalues, .5, 0.);
    }
}
} // namespace

Matz screen_chi0(const Matz &screening_chi0, Matz screening_coulomb)
{
    if (screening_chi0.nr() != screening_chi0.nc() ||
        screening_coulomb.nr() != screening_chi0.nr() ||
        screening_coulomb.nc() != screening_chi0.nc())
    {
        throw std::runtime_error("chi0 and Coulomb dimensions differ");
    }
    const auto screening_sqrt_coulomb = spectral_power(hermitian(screening_coulomb), 0.5, true);
    auto screening_sqrt_coulomb_memory =
        watch_memory("screening_sqrt_coulomb", screening_sqrt_coulomb);
    auto screening_epsilon =
        (screening_sqrt_coulomb * hermitian(screening_chi0) * screening_sqrt_coulomb) * (-1.0);
    auto screening_epsilon_memory = watch_memory("screening_epsilon", screening_epsilon);
    for (std::size_t i = 0; i < screening_epsilon.nr(); ++i)
    {
        screening_epsilon(i, i) += 1.;
    }
    // epsilon = I - V^(1/2) chi0 V^(1/2), W = V^(1/2) epsilon^-1 V^(1/2).
    // Full auxiliary matrices are inverted BEFORE spatial interpolation.
    // Screening individual atom blocks or individual R cells is incorrect:
    // matrix inversion couples all atoms and does not commute with the FT.
    // Hartree units and the original RI basis are preserved. No extra bare
    // Coulomb is added: the expression already gives full W, including V.
    return screening_sqrt_coulomb * spectral_power(hermitian(screening_epsilon), -1., false) *
           screening_sqrt_coulomb;
}

static Matz collect_coulomb(librpa_int::Dataset &ds, const librpa_int::Vector3_Order<double> &q,
                            bool use_cut_coulomb)
{
    const int n = ds.basis_aux.nb_total;
    Matz screening_coulomb(n, n, librpa_int::MAJOR::ROW);
    std::vector<int> screening_coulomb_counts(n * n, 0);
    auto screening_coulomb_memory = watch_memory("screening_coulomb", screening_coulomb);
    auto screening_coulomb_counts_memory =
        watch_memory("screening_coulomb_counts", screening_coulomb_counts);
    // Coulomb blocks may be MPI distributed or replicated. Sum/count avoids
    // multiplying a replicated block by the MPI process count.
    for (const auto &[i, js] : (use_cut_coulomb ? ds.vq_cut : ds.vq))
    {
        for (const auto &[j, qs] : js)
        {
            for (const auto &[qr, screening_atom_pair] : qs)
            {
                bool match = same_screening_q(q, qr), conjugate = false;
                if (!match)
                {
                    auto star = ds.pbc.map_irk_ks.find(qr);
                    if (star != ds.pbc.map_irk_ks.end())
                    {
                        for (const auto &member : star->second)
                        {
                            if (same_screening_q(q, member))
                            {
                                const auto f = (q + qr) * ds.pbc.latvec.Transpose();
                                if (std::abs(f.x - std::round(f.x)) +
                                        std::abs(f.y - std::round(f.y)) +
                                        std::abs(f.z - std::round(f.z)) >
                                    1.e-8)
                                {
                                    throw std::runtime_error(
                                        "chi0 screening cannot restore general Coulomb q stars "
                                        "without basis rotations");
                                }
                                match = true;
                                conjugate = true;
                            }
                        }
                    }
                }
                if (!match)
                {
                    continue;
                }
                for (int a = 0; a < screening_atom_pair->nr; ++a)
                {
                    for (int b = 0; b < screening_atom_pair->nc; ++b)
                    {
                        const int mu = ds.basis_aux.get_global_index(i, a),
                                  nu = ds.basis_aux.get_global_index(j, b);
                        const auto x = conjugate ? std::conj((*screening_atom_pair)(a, b))
                                                 : (*screening_atom_pair)(a, b);
                        screening_coulomb(mu, nu) = x;
                        screening_coulomb_counts[mu * n + nu] = 1;
                        if (i != j)
                        {
                            screening_coulomb(nu, mu) = std::conj(x);
                            screening_coulomb_counts[nu * n + mu] = 1;
                        }
                    }
                }
                break;
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, screening_coulomb.ptr(), n * n, MPI_C_DOUBLE_COMPLEX, MPI_SUM,
                  ds.comm_h.comm);
    MPI_Allreduce(MPI_IN_PLACE, screening_coulomb_counts.data(), n * n, MPI_INT, MPI_SUM,
                  ds.comm_h.comm);
    for (int k = 0; k < n * n; ++k)
    {
        if (!screening_coulomb_counts[k])
        {
            throw std::runtime_error(
                "missing cut Coulomb block for chi0 screening (requires full/inverse q coverage)");
        }
        screening_coulomb.ptr()[k] /= screening_coulomb_counts[k];
    }
    return screening_coulomb;
}

Matz collect_cut_coulomb(librpa_int::Dataset &ds, const librpa_int::Vector3_Order<double> &q)
{
    return collect_coulomb(ds, q, true);
}

// Find the stored Coulomb block that represents V(q) for one atom pair: either
// stored at q itself, or at a q_r whose time-reversal star contains q, in which
// case V(q) = conj(V(q_r)) element-wise (real auxiliary functions).
static const librpa_int::ComplexMatrix *
find_coulomb_q_block(const librpa_int::Dataset &ds,
                     const std::map<librpa_int::Vector3_Order<double>,
                                    std::shared_ptr<librpa_int::ComplexMatrix>> &coulomb_q_blocks,
                     const librpa_int::Vector3_Order<double> &q, bool &coulomb_conjugate)
{
    coulomb_conjugate = false;
    for (const auto &[coulomb_stored_q, coulomb_stored_block] : coulomb_q_blocks)
    {
        if (same_screening_q(q, coulomb_stored_q))
        {
            return coulomb_stored_block.get();
        }
        const auto coulomb_star = ds.pbc.map_irk_ks.find(coulomb_stored_q);
        if (coulomb_star == ds.pbc.map_irk_ks.end())
        {
            continue;
        }
        for (const auto &coulomb_star_member : coulomb_star->second)
        {
            if (!same_screening_q(q, coulomb_star_member))
            {
                continue;
            }
            // Only q = -q_r (mod G) is restored; other star members need a basis rotation.
            const auto coulomb_sum_fractional =
                (q + coulomb_stored_q) * ds.pbc.latvec.Transpose();
            if (std::abs(coulomb_sum_fractional.x - std::round(coulomb_sum_fractional.x)) +
                    std::abs(coulomb_sum_fractional.y - std::round(coulomb_sum_fractional.y)) +
                    std::abs(coulomb_sum_fractional.z - std::round(coulomb_sum_fractional.z)) >
                1.e-8)
            {
                throw std::runtime_error("chi0 screening cannot restore general Coulomb q stars "
                                         "without basis rotations");
            }
            coulomb_conjugate = true;
            return coulomb_stored_block.get();
        }
    }
    return nullptr;
}

// Distributed replacement of hermitian(collect_coulomb(...)) followed by
// scatter_screening_operator: V(q) goes directly from the atom-pair storage
// into the block-cyclic local array of coulomb_descriptor, without a dense
// N_mu x N_mu copy on any rank. Same pattern as the LibRPA GW Coulomb setup:
// LibRI comm_map2_first + LibRPA collect_block_from_ALL_IJ_Tensor_sparse_zero_missing.
//  1. One owner per upper-triangle atom pair (I <= J): the lowest rank that
//     stores V_IJ or V_JI at q. Replicated Coulomb storage is therefore never
//     added twice by the LibRI communication, and missing pairs are detected.
//  2. The owner forms the upper block: V_IJ for I < J, V_JI^H for I > J (the
//     average if both are stored on the owner), (V_II + V_II^H)/2 for I == J.
//  3. Every rank receives the upper pairs its 2D block needs and fills the
//     lower triangle by Hermitian conjugation.
static Matz collect_coulomb_local(librpa_int::Dataset &ds,
                                  const librpa_int::Vector3_Order<double> &q,
                                  bool use_cut_coulomb,
                                  const librpa_int::ArrayDesc &coulomb_descriptor)
{
    const auto &coulomb_basis = ds.basis_aux;
    const int coulomb_atom_count = static_cast<int>(coulomb_basis.n_atoms);
    const int coulomb_my_rank = ds.comm_h.myid;
    // Single-q map: the cell key only labels the current q.
    const std::array<double, 3> coulomb_cell_key = {0., 0., 0.};
    using CoulombTensorMap =
        std::map<int, std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<Complex>>>;
    const auto &coulomb_storage = use_cut_coulomb ? ds.vq_cut : ds.vq;

    // Step 1: owner of each upper pair = lowest rank holding either order at q.
    std::vector<int> coulomb_pair_owner(
        static_cast<std::size_t>(coulomb_atom_count) * coulomb_atom_count, INT_MAX);
    for (const auto &[coulomb_row_atom, coulomb_col_blocks] : coulomb_storage)
    {
        for (const auto &[coulomb_col_atom, coulomb_q_blocks] : coulomb_col_blocks)
        {
            bool coulomb_conjugate = false;
            if (find_coulomb_q_block(ds, coulomb_q_blocks, q, coulomb_conjugate) == nullptr)
            {
                continue;
            }
            const int coulomb_upper_i =
                static_cast<int>(std::min(coulomb_row_atom, coulomb_col_atom));
            const int coulomb_upper_j =
                static_cast<int>(std::max(coulomb_row_atom, coulomb_col_atom));
            coulomb_pair_owner[static_cast<std::size_t>(coulomb_upper_i) * coulomb_atom_count +
                               coulomb_upper_j] = coulomb_my_rank;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, coulomb_pair_owner.data(),
                  static_cast<int>(coulomb_pair_owner.size()), MPI_INT, MPI_MIN, ds.comm_h.comm);
    for (int coulomb_upper_i = 0; coulomb_upper_i < coulomb_atom_count; ++coulomb_upper_i)
    {
        for (int coulomb_upper_j = coulomb_upper_i; coulomb_upper_j < coulomb_atom_count;
             ++coulomb_upper_j)
        {
            if (coulomb_pair_owner[static_cast<std::size_t>(coulomb_upper_i) * coulomb_atom_count +
                                   coulomb_upper_j] == INT_MAX)
            {
                throw std::runtime_error("missing cut Coulomb block for chi0 screening "
                                         "(requires full/inverse q coverage)");
            }
        }
    }

    // Step 2: the owner forms each upper block from its local storage.
    CoulombTensorMap coulomb_upper_owned;
    double coulomb_hermitian_error_local = 0.;
    for (const auto &[coulomb_row_atom, coulomb_col_blocks] : coulomb_storage)
    {
        for (const auto &[coulomb_col_atom, coulomb_q_blocks] : coulomb_col_blocks)
        {
            bool coulomb_conjugate = false;
            const auto *coulomb_stored_block =
                find_coulomb_q_block(ds, coulomb_q_blocks, q, coulomb_conjugate);
            if (coulomb_stored_block == nullptr)
            {
                continue;
            }
            const int coulomb_i = static_cast<int>(coulomb_row_atom);
            const int coulomb_j = static_cast<int>(coulomb_col_atom);
            const int coulomb_upper_i = std::min(coulomb_i, coulomb_j);
            const int coulomb_upper_j = std::max(coulomb_i, coulomb_j);
            if (coulomb_pair_owner[static_cast<std::size_t>(coulomb_upper_i) * coulomb_atom_count +
                                   coulomb_upper_j] != coulomb_my_rank)
            {
                continue;
            }
            const std::size_t coulomb_n_upper_i = coulomb_basis[coulomb_upper_i];
            const std::size_t coulomb_n_upper_j = coulomb_basis[coulomb_upper_j];
            if (static_cast<std::size_t>(coulomb_stored_block->nr) != coulomb_basis[coulomb_i] ||
                static_cast<std::size_t>(coulomb_stored_block->nc) != coulomb_basis[coulomb_j])
            {
                throw std::runtime_error("Coulomb block size differs from the auxiliary basis");
            }
            RI::Tensor<Complex> coulomb_upper_block({coulomb_n_upper_i, coulomb_n_upper_j});
            for (std::size_t a = 0; a < static_cast<std::size_t>(coulomb_stored_block->nr); ++a)
            {
                for (std::size_t b = 0; b < static_cast<std::size_t>(coulomb_stored_block->nc); ++b)
                {
                    // V(q) element of the stored block, conjugated for a -q star member.
                    Complex coulomb_value = (*coulomb_stored_block)(a, b);
                    if (coulomb_conjugate)
                    {
                        coulomb_value = std::conj(coulomb_value);
                    }
                    if (coulomb_i < coulomb_j)
                    {
                        coulomb_upper_block(a, b) = coulomb_value;
                    }
                    else if (coulomb_i > coulomb_j)
                    {
                        // Lower block: V_JI(b, a) = conj(V_IJ(a, b)).
                        coulomb_upper_block(b, a) = std::conj(coulomb_value);
                    }
                    else
                    {
                        // Diagonal atom block: Hermitian average, as hermitian() did:
                        // V(a, b) <- [V(a, b) + conj(V(b, a))] / 2.
                        Complex coulomb_mirror_value = (*coulomb_stored_block)(b, a);
                        if (coulomb_conjugate)
                        {
                            coulomb_mirror_value = std::conj(coulomb_mirror_value);
                        }
                        coulomb_hermitian_error_local +=
                            std::norm(coulomb_value - std::conj(coulomb_mirror_value));
                        coulomb_upper_block(a, b) =
                            0.5 * (coulomb_value + std::conj(coulomb_mirror_value));
                    }
                }
            }
            auto &coulomb_upper_row = coulomb_upper_owned[coulomb_upper_i];
            const auto coulomb_upper_key = std::make_pair(coulomb_upper_j, coulomb_cell_key);
            const auto coulomb_existing = coulomb_upper_row.find(coulomb_upper_key);
            if (coulomb_existing == coulomb_upper_row.end())
            {
                coulomb_upper_row.emplace(coulomb_upper_key, std::move(coulomb_upper_block));
            }
            else
            {
                // Both V_IJ and V_JI^H are stored on the owner: use their average.
                auto &coulomb_existing_block = coulomb_existing->second;
                for (std::size_t a = 0; a < coulomb_n_upper_i; ++a)
                {
                    for (std::size_t b = 0; b < coulomb_n_upper_j; ++b)
                    {
                        coulomb_existing_block(a, b) =
                            0.5 * (coulomb_existing_block(a, b) + coulomb_upper_block(a, b));
                    }
                }
            }
        }
    }

    // Norm of the full matrix: off-diagonal atom blocks appear twice (V_IJ, V_JI).
    double coulomb_norm_local = 0.;
    for (const auto &[coulomb_upper_i, coulomb_upper_row] : coulomb_upper_owned)
    {
        for (const auto &[coulomb_upper_key, coulomb_upper_block] : coulomb_upper_row)
        {
            const double coulomb_block_weight =
                coulomb_upper_key.first == coulomb_upper_i ? 1. : 2.;
            const std::size_t coulomb_block_size = coulomb_upper_block.shape.get_shape_all();
            const Complex *coulomb_block_values = coulomb_upper_block.ptr();
            for (std::size_t k = 0; k < coulomb_block_size; ++k)
            {
                coulomb_norm_local += coulomb_block_weight * std::norm(coulomb_block_values[k]);
            }
        }
    }
    // Same criterion as hermitian(): ||V - V^H|| <= 1e-7 max(1, ||V||). As before,
    // only diagonal atom blocks can violate it; off-diagonal blocks are mirrored.
    double coulomb_check_local[2] = {coulomb_hermitian_error_local, coulomb_norm_local};
    double coulomb_check_global[2] = {0., 0.};
    MPI_Allreduce(coulomb_check_local, coulomb_check_global, 2, MPI_DOUBLE, MPI_SUM,
                  ds.comm_h.comm);
    if (!std::isfinite(coulomb_check_global[1]) ||
        std::sqrt(coulomb_check_global[0]) >
            1.e-7 * std::max(1., std::sqrt(coulomb_check_global[1])))
    {
        throw std::runtime_error("chi0/V is not Hermitian in the supplied auxiliary basis");
    }

    // Step 3: receive the upper pairs needed by this rank's 2D block (LibRI).
    const auto coulomb_needed_pairs =
        librpa_int::get_necessary_IJ_from_block_2D_sy('U', coulomb_basis, coulomb_descriptor);
    std::set<int> coulomb_needed_rows;
    std::set<int> coulomb_needed_cols;
    for (const auto &coulomb_needed_pair : coulomb_needed_pairs)
    {
        coulomb_needed_rows.insert(coulomb_needed_pair.first);
        coulomb_needed_cols.insert(coulomb_needed_pair.second);
    }
    const auto coulomb_received = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
        ds.comm_h.comm, coulomb_upper_owned, coulomb_needed_rows, coulomb_needed_cols);
    coulomb_upper_owned.clear();
    for (const auto &coulomb_needed_pair : coulomb_needed_pairs)
    {
        const auto coulomb_received_row = coulomb_received.find(coulomb_needed_pair.first);
        if (coulomb_received_row == coulomb_received.end() ||
            coulomb_received_row->second.count(
                std::make_pair(coulomb_needed_pair.second, coulomb_cell_key)) == 0)
        {
            throw std::runtime_error("Coulomb block missing after distribution for chi0 screening");
        }
    }

    // Fill the block-cyclic local array: V_IJ for I <= J, V_JI^H for I > J.
    auto coulomb_local =
        librpa_int::init_local_mat<Complex>(coulomb_descriptor, librpa_int::MAJOR::COL);
    librpa_int::collect_block_from_ALL_IJ_Tensor_sparse_zero_missing(
        coulomb_local, coulomb_descriptor, coulomb_basis, coulomb_cell_key, true,
        Complex(1., 0.), coulomb_received, librpa_int::MAJOR::ROW);
    return coulomb_local;
}

TensorMap<Complex> distribute_screened_real_space(const TensorMap<Complex> &screened_owned_r,
                                                  const std::vector<int> &screened_rows,
                                                  const std::vector<int> &screened_cols,
                                                  MPI_Comm screened_communicator)
{
    const std::set<int> screened_row_set(screened_rows.begin(), screened_rows.end());
    const std::set<int> screened_col_set(screened_cols.begin(), screened_cols.end());
    // Rank r receives every (I, (J, R)) with I in its rows and J in its cols.
    return RI::Communicate_Tensors_Map_Judge::comm_map2_first(
        screened_communicator, screened_owned_r, screened_row_set, screened_col_set);
}

Matz effective_inverse_dielectric(const Matz &screening_inverse, double plasma, double binding)
{
    if (!std::isfinite(plasma) || plasma <= 0 || !std::isfinite(binding) || binding < 0)
    {
        throw std::runtime_error(
            "effective screening requires positive plasma energy and nonnegative binding energy");
    }
    if (binding == 0)
    {
        return screening_inverse.copy(); // Exact static limit, including filtered Coulomb channels.
    }
    auto screening_eigenvectors = hermitian(screening_inverse);
    auto screening_eigenvectors_memory =
        watch_memory("screening_eigenvectors", screening_eigenvectors);
    const int n = screening_eigenvectors.nr(), lwork = std::max(1, 64 * n);
    std::vector<double> screening_eigenvalues(n);
    std::vector<double> screening_real_workspace(std::max(1, 3 * n - 2));
    auto screening_eigenvalues_memory =
        watch_memory("screening_eigenvalues", screening_eigenvalues);
    auto screening_real_workspace_memory =
        watch_memory("screening_real_workspace", screening_real_workspace);
    std::vector<Complex> screening_workspace(lwork);
    auto screening_workspace_memory = watch_memory("screening_workspace", screening_workspace);
    int info = 0;
    librpa_int::LapackConnector::heev('V', 'U', n, screening_eigenvectors.ptr(), n,
                                      screening_eigenvalues.data(), screening_workspace.data(),
                                      lwork, screening_real_workspace.data(), info);
    if (info)
    {
        throw std::runtime_error("effective inverse-dielectric eigensolver failed");
    }
    auto screening_weighted_eigenvectors = screening_eigenvectors.get_transpose(true);
    auto screening_weighted_eigenvectors_memory =
        watch_memory("screening_weighted_eigenvectors", screening_weighted_eigenvectors);
    for (int i = 0; i < n; ++i)
    {
        // Evaluate the nonlinear model in screening eigenchannels:
        // f(d)=1-P*(1-d)/(P+Eb*sqrt(1-d)), P=hbar*omega_p.
        // Roundoff near d=0 or 1 is harmless; a genuinely non-passive static
        // dielectric is not silently repaired by clamping physical eigenvalues.
        if (screening_eigenvalues[i] < -1.e-7 || screening_eigenvalues[i] > 1. + 1.e-7)
        {
            throw std::runtime_error("inverse dielectric eigenvalue outside [0,1]: " +
                                     std::to_string(screening_eigenvalues[i]));
        }
        const double d = std::clamp(screening_eigenvalues[i], 0., 1.);
        screening_weighted_eigenvectors.scale_row(
            i, 1. - plasma * (1. - d) / (plasma + binding * std::sqrt(1. - d)));
    }
    return screening_eigenvectors * screening_weighted_eigenvectors;
}

// Visit the live cache, since new q channels are added after construction.
void Chi0Screening::ChannelMemoryObserver::operator()(MemoryVisitor &visitor) const
{
    for (const auto &entry : owner.channels_)
    {
        visit_memory(visitor, entry.second.screening_sqrt_cut, "screening_sqrt_cut");
        visit_memory(visitor, entry.second.screening_inverse, "screening_inverse");
    }
}

Chi0Screening::Chi0Screening(const InputParameters &options, librpa_int::Dataset &ds)
    : pbc(ds.pbc), options_(options), dataset_(ds)
{
    memory_ = std::make_unique<MemoryWatch>(ChannelMemoryObserver{*this});
}

// Defined here, where librpa_int::IndexScheduler is a complete type.
Chi0Screening::~Chi0Screening() = default;

ScreenedOwnedBlocks Chi0Screening::screen(const Matz &screening_chi0,
                                          const librpa_int::Vector3_Order<double> &q,
                                          double omega)
{
    using namespace librpa_int;
    auto &ds = dataset_;
    if (omega_ < 0)
    {
        omega_ = omega;
        if (options_.chi0_headwing)
        {
            // The analytic head and wings MUST use the same physical frequency
            // as the imported response, including exactly zero for chi0_static.
            // This dedicated one-node grid is not used for a GW integral.
            ds.tfg.generate_single_frequency(std::abs(omega));
            librpa::Options opts;
            opts.replace_w_head = LIBRPA_SWITCH_ON;
            opts.option_dielect_func = 3;
            opts.sqrt_coulomb_threshold = 0.;
            opts.use_shrink_abfs = LIBRPA_SWITCH_OFF;
            opts.use_symmetry_rpa = LIBRPA_SWITCH_OFF;
            opts.use_symmetry_gw = LIBRPA_SWITCH_OFF;
            opts.use_kpara_scf_eigvec = LIBRPA_SWITCH_OFF;
            ds.p_headwing.reset();
            const AimsHeadwingVelocityScope aims_headwing_velocity_scope(
                ds.velocity_matrix, options_.input_format == "fhi_aims");
            initialize_ds_headwing(ds, opts, true);
        }
    }
    if (std::abs(omega - omega_) > 1.e-12)
    {
        throw std::runtime_error("inconsistent imported chi0 frequencies");
    }

    const double screening_start_time = MPI_Wtime();
    const int n = screening_chi0.nr();
    ArrayDesc desc(ds.blacs_h);
    desc.init(n, n, std::min(n, 64), std::min(n, 64), 0, 0);
    const bool screening_full_metric = options_.chi0_coulomb_metric == "full";

    // Read both Coulomb metrics: full V builds epsilon, cut V regularizes W.
    // Atom-pair blocks go directly into the block-cyclic local arrays; no rank
    // holds a dense N_mu x N_mu Coulomb matrix.
    Matz screening_sqrt_cut_local = collect_coulomb_local(ds, q, true, desc);
    auto screening_sqrt_cut_local_memory =
        watch_memory("screening_sqrt_cut_local", screening_sqrt_cut_local);
    Matz screening_sqrt_full_local;
    if (screening_full_metric)
    {
        screening_sqrt_full_local = collect_coulomb_local(ds, q, false, desc);
    }
    const double screening_coulomb_end_time = MPI_Wtime();

    auto screening_eigenvectors_local = init_local_mat<Complex>(desc, MAJOR::COL);
    auto screening_eigenvectors_local_memory =
        watch_memory("screening_eigenvectors_local", screening_eigenvectors_local);
    std::vector<double> screening_eigenvalues(n);
    auto screening_eigenvalues_memory =
        watch_memory("screening_eigenvalues", screening_eigenvalues);
    std::size_t filtered = 0;
    // Use exactly the LibRPA GW square-root convention (descending Coulomb
    // eigenvectors; real Gamma decomposition). The largest Coulomb eigenvector
    // is the singular head channel expected by rewrite_eps_abf_space.
    const bool gamma =
        screening_full_metric && std::abs(q.x) + std::abs(q.y) + std::abs(q.z) < 1.e-10;
    // sqrt(V_cut) = Z_cut * diag(lambda_cut^(1/2)) * Z_cut^H.
    sqrt_coulomb(options_.chi0_coulomb_eigensolver, screening_sqrt_cut_local, desc,
                 screening_eigenvectors_local, filtered, screening_eigenvalues.data(), gamma);
    if (screening_full_metric)
    {
        // sqrt(V_full) = Z_full * diag(lambda_full^(1/2)) * Z_full^H.
        // The eigenvectors of V_full stay available for the Gamma head/wing step.
        sqrt_coulomb(options_.chi0_coulomb_eigensolver, screening_sqrt_full_local, desc,
                     screening_eigenvectors_local, filtered, screening_eigenvalues.data(), gamma);
    }
    else
    {
        // A single Coulomb metric has V_full == V_cut: reuse the square root
        // instead of repeating an identical eigendecomposition.
        screening_sqrt_full_local = screening_sqrt_cut_local.copy();
    }
    auto screening_sqrt_full_local_memory =
        watch_memory("screening_sqrt_full_local", screening_sqrt_full_local);
    const double screening_sqrt_end_time = MPI_Wtime();

    // Form epsilon = I - sqrt(V_full) * chi0 * sqrt(V_full).
    auto screening_response_local = scatter_screening_operator(hermitian(screening_chi0), desc);
    auto screening_response_local_memory =
        watch_memory("screening_response_local", screening_response_local);
    auto screening_workspace = init_local_mat<Complex>(desc, MAJOR::COL);
    auto screening_workspace_memory = watch_memory("screening_workspace", screening_workspace);
    // workspace = sqrt(V_full) * chi0
    ScalapackConnector::pgemm_f('N', 'N', n, n, n, 1., screening_sqrt_full_local.ptr(), 1, 1,
                                desc.desc, screening_response_local.ptr(), 1, 1, desc.desc, 0.,
                                screening_workspace.ptr(), 1, 1, desc.desc);
    // response = -workspace * sqrt(V_full), then add the identity below.
    ScalapackConnector::pgemm_f('N', 'N', n, n, n, -1., screening_workspace.ptr(), 1, 1, desc.desc,
                                screening_sqrt_full_local.ptr(), 1, 1, desc.desc, 0.,
                                screening_response_local.ptr(), 1, 1, desc.desc);
    for (int i = 0; i < desc.m_loc(); ++i)
    {
        for (int j = 0; j < desc.n_loc(); ++j)
        {
            if (desc.indx_l2g_r(i) == desc.indx_l2g_c(j))
            {
                screening_response_local(i, j) += 1.;
            }
        }
    }
    const double screening_epsilon_end_time = MPI_Wtime();
    if (gamma && ds.comm_h.myid == 0)
    {
        std::cout << "Gamma Coulomb retained subspace: " << n - filtered << " / " << n << '\n';
    }
    const bool screening_uses_headwing = gamma && options_.chi0_headwing;
    if (screening_uses_headwing)
    {
        // LibRPA option 3 performs the body inverse, analytic wing coupling and
        // angular Gamma-cell average, and returns epsilon^-1 (not epsilon).
        ds.p_headwing->rewrite_eps_abf_space(screening_response_local, 0, screening_sqrt_full_local,
                                             screening_eigenvectors_local, desc, n - filtered, 0.,
                                             false, false);
        // Check and remove the roundoff anti-Hermitian part of epsilon^-1.
        average_screening_hermitian_local(screening_response_local, screening_workspace, desc, n,
                                          ds.comm_h.comm);
    }
    else
    {
        // epsilon is Hermitian positive definite, so epsilon^-1 needs only a
        // Cholesky factorization instead of a full eigendecomposition.
        invert_screening_dielectric_cholesky(screening_response_local, screening_workspace, desc,
                                             n);
    }
    // sqrt(V_full) and the Coulomb eigenvectors are not needed after epsilon^-1.
    screening_sqrt_full_local.clear();
    screening_eigenvectors_local.clear();
    const double screening_inverse_end_time = MPI_Wtime();

    // epsilon=I-sqrt(V_full)*chi0*sqrt(V_full), W=sqrt(V_cut)*epsilon^-1*sqrt(V_cut).
    // The cut only regularizes the external interaction; using it inside epsilon
    // changes all q channels and is not the producer's GW screening convention.
    average_screening_hermitian_local(screening_sqrt_cut_local, screening_workspace, desc, n,
                                      ds.comm_h.comm);
    auto screening_w_local = init_local_mat<Complex>(desc, MAJOR::COL);
    auto screening_w_local_memory = watch_memory("screening_w_local", screening_w_local);
    // workspace = sqrt(V_cut) * epsilon^-1
    ScalapackConnector::pgemm_f('N', 'N', n, n, n, 1., screening_sqrt_cut_local.ptr(), 1, 1,
                                desc.desc, screening_response_local.ptr(), 1, 1, desc.desc, 0.,
                                screening_workspace.ptr(), 1, 1, desc.desc);
    // W = workspace * sqrt(V_cut)
    ScalapackConnector::pgemm_f('N', 'N', n, n, n, 1., screening_workspace.ptr(), 1, 1, desc.desc,
                                screening_sqrt_cut_local.ptr(), 1, 1, desc.desc, 0.,
                                screening_w_local.ptr(), 1, 1, desc.desc);
    screening_workspace.clear();
    const double screening_w_end_time = MPI_Wtime();

    // W goes from the block-cyclic layout to unique atom-pair owners, as in the
    // LibRPA GW Wc export: redistribute to one block per process (pgemr2d), then
    // the LibRPA IndexScheduler assembles each atom pair (I, J) on exactly one
    // rank. No rank holds a dense N_mu x N_mu W. The callers transform the owned
    // blocks to W(R) and send them to the LibRI atom lists
    // (distribute_screened_real_space). sqrt(V_cut) and epsilon^-1 are still
    // replicated, but only for the audit files or the binding-energy correction.
    if (!screening_owner_scheduler_)
    {
        screening_owner_descriptor_ = std::make_unique<ArrayDesc>(ds.blacs_h);
        screening_owner_descriptor_->init_1b1p(n, n, 0, 0);
        const auto screening_owner_pairs = get_balanced_ap_distribution_for_consec_descriptor(
            ds.basis_aux, ds.basis_aux, *screening_owner_descriptor_);
        screening_owner_scheduler_ = std::make_unique<IndexScheduler>();
        screening_owner_scheduler_->init(screening_owner_pairs, ds.basis_aux, ds.basis_aux,
                                         *screening_owner_descriptor_, false);
    }
    auto screening_w_consecutive_local =
        init_local_mat<Complex>(*screening_owner_descriptor_, MAJOR::COL);
    auto screening_w_consecutive_local_memory =
        watch_memory("screening_w_consecutive_local", screening_w_consecutive_local);
    // Block-cyclic (64x64 blocks) -> one block per process.
    ScalapackConnector::pgemr2d_f(n, n, screening_w_local.ptr(), 1, 1, desc.desc,
                                  screening_w_consecutive_local.ptr(), 1, 1,
                                  screening_owner_descriptor_->desc, desc.ictxt());
    screening_w_local.clear();
    ap_p_map<Matz> screening_w_owned_pairs;
    fill_ap_map_from_blacs_dist_scheduler(screening_w_owned_pairs, screening_w_consecutive_local,
                                          *screening_owner_scheduler_, ds.basis_aux,
                                          ds.basis_aux, *screening_owner_descriptor_);
    screening_w_consecutive_local.clear();
    ScreenedOwnedBlocks screening_w_owned;
    auto screening_w_owned_memory = watch_memory("screening_w_owned", screening_w_owned);
    for (auto &screening_owned_entry : screening_w_owned_pairs)
    {
        // The W(q) -> W(R) transform expects row-major atom-pair blocks.
        screening_owned_entry.second.swap_to_row_major();
        screening_w_owned.emplace(
            std::make_pair(static_cast<int>(screening_owned_entry.first.first),
                           static_cast<int>(screening_owned_entry.first.second)),
            std::move(screening_owned_entry.second));
    }
    screening_w_owned_pairs.clear();
    const bool screening_keeps_channels = options_.bse_plasma_energy_ev > 0;
    if (options_.out_screening_matrices || screening_keeps_channels)
    {
        auto screening_sqrt_cut =
            gather_screening_operator(screening_sqrt_cut_local, desc, n, ds.comm_h.comm);
        auto screening_sqrt_cut_memory = watch_memory("screening_sqrt_cut", screening_sqrt_cut);
        auto screening_inverse =
            gather_screening_operator(screening_response_local, desc, n, ds.comm_h.comm);
        auto screening_inverse_memory = watch_memory("screening_inverse", screening_inverse);

        // Optionally save coarse-grid operators for comparison with the producer.
        if (options_.out_screening_matrices && ds.comm_h.myid == 0)
        {
            // Self-describing binary audit record (native int32/float64/complex128,
            // row-major): n, q_fractional[3], omega_Ha, sqrt(V_cut), epsilon^-1.
            // These are the coarse-grid matrices BEFORE spatial interpolation.
            const auto frac = q * ds.pbc.latvec.Transpose();
            const double screening_audit_header[4] = {frac.x, frac.y, frac.z, omega};
            const auto path = std::filesystem::path(options_.output_dir) /
                              ("screening_q_" + std::to_string(diagnostic_index_++) + ".bin");
            std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char *>(&n), sizeof(n));
            out.write(reinterpret_cast<const char *>(screening_audit_header),
                      sizeof(screening_audit_header));
            out.write(reinterpret_cast<const char *>(screening_sqrt_cut.ptr()),
                      screening_sqrt_cut.size() * sizeof(Complex));
            out.write(reinterpret_cast<const char *>(screening_inverse.ptr()),
                      screening_inverse.size() * sizeof(Complex));
            if (!out)
            {
                throw std::runtime_error("cannot write screening audit matrices: " + path.string());
            }
        }

        // Keep the static channels for the later binding-energy correction.
        if (screening_keeps_channels)
        {
            channels_[q] = {std::move(screening_sqrt_cut), std::move(screening_inverse)};
        }
    }
    const double screening_end_time = MPI_Wtime();

    // One progress line per q point, so long screening stages are never silent.
    ++screening_q_count_;
    if (ds.comm_h.myid == 0)
    {
        const auto screening_q_fractional = q * ds.pbc.latvec.Transpose();
        std::cout << "chi0 screening q #" << screening_q_count_ << " = ("
                  << screening_q_fractional.x << ", " << screening_q_fractional.y << ", "
                  << screening_q_fractional.z << "), n_aux = " << n
                  << ", retained Coulomb subspace = " << n - filtered << ", inverse = "
                  << (screening_uses_headwing ? "Gamma head/wing" : "Cholesky") << '\n'
                  << "  seconds: Coulomb " << screening_coulomb_end_time - screening_start_time
                  << ", sqrt(V) " << screening_sqrt_end_time - screening_coulomb_end_time
                  << ", epsilon " << screening_epsilon_end_time - screening_sqrt_end_time
                  << ", epsilon^-1 " << screening_inverse_end_time - screening_epsilon_end_time
                  << ", W " << screening_w_end_time - screening_inverse_end_time << ", distribute "
                  << screening_end_time - screening_w_end_time << ", total "
                  << screening_end_time - screening_start_time
                  << "; node available memory " << screening_node_available_memory_gib()
                  << " GiB" << std::endl;
    }
    return screening_w_owned;
}

TensorMap<Complex> Chi0Screening::effective(double binding_ev, const std::vector<int> &rows,
                                            const std::vector<int> &cols)
{
    const double ha_to_ev = 2 * PARAM.constants.ry_to_ev;
    LibRPA_API::ScreenedQBlocks screening_w_q;
    auto screening_w_q_memory = watch_memory("screening_w_q", screening_w_q);
    for (const auto &[q, channel] : channels_)
    {
        const auto screening_inverse = effective_inverse_dielectric(
            channel.screening_inverse, options_.bse_plasma_energy_ev / ha_to_ev,
            binding_ev / ha_to_ev);
        const auto screening_w =
            channel.screening_sqrt_cut * screening_inverse * channel.screening_sqrt_cut;
        auto screening_w_memory = watch_memory("screening_w", screening_w);
        for (int i : rows)
        {
            for (int j : cols)
            {
                Matz screening_atom_pair(dataset_.basis_aux[i], dataset_.basis_aux[j],
                                         librpa_int::MAJOR::ROW);
                for (std::size_t a = 0; a < screening_atom_pair.nr(); ++a)
                {
                    for (std::size_t b = 0; b < screening_atom_pair.nc(); ++b)
                    {
                        screening_atom_pair(a, b) =
                            screening_w(dataset_.basis_aux.get_global_index(i, a),
                                        dataset_.basis_aux.get_global_index(j, b));
                    }
                }
                screening_w_q[i][j][q] = std::move(screening_atom_pair);
            }
        }
    }
    if (channels_.empty())
    {
        throw std::runtime_error("no cached dielectric channels for effective BSE");
    }
    // Apply the nonlinear spectral model BEFORE W(q)->W(R)->W(q_fine).
    return LibRPA_API::transform_screened_q_to_r(dataset_, pbc, screening_w_q);
}
} // namespace libbse
