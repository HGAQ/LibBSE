#include "bse_calculation.h"

#include "distributed_amplitudes.h"
#include "elpa_solver.h"
#include "matrix_checks.h"
#include "molecular_lri.h"
#include "spectrum.h"
#include "io/bse_files.h"
#include "interface/librpa_api.h"
#include "utils/profiler.h"
#include "utils/progress.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace fs = std::filesystem;

namespace libbse
{
namespace
{

void write_energies(const fs::path &file, const std::vector<double> &energies)
{
    std::ofstream output(file);
    if (!output) throw std::runtime_error("cannot write " + file.string());
    output << std::scientific << std::setprecision(8);
    for (const double energy : energies) output << energy << ' ';
    output << '\n';
}

std::vector<double> read_energies(const fs::path &file, int nstates)
{
    std::ifstream input(file);
    if (!input) throw std::runtime_error("cannot read " + file.string());
    std::vector<double> values(static_cast<std::size_t>(nstates));
    for (double &value : values)
        if (!(input >> value))
            throw std::runtime_error("truncated excitation-energy file "
                                     + file.string());
    return values;
}

void add_qp_diagonal(std::vector<Complex> &matrix,
                     const librpa_int::ArrayDesc &descriptor,
                     const QuasiparticleBands &qp,
                     const InputParameters &options)
{
    const int pair_dimension = options.nocc * options.nvirt;
    for (int ik = 0; ik != qp.nk; ++ik)
    {
        const auto band_offset = static_cast<std::size_t>(ik) * qp.nbands;
        for (int i = 0; i != options.nocc; ++i)
        {
            for (int a = 0; a != options.nvirt; ++a)
            {
                const int global = ik * pair_dimension + i * options.nvirt + a;
                const int local_row = descriptor.indx_g2l_r(global);
                const int local_col = descriptor.indx_g2l_c(global);
                if (local_row < 0 || local_col < 0) continue;
                matrix[static_cast<std::size_t>(local_row)
                       + static_cast<std::size_t>(local_col) * descriptor.lld()]
                    = qp.energies_ry[band_offset + options.nocc + a]
                      - qp.energies_ry[band_offset + i];
            }
        }
    }
}

fs::path energy_file(const InputParameters &options,
                     const std::string &spin_type, bool full)
{
    return fs::path(options.output_dir)
           / ("Excitation_Energy_" + std::string(full ? "full_" : "")
              + spin_type + ".dat");
}

fs::path amplitude_file(const InputParameters &options,
                        const std::string &spin_type,
                        const std::string &component,
                        int rank)
{
    return fs::path(options.output_dir)
           / ("Excitation_Amplitude_" + component + spin_type + "_"
              + std::to_string(rank) + ".dat");
}

void assemble_channel_matrix(
    std::vector<Complex> &matrix,
    const librpa_int::ArrayDesc &descriptor,
    const QuasiparticleBands &qp,
    const InputParameters &options,
    const InteractionCoefficients &coefficients,
    const std::vector<Complex> &hartree,
    const std::vector<Complex> &screened)
{
    matrix.assign(static_cast<std::size_t>(descriptor.lld())
                      * descriptor.n_loc(),
                  Complex{});
    add_qp_diagonal(matrix, descriptor, qp, options);
    if (coefficients.hartree != 0.0 && hartree.size() != matrix.size())
        throw std::runtime_error("missing Hartree matrix for BSE channel");
    if (coefficients.screened != 0.0 && screened.size() != matrix.size())
        throw std::runtime_error("missing screened matrix for BSE channel");
    for (std::size_t index = 0; index < matrix.size(); ++index)
    {
        if (coefficients.hartree != 0.0)
            matrix[index] += coefficients.hartree * hartree[index];
        if (coefficients.screened != 0.0)
            matrix[index] += coefficients.screened * screened[index];
    }
}

struct IpaSolution
{
    std::vector<double> energies;
    DistributedAmplitudes amplitudes;
};

IpaSolution solve_ipa_tda(const QuasiparticleBands &qp,
                          const InputParameters &options, int nstates,
                          MPI_Comm comm)
{
    const int pair_dimension = options.nocc * options.nvirt;
    const int dimension = qp.nk * pair_dimension;
    std::vector<double> gaps(static_cast<std::size_t>(dimension));
    for (int ik = 0; ik < qp.nk; ++ik)
    {
        const auto offset = static_cast<std::size_t>(ik) * qp.nbands;
        for (int i = 0; i < options.nocc; ++i)
            for (int a = 0; a < options.nvirt; ++a)
            {
                const int pair = ik * pair_dimension + i * options.nvirt + a;
                gaps[pair] = qp.energies_ry[offset + options.nocc + a]
                             - qp.energies_ry[offset + i];
            }
    }

    std::vector<int> indices(static_cast<std::size_t>(dimension));
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&gaps](int left, int right) { return gaps[left] < gaps[right]; });

    IpaSolution result;
    result.energies.resize(static_cast<std::size_t>(nstates));
    result.amplitudes = make_distributed_amplitudes(
        comm, dimension, nstates);
    for (int state = 0; state < nstates; ++state)
    {
        const int pair = indices[state];
        result.energies[state] = gaps[pair];
        if (pair >= result.amplitudes.first_pair
            && pair < result.amplitudes.first_pair
                           + result.amplitudes.local_pairs)
            result.amplitudes(
                state, pair - result.amplitudes.first_pair) = Complex(1.0);
    }
    return result;
}

struct ChannelResults
{
    std::string spin_type;
    std::vector<double> energies;
    DistributedAmplitudes amplitudes_x;
    DistributedAmplitudes amplitudes_y;
};

void report_matrix_check(const MatrixCheckResult &result,
                         const std::string &matrix_name,
                         double threshold, int rank)
{
    if (rank != 0) return;
    std::cout << "|  CHECK " << (result.passed ? "PASS" : "WARNING")
              << ": Matrix " << matrix_name << " is "
              << (matrix_name == "A" ? "Hermitian" : "symmetric")
              << " under threshold " << threshold << '\n';
}

bool b_symmetry_diagnostics_enabled()
{
    const char *value = std::getenv("LIBBSE_DIAG_B_SYMMETRY");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

struct FourierInteractionBlock
{
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::vector<Complex> values;
};

FourierInteractionBlock fourier_interaction_block(
    const TensorMap<Complex> &tensors, int iat, int jat,
    const std::array<double, 3> &q)
{
    FourierInteractionBlock result;
    const auto outer = tensors.find(iat);
    if (outer == tensors.end()) return result;
    constexpr double two_pi = 6.283185307179586476925286766559;
    for (const auto &[key, tensor] : outer->second)
    {
        if (key.first != jat || tensor.shape.size() != 2) continue;
        if (result.values.empty())
        {
            result.rows = tensor.shape[0];
            result.columns = tensor.shape[1];
            result.values.assign(result.rows * result.columns, Complex{});
        }
        if (tensor.shape[0] != result.rows
            || tensor.shape[1] != result.columns)
            throw std::runtime_error(
                "inconsistent interaction tensor dimensions in diagnostic");
        const Cell &cell = key.second;
        const double angle = two_pi
                             * (q[0] * cell[0] + q[1] * cell[1]
                                + q[2] * cell[2]);
        const Complex phase = std::polar(1.0, angle);
        for (std::size_t row = 0; row < result.rows; ++row)
            for (std::size_t column = 0; column < result.columns; ++column)
                result.values[row * result.columns + column]
                    += tensor(row, column) * phase;
    }
    return result;
}

void diagnose_interaction_reciprocity(
    const TensorMap<Complex> &tensors,
    const librpa_int::Dataset &dataset, const char *name, int rank)
{
    if (!b_symmetry_diagnostics_enabled()) return;
    int mpi_size = 1;
    MPI_Comm_size(dataset.comm_h.comm, &mpi_size);
    if (mpi_size != 1)
    {
        if (rank == 0)
            std::cout << "|  interaction reciprocity diagnostic for " << name
                      << " requires one MPI rank; skipped\n";
        return;
    }

    const std::array<std::array<double, 3>, 10> q_points{{
        {{0.0, 0.0, 0.0}},
        {{0.5, 0.0, 0.0}},
        {{0.5, 0.5, 0.0}},
        {{0.5, 0.5, 0.5}},
        {{1.0 / 3.0, 0.0, 0.0}},
        {{2.0 / 3.0, 0.0, 0.0}},
        {{1.0 / 3.0, 1.0 / 3.0, 0.0}},
        {{2.0 / 3.0, 2.0 / 3.0, 0.0}},
        {{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}},
        {{2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0}}
    }};
    std::cout << "|  interaction reciprocity diagnostic: " << name << '\n';
    for (const auto &q : q_points)
    {
        const std::array<double, 3> minus_q{{-q[0], -q[1], -q[2]}};
        double transpose_difference_squared = 0.0;
        double transpose_sum_squared = 0.0;
        double hermitian_difference_squared = 0.0;
        double hermitian_sum_squared = 0.0;
        for (int iat = 0; iat < static_cast<int>(dataset.atoms.size()); ++iat)
            for (int jat = 0; jat < static_cast<int>(dataset.atoms.size()); ++jat)
            {
                const FourierInteractionBlock forward =
                    fourier_interaction_block(tensors, iat, jat, q);
                const FourierInteractionBlock reverse_minus =
                    fourier_interaction_block(tensors, jat, iat, minus_q);
                const FourierInteractionBlock reverse_same =
                    fourier_interaction_block(tensors, jat, iat, q);
                if (forward.values.empty() || reverse_minus.values.empty()
                    || reverse_same.values.empty())
                    continue;
                if (forward.rows != reverse_minus.columns
                    || forward.columns != reverse_minus.rows
                    || forward.rows != reverse_same.columns
                    || forward.columns != reverse_same.rows)
                    throw std::runtime_error(
                        "reciprocal interaction tensor dimensions differ");
                for (std::size_t row = 0; row < forward.rows; ++row)
                    for (std::size_t column = 0;
                         column < forward.columns; ++column)
                    {
                        const Complex value =
                            forward.values[row * forward.columns + column];
                        const Complex transposed = reverse_minus.values[
                            column * reverse_minus.columns + row];
                        const Complex adjoint = std::conj(reverse_same.values[
                            column * reverse_same.columns + row]);
                        transpose_difference_squared
                            += std::norm(value - transposed);
                        transpose_sum_squared += std::norm(value + transposed);
                        hermitian_difference_squared
                            += std::norm(value - adjoint);
                        hermitian_sum_squared += std::norm(value + adjoint);
                    }
            }
        const auto ratio = [](double difference, double sum)
        {
            return sum > 0.0 ? std::sqrt(difference / sum) : 0.0;
        };
        std::cout << "|   q=(" << q[0] << ',' << q[1] << ',' << q[2]
                  << ") transpose-reciprocity="
                  << ratio(transpose_difference_squared,
                           transpose_sum_squared)
                  << " Hermiticity="
                  << ratio(hermitian_difference_squared,
                           hermitian_sum_squared) << '\n';
    }
}

void diagnose_b_component(const std::vector<Complex> &matrix,
                          const librpa_int::ArrayDesc &descriptor,
                          const char *name, int rank)
{
    if (!b_symmetry_diagnostics_enabled() || matrix.empty()) return;
    if (rank == 0)
        std::cout << "|  B-symmetry diagnostic component: " << name << '\n';
    const MatrixCheckResult check = check_symmetric(
        matrix, descriptor, PARAM.constants.matrix_symmetry_threshold);
    if (rank == 0)
        std::cout << "|   component status: "
                  << (check.passed ? "PASS" : "FAIL") << '\n';
}

void diagnose_b_k_blocks(const std::vector<Complex> &matrix,
                         const librpa_int::ArrayDesc &descriptor,
                         int pair_dimension,
                         const librpa_int::Dataset &dataset, int rank)
{
    if (!b_symmetry_diagnostics_enabled() || matrix.empty()) return;
    const int nk = dataset.mf_band.get_n_kpoints();
    if (descriptor.m() != nk * pair_dimension) return;

    std::vector<Complex> transposed(matrix.size());
    LibRPA_API::distributed_transpose(
        descriptor.m(), descriptor.n(), matrix.data(), descriptor,
        transposed.data());
    std::vector<double> difference_squared(static_cast<std::size_t>(nk) * nk,
                                           0.0);
    std::vector<double> sum_squared(static_cast<std::size_t>(nk) * nk, 0.0);
    for (int local_column = 0; local_column < descriptor.n_loc(); ++local_column)
    {
        const int global_column = descriptor.indx_l2g_c(local_column);
        const int column_k = global_column / pair_dimension;
        for (int local_row = 0; local_row < descriptor.m_loc(); ++local_row)
        {
            const int global_row = descriptor.indx_l2g_r(local_row);
            const int row_k = global_row / pair_dimension;
            const std::size_t local_index =
                static_cast<std::size_t>(local_column) * descriptor.lld()
                + local_row;
            const std::size_t block_index =
                static_cast<std::size_t>(row_k) * nk + column_k;
            difference_squared[block_index]
                += std::norm(matrix[local_index] - transposed[local_index]);
            sum_squared[block_index]
                += std::norm(matrix[local_index] + transposed[local_index]);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, difference_squared.data(),
                  static_cast<int>(difference_squared.size()), MPI_DOUBLE,
                  MPI_SUM, descriptor.comm());
    MPI_Allreduce(MPI_IN_PLACE, sum_squared.data(),
                  static_cast<int>(sum_squared.size()), MPI_DOUBLE,
                  MPI_SUM, descriptor.comm());
    if (rank != 0) return;

    std::vector<int> order(static_cast<std::size_t>(nk) * nk);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int left, int right)
    {
        return difference_squared[static_cast<std::size_t>(left)]
               > difference_squared[static_cast<std::size_t>(right)];
    });

    double diagonal_difference_squared = 0.0;
    double diagonal_sum_squared = 0.0;
    double offdiagonal_difference_squared = 0.0;
    double offdiagonal_sum_squared = 0.0;
    for (int row_k = 0; row_k < nk; ++row_k)
        for (int column_k = 0; column_k < nk; ++column_k)
        {
            const std::size_t index =
                static_cast<std::size_t>(row_k) * nk + column_k;
            if (row_k == column_k)
            {
                diagonal_difference_squared += difference_squared[index];
                diagonal_sum_squared += sum_squared[index];
            }
            else
            {
                offdiagonal_difference_squared += difference_squared[index];
                offdiagonal_sum_squared += sum_squared[index];
            }
        }
    const auto ratio = [](double difference, double sum)
    {
        return sum > 0.0 ? std::sqrt(difference / sum) : 0.0;
    };
    std::cout << "|  screened-B k-block symmetry diagnostic:\n"
              << "|   same-k relative error: "
              << ratio(diagonal_difference_squared, diagonal_sum_squared)
              << '\n'
              << "|   different-k relative error: "
              << ratio(offdiagonal_difference_squared,
                       offdiagonal_sum_squared)
              << '\n'
              << "|   largest ||B(k1,k2)-B(k2,k1)^T||_F blocks:\n";
    const int count = std::min(12, static_cast<int>(order.size()));
    for (int position = 0; position < count; ++position)
    {
        const int block = order[static_cast<std::size_t>(position)];
        const int row_k = block / nk;
        const int column_k = block % nk;
        const auto &k1 = dataset.kfrac_band_list.at(
            static_cast<std::size_t>(row_k));
        const auto &k2 = dataset.kfrac_band_list.at(
            static_cast<std::size_t>(column_k));
        const auto wrap = [](double value)
        {
            value -= std::floor(value);
            return value;
        };
        const std::size_t block_index = static_cast<std::size_t>(block);
        std::cout << "|    k1=" << row_k + 1 << " k2=" << column_k + 1
                  << " q=(" << wrap(k2.x - k1.x) << ','
                  << wrap(k2.y - k1.y) << ','
                  << wrap(k2.z - k1.z) << ") diff="
                  << std::sqrt(difference_squared[block_index])
                  << " rel="
                  << ratio(difference_squared[block_index],
                           sum_squared[block_index]) << '\n';
    }
}

} // namespace

void run_bse(const InputParameters &options,
             const std::shared_ptr<librpa_int::Dataset> &dataset)
{
    ScopedTimer run_timer(global::profiler, "run_bse", "BSE calculation");
    int rank = 0;
    MPI_Comm_rank(dataset->comm_h.comm, &rank);
    fs::create_directories(options.output_dir);

    const auto qp = [&]()
    {
        ScopedTimer timer(global::profiler, "read_qp_bands",
                          "Read quasiparticle bands");
        return read_qp_bands(options, *dataset);
    }();
    if (rank == 0)
    {
        std::cout << "BSE quasiparticle bands\n"
                  << "  selected core offset: " << qp.ncore << '\n'
                  << "  indirect gap (eV): "
                  << qp.indirect_gap_ry * PARAM.constants.ry_to_ev << '\n'
                  << "  direct gap (eV): "
                  << qp.direct_gap_ry * PARAM.constants.ry_to_ev << '\n';
    }

    const int dimension = qp.nk * options.nocc * options.nvirt;
    const int nstates = options.bse_nstates < 0 ? dimension : options.bse_nstates;
    if (nstates > dimension)
        throw std::invalid_argument(
            "number of requested states exceeds the BSE dimension");

    if (options.spectrum_only())
    {
        if (rank == 0)
            std::cout << "Preparing fine-grid velocity_mo for velocity gauge...\n";
        const auto velocity = [&]()
        {
            ScopedTimer timer(global::profiler, "prepare_velocity_mo",
                              "Prepare fine-grid velocity_mo");
            return prepare_fine_velocity_mo(options, qp, dataset);
        }();
        done("prepare velocity matrix in MO representation",
             dataset->comm_h.comm);

        for (const std::string &spin_type : options.bse_spin_types)
        {
            if (options.solve_tda())
            {
                ChannelResults channel;
                channel.spin_type = spin_type;
                if (rank == 0)
                {
                    ScopedTimer timer(global::profiler, "read_tda_eigenstates",
                                      "Read TDA eigenstates");
                    channel.energies = read_energies(
                        energy_file(options, spin_type, false), nstates);
                }
                {
                    ScopedTimer timer(global::profiler,
                                      "read_tda_amplitudes",
                                      "Read distributed TDA amplitudes");
                    channel.amplitudes_x = read_distributed_amplitudes(
                        amplitude_file(options, spin_type, "", rank),
                        dataset->comm_h.comm, dimension, nstates);
                }
                {
                    ScopedTimer timer(global::profiler, "tda_spectrum",
                                      "Calculate TDA velocity-gauge spectrum");
                    write_velocity_gauge_outputs(
                        options, *dataset, velocity, channel.energies,
                        channel.amplitudes_x, nullptr, spin_type, "tda");
                }
                done("calculate TDA velocity-gauge spectrum " + spin_type,
                     dataset->comm_h.comm);
            }
            if (options.solve_full())
            {
                ChannelResults channel;
                channel.spin_type = spin_type;
                if (rank == 0)
                {
                    ScopedTimer timer(global::profiler, "read_full_eigenstates",
                                      "Read full-BSE eigenstates");
                    channel.energies = read_energies(
                        energy_file(options, spin_type, true), nstates);
                }
                {
                    ScopedTimer timer(global::profiler,
                                      "read_full_amplitudes",
                                      "Read distributed full-BSE amplitudes");
                    channel.amplitudes_x = read_distributed_amplitudes(
                        amplitude_file(options, spin_type, "full_X_", rank),
                        dataset->comm_h.comm, dimension, nstates);
                    channel.amplitudes_y = read_distributed_amplitudes(
                        amplitude_file(options, spin_type, "full_Y_", rank),
                        dataset->comm_h.comm, dimension, nstates);
                }
                {
                    ScopedTimer timer(global::profiler, "full_spectrum",
                                      "Calculate full-BSE velocity-gauge spectrum");
                    write_velocity_gauge_outputs(
                        options, *dataset, velocity, channel.energies,
                        channel.amplitudes_x, &channel.amplitudes_y,
                        spin_type, "full");
                }
                done("calculate full-BSE velocity-gauge spectrum " + spin_type,
                     dataset->comm_h.comm);
            }
        }
        return;
    }

    if (options.ipa_only())
    {
        if (rank == 0)
            std::cout << "Independent particle approximation: "
                         "building sorted transition states directly...\n";
        ChannelResults channel;
        channel.spin_type = "ipa";
        {
            ScopedTimer timer(global::profiler, "solve_ipa",
                              "Build independent-particle TDA states");
            auto solution = solve_ipa_tda(
                qp, options, nstates, dataset->comm_h.comm);
            channel.energies = std::move(solution.energies);
            channel.amplitudes_x = std::move(solution.amplitudes);
        }
        done("build independent-particle TDA states", dataset->comm_h.comm);
        if (rank == 0)
        {
            ScopedTimer timer(global::profiler, "write_tda_results",
                              "Write TDA eigenstates");
            write_energies(energy_file(options, "ipa", false),
                           channel.energies);
        }
        {
            ScopedTimer timer(global::profiler,
                              "write_tda_amplitudes",
                              "Write distributed TDA amplitudes");
            write_distributed_amplitudes(
                amplitude_file(options, "ipa", "", rank),
                channel.amplitudes_x);
        }

        if (rank == 0)
            std::cout << "Preparing fine-grid velocity_mo for velocity gauge...\n";
        const auto velocity = [&]()
        {
            ScopedTimer timer(global::profiler, "prepare_velocity_mo",
                              "Prepare fine-grid velocity_mo");
            return prepare_fine_velocity_mo(options, qp, dataset);
        }();
        done("prepare velocity matrix in MO representation",
             dataset->comm_h.comm);
        {
            ScopedTimer timer(global::profiler, "tda_spectrum",
                              "Calculate TDA velocity-gauge spectrum");
            write_velocity_gauge_outputs(
                options, *dataset, velocity, channel.energies,
                channel.amplitudes_x, nullptr, "ipa", "tda");
        }
        done("calculate TDA velocity-gauge spectrum ipa",
             dataset->comm_h.comm);
        return;
    }

    if (rank == 0)
        std::cout << "Transforming cut Coulomb from q to R with LibRPA FT_Vq...\n";
    auto bare = [&]()
    {
        ScopedTimer timer(global::profiler, "transform_bare_coulomb",
                          "Transform bare Coulomb q to R");
        return LibRPA_API::build_bare_coulomb(*dataset);
    }();
    done("transform cut Coulomb from q to R", dataset->comm_h.comm);

    MolecularLri molecular(*dataset, options, qp);

    if (rank == 0)
        std::cout << "Reading Wc and constructing W = V + Wc...\n";
    auto screened = [&]()
    {
        ScopedTimer timer(global::profiler, "read_screened_interaction",
                          "Read and construct screened interaction");
        return read_screened_interaction(options, bare,
                                         dataset->pbc.Rlist.size(),
                                         molecular.local_i_atoms(),
                                         molecular.local_j_atoms());
    }();
    done("read Wc and construct screened interaction",
         dataset->comm_h.comm);

    if (rank == 0)
        std::cout << "Converting LibRPA Cs tensors for complex LibRI contractions...\n";
    auto coefficients = [&]()
    {
        ScopedTimer timer(global::profiler, "convert_ri_coefficients",
                          "Convert RI coefficients for LibRI");
        return convert_lri_coefficients(*dataset);
    }();
    done("convert RI coefficients for LibRI", dataset->comm_h.comm);

    diagnose_interaction_reciprocity(
        bare, *dataset, "bare interaction before nearest-cell remap", rank);
    diagnose_interaction_reciprocity(
        screened, *dataset,
        "screened interaction before nearest-cell remap", rank);

    {
        ScopedTimer timer(global::profiler, "remap_bvk_cells",
                          "Remap interactions to nearest BvK cells");
        remap_to_nearest_bvk_cell(coefficients, *dataset);
        remap_to_nearest_bvk_cell(bare, *dataset);
        remap_to_nearest_bvk_cell(screened, *dataset);
    }
    done("remap interactions to nearest BvK cells", dataset->comm_h.comm);
    diagnose_interaction_reciprocity(
        bare, *dataset, "bare interaction after nearest-cell remap", rank);
    diagnose_interaction_reciprocity(
        screened, *dataset,
        "screened interaction after nearest-cell remap", rank);

    if (rank == 0)
        std::cout << "Initializing external LibRI RI::LR...\n";
    {
        ScopedTimer timer(global::profiler, "initialize_libri",
                          "Initialize LibRI BSE contractions");
        molecular.initialize(coefficients, bare, screened);
    }
    done("initialize LibRI BSE contractions", dataset->comm_h.comm);

    librpa_int::ArrayDesc descriptor(dataset->blacs_h);
    const int block_size = dimension > 1000 ? 64 : (dimension > 500 ? 32 : 1);
    if (descriptor.init(dimension, dimension, block_size, block_size, 0, 0) != 0)
        throw std::runtime_error("failed to initialize BSE BLACS descriptor");
    const std::size_t local_size = static_cast<std::size_t>(descriptor.lld())
                                   * descriptor.n_loc();

    std::vector<Complex> hartree_a;
    std::vector<Complex> screened_a;
    std::vector<Complex> hartree_b;
    std::vector<Complex> screened_b;
    if (options.requires_hartree())
    {
        hartree_a.assign(local_size, Complex{});
        {
            ScopedTimer timer(global::profiler, "hartree_a", "LibRI Hartree A");
            molecular.add_hartree_a(hartree_a, descriptor, 1.0);
        }
        done("construct Hartree contribution for A", dataset->comm_h.comm);
    }
    if (options.requires_screened())
    {
        screened_a.assign(local_size, Complex{});
        {
            ScopedTimer timer(global::profiler, "screened_a", "LibRI screened A");
            molecular.add_screened_a(screened_a, descriptor, 1.0);
        }
        done("construct screened contribution for A", dataset->comm_h.comm);
    }
    if (options.solve_full() && options.requires_hartree())
    {
        hartree_b.assign(local_size, Complex{});
        {
            ScopedTimer timer(global::profiler, "hartree_b", "LibRI Hartree B");
            molecular.add_hartree_b(hartree_b, descriptor, 1.0);
        }
        done("construct Hartree contribution for B", dataset->comm_h.comm);
        diagnose_b_component(hartree_b, descriptor, "Hartree B", rank);
    }
    if (options.solve_full() && options.requires_screened())
    {
        screened_b.assign(local_size, Complex{});
        {
            ScopedTimer timer(global::profiler, "screened_b", "LibRI screened B");
            molecular.add_screened_b(screened_b, descriptor, 1.0);
        }
        done("construct screened contribution for B", dataset->comm_h.comm);
        diagnose_b_component(screened_b, descriptor, "screened B", rank);
        diagnose_b_k_blocks(screened_b, descriptor,
                            options.nocc * options.nvirt, *dataset, rank);
    }
    molecular.release_interactions();

    std::vector<ChannelResults> tda_results;
    std::vector<ChannelResults> full_results;
    if (options.solve_tda())
    {
        for (const std::string &spin_type : options.bse_spin_types)
        {
            const auto channel_coefficients =
                interaction_coefficients(spin_type);
            std::vector<Complex> matrix;
            {
                ScopedTimer timer(global::profiler, "assemble_tda_matrix",
                                  "Assemble channel TDA matrix");
                assemble_channel_matrix(
                    matrix, descriptor, qp, options, channel_coefficients,
                    hartree_a, screened_a);
            }
            MatrixCheckResult matrix_check;
            {
                ScopedTimer timer(global::profiler, "check_tda_a_matrix",
                                  "Check TDA A-matrix Hermiticity");
                matrix_check = check_hermitian(
                    matrix, descriptor,
                    PARAM.constants.matrix_symmetry_threshold);
            }
            report_matrix_check(
                matrix_check, "A", PARAM.constants.matrix_symmetry_threshold,
                rank);
            done("initialize and check " + spin_type + " TDA A matrix",
                 dataset->comm_h.comm);
            if (rank == 0)
                std::cout << "Diagonalizing " << spin_type
                          << " TDA matrix with ELPA...\n";
            EigenSolution solution;
            {
                ScopedTimer timer(global::profiler, "solve_tda_elpa",
                                  "Diagonalize TDA matrix with ELPA");
                solution = solve_tda_elpa(
                    matrix, descriptor, options.bse_nstates);
            }
            done("diagonalize " + spin_type + " TDA matrix with ELPA",
                 dataset->comm_h.comm);

            ChannelResults channel;
            channel.spin_type = spin_type;
            channel.energies = solution.energies_ry;
            {
                ScopedTimer timer(global::profiler,
                                  "redistribute_tda_eigenvectors",
                                  "Redistribute TDA eigenvectors");
                channel.amplitudes_x = redistribute_amplitudes(
                    dataset->comm_h.comm, solution.vectors_local, descriptor,
                    0, 0, dimension, nstates);
            }
            if (rank == 0)
            {
                ScopedTimer timer(global::profiler, "write_tda_results",
                                  "Write TDA eigenstates");
                write_energies(energy_file(options, spin_type, false),
                               channel.energies);
                std::cout << std::setprecision(10)
                          << spin_type << " TDA lowest excitation (eV): "
                          << channel.energies.front()
                                 * PARAM.constants.ry_to_ev << '\n'
                          << spin_type << " TDA binding energy (eV): "
                          << (qp.direct_gap_ry - channel.energies.front())
                                 * PARAM.constants.ry_to_ev << '\n';
            }
            {
                ScopedTimer timer(global::profiler,
                                  "write_tda_amplitudes",
                                  "Write distributed TDA amplitudes");
                write_distributed_amplitudes(
                    amplitude_file(options, spin_type, "", rank),
                    channel.amplitudes_x);
            }
            done("write TDA states " + spin_type, dataset->comm_h.comm);
            tda_results.push_back(std::move(channel));
        }
    }

    if (options.solve_full())
    {
        librpa_int::ArrayDesc full_descriptor(dataset->blacs_h);
        if (full_descriptor.init(2 * dimension, 2 * dimension,
                                 block_size, block_size, 0, 0) != 0)
            throw std::runtime_error(
                "failed to initialize full-BSE BLACS descriptor");

        for (const std::string &spin_type : options.bse_spin_types)
        {
            const auto channel_coefficients =
                interaction_coefficients(spin_type);
            std::vector<Complex> matrix_a;
            std::vector<Complex> matrix_b(local_size, Complex{});
            {
                ScopedTimer timer(global::profiler, "assemble_full_a_matrix",
                                  "Assemble channel full-BSE A matrix");
                assemble_channel_matrix(
                    matrix_a, descriptor, qp, options, channel_coefficients,
                    hartree_a, screened_a);
            }
            MatrixCheckResult a_check;
            {
                ScopedTimer timer(global::profiler, "check_full_a_matrix",
                                  "Check full-BSE A-matrix Hermiticity");
                a_check = check_hermitian(
                    matrix_a, descriptor,
                    PARAM.constants.matrix_symmetry_threshold);
            }
            report_matrix_check(
                a_check, "A", PARAM.constants.matrix_symmetry_threshold,
                rank);
            done("initialize and check " + spin_type + " full-BSE A matrix",
                 dataset->comm_h.comm);

            {
                ScopedTimer timer(global::profiler, "assemble_full_b_matrix",
                                  "Assemble channel full-BSE B matrix");
                for (std::size_t index = 0; index < local_size; ++index)
                {
                    if (channel_coefficients.hartree != 0.0)
                        matrix_b[index] += channel_coefficients.hartree
                                           * hartree_b[index];
                    if (channel_coefficients.screened != 0.0)
                        matrix_b[index] += channel_coefficients.screened
                                           * screened_b[index];
                }
            }
            MatrixCheckResult b_check;
            {
                ScopedTimer timer(global::profiler, "check_full_b_matrix",
                                  "Check full-BSE B-matrix symmetry");
                b_check = check_symmetric(
                    matrix_b, descriptor,
                    PARAM.constants.matrix_symmetry_threshold);
            }
            report_matrix_check(
                b_check, "B", PARAM.constants.matrix_symmetry_threshold,
                rank);
            done("initialize and check " + spin_type + " full-BSE B matrix",
                 dataset->comm_h.comm);
            if (rank == 0)
                std::cout << "Diagonalizing " << spin_type
                          << " full BSE Hamiltonian with ELPA...\n";
            EigenSolution solution;
            {
                ScopedTimer timer(global::profiler, "solve_full_elpa",
                                  "Solve full BSE with ELPA");
                solution = solve_full_elpa(
                    matrix_a, matrix_b, descriptor, full_descriptor,
                    options.bse_nstates);
            }
            done("solve " + spin_type + " full BSE with ELPA",
                 dataset->comm_h.comm);

            ChannelResults channel;
            channel.spin_type = spin_type;
            channel.energies = solution.energies_ry;
            {
                ScopedTimer timer(global::profiler,
                                  "redistribute_full_eigenvectors",
                                  "Redistribute full-BSE eigenvectors");
                channel.amplitudes_x = redistribute_amplitudes(
                    dataset->comm_h.comm, solution.vectors_local,
                    full_descriptor, 0, dimension, dimension, nstates);
                channel.amplitudes_y = redistribute_amplitudes(
                    dataset->comm_h.comm, solution.vectors_local,
                    full_descriptor, dimension, dimension,
                    dimension, nstates);
            }
            if (rank == 0)
            {
                ScopedTimer timer(global::profiler, "write_full_results",
                                  "Write full-BSE eigenstates");
                write_energies(energy_file(options, spin_type, true),
                               channel.energies);
                std::cout << std::setprecision(10)
                          << spin_type << " full-BSE lowest excitation (eV): "
                          << channel.energies.front()
                                 * PARAM.constants.ry_to_ev << '\n'
                          << spin_type << " full-BSE binding energy (eV): "
                          << (qp.direct_gap_ry - channel.energies.front())
                                 * PARAM.constants.ry_to_ev << '\n';
            }
            {
                ScopedTimer timer(global::profiler,
                                  "write_full_amplitudes",
                                  "Write distributed full-BSE amplitudes");
                write_distributed_amplitudes(
                    amplitude_file(options, spin_type, "full_X_", rank),
                    channel.amplitudes_x);
                write_distributed_amplitudes(
                    amplitude_file(options, spin_type, "full_Y_", rank),
                    channel.amplitudes_y);
            }
            done("write full-BSE states " + spin_type,
                 dataset->comm_h.comm);
            full_results.push_back(std::move(channel));
        }
    }

    if (rank == 0)
        std::cout << "Preparing fine-grid velocity_mo for velocity gauge...\n";
    const auto velocity = [&]()
    {
        ScopedTimer timer(global::profiler, "prepare_velocity_mo",
                          "Prepare fine-grid velocity_mo");
        return prepare_fine_velocity_mo(options, qp, dataset);
    }();
    done("prepare velocity matrix in MO representation",
         dataset->comm_h.comm);

    for (const ChannelResults &channel : tda_results)
    {
        {
            ScopedTimer timer(global::profiler, "tda_spectrum",
                              "Calculate TDA velocity-gauge spectrum");
            write_velocity_gauge_outputs(
                options, *dataset, velocity, channel.energies,
                channel.amplitudes_x, nullptr, channel.spin_type, "tda");
        }
        done("calculate TDA velocity-gauge spectrum " + channel.spin_type,
             dataset->comm_h.comm);
    }
    for (const ChannelResults &channel : full_results)
    {
        {
            ScopedTimer timer(global::profiler, "full_spectrum",
                              "Calculate full-BSE velocity-gauge spectrum");
            write_velocity_gauge_outputs(
                options, *dataset, velocity, channel.energies,
                channel.amplitudes_x, &channel.amplitudes_y,
                channel.spin_type, "full");
        }
        done("calculate full-BSE velocity-gauge spectrum "
                 + channel.spin_type,
             dataset->comm_h.comm);
    }
}

} // namespace libbse
