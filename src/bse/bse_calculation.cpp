#include "io/aims_screened.h"
#include "io/chi0_screening.h"
#include "bse_calculation.h"

#include "distributed_amplitudes.h"
#include "elpa_solver.h"
#include "matrix_checks.h"
#include "molecular_lri.h"
#include "spectrum.h"
#include "io/bse_files.h"
#include "interface/librpa_api.h"
#include "utils/profiler.h"
#include "utils/memory_views.h"

#include <mpi.h>

#include <algorithm>
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
    auto values_memory = libbse::watch_memory("bse_calculation.values", values);
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

IpaSolution solve_ipa_tda(const QuasiparticleBands &qp,
                          const InputParameters &options, int nstates,
                          MPI_Comm comm)
{
    const int pair_dimension = options.nocc * options.nvirt;
    const int dimension = qp.nk * pair_dimension;
    std::vector<double> gaps(static_cast<std::size_t>(dimension));
    auto gaps_memory = libbse::watch_memory("bse_calculation.gaps", gaps);
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
    auto indices_memory = libbse::watch_memory("bse_calculation.indices", indices);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&gaps](int left, int right) { return gaps[left] < gaps[right]; });

    IpaSolution result;
    auto result_memory = watch_memory("result", result);
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

} // namespace

void run_bse(const InputParameters &options,
             const std::shared_ptr<librpa_int::Dataset> &dataset)
{
    ScopedTimer run_timer(global::profiler, "run_bse", "BSE calculation", dataset->comm_h.comm);
    int rank = 0;
    MPI_Comm_rank(dataset->comm_h.comm, &rank);
    fs::create_directories(options.output_dir);

    // Read quasiparticle bands from the dataset and report the gaps.
    QuasiparticleBands qp;
    auto qp_memory = watch_memory("qp", qp);
    {
        ScopedTimer timer(global::profiler, "read_qp_bands",
                          "Read quasiparticle bands", dataset->comm_h.comm);
        qp = read_qp_bands(options, *dataset);
    }
    if (rank == 0)
    {
        std::cout << "BSE quasiparticle bands\n"
                  << "  selected core offset: " << qp.ncore << '\n'
                  << "  indirect gap (eV): "
                  << qp.indirect_gap_ry * PARAM.constants.ry_to_ev << '\n'
                  << "  direct gap (eV): "
                  << qp.direct_gap_ry * PARAM.constants.ry_to_ev << '\n';
    }

    // Apply the band gauge once, before either Hamiltonian construction or a
    // spectrum-only restart.  Keep the phases so a separately supplied
    // same-grid velocity matrix can be transformed covariantly as well.
    std::vector<Complex> band_gauge_phases;
    auto gauge_memory = watch_memory("band_gauge_phases", band_gauge_phases);
    {
        ScopedTimer timer(global::profiler, "wavefunction_gauge", "Apply wavefunction gauge", dataset->comm_h.comm);
        band_gauge_phases = apply_wavefunction_gauge(*dataset, options, qp);
    }

    const int dimension = qp.nk * options.nocc * options.nvirt;
    const int nstates = options.bse_nstates < 0 ? dimension : options.bse_nstates;
    if (nstates > dimension)
        throw std::invalid_argument(
            "number of requested states exceeds the BSE dimension");
    
    //already have the velocity matrix in MO representation, just need to read it and calculate the spectrum
    if (options.spectrum_only())
    {
        if (rank == 0)
            std::cout << "Preparing fine-grid velocity_mo for velocity gauge...\n";
        FineVelocityMo velocity;
        auto velocity_memory = watch_memory("velocity", velocity);
        {
            ScopedTimer timer(global::profiler, "prepare_velocity_mo",
                              "Prepare fine-grid velocity_mo", dataset->comm_h.comm);
            velocity = prepare_fine_velocity_mo(
                options, qp, dataset, band_gauge_phases);
        }

        for (const std::string &spin_type : options.bse_spin_types)
        {
            if (options.solve_tda())
            {
                ChannelSolution channel;
                auto channel_memory = watch_memory("channel", channel);
                channel.spin_type = spin_type;
                {
                    ScopedTimer timer(global::profiler, "read_tda_eigenstates",
                                      "Read TDA eigenstates", dataset->comm_h.comm);
                    if (rank == 0)
                    {
                        channel.energies = read_energies(
                            energy_file(options, spin_type, false), nstates);
                    }
                }
                {
                    ScopedTimer timer(global::profiler,
                                      "read_tda_amplitudes",
                                      "Read distributed TDA amplitudes", dataset->comm_h.comm);
                    channel.amplitudes_x = read_distributed_amplitudes(
                        amplitude_file(options, spin_type, "", rank),
                        dataset->comm_h.comm, dimension, nstates);
                }
                {
                    ScopedTimer timer(global::profiler, "tda_spectrum",
                                      "Calculate TDA velocity-gauge spectrum", dataset->comm_h.comm);
                    write_velocity_gauge_outputs(
                        options, *dataset, velocity, channel.energies,
                        channel.amplitudes_x, nullptr, spin_type, "tda");
                }

            }
            if (options.solve_full())
            {
                ChannelSolution channel;
                auto channel_memory = watch_memory("channel", channel);
                channel.spin_type = spin_type;
                {
                    ScopedTimer timer(global::profiler, "read_full_eigenstates",
                                      "Read full-BSE eigenstates", dataset->comm_h.comm);
                    if (rank == 0)
                    {
                        channel.energies = read_energies(
                            energy_file(options, spin_type, true), nstates);
                    }
                }
                {
                    ScopedTimer timer(global::profiler,
                                      "read_full_amplitudes",
                                      "Read distributed full-BSE amplitudes", dataset->comm_h.comm);
                    channel.amplitudes_x = read_distributed_amplitudes(
                        amplitude_file(options, spin_type, "full_X_", rank),
                        dataset->comm_h.comm, dimension, nstates);
                    channel.amplitudes_y = read_distributed_amplitudes(
                        amplitude_file(options, spin_type, "full_Y_", rank),
                        dataset->comm_h.comm, dimension, nstates);
                }
                {
                    ScopedTimer timer(global::profiler, "full_spectrum",
                                      "Calculate full-BSE velocity-gauge spectrum", dataset->comm_h.comm);
                    write_velocity_gauge_outputs(
                        options, *dataset, velocity, channel.energies,
                        channel.amplitudes_x, &channel.amplitudes_y,
                        spin_type, "full");
                }

            }
        }
        return;
    }

    // ipa calculation
    if (options.ipa_only())
    {
        if (rank == 0)
            std::cout << "Independent particle approximation: "
                         "building sorted transition states directly...\n";
        ChannelSolution channel;
        auto channel_memory = watch_memory("channel", channel);
        channel.spin_type = "ipa";
        {
            ScopedTimer timer(global::profiler, "solve_ipa",
                              "Build independent-particle TDA states", dataset->comm_h.comm);
            auto solution = solve_ipa_tda(
                qp, options, nstates, dataset->comm_h.comm);
            channel.energies = std::move(solution.energies);
            channel.amplitudes_x = std::move(solution.amplitudes);
        }

        {
            ScopedTimer timer(global::profiler, "write_tda_results",
                              "Write TDA eigenstates", dataset->comm_h.comm);
            if (rank == 0)
            {
                write_energies(energy_file(options, "ipa", false),
                               channel.energies);
            }
        }
        if (options.out_bse_eigenvectors)
        {
            ScopedTimer timer(global::profiler,
                              "write_tda_amplitudes",
                              "Write distributed TDA amplitudes", dataset->comm_h.comm);
            write_distributed_amplitudes(
                amplitude_file(options, "ipa", "", rank),
                channel.amplitudes_x);
        }

        if (rank == 0)
            std::cout << "Preparing fine-grid velocity_mo for velocity gauge...\n";
        FineVelocityMo velocity;
        auto velocity_memory = watch_memory("velocity", velocity);
        {
            ScopedTimer timer(global::profiler, "prepare_velocity_mo",
                              "Prepare fine-grid velocity_mo", dataset->comm_h.comm);
            velocity = prepare_fine_velocity_mo(
                options, qp, dataset, band_gauge_phases);
        }

        {
            ScopedTimer timer(global::profiler, "tda_spectrum",
                              "Calculate TDA velocity-gauge spectrum", dataset->comm_h.comm);
            write_velocity_gauge_outputs(
                options, *dataset, velocity, channel.energies,
                channel.amplitudes_x, nullptr, "ipa", "tda");
        }

        return;
    }

    //full and tda calculation
    // 1. Transform the cut Coulomb interaction from q to R space using LibRPA.
    if (rank == 0)
        std::cout << "Transforming cut Coulomb from q to R with LibRPA FT_Vq...\n";
    TensorMap<Complex> bare;
    auto bare_memory = watch_memory("bare", bare);
    {
        ScopedTimer timer(global::profiler, "transform_bare_coulomb",
                          "Transform bare Coulomb q to R", dataset->comm_h.comm);
        bare = LibRPA_API::build_bare_coulomb(*dataset);
    }

    MolecularLri molecular(*dataset, options, qp);
    Chi0Screening dielectric(options, *dataset);
    // 2. Every reader returns full W(R) on the coarse GW cell. Chi0 readers
    // first screen complete q-space matrices; only the legacy Wc reader adds V.
    if (rank == 0)
        std::cout << "Reading screened interaction (" << options.screened_format << ")...\n";
    TensorMap<Complex> screened;
    auto screened_memory = watch_memory("screened", screened);
    {
        ScopedTimer timer(global::profiler, "read_screened_interaction",
                          "Read and construct screened interaction", dataset->comm_h.comm);
        screened = [&]() {
        if (options.screened_format == "fhi_aims_w" || options.screened_format == "fhi_aims_chi0")
            return read_aims_screened_interaction(options, *dataset,
                molecular.local_i_atoms(), molecular.local_j_atoms(), &dielectric);
        if (options.screened_format == "librpa_chi0")
            return read_librpa_chi0(options, *dataset, molecular.local_i_atoms(), molecular.local_j_atoms(), &dielectric);
        return read_screened_interaction(options, bare,
                                         dataset->pbc.Rlist.size(),
                                         molecular.local_i_atoms(),
                                         molecular.local_j_atoms());
        }();
    }

    //3. Convert the LibRPA Cs tensors to LibRI coefficients for the BSE contractions.
    if (rank == 0)
        std::cout << "Converting LibRPA Cs tensors for complex LibRI contractions...\n";
    TensorMap<Complex> coefficients;
    auto coefficients_memory = watch_memory("coefficients", coefficients);
    {
        ScopedTimer timer(global::profiler, "convert_ri_coefficients",
                          "Convert RI coefficients for LibRI", dataset->comm_h.comm);
        coefficients = convert_lri_coefficients(*dataset);
    }

    //4. Remap the interactions to the nearest BvK cells.

    {
        ScopedTimer timer(global::profiler, "remap_bvk_cells",
                          "Remap interactions to nearest BvK cells", dataset->comm_h.comm);
        remap_to_nearest_bvk_cell(coefficients, *dataset);
        remap_to_nearest_bvk_cell(bare, *dataset);
        remap_to_nearest_bvk_cell(screened, *dataset);
    }

    //5. Initialize the LibRI BSE contractions and construct the Hartree and screened contributions.
    if (rank == 0)
        std::cout << "Initializing external LibRI RI::LR...\n";
    {
        ScopedTimer timer(global::profiler, "initialize_libri",
                          "Initialize LibRI BSE contractions", dataset->comm_h.comm);
        molecular.initialize(coefficients, bare, screened);
    }

    librpa_int::ArrayDesc descriptor(dataset->blacs_h);
    const int block_size = dimension > 1000 ? 64 : (dimension > 500 ? 32 : 1);
    if (descriptor.init(dimension, dimension, block_size, block_size, 0, 0) != 0)
        throw std::runtime_error("failed to initialize BSE BLACS descriptor");
    const std::size_t local_size = static_cast<std::size_t>(descriptor.lld())
                                   * descriptor.n_loc();

    std::vector<Complex> hartree_a;
    auto hartree_a_memory = libbse::watch_memory("bse_calculation.hartree_a", hartree_a);
    std::vector<Complex> screened_a;
    auto screened_a_memory = libbse::watch_memory("bse_calculation.screened_a", screened_a);
    std::vector<Complex> hartree_b;
    auto hartree_b_memory = libbse::watch_memory("bse_calculation.hartree_b", hartree_b);
    std::vector<Complex> screened_b;
    auto screened_b_memory = libbse::watch_memory("bse_calculation.screened_b", screened_b);
    if (options.requires_hartree())
    {
        hartree_a.assign(local_size, Complex{});
        {
            ScopedTimer timer(global::profiler, "hartree_a", "LibRI Hartree A", dataset->comm_h.comm);
            molecular.add_hartree_a(hartree_a, descriptor, 1.0);
        }

    }
    if (options.requires_screened())
    {
        screened_a.assign(local_size, Complex{});
        {
            ScopedTimer timer(global::profiler, "screened_a", "LibRI screened A", dataset->comm_h.comm);
            molecular.add_screened_a(screened_a, descriptor, 1.0);
        }

    }
    // Construct the B contributions only if the full BSE is requested, since they are not needed for TDA.
    if (options.solve_full() && options.requires_hartree())
    {
        hartree_b.assign(local_size, Complex{});
        {
            ScopedTimer timer(global::profiler, "hartree_b", "LibRI Hartree B", dataset->comm_h.comm);
            molecular.add_hartree_b(hartree_b, descriptor, 1.0);
        }

    }
    if (options.solve_full() && options.requires_screened())
    {
        screened_b.assign(local_size, Complex{});
        {
            ScopedTimer timer(global::profiler, "screened_b", "LibRI screened B", dataset->comm_h.comm);
            molecular.add_screened_b(screened_b, descriptor, 1.0);
        }

    }
    molecular.release_interactions();
    //6. Assemble the channel matrices, check their symmetry, and diagonalize them with ELPA.
    std::vector<ChannelSolution> tda_solutions;
    auto tda_solutions_memory = watch_memory("tda_solutions", tda_solutions);
    std::vector<ChannelSolution> full_solutions;
    auto full_solutions_memory = watch_memory("full_solutions", full_solutions);
    if (options.solve_tda())
    {
        for (const std::string &spin_type : options.bse_spin_types)
        {
            const auto channel_coefficients =
                interaction_coefficients(spin_type);
            std::vector<Complex> matrix;
            auto matrix_memory = watch_memory("matrix", matrix);
            {
                ScopedTimer timer(global::profiler, "assemble_tda_matrix",
                                  "Assemble channel TDA matrix", dataset->comm_h.comm);
                assemble_channel_matrix(
                    matrix, descriptor, qp, options, channel_coefficients,
                    hartree_a, screened_a);
            }
            if (options.bse_memory_optimized && !options.solve_full()
                && options.bse_spin_types.size()==1 && options.bse_plasma_energy_ev<=0.)
            {
                std::vector<Complex>().swap(hartree_a);
                std::vector<Complex>().swap(screened_a);
            }
            {
                ScopedTimer timer(global::profiler, "check_tda_a_matrix",
                                  "Check TDA A-matrix Hermiticity", dataset->comm_h.comm);
                check_hermitian(
                    matrix, descriptor,
                    PARAM.constants.matrix_symmetry_threshold);
            }

            if (rank == 0)
                std::cout << "Diagonalizing " << spin_type
                          << " TDA matrix with ELPA...\n";
            EigenSolution solution;
            auto solution_memory = watch_memory("solution", solution);
            {
                ScopedTimer timer(global::profiler, "solve_tda_elpa",
                                  "Diagonalize TDA matrix with ELPA", dataset->comm_h.comm);
                solution = solve_tda_elpa(
                    matrix, descriptor, options.bse_nstates);
            }

            if (options.bse_plasma_energy_ev > 0 && channel_coefficients.screened != 0.)
            {
                ScopedTimer effective_timer(global::profiler, "effective_bse", "One-shot effective BSE", dataset->comm_h.comm);
                // Match the reference aims one-shot prescription: Eb=Eg-E_static.
                // Eg is the smallest DIRECT transition on the actual BSE QP grid,
                // not Si's indirect gap. The kernel targets the lowest static
                // exciton in this spin channel; higher levels share that kernel.
                const auto static_energies = solution.energies_ry;
                const double binding_ev = (qp.direct_gap_ry - static_energies.front())
                                          * PARAM.constants.ry_to_ev;
                if (binding_ev < 0.)
                    throw std::runtime_error("lowest static exciton is unbound: effective BSE requires Eb >= 0");
                auto effective_w = dielectric.effective(binding_ev,
                    molecular.local_i_atoms(), molecular.local_j_atoms());
                remap_to_nearest_bvk_cell(effective_w, *dataset);
                molecular.replace_screened(effective_w);
                std::vector<Complex> effective_a(local_size, Complex{});
                auto effective_memory = watch_memory("effective_A", effective_a);
                molecular.add_screened_a(effective_a, descriptor, 1.0);
                molecular.release_interactions();
                assemble_channel_matrix(matrix, descriptor, qp, options,
                    channel_coefficients, hartree_a, effective_a);
                check_hermitian(matrix, descriptor, PARAM.constants.matrix_symmetry_threshold);
                solution = solve_tda_elpa(matrix, descriptor, options.bse_nstates);
                if(rank == 0)
                {
                    const auto static_file = fs::path(options.output_dir) / ("static_excitation_" + spin_type + ".dat");
                    write_energies(static_file, static_energies);
                    std::ofstream comparison(fs::path(options.output_dir) / ("dynamical_" + spin_type + ".dat"));
                    comparison << std::setprecision(16)
                        << "# One-shot spectral effective BSE; all energies in eV\n"
                        << "# plasma " << options.bse_plasma_energy_ev << "\n"
                        << "# direct_QP_gap " << qp.direct_gap_ry * PARAM.constants.ry_to_ev << "\n"
                        << "# binding_used " << binding_ev << "\n"
                        << "# state static effective shift\n";
                    for(std::size_t i=0;i<static_energies.size();++i)
                        comparison << i+1 << ' ' << static_energies[i]*PARAM.constants.ry_to_ev << ' '
                            << solution.energies_ry[i]*PARAM.constants.ry_to_ev << ' '
                            << (solution.energies_ry[i]-static_energies[i])*PARAM.constants.ry_to_ev << '\n';
                    std::cout << "Effective BSE: plasma = " << options.bse_plasma_energy_ev
                              << " eV, binding from static lowest " << spin_type << " = " << binding_ev << " eV\n";
                }
            }
            if (options.bse_memory_optimized)
                std::vector<Complex>().swap(matrix);
            if (options.bse_memory_optimized && !options.out_bse_eigenvectors)
            {
                {
                    ScopedTimer timer(global::profiler, "write_tda_results",
                                      "Write TDA eigenstates", dataset->comm_h.comm);
                    if (rank==0)
                    {
                        write_energies(energy_file(options, spin_type, false), solution.energies_ry);
                        std::cout << std::setprecision(10) << spin_type
                                  << " TDA lowest excitation (eV): "
                                  << solution.energies_ry.front()*PARAM.constants.ry_to_ev << '\n'
                                  << spin_type << " TDA binding energy (eV): "
                                  << (qp.direct_gap_ry-solution.energies_ry.front())*PARAM.constants.ry_to_ev << '\n';
                    }
                }
                FineVelocityMo velocity;
                auto velocity_memory = watch_memory("velocity", velocity);
                {
                    ScopedTimer timer(global::profiler, "prepare_velocity_mo", "Prepare fine-grid velocity_mo", dataset->comm_h.comm);
                    velocity = prepare_fine_velocity_mo(options, qp, dataset, band_gauge_phases);
                }
                {
                    ScopedTimer timer(global::profiler, "tda_spectrum_block_cyclic",
                                      "TDA optics on block-cyclic eigenvectors", dataset->comm_h.comm);
                    write_tda_block_cyclic_outputs(options, *dataset, velocity,
                        solution.energies_ry, solution.vectors_local, descriptor, spin_type);
                }

                continue;
            }
            //7. Redistribute the eigenvectors to the BSE grid and write the results to disk.
            ChannelSolution channel;
            auto channel_memory = watch_memory("channel", channel);
            channel.spin_type = spin_type;
            channel.energies = solution.energies_ry;
            {
                ScopedTimer timer(global::profiler,
                                  "redistribute_tda_eigenvectors",
                                  "Redistribute TDA eigenvectors", dataset->comm_h.comm);
                channel.amplitudes_x = redistribute_amplitudes(
                    dataset->comm_h.comm, solution.vectors_local, descriptor,
                    0, 0, dimension, nstates);
            }
            {
                ScopedTimer timer(global::profiler, "write_tda_results",
                                  "Write TDA eigenstates", dataset->comm_h.comm);
                if (rank == 0)
                {
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
            }
            if (options.out_bse_eigenvectors)
            {
                ScopedTimer timer(global::profiler,
                                  "write_tda_amplitudes",
                                  "Write distributed TDA amplitudes", dataset->comm_h.comm);
                write_distributed_amplitudes(
                    amplitude_file(options, spin_type, "", rank),
                    channel.amplitudes_x);
            }

            tda_solutions.push_back(std::move(channel));
        }
    }
    // full BSE calculation
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
            auto matrix_a_memory = watch_memory("matrix_a", matrix_a);
            std::vector<Complex> matrix_b(local_size, Complex{});
            auto matrix_b_memory = libbse::watch_memory("bse_calculation.matrix_b", matrix_b);
            {
                ScopedTimer timer(global::profiler, "assemble_full_a_matrix",
                                  "Assemble channel full-BSE A matrix", dataset->comm_h.comm);
                assemble_channel_matrix(
                    matrix_a, descriptor, qp, options, channel_coefficients,
                    hartree_a, screened_a);
            }
            {
                ScopedTimer timer(global::profiler, "check_full_a_matrix",
                                  "Check full-BSE A-matrix Hermiticity", dataset->comm_h.comm);
                check_hermitian(
                    matrix_a, descriptor,
                    PARAM.constants.matrix_symmetry_threshold);
            }

            {
                ScopedTimer timer(global::profiler, "assemble_full_b_matrix",
                                  "Assemble channel full-BSE B matrix", dataset->comm_h.comm);
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
            {
                ScopedTimer timer(global::profiler, "check_full_b_matrix",
                                  "Check full-BSE B-matrix symmetry", dataset->comm_h.comm);
                check_symmetric(
                    matrix_b, descriptor,
                    PARAM.constants.matrix_symmetry_threshold);
            }

            if (rank == 0)
                std::cout << "Diagonalizing " << spin_type
                          << " full BSE Hamiltonian with ELPA...\n";
            EigenSolution solution;
            auto solution_memory = watch_memory("solution", solution);
            {
                ScopedTimer timer(global::profiler, "solve_full_elpa",
                                  "Solve full BSE with ELPA", dataset->comm_h.comm);
                // The full BSE Hamiltonian is NOT Hermitian AND not positive definite, so we use the generalized eigenvalue solver.
                solution = solve_full_elpa(
                    matrix_a, matrix_b, descriptor, full_descriptor,
                    options.bse_nstates);
            }

            ChannelSolution channel;
            auto channel_memory = watch_memory("channel", channel);
            channel.spin_type = spin_type;
            channel.energies = solution.energies_ry;
            {
                ScopedTimer timer(global::profiler,
                                  "redistribute_full_eigenvectors",
                                  "Redistribute full-BSE eigenvectors", dataset->comm_h.comm);
                channel.amplitudes_x = redistribute_amplitudes(
                    dataset->comm_h.comm, solution.vectors_local,
                    full_descriptor, 0, dimension, dimension, nstates);
                channel.amplitudes_y = redistribute_amplitudes(
                    dataset->comm_h.comm, solution.vectors_local,
                    full_descriptor, dimension, dimension,
                    dimension, nstates);
            }
            {
                ScopedTimer timer(global::profiler, "write_full_results",
                                  "Write full-BSE eigenstates", dataset->comm_h.comm);
                if (rank == 0)
                {
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
            }
            if (options.out_bse_eigenvectors)
            {
                ScopedTimer timer(global::profiler,
                                  "write_full_amplitudes",
                                  "Write distributed full-BSE amplitudes", dataset->comm_h.comm);
                write_distributed_amplitudes(
                    amplitude_file(options, spin_type, "full_X_", rank),
                    channel.amplitudes_x);
                write_distributed_amplitudes(
                    amplitude_file(options, spin_type, "full_Y_", rank),
                    channel.amplitudes_y);
            }

            full_solutions.push_back(std::move(channel));
        }
    }

    if (tda_solutions.empty() && full_solutions.empty()) return;
    if (rank == 0)
        std::cout << "Preparing fine-grid velocity_mo for velocity gauge...\n";
    FineVelocityMo velocity;
    auto velocity_memory = watch_memory("velocity", velocity);
    {
        ScopedTimer timer(global::profiler, "prepare_velocity_mo",
                          "Prepare fine-grid velocity_mo", dataset->comm_h.comm);
        velocity = prepare_fine_velocity_mo(
            options, qp, dataset, band_gauge_phases);
    }

    for (const ChannelSolution &channel : tda_solutions)
    {
        {
            ScopedTimer timer(global::profiler, "tda_spectrum",
                              "Calculate TDA velocity-gauge spectrum", dataset->comm_h.comm);
            write_velocity_gauge_outputs(
                options, *dataset, velocity, channel.energies,
                channel.amplitudes_x, nullptr, channel.spin_type, "tda");
        }

    }
    for (const ChannelSolution &channel : full_solutions)
    {
        {
            ScopedTimer timer(global::profiler, "full_spectrum",
                              "Calculate full-BSE velocity-gauge spectrum", dataset->comm_h.comm);
            write_velocity_gauge_outputs(
                options, *dataset, velocity, channel.energies,
                channel.amplitudes_x, &channel.amplitudes_y,
                channel.spin_type, "full");
        }

    }
}

} // namespace libbse
