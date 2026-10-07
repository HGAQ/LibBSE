#include "bse/bse_calculation.h"
#include "interface/librpa_api.h"
#include "io/fhi_aims_adapter.h"
#include "parameter/parameter.h"
#include "utils/profiler.h"
#include "utils/memory_views.h"

#include <mpi.h>
#include <omp.h>

#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace
{

    void print_banner()
    {
        std::cout << R"(
██╗     ██╗██╗     ██████╗ ███████╗███████╗
██║     ╚═╝██║     ██╔══██╗██╔════╝██╔════╝
██║     ██║██████╗ ██████╔╝███████╗█████╗
██║     ██║██╔══██╗██╔══██╗╚════██║██╔══╝
███████╗██║██████╔╝██████╔╝███████║███████╗
╚══════╝╚═╝╚═════╝ ╚═════╝ ╚══════╝╚══════╝)"
    <<  "\n"
    << "==================================================\n"
    <<  "Start Running LibBSE ...\n"
    <<std::flush;
    }

    const char *mpi_thread_level_name(int level)
    {
        switch (level)
        {
            case MPI_THREAD_SINGLE: return "MPI_THREAD_SINGLE";
            case MPI_THREAD_FUNNELED: return "MPI_THREAD_FUNNELED";
            case MPI_THREAD_SERIALIZED: return "MPI_THREAD_SERIALIZED";
            case MPI_THREAD_MULTIPLE: return "MPI_THREAD_MULTIPLE";
            default: return "unknown";
        }
    }

    void print_parallel_configuration(int mpi_size, int mpi_thread_level)
    {
        int team_size = 1;
#pragma omp parallel
        {
#pragma omp single
            team_size = omp_get_num_threads();
        }
        std::cout << "LibBSE parallel configuration\n"
                  << "  MPI processes: " << mpi_size << '\n'
                  << "  MPI thread level requested: MPI_THREAD_FUNNELED\n"
                  << "  MPI thread level provided: "
                  << mpi_thread_level_name(mpi_thread_level) << '\n'
                  << "  OpenMP max threads per MPI process: "
                  << omp_get_max_threads() << '\n'
                  << "  OpenMP observed team size: " << team_size << '\n'
                  << "  OpenMP available processors: " << omp_get_num_procs() << '\n'
                  << "  OpenMP dynamic adjustment: "
                  << (omp_get_dynamic() ? "enabled" : "disabled") << '\n';
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
    if (rank == 0) print_banner();
    int status = 0;
    {
        libbse::ScopedTimer total(libbse::global::profiler, "libbse_total",
                                  "Total LibBSE execution", MPI_COMM_WORLD);
        bool librpa_initialized = false;
        std::shared_ptr<librpa_int::Dataset> dataset;
        auto dataset_memory = libbse::watch_memory("dataset", dataset);
        try
        {
            // check if MPI provides the required thread level (MPI_THREAD_FUNNELED)
            if (provided < MPI_THREAD_FUNNELED)
                throw std::runtime_error("MPI does not provide MPI_THREAD_FUNNELED");
            if (argc != 1)
                throw std::invalid_argument(
                    "LibBSE takes no command-line parameters; configure ./libbse.in");
            // read param file and initialize LibRPA
            {
                libbse::ScopedTimer timer(libbse::global::profiler,
                                          "read_parameters", "Read libbse.in", MPI_COMM_WORLD);
                libbse::PARAM.read();
                if (rank == 0) libbse::PARAM.print(std::cout);
                libbse::resolve_input_format(libbse::PARAM.inp);
            }
            {
                libbse::ScopedTimer timer(libbse::global::profiler,
                                          "initialize_librpa", "Initialize LibRPA", MPI_COMM_WORLD);
                LibRPA_API::initialize();
                librpa_initialized = true;
                if (rank == 0) print_parallel_configuration(mpi_size, provided);
            }

            // translate libbse.in options into LibRPA reader options
            LibRPA_API::ReaderOptions reader_options;
            reader_options.input_dir = libbse::PARAM.inp.input_dir;
            reader_options.output_dir = libbse::PARAM.inp.output_dir;
            reader_options.input_format = libbse::PARAM.inp.input_format;
            reader_options.read_ri = !libbse::PARAM.inp.spectrum_only()
                                     && !libbse::PARAM.inp.ipa_only();
            reader_options.read_band_data
                = libbse::PARAM.inp.bse_use_fine_kgrid == 1;
            // read the dataset from the input files
            {
                libbse::ScopedTimer timer(libbse::global::profiler,
                                          "read_dataset", "Read calculation files with LibRPA", MPI_COMM_WORLD);
                dataset = LibRPA_API::read_dataset(MPI_COMM_WORLD, reader_options);
            }
            if (rank == 0)
            {
                std::cout << "|| LibBSE data summary:\n"
                          << "||   atoms: " << dataset->atoms.size() << '\n'
                          << "||   AO basis: " << dataset->basis_wfc.nb_total << '\n'
                          << "||   auxiliary basis: " << dataset->basis_aux.nb_total << '\n'
                          << "||   coarse k-points: " << dataset->mf.get_n_kpoints() << '\n'
                          << "||   fine k-points: " << dataset->mf_band.get_n_kpoints() << '\n'
                          << "||   states: " << dataset->mf_band.get_n_states() << '\n'
                          << "||   Cs keys: " << dataset->cs_data.n_keys() << '\n'
                          << "||   cut-Coulomb atom pairs: " << dataset->vq_cut.size() << '\n'
                          << "|| \n"
                          << "|| LibBSE input parphase ends.\n\n"
                          << std::flush;
            }
            // run the BSE calculation
            libbse::run_bse(libbse::PARAM.inp, dataset);
        }
        catch (const std::exception &error)
        {
            std::cerr << "LibBSE error on task " << rank << ": " << error.what() << '\n';
            if (mpi_size > 1) MPI_Abort(MPI_COMM_WORLD, 1);
            status = 1;
        }

        {
            libbse::ScopedTimer release(libbse::global::profiler, "release_dataset",
                                        "Release calculation dataset", MPI_COMM_WORLD);
            dataset.reset();
        }
        if (librpa_initialized)
        {
            libbse::ScopedTimer timer(libbse::global::profiler,
                                      "finalize_librpa", "Finalize LibRPA", MPI_COMM_WORLD);
            LibRPA_API::finalize();
        }
    } // All tracked calculation owners and the total ScopedTimer have ended.
    if (rank == 0) libbse::global::profiler.display(std::cout);
    libbse::MemoryTracker::instance().report_final(std::cout, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
