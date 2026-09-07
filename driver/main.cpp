#include "bse/bse_calculation.h"
#include "interface/librpa_api.h"
#include "parameter/parameter.h"
#include "utils/profiler.h"
#include "utils/progress.h"

#include <mpi.h>
#include <omp.h>

#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace
{

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
        std::cout << "LibBSE parallel configuration\n"
                  << "  MPI processes: " << mpi_size << '\n'
                  << "  MPI thread level requested: MPI_THREAD_FUNNELED\n"
                  << "  MPI thread level provided: "
                  << mpi_thread_level_name(mpi_thread_level) << '\n'
                  << "  OpenMP max threads per MPI process: "
                  << omp_get_max_threads() << '\n'
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
    libbse::global::profiler.start("libbse_total", "Total LibBSE execution");

    int status = 0;
    bool librpa_initialized = false;
    std::shared_ptr<librpa_int::Dataset> dataset;
    try
    {
        if (provided < MPI_THREAD_FUNNELED)
            throw std::runtime_error("MPI does not provide MPI_THREAD_FUNNELED");
        if (argc != 1)
            throw std::invalid_argument(
                "LibBSE takes no command-line parameters; configure ./libbse.in");

        {
            libbse::ScopedTimer timer(libbse::global::profiler,
                                      "read_parameters", "Read libbse.in");
            libbse::PARAM.read();
        }
        if (rank == 0) libbse::PARAM.print(std::cout);

        {
            libbse::ScopedTimer timer(libbse::global::profiler,
                                      "initialize_librpa", "Initialize LibRPA");
            LibRPA_API::initialize();
        }
        librpa_initialized = true;
        if (rank == 0) print_parallel_configuration(mpi_size, provided);

        LibRPA_API::ReaderOptions reader_options;
        reader_options.input_dir = libbse::PARAM.inp.input_dir;
        reader_options.read_ri = !libbse::PARAM.inp.spectrum_only()
                                 && !libbse::PARAM.inp.ipa_only();
        reader_options.read_band_data
            = libbse::PARAM.inp.bse_use_fine_kgrid == 1;
        {
            libbse::ScopedTimer timer(libbse::global::profiler,
                                      "read_dataset", "Read calculation files with LibRPA");
            dataset = LibRPA_API::read_dataset(MPI_COMM_WORLD, reader_options);
        }
        libbse::done("read calculation files with LibRPA", MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::cout << "LibBSE data summary\n"
                      << "  atoms: " << dataset->atoms.size() << '\n'
                      << "  AO basis: " << dataset->basis_wfc.nb_total << '\n'
                      << "  auxiliary basis: " << dataset->basis_aux.nb_total << '\n'
                      << "  coarse k-points: " << dataset->mf.get_n_kpoints() << '\n'
                      << "  fine k-points: " << dataset->mf_band.get_n_kpoints() << '\n'
                      << "  states: " << dataset->mf_band.get_n_states() << '\n'
                      << "  Cs keys: " << dataset->cs_data.n_keys() << '\n'
                      << "  cut-Coulomb atom pairs: " << dataset->vq_cut.size() << '\n';
        }
        libbse::run_bse(libbse::PARAM.inp, dataset);
        libbse::done("LibBSE calculation", MPI_COMM_WORLD);
    }
    catch (const std::exception &error)
    {
        if (rank == 0) std::cerr << "LibBSE error: " << error.what() << '\n';
        status = 1;
    }

    dataset.reset();
    if (librpa_initialized)
    {
        libbse::ScopedTimer timer(libbse::global::profiler,
                                  "finalize_librpa", "Finalize LibRPA");
        LibRPA_API::finalize();
    }
    libbse::global::profiler.stop("libbse_total");
    if (rank == 0) libbse::global::profiler.display(std::cout);
    MPI_Finalize();
    return status;
}
