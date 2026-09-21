// Compatibility exporter for checking the same W with a legacy BSE consumer.
// The production LibBSE reader consumes full aims W directly. ABACUS expects
// Wc(R) and adds V(R), so this tool writes Wc(R)=FT[W_aims](R)-FT[v_cut](R)
// before the nearest-cell remap. This changes only the container, not W.
#include "interface/librpa_api.h"
#include "io/aims_screened.h"
#include "io/fhi_aims_adapter.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int size; MPI_Comm_size(MPI_COMM_WORLD, &size);
    try {
        if (argc != 3 || size != 1)
            throw std::runtime_error("usage (one MPI rank): aims_w_to_librpa libbse.in output_directory");
        libbse::PARAM.read(argv[1]);
        auto options = libbse::PARAM.inp;
        libbse::resolve_input_format(options);
        if (options.screened_format != "fhi_aims_w")
            throw std::runtime_error("converter requires screened_format fhi_aims_w");
        LibRPA_API::initialize();
        LibRPA_API::ReaderOptions reader;
        reader.input_dir=options.input_dir; reader.output_dir=options.output_dir;
        reader.input_format=options.input_format;
        auto dataset=LibRPA_API::read_dataset(MPI_COMM_WORLD,reader);
        auto bare=LibRPA_API::build_bare_coulomb(*dataset);
        std::vector<int> atoms(dataset->atoms.size());
        std::iota(atoms.begin(),atoms.end(),0);
        auto screened=libbse::read_aims_screened_interaction(options,*dataset,atoms,atoms);
        std::filesystem::create_directories(argv[2]);
        for (int i:atoms) for (int j:atoms)
            for (std::size_t ir=0; ir<dataset->pbc.Rlist.size(); ++ir) {
                const auto r=dataset->pbc.Rlist[ir];
                const libbse::AtomCell key{j,{r.x,r.y,r.z}};
                const auto &w=screened.at(i).at(key);
                const auto &v=bare.at(i).at(key);
                const auto filename=std::filesystem::path(argv[2])/(
                    "Wc_Mu_"+std::to_string(i)+"_Nu_"+std::to_string(j)+"_iR_"+
                    std::to_string(ir)+"_ifreq_0.mtx");
                std::ofstream out(filename);
                if (!out) throw std::runtime_error("cannot write "+filename.string());
                out << "%%MatrixMarket matrix coordinate complex general\n%\n"
                    << "% Wc = W_aims - v_cut, Hartree; R = (" << r.x << ' ' << r.y << ' ' << r.z << ")\n"
                    << w.shape[0] << ' ' << w.shape[1] << ' ' << w.shape[0]*w.shape[1] << '\n'
                    << std::scientific << std::setprecision(17);
                for (std::size_t mu=0; mu<w.shape[0]; ++mu)
                    for (std::size_t nu=0; nu<w.shape[1]; ++nu) {
                        const auto x=w(mu,nu)-v(mu,nu);
                        out << mu+1 << ' ' << nu+1 << ' ' << x.real() << ' ' << x.imag() << '\n';
                    }
                if (!out) throw std::runtime_error("failed writing "+filename.string());
            }
        dataset.reset();
        LibRPA_API::finalize();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();
}
