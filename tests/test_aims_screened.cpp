#include "io/aims_screened.h"
#include "interface/librpa_api.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unistd.h>

namespace fs=std::filesystem;
// A three-point trigonometric polynomial is reconstructed exactly at arbitrary
// fine q. Split each q over two producer ranks and put the lowest frequency at
// index 2 to test assembly, node selection, conjugation, sign, and 1/Nq.
void fixture(const fs::path &root) {
    for (int iq=0; iq<2; ++iq) for (int rank=0; rank<2; ++rank) {
        std::ofstream out(root/("periodic_gw_w_q_"+std::to_string(iq+1)+"_rank_"+std::to_string(rank)+".dat"));
        const double q=iq/3.0, angle=2*libbse::PARAM.constants.pi*q;
        out << std::setprecision(17)
            << "# quantity: w\n# n_basbas n_freq q_index: 2 2 " << iq+1
            << "\n# q_fractional: " << q << " 0 0\n"
            << "# Original RI auxiliary basis; atomic units; indices start at 1.\n";
        for (int iw=1; iw<=2; ++iw) for (int col=0; col<2; ++col) {
            const int row=rank;
            libbse::Complex x;
            if (row==col) x=row==0 ? 2+std::cos(angle) : 4;
            else x=libbse::Complex(0,(row==0 ? 1:-1)*std::sin(angle));
            if (iw==1) x*=99;
            out << iw << ' ' << (iw==1 ? 0.9:0.001) << ' ' << row+1 << ' ' << col+1 << ' '
                << x.real() << ' ' << x.imag() << '\n';
        }
    }
}
int main(int argc,char **argv) {
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    long id=rank==0 ? getpid():0; MPI_Bcast(&id,1,MPI_LONG,0,MPI_COMM_WORLD);
    const auto root=fs::temp_directory_path()/("libbse_aims_w_"+std::to_string(id));
    try {
        LibRPA_API::initialize();
        if (rank==0) { fs::create_directories(root); fixture(root); }
        MPI_Barrier(MPI_COMM_WORLD);
        {
            librpa_int::Dataset dataset(MPI_COMM_WORLD);
            dataset.basis_aux=librpa_int::AtomicBasis(std::vector<std::size_t>{1,1});
            dataset.pbc.set_latvec({1,0,0,0,1,0,0,0,1});
            dataset.pbc.set_period(3,1,1);
            dataset.mf_band=librpa_int::MeanField(1,4,2,2);
            libbse::InputParameters options;
            options.input_format="fhi_aims"; options.screened_dir=root.string();
            std::vector<int> rows;
            for (int i=0;i<2;++i) if (i%size==rank) rows.push_back(i);
            const auto wr=libbse::read_aims_screened_interaction(options,dataset,rows,{0,1});
            for (int i:rows) for (int j=0;j<2;++j) for (int iq=0;iq<4;++iq) {
                const double q=iq/4.0, angle=2*libbse::PARAM.constants.pi*q;
                libbse::Complex x=0;
                for (const auto &r:dataset.pbc.Rlist)
                    x+=std::exp(libbse::Complex(0,angle*r.x))*wr.at(i).at({j,{r.x,r.y,r.z}})(0,0);
                const libbse::Complex expected=i==j ? libbse::Complex(i==0 ? 2+std::cos(angle):4)
                    : libbse::Complex(0,(i==0 ? 1:-1)*std::sin(angle));
                if (std::abs(x-expected)>1.e-12) throw std::runtime_error("3->4 Fourier interpolation or W normalization failed");
            }
            // Remove one producer block: missing values must not silently turn
            // into zeros in the BSE kernel. All ranks see the same invalid input.
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) fs::remove(root/"periodic_gw_w_q_1_rank_1.dat");
            MPI_Barrier(MPI_COMM_WORLD);
            bool rejected=false;
            try { (void)libbse::read_aims_screened_interaction(options,dataset,rows,{0,1}); }
            catch (const std::runtime_error &) { rejected=true; }
            if (!rejected) throw std::runtime_error("incomplete rank blocks were accepted");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank==0) fs::remove_all(root);
        LibRPA_API::finalize();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();
}
