#include "io/chi0_screening.h"
#include "io/aims_screened.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <unistd.h>
int main(int argc,char **argv){
    MPI_Init(&argc,&argv);
    try {
        LibRPA_API::initialize();
        using namespace libbse; using namespace librpa_int;
        Matz v(2,2,MAJOR::ROW),chi(2,2,MAJOR::ROW);
        v(0,0)=2.;v(1,1)=3.;v(0,1)=Complex(0,0.7);v(1,0)=std::conj(v(0,1));
        chi(0,0)=-0.2;chi(1,1)=-0.5;chi(0,1)=0.1;chi(1,0)=0.1;
        const auto w=screen_chi0(chi,v);
        // Independent Dyson equation check, with noncommuting complex V/chi.
        // W - V chi W = V exercises matrix order, full W, and square roots.
        const auto residual=w-v*chi*w-v;
        for(std::size_t i=0;i<residual.size();++i)
            if(std::abs(residual.ptr()[i])>1.e-12) throw std::runtime_error("Dyson identity failed");
        Matz zero(2,2,MAJOR::ROW); const auto bare=screen_chi0(zero,v);
        for(std::size_t i=0;i<v.size();++i) if(std::abs(bare.ptr()[i]-v.ptr()[i])>1.e-12)
            throw std::runtime_error("zero response did not recover full bare V");
        // Dense complex eigenchannels: entry-wise nonlinear functions fail
        // this unitary-covariance check even if diagonal scalar tests pass.
        Matz inv(2,2,MAJOR::ROW);inv(0,0)=0.4;inv(1,1)=0.6;inv(0,1)=Complex(0,0.2);inv(1,0)=std::conj(inv(0,1));
        const auto unchanged=effective_inverse_dielectric(inv,15.,0.);
        for(std::size_t k=0;k<inv.size();++k) if(unchanged.ptr()[k]!=inv.ptr()[k])
            throw std::runtime_error("effective model violates exact Eb=0 limit");
        Matz rotation(2,2,MAJOR::ROW);const double c=std::sqrt(0.5);
        rotation(0,0)=c;rotation(0,1)=c;rotation(1,0)=-c;rotation(1,1)=c;
        const auto effective=effective_inverse_dielectric(inv,15.,0.5);
        const auto rotated=effective_inverse_dielectric(rotation*inv*rotation.get_transpose(true),15.,0.5);
        const auto reference=rotation*effective*rotation.get_transpose(true);
        for(std::size_t k=0;k<inv.size();++k) if(std::abs(rotated.ptr()[k]-reference.ptr()[k])>1.e-12)
            throw std::runtime_error("effective model is not unitarily covariant");
        Matz diagonal(2,2,MAJOR::ROW);diagonal(0,0)=.2;diagonal(1,1)=1.;
        const auto expected=effective_inverse_dielectric(diagonal,15.,.5);
        if(std::abs(expected(0,0)-(1.-15.*.8/(15.+.5*std::sqrt(.8))))>1.e-12 || std::abs(expected(1,1)-1.)>1.e-12)
            throw std::runtime_error("effective model scalar limit failed");
        bool rejected=false;
        try { effective_inverse_dielectric(diagonal,15.,-0.1); }
        catch(const std::runtime_error &) { rejected=true; }
        if(!rejected) throw std::runtime_error("negative exciton binding was accepted");
        diagonal(0,0)=1.1; rejected=false;
        try { effective_inverse_dielectric(diagonal,15.,0.5); }
        catch(const std::runtime_error &) { rejected=true; }
        if(!rejected) throw std::runtime_error("nonpassive dielectric was silently clamped");
        TFGrids imported; imported.generate_single_frequency(0.);
        if(imported.size()!=1 || imported.get_freq_nodes()[0]!=0. || imported.has_time_grids())
            throw std::runtime_error("zero-frequency head/wing grid is invalid");
        TFGrids source(16),stat;
        source.generate_minimax(0.1,10.0); stat.generate_static_export(source);
        if(stat.get_freq_nodes()[0]!=0 || source.get_freq_nodes()[0]<=0)
            throw std::runtime_error("static export changed original frequency grid");
        // For chi(tau)=exp(-E|tau|), chi(i0)=2/E analytically.
        for(double e:{0.1,0.2,1.,3.,10.}) {
            double integral=0;
            for(std::size_t it=0;it<stat.size();++it)
                integral+=stat.get_costrans_t2f()(0,it)*std::exp(-e*stat.get_time_nodes()[it]);
            std::cout<<"static quadrature E="<<e<<" relative="<<integral*e/2<<std::endl;
            if(std::abs(integral*e/2-1)>1.e-4) throw std::runtime_error("static time quadrature normalization failed");
        }
        // One-cell real-space response plus MPI-distributed bare V exercises
        // the actual LibRPA reader, metadata, screening and Fourier interface.
        int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank); long id=rank==0?getpid():0;
        MPI_Bcast(&id,1,MPI_LONG,0,MPI_COMM_WORLD);
        auto dir=std::filesystem::temp_directory_path()/("chi0_reader_"+std::to_string(id));
        if(rank==0){std::filesystem::create_directories(dir);
            std::ofstream info(dir/"chi0_rf.info");info<<"LIBRPA_CHI0_RF 1\n0 2 1 1 1\n";
            std::ofstream out(dir/"Chi0_Mu_0_Nu_0_iR_0_ifreq_0.mtx");
            out<<"%%MatrixMarket matrix coordinate complex general\n%\n% R = ( 0 0 0 )\n2 2 4\n1 1 -0.2 0\n1 2 0.1 0\n2 1 0.1 0\n2 2 -0.5 0\n";
        }
        if(rank==0){std::ofstream out(dir/"periodic_gw_chi0_q_1_rank_0.dat");
            out<<"# quantity: chi0\n# n_basbas n_freq q_index: 2 1 1\n# q_fractional: 0 0 0\n# Original RI auxiliary basis; atomic units; indices start at 1.\n"
               <<"1 0 1 1 -0.2 0\n1 0 1 2 0.1 0\n1 0 2 1 0.1 0\n1 0 2 2 -0.5 0\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        {Dataset ds(MPI_COMM_WORLD);ds.basis_aux=AtomicBasis(std::vector<std::size_t>{2});
            ds.pbc.set_latvec({1,0,0,0,1,0,0,0,1});ds.pbc.set_period(1,1,1);
            if(rank==0){auto m=std::make_shared<ComplexMatrix>(2,2);
                for(int i=0;i<2;++i)for(int j=0;j<2;++j)(*m)(i,j)=v(i,j);
                ds.vq_cut[0][0][{0,0,0}]=m;}
            InputParameters opts;opts.screened_dir=dir.string();opts.chi0_headwing=false;opts.chi0_coulomb_metric="single_cut";
            auto result=read_librpa_chi0(opts,ds,{0},{0});
            opts.input_format="fhi_aims";opts.screened_format="fhi_aims_chi0";
            const auto aims=read_aims_screened_interaction(opts,ds,{0},{0});
            const auto &am=aims.at(0).at({0,{0,0,0}});
            for(int i=0;i<2;++i)for(int j=0;j<2;++j)if(std::abs(am(i,j)-w(i,j))>1.e-12)
                throw std::runtime_error("aims chi0 reader differs from Dyson solution");
            const auto &m=result.at(0).at({0,{0,0,0}});
            for(int i=0;i<2;++i)for(int j=0;j<2;++j)if(std::abs(m(i,j)-w(i,j))>1.e-12)
                throw std::runtime_error("real-space chi0 reader differs from Dyson solution");
        }
        MPI_Barrier(MPI_COMM_WORLD);if(rank==0)std::filesystem::remove_all(dir);
        LibRPA_API::finalize();
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);}
    MPI_Finalize();
}
