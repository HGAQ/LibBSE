#include "bse/spectrum.h"
#include "bse/distributed_amplitudes.h"
#include "interface/librpa_api.h"
#include <mpi.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>
using libbse::Complex;
static Complex amplitude(int pair,int state,int d) {
    return std::polar(1./std::sqrt(double(d)), 2.*libbse::Constants::pi*pair*state/d);
}
static Complex velocity(int pair,int axis) {return {0.25*(pair+1)*(axis+1),0.1*(pair-axis)};}
int main(int argc,char **argv) {
    int provided; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    int status=0;
    LibRPA_API::initialize();
    try {
        librpa_int::Dataset ds(MPI_COMM_WORLD);
        librpa_int::BlacsCtxtHandler blacs(MPI_COMM_WORLD);blacs.init();blacs.set_square_grid();
        long long pid=rank==0?getpid():0;MPI_Bcast(&pid,1,MPI_LONG_LONG,0,MPI_COMM_WORLD);
        // D=2 gives empty owners at 3/4 MPI; D=7 covers uneven partitions.
        for(int d:{2,7}) for(int ns:{1,d}) for(int block:{1,2}) {
            librpa_int::ArrayDesc desc(blacs);
            if(desc.init(d,d,block,block,0,0)) throw std::runtime_error("descriptor");
            libbse::InputParameters opt;opt.nocc=1;opt.nvirt=1;
            opt.spectrum_energy_max_ev=4.;
            ds.kfrac_band_list.clear();
            for(int k=0;k<d;++k) ds.kfrac_band_list.emplace_back(double(k)/d,0.,0.);
            auto amps=libbse::make_distributed_amplitudes(MPI_COMM_WORLD,d,ns);
            libbse::FineVelocityMo v;v.nk=d;v.nbands=2;
            v.first_pair=amps.first_pair;v.local_pairs=amps.local_pairs;
            v.values.resize(3*v.local_pairs);v.gaps_ha.resize(v.local_pairs);
            for(int i=0;i<v.local_pairs;++i) {
                const int pair=i+v.first_pair;v.gaps_ha[i]=0.5+0.1*pair;
                for(int axis=0;axis<3;++axis)v.values[axis*v.local_pairs+i]=velocity(pair,axis);
                for(int st=0;st<ns;++st)amps(st,i)=amplitude(pair,st,d);
            }
            std::vector<Complex> vec(static_cast<std::size_t>(desc.lld())*desc.n_loc());
            for(int c=0;c<desc.n_loc();++c)for(int row=0;row<desc.m_loc();++row)
                vec[static_cast<std::size_t>(c)*desc.lld()+row]=amplitude(desc.indx_l2g_r(row),desc.indx_l2g_c(c),d);
            std::vector<double> energies(ns);for(int st=0;st<ns;++st)energies[st]=0.1+0.02*st;
            const auto base=std::filesystem::current_path()/ ("optics_test_"+std::to_string(pid)+"_"+std::to_string(d)+"_"+std::to_string(ns)+"_"+std::to_string(block));
            for(const std::string spin:{"singlet","triplet"}) {
                auto old_dir=base/(spin+"_pair"),new_dir=base/(spin+"_block");
                if(rank==0){std::filesystem::create_directories(old_dir);std::filesystem::create_directories(new_dir);}
                MPI_Barrier(MPI_COMM_WORLD);
                opt.output_dir=old_dir.string();
                libbse::write_velocity_gauge_outputs(opt,ds,v,energies,amps,nullptr,spin,"tda");
                opt.output_dir=new_dir.string();
                libbse::write_tda_block_cyclic_outputs(opt,ds,v,energies,vec,desc,spin);
                if(rank==0) {
                    for(const auto &dir:{old_dir,new_dir}) {
                        std::ifstream in(dir/("momentum_strength_"+spin+"_tda.dat"));
                        std::string line;int count=0;
                        while(std::getline(in,line)){
                            if(line.empty()||line[0]=='#')continue;
                            std::istringstream row(line);int st;double en;row>>st>>en;
                            for(int axis=0;axis<3;++axis) {
                                double actual;row>>actual;Complex sum{};
                                for(int pair=0;pair<d;++pair)sum+=velocity(pair,axis)*amplitude(pair,st-1,d);
                                double expected=spin=="triplet"?0.:std::norm(sum);
                                if(!row || std::abs(actual-expected)>1e-11)throw std::runtime_error("analytic complex momentum mismatch");
                            }++count;
                        }
                        if(count!=ns)throw std::runtime_error("missing states");
                    }
                    for(const std::string file:{"oscillator_strength_","spectrum_"}) {
                        std::ifstream a(old_dir/(file+spin+"_tda.dat")),b(new_dir/(file+spin+"_tda.dat"));
                        std::string x,y;
                        while(std::getline(a,x)) {
                            if(!std::getline(b,y))throw std::runtime_error("missing output rows");
                            if(x.empty()||x[0]=='#')continue;
                            std::istringstream xs(x),ys(y);double xv,yv;
                            while(xs>>xv)if(!(ys>>yv)||std::abs(xv-yv)>1e-9*std::max(1.,std::abs(xv)))throw std::runtime_error("optical output mismatch");
                        }
                    }
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);if(rank==0)std::filesystem::remove_all(base);
        }
    } catch(const std::exception &e) {std::cerr<<rank<<": "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);status=1;}
    LibRPA_API::finalize();MPI_Finalize();return status;
}
