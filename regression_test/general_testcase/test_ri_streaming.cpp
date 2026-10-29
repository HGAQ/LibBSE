// Integration check: run in a small, complete FHI-aims LibBSE input directory.
// Compare all distributed matrix entries, before eigensolver/phase conventions.
#include "bse/molecular_lri.h"
#include "io/aims_screened.h"
#include "io/chi0_screening.h"
#include "io/bse_files.h"
#include "io/fhi_aims_adapter.h"
#include "interface/librpa_api.h"
#include <mpi.h>
#include <iostream>
#include <algorithm>
int main(int argc,char **argv) {
    int provided;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    LibRPA_API::initialize();
    try {
        auto &options=libbse::PARAM.inp;
        libbse::PARAM.read();libbse::resolve_input_format(options);
        LibRPA_API::ReaderOptions ro;
        ro.input_dir=options.input_dir;ro.output_dir=options.output_dir;
        ro.input_format=options.input_format;ro.read_band_data=options.bse_use_fine_kgrid==1;
        auto ds=LibRPA_API::read_dataset(MPI_COMM_WORLD,ro);
        const auto qp=libbse::read_qp_bands(options,*ds);
        libbse::apply_wavefunction_gauge(*ds,options,qp);
        auto bare=LibRPA_API::build_bare_coulomb(*ds);
        libbse::MolecularLri molecular(*ds,options,qp);
        libbse::Chi0Screening dielectric(options,*ds);
        auto screened=libbse::read_aims_screened_interaction(options,*ds,
            molecular.local_i_atoms(),molecular.local_j_atoms(),&dielectric);
        auto coefficients=libbse::convert_lri_coefficients(*ds);
        libbse::remap_to_nearest_bvk_cell(coefficients,*ds);
        libbse::remap_to_nearest_bvk_cell(bare,*ds);
        libbse::remap_to_nearest_bvk_cell(screened,*ds);
        molecular.initialize(coefficients,bare,screened);
        const int d=qp.nk*options.nocc*options.nvirt;
        librpa_int::ArrayDesc desc(ds->blacs_h);
        if(desc.init(d,d,3,3,0,0)) throw std::runtime_error("descriptor failed");
        const auto local=static_cast<std::size_t>(desc.lld())*desc.n_loc();
        const auto add=[&](int which,std::vector<libbse::Complex> &mat){
            if(which==0)molecular.add_hartree_a(mat,desc,1.);
            if(which==1)molecular.add_screened_a(mat,desc,1.);
            if(which==2)molecular.add_hartree_b(mat,desc,1.);
            if(which==3)molecular.add_screened_b(mat,desc,1.);
        };
        int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        for(int which=0;which<4;++which) {
            std::vector<libbse::Complex> reference(local),candidate(local);
            options.bse_memory_optimized=false;add(which,reference);
            for(int batch:{1,3,64}) {
                std::fill(candidate.begin(),candidate.end(),libbse::Complex{});
                options.bse_memory_optimized=true;options.bse_ri_batch_blocks=batch;
                add(which,candidate);
                double error=0.,all_error=0.;
                for(std::size_t i=0;i<local;++i)error=std::max(error,std::abs(candidate[i]-reference[i]));
                MPI_Allreduce(&error,&all_error,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
                if(rank==0)std::cout<<"RI_MATRIX_COMPARE kind="<<which<<" batch="<<batch<<" max_abs="<<all_error<<'\n';
                if(all_error>5e-12)throw std::runtime_error("streamed RI matrix differs from unbatched result");
            }
        }
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);}
    LibRPA_API::finalize();MPI_Finalize();return 0;
}
