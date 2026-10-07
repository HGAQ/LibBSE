#include "utils/memory_views.h"
#include "chi0_screening.h"
#include <../src/api/dataset_helper.h>
#include <../src/math/utils_matrix_m_mpi.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace libbse {
using librpa_int::Matz;
namespace {
Matz hermitian(const Matz &a) {
    double error=0, norm=0;
    for (std::size_t i=0;i<a.nr();++i) for(std::size_t j=0;j<a.nc();++j) {
        error+=std::norm(a(i,j)-std::conj(a(j,i))); norm+=std::norm(a(i,j));
    }
    if (!std::isfinite(norm) || std::sqrt(error)>1.e-7*std::max(1.,std::sqrt(norm)))
        throw std::runtime_error("chi0/V is not Hermitian in the supplied auxiliary basis");
    return (a+a.get_transpose(true))*0.5;
}
Matz spectral_power(Matz a, double power, bool clip) {
    auto a_memory = watch_memory("screening.spectral_input", a);
    // Reuse LibRPA's Hermitian matrix function, including its positive-Coulomb
    // projection. This is a matrix power, never an entry-wise power.
    return librpa_int::power_hemat(a, power, false, false, clip ? 0. : -1.e5);
}
}
Matz screen_chi0(const Matz &chi, Matz v) {
    if(chi.nr()!=chi.nc() || v.nr()!=chi.nr() || v.nc()!=chi.nc())
        throw std::runtime_error("chi0 and Coulomb dimensions differ");
    const auto sqrtv=spectral_power(hermitian(v),0.5,true);
    auto sqrtv_memory = watch_memory("screening.sqrtv", sqrtv);
    auto epsilon=(sqrtv*hermitian(chi)*sqrtv)*(-1.0);
    auto epsilon_memory = watch_memory("screening.epsilon", epsilon);
    for(std::size_t i=0;i<epsilon.nr();++i) epsilon(i,i)+=1.;
    // epsilon = I - V^(1/2) chi0 V^(1/2), W = V^(1/2) epsilon^-1 V^(1/2).
    // Full auxiliary matrices are inverted BEFORE spatial interpolation.
    // Screening individual atom blocks or individual R cells is incorrect:
    // matrix inversion couples all atoms and does not commute with the FT.
    // Hartree units and the original RI basis are preserved. No extra bare
    // Coulomb is added: the expression already gives full W, including V.
    return sqrtv*spectral_power(hermitian(epsilon),-1.,false)*sqrtv;
}
static Matz collect_coulomb(librpa_int::Dataset &ds,const librpa_int::Vector3_Order<double> &q, bool cut) {
    const int n=ds.basis_aux.nb_total;
    Matz v(n,n,librpa_int::MAJOR::ROW); std::vector<int> counts(n*n,0);
    auto v_memory = watch_memory("screening.v", v);
    auto counts_memory = watch_memory("screening.counts", counts);
    auto same=[](const auto &a,const auto &b){return std::abs(a.x-b.x)+std::abs(a.y-b.y)+std::abs(a.z-b.z)<1.e-8;};
    // Coulomb blocks may be MPI distributed or replicated. Sum/count avoids
    // multiplying a replicated block by the MPI process count.
    for(const auto &[i,js]:(cut ? ds.vq_cut : ds.vq)) for(const auto &[j,qs]:js) {
        for(const auto &[qr,block]:qs) {
            bool match=same(q,qr), conjugate=false;
            if(!match) {
                auto star=ds.pbc.map_irk_ks.find(qr);
                if(star!=ds.pbc.map_irk_ks.end()) for(const auto &member:star->second)
                    if(same(q,member)) {
                        const auto f=(q+qr)*ds.pbc.latvec.Transpose();
                        if(std::abs(f.x-std::round(f.x))+std::abs(f.y-std::round(f.y))+std::abs(f.z-std::round(f.z))>1.e-8)
                            throw std::runtime_error("chi0 screening cannot restore general Coulomb q stars without basis rotations");
                        match=true; conjugate=true;
                    }
            }
            if(!match) continue;
            for(int a=0;a<block->nr;++a) for(int b=0;b<block->nc;++b) {
                const int mu=ds.basis_aux.get_global_index(i,a),nu=ds.basis_aux.get_global_index(j,b);
                const auto x=conjugate ? std::conj((*block)(a,b)):(*block)(a,b);
                v(mu,nu)=x; counts[mu*n+nu]=1;
                if(i!=j){v(nu,mu)=std::conj(x);counts[nu*n+mu]=1;}
            }
            break;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE,v.ptr(),n*n,MPI_C_DOUBLE_COMPLEX,MPI_SUM,ds.comm_h.comm);
    MPI_Allreduce(MPI_IN_PLACE,counts.data(),n*n,MPI_INT,MPI_SUM,ds.comm_h.comm);
    for(int k=0;k<n*n;++k){
        if(!counts[k]) throw std::runtime_error("missing cut Coulomb block for chi0 screening (requires full/inverse q coverage)");
        v.ptr()[k]/=counts[k];
    }
    return v;
}
Matz collect_cut_coulomb(librpa_int::Dataset &ds,const librpa_int::Vector3_Order<double> &q) {
    return collect_coulomb(ds,q,true);
}

Matz effective_inverse_dielectric(const Matz &inverse, double plasma, double binding) {
    if (!std::isfinite(plasma) || plasma<=0 || !std::isfinite(binding) || binding<0)
        throw std::runtime_error("effective screening requires positive plasma energy and nonnegative binding energy");
    if(binding==0) return inverse.copy(); // Exact static limit, including filtered Coulomb channels.
    auto vectors=hermitian(inverse);
    auto vectors_memory = watch_memory("screening.vectors", vectors);
    const int n=vectors.nr(), lwork=std::max(1,64*n);
    std::vector<double> eigen(n),rwork(std::max(1,3*n-2));
    auto eigen_memory = watch_memory("screening.eigen", eigen);
    auto rwork_memory = watch_memory("screening.rwork", rwork);
    std::vector<Complex> work(lwork);
    auto work_memory = watch_memory("screening.work", work); int info=0;
    librpa_int::LapackConnector::heev('V','U',n,vectors.ptr(),n,eigen.data(),work.data(),lwork,rwork.data(),info);
    if(info) throw std::runtime_error("effective inverse-dielectric eigensolver failed");
    auto weighted=vectors.get_transpose(true);
    auto weighted_memory = watch_memory("screening.weighted", weighted);
    for(int i=0;i<n;++i) {
        // Evaluate the nonlinear model in screening eigenchannels:
        // f(d)=1-P*(1-d)/(P+Eb*sqrt(1-d)), P=hbar*omega_p.
        // Roundoff near d=0 or 1 is harmless; a genuinely non-passive static
        // dielectric is not silently repaired by clamping physical eigenvalues.
        if(eigen[i]<-1.e-7 || eigen[i]>1.+1.e-7)
            throw std::runtime_error("inverse dielectric eigenvalue outside [0,1]: "+std::to_string(eigen[i]));
        const double d=std::clamp(eigen[i],0.,1.);
        weighted.scale_row(i,1.-plasma*(1.-d)/(plasma+binding*std::sqrt(1.-d)));
    }
    return vectors*weighted;
}

Chi0Screening::Chi0Screening(const InputParameters &options,librpa_int::Dataset &ds)
    :pbc(ds.pbc),options_(options),dataset_(ds) {
    memory_ = std::make_unique<MemoryWatch>([this](MemoryVisitor &v) {
        for (const auto &entry : channels_) {
            visit_memory(v, entry.second.sqrt_cut, "screening.cached_sqrt_cut");
            visit_memory(v, entry.second.inverse, "screening.cached_inverse");
        }
    });
}

Matz Chi0Screening::screen(const Matz &chi,const librpa_int::Vector3_Order<double> &q,double omega) {
    using namespace librpa_int;
    auto &ds=dataset_;
    if(omega_<0) {
        omega_=omega;
        if(options_.chi0_headwing) {
            // The analytic head and wings MUST use the same physical frequency
            // as the imported response, including exactly zero for chi0_static.
            // This dedicated one-node grid is not used for a GW integral.
            ds.tfg.generate_single_frequency(std::abs(omega));
            librpa::Options opts;
            opts.replace_w_head=LIBRPA_SWITCH_ON; opts.option_dielect_func=3;
            opts.sqrt_coulomb_threshold=0.; opts.use_shrink_abfs=LIBRPA_SWITCH_OFF;
            opts.use_symmetry_rpa=LIBRPA_SWITCH_OFF; opts.use_symmetry_gw=LIBRPA_SWITCH_OFF;
            opts.use_kpara_scf_eigvec=LIBRPA_SWITCH_OFF;
            ds.p_headwing.reset();
            initialize_ds_headwing(ds,opts,true);
        }
    }
    if(std::abs(omega-omega_)>1.e-12) throw std::runtime_error("inconsistent imported chi0 frequencies");
    auto cut=hermitian(collect_coulomb(ds,q,true));
    auto cut_memory = watch_memory("screening.cut", cut);
    auto full=options_.chi0_coulomb_metric=="full" ? hermitian(collect_coulomb(ds,q,false)) : cut.copy();
    auto full_memory = watch_memory("screening.full", full);
    const int n=chi.nr();
    ArrayDesc desc(ds.blacs_h); desc.init(n,n,std::min(n,64),std::min(n,64),0,0);
    auto scatter=[&](const Matz &dense) {
        auto local=init_local_mat<Complex>(desc,MAJOR::COL);
        for(int i=0;i<desc.m_loc();++i) for(int j=0;j<desc.n_loc();++j)
            local(i,j)=dense(desc.indx_l2g_r(i),desc.indx_l2g_c(j));
        return local;
    };
    auto gather=[&](const Matz &local) {
        Matz dense(n,n,MAJOR::ROW);
        auto dense_memory = watch_memory("screening.dense", dense);
        for(int i=0;i<desc.m_loc();++i) for(int j=0;j<desc.n_loc();++j)
            dense(desc.indx_l2g_r(i),desc.indx_l2g_c(j))=local(i,j);
        MPI_Allreduce(MPI_IN_PLACE,dense.ptr(),n*n,MPI_C_DOUBLE_COMPLEX,MPI_SUM,ds.comm_h.comm);
        return dense;
    };
    auto sqrtfull=scatter(full), sqrtcut=scatter(cut);
    auto sqrtfull_memory = watch_memory("screening.sqrtfull", sqrtfull);
    auto sqrtcut_memory = watch_memory("screening.sqrtcut", sqrtcut);
    auto eigenvectors=init_local_mat<Complex>(desc,MAJOR::COL);
    auto eigenvectors_memory = watch_memory("screening.eigenvectors", eigenvectors);
    std::vector<double> eigen(n);
    auto eigen_memory = watch_memory("screening.eigen", eigen); std::size_t filtered=0;
    // Use exactly the LibRPA GW square-root convention (descending Coulomb
    // eigenvectors; real Gamma decomposition). The largest Coulomb eigenvector
    // is the singular head channel expected by rewrite_eps_abf_space.
    const bool gamma=options_.chi0_coulomb_metric=="full" && std::abs(q.x)+std::abs(q.y)+std::abs(q.z)<1.e-10;
    if(gamma) power_hemat_blacs_real(sqrtcut,desc,eigenvectors,desc,filtered,eigen.data(),.5,0.);
    else power_hemat_blacs_desc(sqrtcut,desc,eigenvectors,desc,filtered,eigen.data(),.5,0.);
    if(gamma) power_hemat_blacs_real(sqrtfull,desc,eigenvectors,desc,filtered,eigen.data(),.5,0.);
    else power_hemat_blacs_desc(sqrtfull,desc,eigenvectors,desc,filtered,eigen.data(),.5,0.);
    auto response=scatter(hermitian(chi));
    auto response_memory = watch_memory("screening.response", response);
    auto work=init_local_mat<Complex>(desc,MAJOR::COL);
    auto work_memory = watch_memory("screening.work", work);
    ScalapackConnector::pgemm_f('N','N',n,n,n,1.,sqrtfull.ptr(),1,1,desc.desc,response.ptr(),1,1,desc.desc,0.,work.ptr(),1,1,desc.desc);
    ScalapackConnector::pgemm_f('N','N',n,n,n,-1.,work.ptr(),1,1,desc.desc,sqrtfull.ptr(),1,1,desc.desc,0.,response.ptr(),1,1,desc.desc);
    for(int i=0;i<desc.m_loc();++i) for(int j=0;j<desc.n_loc();++j)
        if(desc.indx_l2g_r(i)==desc.indx_l2g_c(j)) response(i,j)+=1.;
    if(gamma && ds.comm_h.myid==0)
        std::cout << "Gamma Coulomb retained subspace: " << n-filtered << " / " << n << '\n';
    if(gamma && options_.chi0_headwing) {
        // LibRPA option 3 performs the body inverse, analytic wing coupling and
        // angular Gamma-cell average, and returns epsilon^-1 (not epsilon).
        ds.p_headwing->rewrite_eps_abf_space(response,0,sqrtfull,eigenvectors,desc,n-filtered,0.,false,false);
    } else {
        power_hemat_blacs_desc(response,desc,eigenvectors,desc,filtered,eigen.data(),-1.,0.);
        if(filtered || *std::min_element(eigen.begin(),eigen.end())<=0.)
            throw std::runtime_error("non-positive dielectric eigenvalue");
    }
    auto inverse=hermitian(gather(response));
    auto inverse_memory = watch_memory("screening.inverse", inverse);
    auto sqrt_cut=hermitian(gather(sqrtcut));
    auto sqrt_cut_memory = watch_memory("screening.sqrt_cut", sqrt_cut);
    // epsilon=I-sqrt(V_full)*chi0*sqrt(V_full), W=sqrt(V_cut)*epsilon^-1*sqrt(V_cut).
    // The cut only regularizes the external interaction; using it inside epsilon
    // changes all q channels and is not the producer's GW screening convention.
    const auto w=sqrt_cut*inverse*sqrt_cut;
        auto w_memory = watch_memory("screening.W", w);
    if(options_.out_screening_matrices && ds.comm_h.myid==0) {
        // Self-describing binary audit record (native int32/float64/complex128,
        // row-major): n, q_fractional[3], omega_Ha, sqrt(V_cut), epsilon^-1.
        // These are the coarse-grid matrices BEFORE spatial interpolation.
        const auto frac=q*ds.pbc.latvec.Transpose();
        const double header[4]={frac.x,frac.y,frac.z,omega};
        const auto path=std::filesystem::path(options_.output_dir)/("screening_q_"+std::to_string(diagnostic_index_++)+".bin");
        std::ofstream out(path,std::ios::binary);
        out.write(reinterpret_cast<const char*>(&n),sizeof(n));
        out.write(reinterpret_cast<const char*>(header),sizeof(header));
        out.write(reinterpret_cast<const char*>(sqrt_cut.ptr()),sqrt_cut.size()*sizeof(Complex));
        out.write(reinterpret_cast<const char*>(inverse.ptr()),inverse.size()*sizeof(Complex));
        if(!out) throw std::runtime_error("cannot write screening audit matrices: "+path.string());
    }
    if(options_.bse_plasma_energy_ev>0) channels_[q]={std::move(sqrt_cut),std::move(inverse)};
    return w;
}

TensorMap<Complex> Chi0Screening::effective(double binding_ev,const std::vector<int> &rows,const std::vector<int> &cols) {
    const double ha_to_ev=2*PARAM.constants.ry_to_ev;
    LibRPA_API::ScreenedQBlocks wq;
    auto wq_memory = watch_memory("screening.wq", wq);
    for(const auto &[q,channel]:channels_) {
        const auto inverse=effective_inverse_dielectric(channel.inverse,options_.bse_plasma_energy_ev/ha_to_ev,binding_ev/ha_to_ev);
        const auto w=channel.sqrt_cut*inverse*channel.sqrt_cut;
        auto w_memory = watch_memory("screening.W", w);
        for(int i:rows) for(int j:cols) {
            Matz block(dataset_.basis_aux[i],dataset_.basis_aux[j],librpa_int::MAJOR::ROW);
            for(std::size_t a=0;a<block.nr();++a) for(std::size_t b=0;b<block.nc();++b)
                block(a,b)=w(dataset_.basis_aux.get_global_index(i,a),dataset_.basis_aux.get_global_index(j,b));
            wq[i][j][q]=std::move(block);
        }
    }
    if(channels_.empty()) throw std::runtime_error("no cached dielectric channels for effective BSE");
    // Apply the nonlinear spectral model BEFORE W(q)->W(R)->W(q_fine).
    return LibRPA_API::transform_screened_q_to_r(dataset_,pbc,wq);
}
TensorMap<Complex> read_librpa_chi0(const InputParameters &opts,librpa_int::Dataset &ds,
                                 const std::vector<int> &rows,const std::vector<int> &cols,Chi0Screening *screening) {
    Chi0Screening owned(opts,ds); if(!screening) screening=&owned;
    namespace fs=std::filesystem;
    std::ifstream metadata(fs::path(opts.screened_dir)/"chi0_rf.info");
    std::string magic; int version,n,px,py,pz; double omega;
    if(!(metadata>>magic>>version>>omega>>n>>px>>py>>pz) || magic!="LIBRPA_CHI0_RF" || version!=1
       || n!=int(ds.basis_aux.nb_total) || px!=ds.pbc.period.x || py!=ds.pbc.period.y || pz!=ds.pbc.period.z
       || !std::isfinite(omega)) throw std::runtime_error("invalid/incompatible chi0_rf.info");
    std::vector<Matz> chir;
    auto chir_memory = watch_memory("screening.chir", chir);
    for(std::size_t ir=0;ir<ds.pbc.Rlist.size();++ir) {
        Matz matrix(n,n,librpa_int::MAJOR::ROW);
        auto matrix_memory = watch_memory("screening.matrix", matrix); const auto r=ds.pbc.Rlist[ir];
        for(int i=0;i<ds.basis_aux.n_atoms;++i) for(int j=0;j<ds.basis_aux.n_atoms;++j) {
            const auto path=fs::path(opts.screened_dir)/("Chi0_Mu_"+std::to_string(i)+"_Nu_"+std::to_string(j)+"_iR_"+std::to_string(ir)+"_ifreq_0.mtx");
            std::ifstream in(path); std::string line;
            if(!std::getline(in,line) || line!="%%MatrixMarket matrix coordinate complex general")
                throw std::runtime_error("invalid chi0 MatrixMarket file: "+path.string());
            bool found=false; int x=0,y=0,z=0;
            while(std::getline(in,line) && !line.empty() && line[0]=='%') {
                const auto pos=line.find('(');
                if(pos!=std::string::npos){std::istringstream s(line.substr(pos+1));if(s>>x>>y>>z) found=true;}
            }
            int nr,nc,nz; std::istringstream dims(line);
            if(!found || x!=r.x || y!=r.y || z!=r.z || !(dims>>nr>>nc>>nz)
               || nr!=int(ds.basis_aux[i]) || nc!=int(ds.basis_aux[j]) || nz<0 || nz>nr*nc)
                throw std::runtime_error("invalid chi0 block dimensions/R: "+path.string());
            std::vector<bool> seen(nr*nc,false);
            for(int k=0;k<nz;++k){int a,b;double re,im;
                if(!(in>>a>>b>>re>>im)||a<1||a>nr||b<1||b>nc||!std::isfinite(re)||!std::isfinite(im)||seen[(a-1)*nc+b-1])
                    throw std::runtime_error("invalid/duplicate chi0 entry: "+path.string());
                seen[(a-1)*nc+b-1]=true;
                matrix(ds.basis_aux.get_global_index(i,a-1),ds.basis_aux.get_global_index(j,b-1))={re,im};
            }
        }
        chir.push_back(std::move(matrix));
    }
    auto pbc=ds.pbc; pbc.map_irk_ks.clear();
    LibRPA_API::ScreenedQBlocks wq;
    auto wq_memory = watch_memory("screening.wq", wq);
    const auto grid=librpa_int::build_uniform_kmesh_frac(pbc.period);
    for(const auto &frac:grid) {
        const auto q=frac*pbc.G; pbc.map_irk_ks[q].push_back(q);
        Matz chi(n,n,librpa_int::MAJOR::ROW);
        auto chi_memory = watch_memory("screening.chi", chi);
        // chi0(q)=sum_R exp(+2*pi*i*q.R) chi0(R), on the ORIGINAL coarse mesh.
        // Only W, after the nonlinear screening step, is interpolated to BSE k.
        for(std::size_t ir=0;ir<chir.size();++ir){const auto r=pbc.Rlist[ir];
            chi+=chir[ir]*std::exp(Complex(0,2*PARAM.constants.pi*(frac.x*r.x+frac.y*r.y+frac.z*r.z)));}
        const auto w=screening->screen(chi,q,omega);
        auto w_memory = watch_memory("screening.W", w);
        for(int i:rows) for(int j:cols){Matz block(ds.basis_aux[i],ds.basis_aux[j],librpa_int::MAJOR::ROW);
            for(std::size_t a=0;a<block.nr();++a) for(std::size_t b=0;b<block.nc();++b)
                block(a,b)=w(ds.basis_aux.get_global_index(i,a),ds.basis_aux.get_global_index(j,b));
            wq[i][j][q]=std::move(block);}
    }
    if(ds.comm_h.myid==0) std::cout<<"LibRPA chi0(R) -> chi0(q) -> W(q): omega = "<<omega<<" Ha; LibRPA dielectric screening\n";
    screening->pbc=pbc;
    return LibRPA_API::transform_screened_q_to_r(ds,pbc,wq);
}
}
