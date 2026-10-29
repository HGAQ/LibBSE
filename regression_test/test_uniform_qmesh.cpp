#include "bse/uniform_qmesh.h"
#include "bse/w_boundary_phase.h"
#include <iostream>
#include <stdexcept>

int main()
{
    for (const int qmesh_n:{3,7,8,9}) {
        for (const double qmesh_shift:{0.,0.5}) {
            std::vector<std::array<double,3>> qmesh_kpoints;
            for(int qmesh_i=0;qmesh_i<qmesh_n;++qmesh_i)
              for(int qmesh_j=0;qmesh_j<qmesh_n;++qmesh_j)
                for(int qmesh_k=0;qmesh_k<qmesh_n;++qmesh_k)
                    qmesh_kpoints.push_back({(qmesh_i+qmesh_shift)/qmesh_n,
                                            (qmesh_j+qmesh_shift)/qmesh_n,
                                            (qmesh_k+qmesh_shift)/qmesh_n});
            const auto qmesh_shape=libbse::infer_uniform_qmesh(qmesh_kpoints);
            if(qmesh_shape!=std::array<int,3>{qmesh_n,qmesh_n,qmesh_n})
                throw std::runtime_error("Uniform or shifted mesh not recognized");
            std::set<std::array<double,3>> qmesh_keys;
            for(const auto &qmesh_k1:qmesh_kpoints)
              for(const auto &qmesh_k2:qmesh_kpoints) {
                std::array<double,3> qmesh_transfer;
                for(int qmesh_axis=0;qmesh_axis<3;++qmesh_axis)
                    qmesh_transfer[qmesh_axis]=qmesh_k2[qmesh_axis]-qmesh_k1[qmesh_axis];
                const auto qmesh_canonical=libbse::canonical_uniform_q(qmesh_transfer,qmesh_shape);
                for(int qmesh_axis=0;qmesh_axis<3;++qmesh_axis) {
                    const double qmesh_difference=qmesh_transfer[qmesh_axis]-qmesh_canonical[qmesh_axis];
                    if(std::abs(qmesh_difference-std::round(qmesh_difference))>1.e-14)
                        throw std::runtime_error("Canonical q changed physical transfer");
                }
                qmesh_keys.insert(qmesh_canonical);
              }
            if(qmesh_keys.size()!=qmesh_kpoints.size())
                throw std::runtime_error("Transfer roundoff duplicates remain");
            qmesh_kpoints[0][0]+=.001;
            if(libbse::infer_uniform_qmesh(qmesh_kpoints)!=std::array<int,3>{})
                throw std::runtime_error("Irregular mesh incorrectly canonicalized");
        }
    }
    std::vector<std::array<double,3>> qmesh_rectangular;
    for(int qmesh_i=0;qmesh_i<2;++qmesh_i)
      for(int qmesh_j=0;qmesh_j<3;++qmesh_j)
        qmesh_rectangular.push_back({qmesh_i/2.,qmesh_j/3.,0.});
    if(libbse::infer_uniform_qmesh(qmesh_rectangular)!=std::array<int,3>{2,3,1})
        throw std::runtime_error("Rectangular mesh not recognized");
    qmesh_rectangular.back()=qmesh_rectangular.front();
    if(libbse::infer_uniform_qmesh(qmesh_rectangular)!=std::array<int,3>{})
        throw std::runtime_error("Duplicate/missing k point accepted");
    // Compare endpoint averaging against the explicit mean of +/- images.
    const std::array<int,3> w_period{4,3,2}, w_cell{-2,1,-1};
    const std::array<double,3> w_fine_q{0.13,0.21,0.37};
    const double w_two_pi=2*std::acos(-1.);
    std::complex<double> w_image_average{};
    for (int w_sign_x:{-1,1}) for (int w_sign_z:{-1,1})
        w_image_average+=std::polar(0.25,w_two_pi*(w_sign_x*2*w_fine_q[0]+w_fine_q[1]+w_sign_z*w_fine_q[2]));
    const auto w_single_phase=std::polar(1.,w_two_pi*(-2*w_fine_q[0]+w_fine_q[1]-w_fine_q[2]));
    if (std::abs(w_single_phase+libbse::w_boundary_phase_delta(w_fine_q,w_cell,w_period)-w_image_average)>1.e-14)
        throw std::runtime_error("boundary phase does not match equivalent-image average");
    if (std::abs(libbse::w_boundary_phase_delta({0.25,1./3.,0.5},w_cell,w_period))>1.e-14)
        throw std::runtime_error("boundary average changed coarse-grid W");
    if (std::abs(libbse::w_boundary_phase_delta(w_fine_q,{-1,1,-1},{3,3,3}))>1.e-14)
        throw std::runtime_error("boundary average changed odd-grid phases");
    std::cout<<"Uniform transfer mesh checks passed\n";
}
