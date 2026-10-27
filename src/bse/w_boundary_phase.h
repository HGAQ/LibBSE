#pragma once
#include <array>
#include <complex>
#include <cmath>
namespace libbse {
// Difference between symmetric endpoint averaging and the single -N/2 image.
// (exp(i*a)+exp(-i*a))/2 = cos(a): averaging each boundary axis also averages
// all equivalent corner/edge images. Coarse samples are unchanged exactly.
inline std::complex<double> w_boundary_phase_delta(
    const std::array<double,3> &w_qfrac, const std::array<int,3> &w_cell,
    const std::array<int,3> &w_period) {
    bool w_on_coarse_grid=true;
    for(int w_axis=0;w_axis<3;++w_axis)
        w_on_coarse_grid=w_on_coarse_grid && std::abs(w_qfrac[w_axis]*w_period[w_axis]-std::round(w_qfrac[w_axis]*w_period[w_axis]))<1.e-12;
    if(w_on_coarse_grid) return 0.;
    std::complex<double> w_original_phase=1.,w_averaged_phase=1.;
    for(int w_axis=0;w_axis<3;++w_axis) {
        const double w_angle=2.*std::acos(-1.)*w_qfrac[w_axis]*w_cell[w_axis];
        const std::complex<double> w_axis_phase=std::polar(1.,w_angle);
        w_original_phase*=w_axis_phase;
        w_averaged_phase*= (w_period[w_axis]%2==0 && w_cell[w_axis]==-w_period[w_axis]/2)
            ? std::complex<double>(std::cos(w_angle),0.) : w_axis_phase;
    }
    return w_averaged_phase-w_original_phase;
}
}
