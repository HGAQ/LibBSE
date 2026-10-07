#pragma once
#include "utils/memory.h"
#include "interface/librpa_api.h"
namespace libbse {
// Dense test/reference entry point: a single Coulomb metric, without Gamma repair.
librpa_int::Matz screen_chi0(const librpa_int::Matz &chi, librpa_int::Matz v);
librpa_int::Matz effective_inverse_dielectric(const librpa_int::Matz &inverse,
                                             double plasma_ha, double binding_ha);
librpa_int::Matz collect_cut_coulomb(librpa_int::Dataset &dataset,
                                    const librpa_int::Vector3_Order<double> &q);
// Own the static screening channels through the static BSE solve. This avoids
// rereading response files or recomputing head/wing when Eb becomes available.
class Chi0Screening {
public:
    Chi0Screening(const InputParameters &options, librpa_int::Dataset &dataset);
    librpa_int::Matz screen(const librpa_int::Matz &chi,
                           const librpa_int::Vector3_Order<double> &q, double omega);
    TensorMap<Complex> effective(double binding_ev, const std::vector<int> &rows,
                                const std::vector<int> &cols);
    librpa_int::PeriodicBoundaryData pbc;
private:
    const InputParameters &options_;
    librpa_int::Dataset &dataset_;
    double omega_ = -1;
    int diagnostic_index_ = 0;
    struct Channel { librpa_int::Matz sqrt_cut, inverse; };
    std::map<librpa_int::Vector3_Order<double>, Channel> channels_;
    std::unique_ptr<MemoryWatch> memory_;
};
TensorMap<Complex> read_librpa_chi0(const InputParameters &, librpa_int::Dataset &,
                                 const std::vector<int> &, const std::vector<int> &,
                                 Chi0Screening *screening = nullptr);
}
