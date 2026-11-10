#pragma once

#include "utils/memory.h"
#include "interface/librpa_api.h"

namespace libbse
{
// Dense test/reference entry point: a single Coulomb metric, without Gamma repair.
librpa_int::Matz screen_chi0(const librpa_int::Matz &screening_chi0,
                             librpa_int::Matz screening_coulomb);
librpa_int::Matz effective_inverse_dielectric(const librpa_int::Matz &screening_inverse,
                                              double plasma_ha, double binding_ha);
librpa_int::Matz collect_cut_coulomb(librpa_int::Dataset &dataset,
                                     const librpa_int::Vector3_Order<double> &q);

// Own the static screening channels through the static BSE solve. This avoids
// rereading response files or recomputing head/wing when Eb becomes available.
class Chi0Screening
{
  public:
    Chi0Screening(const InputParameters &options, librpa_int::Dataset &dataset);
    librpa_int::Matz screen(const librpa_int::Matz &screening_chi0,
                            const librpa_int::Vector3_Order<double> &q, double omega);
    TensorMap<Complex> effective(double binding_ev, const std::vector<int> &rows,
                                 const std::vector<int> &cols);
    librpa_int::PeriodicBoundaryData pbc;

  private:
    const InputParameters &options_;
    librpa_int::Dataset &dataset_;
    double omega_ = -1;
    int diagnostic_index_ = 0;

    struct Channel
    {
        librpa_int::Matz screening_sqrt_cut;
        librpa_int::Matz screening_inverse;
    };
    std::map<librpa_int::Vector3_Order<double>, Channel> channels_;

    struct ChannelMemoryObserver
    {
        const Chi0Screening &owner;
        void operator()(MemoryVisitor &visitor) const;
    };
    std::unique_ptr<MemoryWatch> memory_;
};
} // namespace libbse
