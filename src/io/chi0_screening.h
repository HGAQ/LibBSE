#pragma once

#include "utils/memory.h"
#include "interface/librpa_api.h"

#include <mpi.h>

#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace librpa_int
{
class IndexScheduler;
}

namespace libbse
{
// Dense test/reference entry point: a single Coulomb metric, without Gamma repair.
librpa_int::Matz screen_chi0(const librpa_int::Matz &screening_chi0,
                             librpa_int::Matz screening_coulomb);
librpa_int::Matz effective_inverse_dielectric(const librpa_int::Matz &screening_inverse,
                                              double plasma_ha, double binding_ha);
librpa_int::Matz collect_cut_coulomb(librpa_int::Dataset &dataset,
                                     const librpa_int::Vector3_Order<double> &q);

// W(q) atom-pair blocks owned by this MPI rank, row major. Every atom pair
// (I, J) of the full auxiliary matrix is owned by exactly one rank.
using ScreenedOwnedBlocks = std::map<std::pair<int, int>, librpa_int::Matz>;

// Send owner-distributed W(R) tensors to the ranks whose LibRI atom lists need
// them: rank r receives every (I, J, R) with I in its rows and J in its cols.
// Input keys must be unique across ranks (LibRI adds duplicate keys).
TensorMap<Complex> distribute_screened_real_space(const TensorMap<Complex> &screened_owned_r,
                                                  const std::vector<int> &screened_rows,
                                                  const std::vector<int> &screened_cols,
                                                  MPI_Comm screened_communicator);

// Own the static screening channels through the static BSE solve. This avoids
// rereading response files or recomputing head/wing when Eb becomes available.
class Chi0Screening
{
  public:
    Chi0Screening(const InputParameters &options, librpa_int::Dataset &dataset);
    ~Chi0Screening();
    // Returns only the W(q) atom-pair blocks owned by this rank; see
    // distribute_screened_real_space for delivering W(R) to the LibRI lists.
    ScreenedOwnedBlocks screen(const librpa_int::Matz &screening_chi0,
                               const librpa_int::Vector3_Order<double> &q, double omega);
    TensorMap<Complex> effective(double binding_ev, const std::vector<int> &rows,
                                 const std::vector<int> &cols);
    librpa_int::PeriodicBoundaryData pbc;

  private:
    const InputParameters &options_;
    librpa_int::Dataset &dataset_;
    double omega_ = -1;
    int diagnostic_index_ = 0;
    // Number of q points screened so far; labels the per-q progress line.
    int screening_q_count_ = 0;

    struct Channel
    {
        librpa_int::Matz screening_sqrt_cut;
        librpa_int::Matz screening_inverse;
    };
    std::map<librpa_int::Vector3_Order<double>, Channel> channels_;

    // BLACS -> unique atom-pair owner schedule for W, identical for every q.
    // Built on the first screen() call on a one-block-per-process descriptor.
    std::unique_ptr<librpa_int::ArrayDesc> screening_owner_descriptor_;
    std::unique_ptr<librpa_int::IndexScheduler> screening_owner_scheduler_;

    struct ChannelMemoryObserver
    {
        const Chi0Screening &owner;
        void operator()(MemoryVisitor &visitor) const;
    };
    std::unique_ptr<MemoryWatch> memory_;
};
} // namespace libbse
