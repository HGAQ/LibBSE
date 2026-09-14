#pragma once

#include "bse_types.h"
#include "molecular_lri_comm.h"

#include <librpa_file_reader.hpp>

#include <RI/physics/LR.h>

#include <fstream>
#include <map>
#include <vector>

namespace libbse
{

// Put the fine-grid KS states in the producer-compatible band gauge and
// return p_n(k) for psi_n(k) -> p_n(k) psi_n(k).  The returned phases are
// replicated on every MPI rank in (k, selected-band) order so observables
// built from a separately supplied MO operator can be transformed with the
// same gauge.
std::vector<Complex> apply_wavefunction_gauge(
    librpa_int::Dataset &dataset,
    const InputParameters &options,
    const QuasiparticleBands &qp);

class MolecularLri
{
public:
    MolecularLri(librpa_int::Dataset &dataset,
                 const InputParameters &options,
                 const QuasiparticleBands &qp);

    const std::vector<int> &local_i_atoms() const { return lr_.list_I; }
    const std::vector<int> &local_j_atoms() const { return lr_.list_J; }

    void initialize(TensorMap<Complex> &coefficients,
                    TensorMap<Complex> &bare_coulomb,
                    TensorMap<Complex> &screened_interaction);

    void add_hartree_a(std::vector<Complex> &matrix,
                       const librpa_int::ArrayDesc &descriptor,
                       double coefficient);
    void add_hartree_b(std::vector<Complex> &matrix,
                       const librpa_int::ArrayDesc &descriptor,
                       double coefficient);
    void add_screened_a(std::vector<Complex> &matrix,
                        const librpa_int::ArrayDesc &descriptor,
                        double coefficient);
    void add_screened_b(std::vector<Complex> &matrix,
                        const librpa_int::ArrayDesc &descriptor,
                        double coefficient);

    void release_interactions();

private:
    void build_exact_q_map();
    void build_wavefunctions();

    librpa_int::Dataset &dataset_;
    const InputParameters &options_;
    const QuasiparticleBands &qp_;
    int nk_ = 0;
    int pair_dimension_ = 0;
    RI::LR<int, int, 3, Complex> lr_;
    std::ofstream log_;
};

} // namespace libbse
