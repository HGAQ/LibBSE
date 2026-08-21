#pragma once

#include "bse_types.h"

#include <mpi.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace librpa_int
{
class Dataset;
}

namespace libbse
{

struct FineVelocityMo
{
    int nk = 0;
    int nbands = 0;
    int first_pair = 0;
    int local_pairs = 0;
    // Direction-major velocity elements for the locally owned (k,i,a) pairs.
    // Only <i|v|a>, which enters the velocity-gauge spectrum, is retained.
    std::vector<Complex> values;
    std::vector<double> gaps_ha;
};

// Return the velocity matrix elements needed by the locally owned BSE pairs.
// When the SCF and BSE grids coincide this selects the supplied velocity_mo
// directly; otherwise it interpolates the operator through its localized
// AO/R form and sends each pair only to its excitation-amplitude owner.
FineVelocityMo prepare_fine_velocity_mo(
    const InputParameters &options, const QuasiparticleBands &qp,
    const std::shared_ptr<librpa_int::Dataset> &dataset);

std::array<Complex, 3> velocity_gauge_transition_dipole(
    int state, const InputParameters &options, const FineVelocityMo &velocity_mo,
    const std::vector<Complex> &amplitudes_x,
    const std::vector<Complex> *amplitudes_y);

// Contract each rank's local electron-hole-pair block and reduce the
// transition dipoles to rank 0. Excitation amplitudes remain distributed.
std::vector<std::array<Complex, 3>> velocity_gauge_transition_dipoles_mpi(
    MPI_Comm comm, const InputParameters &options,
    const FineVelocityMo &velocity_mo,
    const DistributedAmplitudes &amplitudes_x,
    const DistributedAmplitudes *amplitudes_y);

void write_velocity_gauge_outputs(
    const InputParameters &options,
    const librpa_int::Dataset &dataset,
    const FineVelocityMo &velocity_mo,
    const std::vector<double> &energies_ry,
    const DistributedAmplitudes &amplitudes_x,
    const DistributedAmplitudes *amplitudes_y,
    const std::string &spin_type,
    const std::string &solution_type);

} // namespace libbse
