#pragma once

#include "parameter/parameter.h"

#include <RI/global/Tensor.h>

#include <array>
#include <complex>
#include <map>
#include <utility>
#include <vector>

namespace libbse
{

using Complex = std::complex<double>;
using Cell = std::array<int, 3>;
using KPoint = std::array<double, 3>;
using AtomCell = std::pair<int, Cell>;

template <typename T>
using TensorMap = std::map<int, std::map<AtomCell, RI::Tensor<T>>>;

struct QuasiparticleBands
{
    int ncore = 0;
    int nk = 0;
    int nbands = 0;
    std::vector<double> energies_ry;
    double indirect_gap_ry = 0.0;
    double direct_gap_ry = 0.0;
};

struct EigenSolution
{
    std::vector<double> energies_ry;
    // Local block-cyclic eigenvectors, column-major, with the descriptor used
    // by the solver.  Kept here only long enough for output/analysis.
    std::vector<Complex> vectors_local;
};

// Excitation amplitudes distributed by contiguous electron-hole-pair blocks.
// Every MPI rank stores all requested states for only its local pair interval.
struct DistributedAmplitudes
{
    int dimension = 0;
    int nstates = 0;
    int first_pair = 0;
    int local_pairs = 0;
    std::vector<Complex> values;

    Complex &operator()(int state, int local_pair)
    {
        return values[static_cast<std::size_t>(state) * local_pairs
                      + local_pair];
    }

    const Complex &operator()(int state, int local_pair) const
    {
        return values[static_cast<std::size_t>(state) * local_pairs
                      + local_pair];
    }
};

} // namespace libbse
