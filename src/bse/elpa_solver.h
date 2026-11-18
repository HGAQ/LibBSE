#pragma once

#include "bse_types.h"

#include <librpa_file_reader.hpp>

#include <vector>

namespace libbse
{

EigenSolution solve_tda_elpa(std::vector<Complex> &matrix,
                             librpa_int::ArrayDesc &descriptor,
                             int nstates);

EigenSolution solve_full_elpa(std::vector<Complex> &matrix_a,
                              std::vector<Complex> &matrix_b,
                              const librpa_int::ArrayDesc &pair_descriptor,
                              librpa_int::ArrayDesc &full_descriptor,
                              int nstates);

// All eigenpairs of a distributed Hermitian matrix with the ELPA two-stage
// solver: matrix * eigenvectors = eigenvectors * diag(eigenvalues), with the
// eigenvalues in ascending order. The input matrix is overwritten by ELPA.
// matrix_local and eigenvectors_local use the same block-cyclic descriptor.
void solve_hermitian_eigenvectors_elpa(Complex *matrix_local,
                                       const librpa_int::ArrayDesc &descriptor,
                                       double *eigenvalues,
                                       Complex *eigenvectors_local);

// Real symmetric version of solve_hermitian_eigenvectors_elpa.
void solve_symmetric_eigenvectors_elpa(double *matrix_local,
                                       const librpa_int::ArrayDesc &descriptor,
                                       double *eigenvalues,
                                       double *eigenvectors_local);

} // namespace libbse
