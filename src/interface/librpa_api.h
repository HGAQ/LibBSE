#pragma once

#include "bse/bse_types.h"

#include <librpa_file_reader.hpp>

#include <memory>
#include <string>

namespace LibRPA_API
{

void initialize();
void finalize();

struct ReaderOptions
{
    std::string input_dir;
    bool read_ri = true;
    bool read_band_data = true;
};

std::shared_ptr<librpa_int::Dataset> read_dataset(
    MPI_Comm comm, const ReaderOptions &options);

libbse::TensorMap<libbse::Complex> build_bare_coulomb(
    librpa_int::Dataset &dataset);

librpa_int::ComplexMatrix inverse(const librpa_int::ComplexMatrix &matrix);
librpa_int::ComplexMatrix conjugate(const librpa_int::ComplexMatrix &matrix);
librpa_int::ComplexMatrix transpose(const librpa_int::ComplexMatrix &matrix);
void scale_accumulate(libbse::Complex factor,
                      const librpa_int::ComplexMatrix &source,
                      librpa_int::ComplexMatrix &target);

void redistribute(int rows, int columns,
                  const libbse::Complex *source, int source_row, int source_column,
                  const librpa_int::ArrayDesc &source_descriptor,
                  libbse::Complex *target, int target_row, int target_column,
                  const librpa_int::ArrayDesc &target_descriptor);
void redistribute(int rows, int columns,
                  const double *source, int source_row, int source_column,
                  const librpa_int::ArrayDesc &source_descriptor,
                  double *target, int target_row, int target_column,
                  const librpa_int::ArrayDesc &target_descriptor);

void distributed_transpose(
    int rows, int columns, const libbse::Complex *source,
    const librpa_int::ArrayDesc &descriptor, libbse::Complex *target);
void distributed_conjugate_transpose(
    int rows, int columns, const libbse::Complex *source,
    const librpa_int::ArrayDesc &descriptor, libbse::Complex *target);

void multiply(char trans_a, char trans_b, int m, int n, int k,
              double alpha,
              const double *a, const librpa_int::ArrayDesc &a_descriptor,
              const double *b, const librpa_int::ArrayDesc &b_descriptor,
              double beta,
              double *c, const librpa_int::ArrayDesc &c_descriptor);
void multiply(char trans_a, char trans_b, int m, int n, int k,
              libbse::Complex alpha,
              const libbse::Complex *a, const librpa_int::ArrayDesc &a_descriptor,
              const libbse::Complex *b, const librpa_int::ArrayDesc &b_descriptor,
              libbse::Complex beta,
              libbse::Complex *c, const librpa_int::ArrayDesc &c_descriptor);

} // namespace LibRPA_API
