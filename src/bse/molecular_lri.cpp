#include "molecular_lri.h"

#include <RI/global/Array_Operator.h>

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace fs = std::filesystem;

namespace libbse
{
namespace
{

struct BlockHead
{
    int global_row;
    int global_column;
    int rows;
    int columns;
};

MPI_Datatype block_head_mpi_type()
{
    static MPI_Datatype datatype = MPI_DATATYPE_NULL;
    if (datatype == MPI_DATATYPE_NULL)
    {
        const int block_lengths[4] = {1, 1, 1, 1};
        const MPI_Aint displacements[4] = {
            static_cast<MPI_Aint>(offsetof(BlockHead, global_row)),
            static_cast<MPI_Aint>(offsetof(BlockHead, global_column)),
            static_cast<MPI_Aint>(offsetof(BlockHead, rows)),
            static_cast<MPI_Aint>(offsetof(BlockHead, columns))};
        const MPI_Datatype types[4] = {
            MPI_INT, MPI_INT, MPI_INT, MPI_INT};
        MPI_Type_create_struct(4, block_lengths, displacements, types,
                               &datatype);
        MPI_Type_commit(&datatype);
    }
    return datatype;
}

int checked_total(const std::vector<int> &counts,
                  const char *description)
{
    const long long total = std::accumulate(
        counts.begin(), counts.end(), 0LL);
    if (total > std::numeric_limits<int>::max())
        throw std::overflow_error(std::string(description)
                                  + " exceeds the MPI count limit");
    return static_cast<int>(total);
}

} // namespace

MolecularLri::MolecularLri(librpa_int::Dataset &dataset,
                           const InputParameters &options,
                           const QuasiparticleBands &qp)
    : dataset_(dataset), options_(options), qp_(qp)
{
    int rank = 0;
    MPI_Comm_rank(dataset_.comm_h.comm, &rank);
    nk_ = dataset_.mf_band.get_n_kpoints();
    pair_dimension_ = options_.nocc * options_.nvirt;
    log_.open(fs::path(options_.output_dir)
              / ("libri_rank_" + std::to_string(rank) + ".log"));
    if (!log_) throw std::runtime_error("cannot create LibRI log file");

    std::vector<KPoint> kpoints(static_cast<std::size_t>(nk_));
    for (int ik = 0; ik != nk_; ++ik)
    {
        const auto &k = dataset_.kfrac_band_list.at(static_cast<std::size_t>(ik));
        kpoints[static_cast<std::size_t>(ik)] = {k.x, k.y, k.z};
    }
    lr_.init(std::move(kpoints), options_.nocc, options_.nvirt);
    lr_.set_parallel(dataset_.comm_h.comm, dataset_.atoms.size(),
                     static_cast<std::size_t>(nk_), dataset_.pbc.period_array);
    build_exact_q_map();
}

void MolecularLri::initialize(TensorMap<Complex> &Cs_in,
                              TensorMap<Complex> &Vs_in,
                              TensorMap<Complex> &Ws_in)
{
    std::set<int> all_atoms;
    for (int iat = 0; iat != static_cast<int>(dataset_.atoms.size()); ++iat)
        all_atoms.insert(iat);
    const std::set<int> set_i(lr_.list_I.begin(), lr_.list_I.end());
    const std::set<int> set_j(lr_.list_J.begin(), lr_.list_J.end());
    const std::set<int> set_ij(lr_.list_IJ.begin(), lr_.list_IJ.end());

    lr_.set_Cs(Cs_in, PARAM.constants.cs_threshold, set_ij, all_atoms);
    Cs_in.clear();
    lr_.set_Vs(Vs_in, PARAM.constants.coulomb_threshold, set_i, set_j);
    Vs_in.clear();
    lr_.set_Ws(Ws_in, PARAM.constants.coulomb_threshold, set_i, set_j);
    Ws_in.clear();

    build_wavefunctions();
    lr_.cal_Csk_ao_mo("Cs_", log_);
    lr_.free_Cs();
}

void MolecularLri::build_exact_q_map()
{
    using namespace RI::Array_Operator;
    const KPoint period{1.0, 1.0, 1.0};
    for (const int k1 : lr_.k1_indices)
    {
        for (const int k2 : lr_.k2_indices)
        {
            const KPoint q = (lr_.kindex_map[static_cast<std::size_t>(k2)]
                              - lr_.kindex_map[static_cast<std::size_t>(k1)]) % period;
            lr_.q2kpair[q].emplace_back(k1, k2);
        }
    }
    for (const auto &entry : lr_.q2kpair) lr_.q_list.push_back(entry.first);
}

void MolecularLri::build_wavefunctions()
{
    const auto atom_sizes = dataset_.basis_wfc.get_atom_nbs();
    std::vector<std::size_t> offsets(atom_sizes.size() + 1, 0);
    for (std::size_t i = 0; i != atom_sizes.size(); ++i)
        offsets[i + 1] = offsets[i] + atom_sizes[i];

    for (const int ik : lr_.k_indices)
    {
        const auto *wavefunctions = dataset_.mf_band.find_wfc(0, 0, ik);
        if (wavefunctions == nullptr)
            throw std::runtime_error("fine-grid wavefunction is missing at k-point "
                                     + std::to_string(ik));
        if (qp_.ncore + options_.nocc + options_.nvirt > wavefunctions->nr
            || static_cast<int>(offsets.back()) != wavefunctions->nc)
            throw std::runtime_error("fine-grid wavefunction dimensions do not cover BSE bands");
        for (std::size_t iat = 0; iat != atom_sizes.size(); ++iat)
        {
            RI::Tensor<Complex> tensor(
                {static_cast<std::size_t>(options_.nocc + options_.nvirt), atom_sizes[iat]});
            for (int ib = 0; ib != options_.nocc + options_.nvirt; ++ib)
                for (std::size_t iw = 0; iw != atom_sizes[iat]; ++iw)
                    tensor(ib, iw) = (*wavefunctions)(qp_.ncore + ib,
                                                     static_cast<int>(offsets[iat] + iw));
            lr_.map_psi[ik][static_cast<int>(iat)] = std::move(tensor);
        }
    }
}

void transform_k_2dlocal(
    std::vector<Complex> &matrix,
    const KMatrixMap &blocks,
    const librpa_int::ArrayDesc &descriptor,
    int nk, int pair_dimension, double coefficient)
{
    if (matrix.size()
        != static_cast<std::size_t>(descriptor.lld()) * descriptor.n_loc())
        throw std::invalid_argument("invalid local BSE matrix storage");

    int rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(descriptor.comm(), &rank);
    MPI_Comm_size(descriptor.comm(), &mpi_size);
    const double factor = coefficient * PARAM.constants.ha_to_ry
                          / static_cast<double>(nk);
    constexpr int k1_batch_size = 64;
    const MPI_Datatype block_head_type = block_head_mpi_type();
    std::vector<int> send_head_counts(static_cast<std::size_t>(mpi_size));
    std::vector<int> receive_head_counts(static_cast<std::size_t>(mpi_size));
    std::vector<int> send_value_counts(static_cast<std::size_t>(mpi_size));
    std::vector<int> receive_value_counts(static_cast<std::size_t>(mpi_size));
    std::vector<int> send_head_offsets(static_cast<std::size_t>(mpi_size));
    std::vector<int> receive_head_offsets(static_cast<std::size_t>(mpi_size));
    std::vector<int> send_value_offsets(static_cast<std::size_t>(mpi_size));
    std::vector<int> receive_value_offsets(static_cast<std::size_t>(mpi_size));

    const auto owner = [&](int global_row, int global_column)
    {
        return descriptor.get_pnum(descriptor.g2p_r()[global_row],
                                   descriptor.g2p_c()[global_column]);
    };

    for (int k1_start = 0; k1_start < nk; k1_start += k1_batch_size)
    {
        const int k1_end = std::min(k1_start + k1_batch_size, nk);
        std::fill(send_head_counts.begin(), send_head_counts.end(), 0);
        std::fill(send_value_counts.begin(), send_value_counts.end(), 0);

        for (const auto &[k1, k2_blocks] : blocks)
        {
            if (k1 < k1_start || k1 >= k1_end) continue;
            const int row_base = k1 * pair_dimension;
            for (const auto &[k2, tensor] : k2_blocks)
            {
                if (tensor.shape.get_shape_all()
                    != static_cast<std::size_t>(pair_dimension)
                           * pair_dimension)
                    throw std::runtime_error(
                        "LibRI returned a BSE block with unexpected dimensions");
                const int column_base = k2 * pair_dimension;
                for (int j = 0; j < pair_dimension;)
                {
                    const int global_column = column_base + j;
                    const int next_j = std::min(
                        ((global_column / descriptor.nb()) + 1)
                                * descriptor.nb()
                            - column_base,
                        pair_dimension);
                    for (int i = 0; i < pair_dimension;)
                    {
                        const int global_row = row_base + i;
                        const int next_i = std::min(
                            ((global_row / descriptor.mb()) + 1)
                                    * descriptor.mb()
                                - row_base,
                            pair_dimension);
                        const int destination = owner(global_row, global_column);
                        if (destination != rank)
                        {
                            ++send_head_counts[destination];
                            const std::size_t block_values =
                                static_cast<std::size_t>(next_i - i)
                                * (next_j - j);
                            if (block_values
                                > static_cast<std::size_t>(
                                      std::numeric_limits<int>::max()
                                      - send_value_counts[destination]))
                                throw std::overflow_error(
                                    "BSE matrix block message exceeds the MPI count limit");
                            send_value_counts[destination]
                                += static_cast<int>(block_values);
                        }
                        i = next_i;
                    }
                    j = next_j;
                }
            }
        }

        MPI_Alltoall(send_head_counts.data(), 1, MPI_INT,
                     receive_head_counts.data(), 1, MPI_INT,
                     descriptor.comm());
        MPI_Alltoall(send_value_counts.data(), 1, MPI_INT,
                     receive_value_counts.data(), 1, MPI_INT,
                     descriptor.comm());

        for (int process = 1; process < mpi_size; ++process)
        {
            send_head_offsets[process] = send_head_offsets[process - 1]
                                         + send_head_counts[process - 1];
            receive_head_offsets[process] = receive_head_offsets[process - 1]
                                            + receive_head_counts[process - 1];
            send_value_offsets[process] = send_value_offsets[process - 1]
                                          + send_value_counts[process - 1];
            receive_value_offsets[process] = receive_value_offsets[process - 1]
                                             + receive_value_counts[process - 1];
        }
        const int send_head_total = checked_total(
            send_head_counts, "BSE matrix block-head send buffer");
        const int receive_head_total = checked_total(
            receive_head_counts, "BSE matrix block-head receive buffer");
        const int send_value_total = checked_total(
            send_value_counts, "BSE matrix-value send buffer");
        const int receive_value_total = checked_total(
            receive_value_counts, "BSE matrix-value receive buffer");

        std::vector<BlockHead> send_heads(
            static_cast<std::size_t>(send_head_total));
        std::vector<BlockHead> receive_heads(
            static_cast<std::size_t>(receive_head_total));
        std::vector<Complex> send_values(
            static_cast<std::size_t>(send_value_total));
        std::vector<Complex> receive_values(
            static_cast<std::size_t>(receive_value_total));
        std::vector<int> head_cursor = send_head_offsets;
        std::vector<int> value_cursor = send_value_offsets;

        for (const auto &[k1, k2_blocks] : blocks)
        {
            if (k1 < k1_start || k1 >= k1_end) continue;
            const int row_base = k1 * pair_dimension;
            for (const auto &[k2, tensor] : k2_blocks)
            {
                const int column_base = k2 * pair_dimension;
                for (int j = 0; j < pair_dimension;)
                {
                    const int global_column = column_base + j;
                    const int next_j = std::min(
                        ((global_column / descriptor.nb()) + 1)
                                * descriptor.nb()
                            - column_base,
                        pair_dimension);
                    for (int i = 0; i < pair_dimension;)
                    {
                        const int global_row = row_base + i;
                        const int next_i = std::min(
                            ((global_row / descriptor.mb()) + 1)
                                    * descriptor.mb()
                                - row_base,
                            pair_dimension);
                        const int destination = owner(global_row, global_column);
                        if (destination == rank)
                        {
                            const int local_row =
                                descriptor.indx_g2l_r(global_row);
                            const int local_column =
                                descriptor.indx_g2l_c(global_column);
                            for (int jj = j; jj < next_j; ++jj)
                                for (int ii = i; ii < next_i; ++ii)
                                    matrix[static_cast<std::size_t>(local_row + ii - i)
                                           + static_cast<std::size_t>(
                                                 local_column + jj - j)
                                                 * descriptor.lld()]
                                        += (*tensor.data)[static_cast<std::size_t>(
                                               ii + jj * pair_dimension)]
                                           * factor;
                        }
                        else
                        {
                            send_heads[head_cursor[destination]++] = {
                                global_row, global_column,
                                next_i - i, next_j - j};
                            for (int jj = j; jj < next_j; ++jj)
                                for (int ii = i; ii < next_i; ++ii)
                                    send_values[value_cursor[destination]++]
                                        = (*tensor.data)[static_cast<std::size_t>(
                                              ii + jj * pair_dimension)]
                                          * factor;
                        }
                        i = next_i;
                    }
                    j = next_j;
                }
            }
        }

        for (int process = 0; process < mpi_size; ++process)
            if (head_cursor[process]
                    != send_head_offsets[process]
                           + send_head_counts[process]
                || value_cursor[process]
                       != send_value_offsets[process]
                              + send_value_counts[process])
                throw std::runtime_error(
                    "inconsistent BSE matrix send-buffer packing");

        MPI_Alltoallv(send_heads.data(), send_head_counts.data(),
                      send_head_offsets.data(), block_head_type,
                      receive_heads.data(), receive_head_counts.data(),
                      receive_head_offsets.data(), block_head_type,
                      descriptor.comm());
        MPI_Alltoallv(send_values.data(), send_value_counts.data(),
                      send_value_offsets.data(), MPI_C_DOUBLE_COMPLEX,
                      receive_values.data(), receive_value_counts.data(),
                      receive_value_offsets.data(), MPI_C_DOUBLE_COMPLEX,
                      descriptor.comm());

        int receive_cursor = 0;
        for (const BlockHead &head : receive_heads)
        {
            const int local_row = descriptor.indx_g2l_r(head.global_row);
            const int local_column =
                descriptor.indx_g2l_c(head.global_column);
            if (local_row < 0 || local_column < 0)
                throw std::runtime_error(
                    "received a BSE matrix block on the wrong MPI rank");
            for (int j = 0; j < head.columns; ++j)
                for (int i = 0; i < head.rows; ++i)
                    matrix[static_cast<std::size_t>(local_row + i)
                           + static_cast<std::size_t>(local_column + j)
                                 * descriptor.lld()]
                        += receive_values[static_cast<std::size_t>(receive_cursor)
                                          + i + j * head.rows];
            receive_cursor += head.rows * head.columns;
        }
        if (receive_cursor != receive_value_total)
            throw std::runtime_error(
                "inconsistent BSE matrix block communication");
    }
}

void MolecularLri::add_hartree_a(std::vector<Complex> &matrix,
                                 const librpa_int::ArrayDesc &descriptor,
                                 double coefficient)
{
    auto blocks = lr_.cal_cvc_mo_k_hartree_onthefly(
        {"O", "V", "O", "V"}, "Vs_", true);
    transform_k_2dlocal(
        matrix, blocks, descriptor, nk_, pair_dimension_, coefficient);
}

void MolecularLri::add_hartree_b(std::vector<Complex> &matrix,
                                 const librpa_int::ArrayDesc &descriptor,
                                 double coefficient)
{
    auto blocks = lr_.cal_cvc_mo_k_hartree_onthefly(
        {"O", "V", "O", "V"}, "Vs_", false);
    transform_k_2dlocal(
        matrix, blocks, descriptor, nk_, pair_dimension_, coefficient);
}

void MolecularLri::add_screened_a(std::vector<Complex> &matrix,
                                  const librpa_int::ArrayDesc &descriptor,
                                  double coefficient)
{
    auto blocks = lr_.cal_cvc_mo_k_onthefly(
        {"O", "O", "V", "V"}, "Ws_", true);
    transform_k_2dlocal(
        matrix, blocks, descriptor, nk_, pair_dimension_, coefficient);
}

void MolecularLri::add_screened_b(std::vector<Complex> &matrix,
                                  const librpa_int::ArrayDesc &descriptor,
                                  double coefficient)
{
    auto blocks = lr_.cal_cvc_mo_k_onthefly(
        {"V", "O", "O", "V"}, "Ws_", false);
    transform_k_2dlocal(
        matrix, blocks, descriptor, nk_, pair_dimension_, coefficient);
}

void MolecularLri::release_interactions()
{
    lr_.free_Vs();
    lr_.free_Ws();
}

} // namespace libbse
