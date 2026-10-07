#include "utils/memory_views.h"
#include "utils/profiler.h"
#include "molecular_lri.h"

#include <RI/global/Array_Operator.h>

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <iostream>
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


// Apply a wavefunction gauge to the dataset's wavefunctions, if requested by the options.
std::vector<Complex> apply_wavefunction_gauge(
    librpa_int::Dataset &dataset,
    const InputParameters &options,
    const QuasiparticleBands &qp)
{
    // The wavefunction gauge is applied to the dataset's wavefunctions in-place, and
    // the phases are returned for use in transforming a separately supplied velocity matrix.
    const int nk = dataset.mf_band.get_n_kpoints();
    const int selected_bands = options.nocc + options.nvirt;
    const int basis_size = dataset.mf_band.get_n_aos();
    std::vector<Complex> phases(
        static_cast<std::size_t>(nk) * selected_bands, Complex(1.0, 0.0));
    auto phases_memory = libbse::watch_memory("molecular_lri.phases", phases);
    // If the wavefunction gauge is not requested, return the default phases.
    const bool align_to_first_k = options.wavefunction_gauge == "first_k"
        || (options.wavefunction_gauge == "auto"
            && options.input_format != "fhi_aims");
    if (!align_to_first_k) return phases;

    int rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(dataset.comm_h.comm, &rank);
    MPI_Comm_size(dataset.comm_h.comm, &mpi_size);
    
    // Determine which MPI rank owns the wavefunction for each k-point,
    // and check that every k-point has an owner.
    std::vector<int> local_owners(static_cast<std::size_t>(nk), mpi_size);
    auto local_owners_memory = libbse::watch_memory("molecular_lri.local_owners", local_owners);
    for (int ik = 0; ik != nk; ++ik)
        if (dataset.mf_band.find_wfc(0, 0, ik) != nullptr)
            local_owners[static_cast<std::size_t>(ik)] = rank;
    std::vector<int> owners(static_cast<std::size_t>(nk), mpi_size);
    auto owners_memory = libbse::watch_memory("molecular_lri.owners", owners);
    MPI_Allreduce(local_owners.data(), owners.data(), nk, MPI_INT, MPI_MIN,
                  dataset.comm_h.comm);
    if (std::find(owners.begin(), owners.end(), mpi_size) != owners.end())
        throw std::runtime_error(
            "a fine-grid wavefunction is absent on every MPI rank");
    // Broadcast the k=0 wavefunction to all ranks, and align the phases of all other k-points
    // to the k=0 wavefunction.  
    // The phases are stored in the returned vector, which is indexed by (ik * selected_bands + ib).
    std::vector<Complex> reference(
        static_cast<std::size_t>(selected_bands) * basis_size, Complex{});
    auto reference_memory = libbse::watch_memory("molecular_lri.reference", reference);
    if (rank == owners[0]){
        const auto *wavefunctions = dataset.mf_band.find_wfc(0, 0, 0);
        if (wavefunctions == nullptr
            || qp.ncore + selected_bands > wavefunctions->nr
            || basis_size != wavefunctions->nc)
            throw std::runtime_error(
                "reference wavefunction dimensions do not cover BSE bands");
        // Copy the k=0 wavefunction for the selected bands into the reference array.
        for (int ib = 0; ib != selected_bands; ++ib)
            for (int iw = 0; iw != basis_size; ++iw)
                reference[static_cast<std::size_t>(ib) * basis_size + iw]
                    = (*wavefunctions)(qp.ncore + ib, iw);
    }
    // Broadcast the reference wavefunction to all ranks.
    MPI_Bcast(reference.data(), static_cast<int>(reference.size()),
              MPI_C_DOUBLE_COMPLEX, owners[0], dataset.comm_h.comm);

    std::vector<Complex> owner_phases(phases.size(), Complex{});
    auto owner_phases_memory = libbse::watch_memory("molecular_lri.owner_phases", owner_phases);
    for (int ik = 0; ik != nk; ++ik)
    {
        auto *wavefunctions = dataset.mf_band.find_wfc(0, 0, ik);
        if (wavefunctions == nullptr) continue;
        if (qp.ncore + selected_bands > wavefunctions->nr
            || basis_size != wavefunctions->nc)
            throw std::runtime_error(
                "fine-grid wavefunction dimensions do not cover BSE bands");
        for (int ib = 0; ib != selected_bands; ++ib)
        {
            Complex band_phase(1.0, 0.0);
            if (ik != 0)
            {
                // Align the phase of the wavefunction for this band to the reference wavefunction.
                // The formula is: band_phase = <psi|phi> / |<psi|phi>|, where |phi> is the reference wavefunction
                Complex overlap{};
                for (int iw = 0; iw != basis_size; ++iw)
                    overlap += std::conj((*wavefunctions)(qp.ncore + ib, iw))
                               * reference[static_cast<std::size_t>(ib)
                                           * basis_size + iw];
                const double magnitude = std::abs(overlap);
                if (magnitude == 0.0)
                    throw std::runtime_error(
                        "cannot align a BSE wavefunction with the k=0 phase reference");
                band_phase = overlap / magnitude;
                for (int iw = 0; iw != basis_size; ++iw)
                    (*wavefunctions)(qp.ncore + ib, iw) *= band_phase;
            }
            if (rank == owners[static_cast<std::size_t>(ik)])
                owner_phases[static_cast<std::size_t>(ik) * selected_bands + ib]
                    = band_phase;
        }
    }
    // Reduce the phases from the owner ranks to all ranks.
    MPI_Allreduce(owner_phases.data(), phases.data(),
                  static_cast<int>(phases.size()), MPI_C_DOUBLE_COMPLEX,
                  MPI_SUM, dataset.comm_h.comm);
    return phases;
}

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
    auto kpoints_memory = libbse::watch_memory("molecular_lri.kpoints", kpoints);
    for (int ik = 0; ik != nk_; ++ik)
    {
        const auto &k = dataset_.kfrac_band_list.at(static_cast<std::size_t>(ik));
        kpoints[static_cast<std::size_t>(ik)] = {k.x, k.y, k.z};
    }
    lr_.init(std::move(kpoints), options_.nocc, options_.nvirt);
    lr_.set_parallel(dataset_.comm_h.comm, dataset_.atoms.size(),
                     static_cast<std::size_t>(nk_), dataset_.pbc.period_array);
    memory_ = std::make_unique<MemoryWatch>([this](MemoryVisitor &v) {
        visit_memory(v, lr_.map_psi, "LibRI.wavefunctions");
        visit_memory(v, lr_.Csk_ao_mo, "LibRI.Csk_ao_mo");
        for (const auto &entry : lr_.lrik.data_pool)
            visit_memory(v, entry.second.Ds_ab, "LibRI." + entry.first);
    });
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

    if (options_.screened_format.find("chi0") != std::string::npos && options_.chi0_headwing)
    {
        // Analytic wings need uniquely owned C(R) blocks. The LR interface
        // disables its own tensor communication, so gather the atom rows needed
        // by this rank before its AO->MO Fourier transform. Do this once, AFTER
        // head/wing construction; replicating earlier would overcount the wing.
        Cs_in = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
            dataset_.comm_h.comm, Cs_in, set_ij, all_atoms);
    }
    lr_.set_Cs(Cs_in, PARAM.constants.cs_threshold, set_ij, all_atoms);
    libbse::MemoryTracker::instance().checkpoint();
    Cs_in.clear();
    lr_.set_Vs(Vs_in, PARAM.constants.coulomb_threshold, set_i, set_j);
    libbse::MemoryTracker::instance().checkpoint();
    Vs_in.clear();
    lr_.set_Ws(Ws_in, PARAM.constants.coulomb_threshold, set_i, set_j);
    libbse::MemoryTracker::instance().checkpoint();
    Ws_in.clear();

    {
        ScopedTimer timer(global::profiler, "libri_wavefunctions", "Prepare LibRI wavefunctions", dataset_.comm_h.comm);
        build_wavefunctions();
    }
    {
        ScopedTimer timer(global::profiler, "libri_Csk_ao_mo", "Transform RI coefficients AO to MO", dataset_.comm_h.comm);
        lr_.cal_Csk_ao_mo("Cs_", log_);
    }
    MemoryTracker::instance().checkpoint();
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
    auto offsets_memory = libbse::watch_memory("molecular_lri.offsets", offsets);
    for (std::size_t i = 0; i != atom_sizes.size(); ++i)
        offsets[i + 1] = offsets[i] + atom_sizes[i];

    const int basis_size = static_cast<int>(offsets.back());

    for (const int ik : lr_.k_indices)
    {
        auto *wavefunctions = dataset_.mf_band.find_wfc(0, 0, ik);
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
                    tensor(ib, iw) = (*wavefunctions)(
                        qp_.ncore + ib, static_cast<int>(offsets[iat] + iw));
            lr_.map_psi[ik][static_cast<int>(iat)] = std::move(tensor);
        }
    }
}

void transform_k_2dlocal(
    std::vector<Complex> &matrix,
    const KMatrixMap &blocks,
    const librpa_int::ArrayDesc &descriptor,
    int nk, int pair_dimension, double coefficient, int k1_batch_size)
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
    if (k1_batch_size <= 0) throw std::invalid_argument("invalid k batch size");
    const MPI_Datatype block_head_type = block_head_mpi_type();
    std::vector<int> send_head_counts(static_cast<std::size_t>(mpi_size));
    auto send_head_counts_memory = libbse::watch_memory("molecular_lri.send_head_counts", send_head_counts);
    std::vector<int> receive_head_counts(static_cast<std::size_t>(mpi_size));
    auto receive_head_counts_memory = libbse::watch_memory("molecular_lri.receive_head_counts", receive_head_counts);
    std::vector<int> send_value_counts(static_cast<std::size_t>(mpi_size));
    auto send_value_counts_memory = libbse::watch_memory("molecular_lri.send_value_counts", send_value_counts);
    std::vector<int> receive_value_counts(static_cast<std::size_t>(mpi_size));
    auto receive_value_counts_memory = libbse::watch_memory("molecular_lri.receive_value_counts", receive_value_counts);
    std::vector<int> send_head_offsets(static_cast<std::size_t>(mpi_size));
    auto send_head_offsets_memory = libbse::watch_memory("molecular_lri.send_head_offsets", send_head_offsets);
    std::vector<int> receive_head_offsets(static_cast<std::size_t>(mpi_size));
    auto receive_head_offsets_memory = libbse::watch_memory("molecular_lri.receive_head_offsets", receive_head_offsets);
    std::vector<int> send_value_offsets(static_cast<std::size_t>(mpi_size));
    auto send_value_offsets_memory = libbse::watch_memory("molecular_lri.send_value_offsets", send_value_offsets);
    std::vector<int> receive_value_offsets(static_cast<std::size_t>(mpi_size));
    auto receive_value_offsets_memory = libbse::watch_memory("molecular_lri.receive_value_offsets", receive_value_offsets);

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

        const int send_head_total = checked_total(
            send_head_counts, "BSE matrix block-head send buffer");
        const int receive_head_total = checked_total(
            receive_head_counts, "BSE matrix block-head receive buffer");
        const int send_value_total = checked_total(
            send_value_counts, "BSE matrix-value send buffer");
        const int receive_value_total = checked_total(
            receive_value_counts, "BSE matrix-value receive buffer");

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
        std::vector<BlockHead> send_heads(
            static_cast<std::size_t>(send_head_total));
        auto send_heads_memory = libbse::watch_memory("molecular_lri.send_heads", send_heads);
        std::vector<BlockHead> receive_heads(
            static_cast<std::size_t>(receive_head_total));
        auto receive_heads_memory = libbse::watch_memory("molecular_lri.receive_heads", receive_heads);
        std::vector<Complex> send_values(
            static_cast<std::size_t>(send_value_total));
        auto send_values_memory = libbse::watch_memory("molecular_lri.send_values", send_values);
        std::vector<Complex> receive_values(
            static_cast<std::size_t>(receive_value_total));
        auto receive_values_memory = libbse::watch_memory("molecular_lri.receive_values", receive_values);
        std::vector<int> head_cursor = send_head_offsets;
        auto head_cursor_memory = libbse::watch_memory("molecular_lri.head_cursor", head_cursor);
        std::vector<int> value_cursor = send_value_offsets;
        auto value_cursor_memory = libbse::watch_memory("molecular_lri.value_cursor", value_cursor);

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

void MolecularLri::add_batched(std::vector<Complex> &matrix,
                                const librpa_int::ArrayDesc &descriptor,
                                double coefficient, bool hartree, bool is_a)
{
    const std::size_t limit = options_.bse_ri_batch_blocks;
    const std::vector<std::string> psi = hartree
        ? (is_a ? std::vector<std::string>{"O","V","O","V"}
                : std::vector<std::string>{"O","V","O","V"})
        : (is_a ? std::vector<std::string>{"O","O","V","V"}
                : std::vector<std::string>{"V","O","O","V"});
    // Only metadata is retained across batches, never all dense k-pair blocks.
    std::vector<std::pair<KPoint, std::pair<int,int>>> pairs;
    auto pairs_memory = libbse::watch_memory("molecular_lri.pairs", pairs);
    if (!hartree)
        for (const auto &[q, entries] : lr_.q2kpair)
            for (const auto &entry : entries) pairs.emplace_back(q, entry);
    const std::size_t col_batches = (lr_.k2_indices.size() + limit - 1) / limit;
    unsigned long long batches = hartree ? lr_.k1_indices.size() * col_batches
                                         : (pairs.size() + limit - 1) / limit;
    unsigned long long rounds = 0;
    MPI_Allreduce(&batches, &rounds, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                  descriptor.comm());
    int rank = 0; MPI_Comm_rank(descriptor.comm(), &rank);
    if (rank == 0)
        std::cout << "RI streaming: " << (hartree ? "Hartree" : "screened")
                  << " max " << limit << " k-pair blocks/rank, " << rounds
                  << " collective batches\n";
    for (unsigned long long batch = 0; batch < rounds; ++batch)
    {
        KMatrixMap blocks;
        auto blocks_memory = watch_memory("LibRI.batch_blocks", blocks);
        if (batch < batches)
        {
            if (hartree)
            {
                const std::vector<int> rows{lr_.k1_indices[batch / col_batches]};
                auto rows_memory = libbse::watch_memory("molecular_lri.rows", rows);
                const auto begin = (batch % col_batches) * limit;
                const auto end = std::min<std::size_t>(begin + limit, lr_.k2_indices.size());
                const std::vector<int> cols(lr_.k2_indices.begin() + begin,
                                            lr_.k2_indices.begin() + end);
                auto cols_memory = libbse::watch_memory("molecular_lri.cols", cols);
                blocks = lr_.lrik.cal_cvc_mo_k_hartree_onthefly(
                    lr_.Csk_ao_mo, lr_.map_psi, rows, cols, lr_.list_I, lr_.list_J,
                    psi, options_.nocc, options_.nvirt, "Vs_", is_a);
            }
            else
            {
                std::map<KPoint, std::vector<std::pair<int,int>>> q_pairs;
                const auto end = std::min<std::size_t>((batch + 1) * limit, pairs.size());
                for (std::size_t i = batch * limit; i < end; ++i)
                    q_pairs[pairs[i].first].push_back(pairs[i].second);
                std::vector<KPoint> qs;
                auto qs_memory = libbse::watch_memory("molecular_lri.qs", qs);
                for (const auto &entry : q_pairs) qs.push_back(entry.first);
                blocks = lr_.lrik.cal_cvc_mo_k_onthefly(
                    lr_.Csk_ao_mo, lr_.map_psi, lr_.k1_indices, lr_.k2_indices,
                    lr_.list_I, lr_.list_J, psi, options_.nocc, options_.nvirt,
                    "Ws_", is_a, qs, q_pairs);
            }
        }
        MemoryTracker::instance().checkpoint();
        // Empty ranks participate too. Every rank uses the same global rounds.
        // The generated block set is already bounded, so exchange it once.
        transform_k_2dlocal(matrix, blocks, descriptor, nk_, pair_dimension_,
                           coefficient, nk_);
    }
}

void MolecularLri::add_hartree_a(std::vector<Complex> &matrix,
                                 const librpa_int::ArrayDesc &descriptor,
                                 double coefficient)
{
    if (options_.bse_memory_optimized && options_.bse_ri_batch_blocks > 0)
    {
        add_batched(matrix, descriptor, coefficient, true, true);
        return;
    }

    auto blocks = lr_.cal_cvc_mo_k_hartree_onthefly(
        {"O", "V", "O", "V"}, "Vs_", true);
    auto blocks_memory = watch_memory("LibRI.blocks", blocks);
    transform_k_2dlocal(
        matrix, blocks, descriptor, nk_, pair_dimension_, coefficient);
}

void MolecularLri::add_hartree_b(std::vector<Complex> &matrix,
                                 const librpa_int::ArrayDesc &descriptor,
                                 double coefficient)
{
    if (options_.bse_memory_optimized && options_.bse_ri_batch_blocks > 0)
    {
        add_batched(matrix, descriptor, coefficient, true, false);
        return;
    }

    auto blocks = lr_.cal_cvc_mo_k_hartree_onthefly(
        {"O", "V", "O", "V"}, "Vs_", false);
    auto blocks_memory = watch_memory("LibRI.blocks", blocks);
    transform_k_2dlocal(
        matrix, blocks, descriptor, nk_, pair_dimension_, coefficient);
}

void MolecularLri::add_screened_a(std::vector<Complex> &matrix,
                                  const librpa_int::ArrayDesc &descriptor,
                                  double coefficient)
{
    if (options_.bse_memory_optimized && options_.bse_ri_batch_blocks > 0)
    {
        add_batched(matrix, descriptor, coefficient, false, true);
        return;
    }

    auto blocks = lr_.cal_cvc_mo_k_onthefly(
        {"O", "O", "V", "V"}, "Ws_", true);
    auto blocks_memory = watch_memory("LibRI.blocks", blocks);
    transform_k_2dlocal(
        matrix, blocks, descriptor, nk_, pair_dimension_, coefficient);
}

void MolecularLri::add_screened_b(std::vector<Complex> &matrix,
                                  const librpa_int::ArrayDesc &descriptor,
                                  double coefficient)
{
    if (options_.bse_memory_optimized && options_.bse_ri_batch_blocks > 0)
    {
        add_batched(matrix, descriptor, coefficient, false, false);
        return;
    }

    auto blocks = lr_.cal_cvc_mo_k_onthefly(
        {"V", "O", "O", "V"}, "Ws_", false);
    auto blocks_memory = watch_memory("LibRI.blocks", blocks);
    transform_k_2dlocal(
        matrix, blocks, descriptor, nk_, pair_dimension_, coefficient);
}

void MolecularLri::replace_screened(TensorMap<Complex> &screened)
{
    lr_.free_Ws();
    const std::set<int> rows(lr_.list_I.begin(), lr_.list_I.end());
    const std::set<int> cols(lr_.list_J.begin(), lr_.list_J.end());
    lr_.set_Ws(screened, PARAM.constants.coulomb_threshold, rows, cols);
    libbse::MemoryTracker::instance().checkpoint();
    screened.clear();
}

void MolecularLri::release_interactions()
{
    lr_.free_Vs();
    lr_.free_Ws();
}

} // namespace libbse
