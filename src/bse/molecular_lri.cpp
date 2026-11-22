#include "w_boundary_phase.h"
#include "utils/memory_views.h"
#include "utils/profiler.h"
#include "molecular_lri.h"
#include "uniform_qmesh.h"

#include <RI/global/Array_Operator.h>
#include <RI/global/Blas_Interface.h>
#include <RI/ri/LRI_Cal_Aux.h>
#include <RI/distribute/Divide_Atoms.h>

#include <malloc.h>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
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
        visit_memory(v, w_boundary_tensors_, "LibRI.w_boundary_tensors");
        visit_memory(v, lr_.map_psi, "LibRI.wavefunctions");
        visit_memory(v, lr_.Csk_ao_mo, "LibRI.Csk_ao_mo");
        for (const auto &entry : lr_.lrik.data_pool)
            visit_memory(v, entry.second.Ds_ab, "LibRI." + entry.first);
    });
    // Correct W(q) after its Fourier transform and before the RI contraction.
    if (options_.bse_w_boundary_average)
        lr_.lrik.screened_q_correction = WBoundaryCorrection{this};
    build_exact_q_map();
}

// Add [averaged phase - original phase] * W_ij(R) for boundary cells.
// Use fractional q coordinates and the coarse BvK period.
void MolecularLri::apply_w_boundary_correction(
    int w_boundary_iatom, int w_boundary_jatom,
    const KPoint &w_boundary_qfrac,
    RI::Tensor<Complex> &w_boundary_pair) const
{
    if (!options_.bse_w_boundary_average) return;

    // Cache layout: iatom -> {(jatom, R) -> W_ij(R)}.
    const auto w_boundary_row = w_boundary_tensors_.find(w_boundary_iatom);
    if (w_boundary_row == w_boundary_tensors_.end()) return;

    for (const auto &[w_boundary_key, w_boundary_tensor] : w_boundary_row->second)
    {
        if (w_boundary_key.first != w_boundary_jatom) continue;

        // Average the +/-N/2 images on even boundary axes; coarse q is unchanged.
        const auto w_phase_delta = w_boundary_phase_delta(
            w_boundary_qfrac, w_boundary_key.second, dataset_.pbc.period_array);
        if (std::abs(w_phase_delta) < 1.e-15) continue; // Negligible phase change.

        // Allocate zeros if needed; otherwise add to the existing W(q).
        if (w_boundary_pair.empty())
            w_boundary_pair = RI::Tensor<Complex>(w_boundary_tensor.shape);
        for (std::size_t w_entry = 0;
             w_entry < w_boundary_tensor.get_shape_all(); ++w_entry)
            w_boundary_pair.ptr()[w_entry]
                += w_phase_delta * w_boundary_tensor.ptr()[w_entry];
    }
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
    if (options_.requires_screened())
        lr_.set_Ws(Ws_in, PARAM.constants.coulomb_threshold, set_i, set_j);
    libbse::MemoryTracker::instance().checkpoint();
    // Retain only endpoint tensors needed by the phase correction. RI tensors
    // share their payload here; the original input map can still be cleared.
    if(options_.bse_w_boundary_average && options_.requires_screened()) {
        w_boundary_tensors_.clear();
        for(const auto &[w_iatom,w_atom_pairs]:lr_.lrik.data_pool.at("Ws_").Ds_ab)
            for(const auto &[w_pair_key,w_pair_tensor]:w_atom_pairs) {
                bool w_is_boundary=false;
                for(int w_axis=0;w_axis<3;++w_axis)
                    w_is_boundary=w_is_boundary || (dataset_.pbc.period_array[w_axis]%2==0 && w_pair_key.second[w_axis]==-dataset_.pbc.period_array[w_axis]/2);
                if(w_is_boundary) w_boundary_tensors_[w_iatom][w_pair_key]=w_pair_tensor;
            }
    }
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
    // Mathematically identical k2-k1 transfers can differ in their last floating
    // digits. Canonical fine-mesh keys let them share one Fourier transform;
    // they do not round transfers to the coarser screening mesh.
    const auto qmesh_shape=infer_uniform_qmesh(lr_.kindex_map);
    for (const int k1 : lr_.k1_indices)
    {
        for (const int k2 : lr_.k2_indices)
        {
            const KPoint q = canonical_uniform_q(
                (lr_.kindex_map[static_cast<std::size_t>(k2)]
                 - lr_.kindex_map[static_cast<std::size_t>(k1)]) % period,
                qmesh_shape);
            lr_.q2kpair[q].emplace_back(k1, k2);
        }
    }
    for (const auto &entry : lr_.q2kpair) lr_.q_list.push_back(entry.first);
    if (dataset_.comm_h.myid==0 && qmesh_shape[0]>0)
        std::cout << "Uniform fine q mesh: " << qmesh_shape[0] << ' '
                  << qmesh_shape[1] << ' ' << qmesh_shape[2]
                  << "; rank-0 transfer keys: " << lr_.q_list.size() << '\n';
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
    // Only the two value buffers are memory-tracked here: every watch
    // attach/detach snapshots all tracked LibRI tensors, and this function
    // runs once per streaming batch. The per-rank count/offset vectors are tiny.
    const double factor = coefficient * PARAM.constants.ha_to_ry
                          / static_cast<double>(nk);
    if (k1_batch_size <= 0) throw std::invalid_argument("invalid k batch size");
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
            std::vector<BlockHead> receive_heads(
            static_cast<std::size_t>(receive_head_total));
            std::vector<Complex> send_values(
            static_cast<std::size_t>(send_value_total));
        auto send_values_memory = libbse::watch_memory("molecular_lri.send_values", send_values);
        std::vector<Complex> receive_values(
            static_cast<std::size_t>(receive_value_total));
        auto receive_values_memory = libbse::watch_memory("molecular_lri.receive_values", receive_values);
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

namespace
{

// Rank-local stacking of the auxiliary functions of one atom list.
// Auxiliary function P of atom auxiliary_layout.atoms[i] is dense row
// auxiliary_layout.offsets[i] + P.
struct AuxiliaryLayout
{
    std::vector<int> atoms;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> sizes;
    std::size_t total = 0;
};

AuxiliaryLayout build_auxiliary_layout(
    const std::vector<int> &layout_atoms,
    const std::map<int, std::map<int, RI::Tensor<Complex>>> &layout_csk_ao_mo,
    const std::vector<int> &layout_k_indices)
{
    AuxiliaryLayout auxiliary_layout;
    auxiliary_layout.atoms = layout_atoms;
    if (layout_atoms.empty() || layout_k_indices.empty())
    {
        auxiliary_layout.atoms.clear();
        return auxiliary_layout;
    }
    const auto &layout_reference_k = layout_csk_ao_mo.at(layout_k_indices.front());
    for (const int layout_atom : layout_atoms)
    {
        // C^mu(P, s, m)[k]: shape[0] is the number of auxiliary functions of atom mu.
        const std::size_t layout_atom_auxiliary_count =
            layout_reference_k.at(layout_atom).shape[0];
        auxiliary_layout.offsets.push_back(auxiliary_layout.total);
        auxiliary_layout.sizes.push_back(layout_atom_auxiliary_count);
        auxiliary_layout.total += layout_atom_auxiliary_count;
    }
    return auxiliary_layout;
}

std::size_t mo_type_count(const std::string &mo_type,
                          std::size_t nocc, std::size_t nvirt)
{
    if (mo_type == "O") return nocc;
    if (mo_type == "V") return nvirt;
    throw std::invalid_argument("unknown MO type " + mo_type);
}

int checked_blas_dimension(std::size_t blas_dimension)
{
    if (blas_dimension > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("BLAS dimension exceeds the int range");
    return static_cast<int>(blas_dimension);
}

// interaction_dense_q[P][Q] = sum_R exp(i 2 pi q.R) X_{mu P, nu Q}(R), X = V or W.
// Rows P run over row_layout (list_I), columns Q over column_layout (list_J).
// The phase convention and the optional W(q) correction are those of LibRI
// LRI_k::cal_cvc_mo_k_onthefly; the correction is applied after the Fourier
// sum and before the contraction, also when the Fourier sum is empty.
void build_dense_interaction_q(
    const RI::LRI_k<int, int, 3, Complex> &interaction_lrik,
    const std::string &interaction_save_name,
    const KPoint &interaction_q,
    const bool interaction_apply_correction,
    const AuxiliaryLayout &interaction_row_layout,
    const AuxiliaryLayout &interaction_column_layout,
    std::vector<Complex> &interaction_dense_q)
{
    using InteractionAtomCell = std::pair<int, std::array<int, 3>>;
    const auto &interaction_real_space =
        interaction_lrik.data_pool.at(interaction_save_name).Ds_ab;
    const std::size_t interaction_dense_columns = interaction_column_layout.total;
    interaction_dense_q.assign(
        interaction_row_layout.total * interaction_dense_columns, Complex(0.0, 0.0));
    const std::size_t interaction_column_atom_count = interaction_column_layout.atoms.size();
    const long long interaction_atom_pair_count =
        static_cast<long long>(interaction_row_layout.atoms.size())
        * static_cast<long long>(interaction_column_atom_count);
    int interaction_shape_error = 0;

#pragma omp parallel for schedule(dynamic)
    for (long long interaction_atom_pair = 0;
         interaction_atom_pair < interaction_atom_pair_count;
         ++interaction_atom_pair)
    {
        const std::size_t interaction_row_index =
            static_cast<std::size_t>(interaction_atom_pair) / interaction_column_atom_count;
        const std::size_t interaction_column_index =
            static_cast<std::size_t>(interaction_atom_pair) % interaction_column_atom_count;
        const int interaction_row_atom = interaction_row_layout.atoms[interaction_row_index];
        const int interaction_column_atom =
            interaction_column_layout.atoms[interaction_column_index];

        RI::Tensor<Complex> interaction_pair_q;
        const auto interaction_row_entry = interaction_real_space.find(interaction_row_atom);
        if (interaction_row_entry != interaction_real_space.end())
        {
            const std::vector<InteractionAtomCell> interaction_column_cells =
                RI::Divide_Atoms::traversal_atom_period(
                    std::vector<int>{interaction_column_atom}, interaction_lrik.period);
            for (const InteractionAtomCell &interaction_column_cell : interaction_column_cells)
            {
                const auto interaction_cell_entry =
                    interaction_row_entry->second.find(interaction_column_cell);
                if (interaction_cell_entry == interaction_row_entry->second.end()
                    || interaction_cell_entry->second.empty())
                    continue;
                const std::array<int, 3> &interaction_cell = interaction_column_cell.second;
                // X_{mu nu}(q) += exp(i 2 pi q.R) X_{mu nu}(R)
                const double interaction_phase_argument =
                    2.0 * M_PI
                    * (interaction_q[0] * interaction_cell[0]
                       + interaction_q[1] * interaction_cell[1]
                       + interaction_q[2] * interaction_cell[2]);
                const Complex interaction_phase(std::cos(interaction_phase_argument),
                                                std::sin(interaction_phase_argument));
                RI::LRI_Cal_Aux::add_Ds(interaction_cell_entry->second,
                                        interaction_pair_q, interaction_phase);
            }
        }
        if (interaction_apply_correction && interaction_lrik.screened_q_correction)
            interaction_lrik.screened_q_correction(interaction_row_atom, interaction_column_atom,
                                                   interaction_q, interaction_pair_q);
        if (interaction_pair_q.empty()) continue;

        const std::size_t interaction_block_rows =
            interaction_row_layout.sizes[interaction_row_index];
        const std::size_t interaction_block_columns =
            interaction_column_layout.sizes[interaction_column_index];
        if (interaction_pair_q.shape.size() != 2
            || interaction_pair_q.shape[0] != interaction_block_rows
            || interaction_pair_q.shape[1] != interaction_block_columns)
        {
#pragma omp atomic write
            interaction_shape_error = 1;
            continue;
        }
        const Complex *interaction_pair_values = interaction_pair_q.ptr();
        for (std::size_t interaction_p = 0; interaction_p < interaction_block_rows; ++interaction_p)
        {
            Complex *interaction_dense_row =
                interaction_dense_q.data()
                + (interaction_row_layout.offsets[interaction_row_index] + interaction_p)
                      * interaction_dense_columns
                + interaction_column_layout.offsets[interaction_column_index];
            for (std::size_t interaction_c = 0; interaction_c < interaction_block_columns;
                 ++interaction_c)
                interaction_dense_row[interaction_c] =
                    interaction_pair_values[interaction_p * interaction_block_columns
                                            + interaction_c];
        }
    }
    if (interaction_shape_error)
        throw std::runtime_error(
            "interaction block does not match the auxiliary sizes of its atoms");
}

// stacked_coefficients[P][(m1 m2)] = C^mu_P(m1, m2)[k_left, k_right] from
// RI::Cs_ao_mo_to_Cs_mo, one row block per atom of the layout.
void build_stacked_mo_coefficients(
    const std::map<int, std::map<int, RI::Tensor<Complex>>> &stacked_csk_ao_mo,
    const std::map<int, std::map<int, RI::Tensor<Complex>>> &stacked_wavefunctions,
    const AuxiliaryLayout &stacked_layout,
    const int stacked_k_left, const int stacked_k_right,
    const std::string &stacked_first_type, const std::string &stacked_second_type,
    const std::size_t nocc, const std::size_t nvirt, const bool stacked_left_conj,
    std::vector<Complex> &stacked_coefficients)
{
    const std::size_t stacked_width = mo_type_count(stacked_first_type, nocc, nvirt)
                                      * mo_type_count(stacked_second_type, nocc, nvirt);
    stacked_coefficients.assign(stacked_layout.total * stacked_width, Complex(0.0, 0.0));
    const long long stacked_atom_count = static_cast<long long>(stacked_layout.atoms.size());
    int stacked_shape_error = 0;

#pragma omp parallel for schedule(dynamic)
    for (long long stacked_atom_index = 0; stacked_atom_index < stacked_atom_count;
         ++stacked_atom_index)
    {
        const RI::Tensor<Complex> stacked_atom_coefficients = RI::Cs_ao_mo_to_Cs_mo(
            stacked_csk_ao_mo, stacked_wavefunctions, stacked_k_left, stacked_k_right,
            stacked_layout.atoms[static_cast<std::size_t>(stacked_atom_index)],
            stacked_first_type, stacked_second_type, nocc, nvirt, stacked_left_conj);
        const std::size_t stacked_atom_values =
            stacked_layout.sizes[static_cast<std::size_t>(stacked_atom_index)] * stacked_width;
        if (stacked_atom_coefficients.get_shape_all() != stacked_atom_values)
        {
#pragma omp atomic write
            stacked_shape_error = 1;
            continue;
        }
        std::copy(stacked_atom_coefficients.ptr(),
                  stacked_atom_coefficients.ptr() + stacked_atom_values,
                  stacked_coefficients.data()
                      + stacked_layout.offsets[static_cast<std::size_t>(stacked_atom_index)]
                            * stacked_width);
    }
    if (stacked_shape_error)
        throw std::runtime_error("RI MO coefficients do not match the auxiliary layout");
}

// Row-major product product_out[m][n] = op(left)[m][k] op(right)[k][n]
// through LibRI's row-major BLAS wrapper. Called outside OpenMP regions, so
// the OpenMP OpenBLAS uses all threads of the rank.
void row_major_gemm(const char left_transpose, const char right_transpose,
                    std::size_t product_rows, std::size_t product_columns,
                    std::size_t product_inner,
                    const Complex *product_left, const Complex *product_right,
                    Complex *product_out)
{
    if (product_rows == 0 || product_columns == 0) return;
    if (product_inner == 0)
    {
        std::fill(product_out, product_out + product_rows * product_columns,
                  Complex(0.0, 0.0));
        return;
    }
    RI::Blas_Interface::gemm(left_transpose, right_transpose,
                             checked_blas_dimension(product_rows),
                             checked_blas_dimension(product_columns),
                             checked_blas_dimension(product_inner),
                             Complex(1.0, 0.0), product_left, product_right,
                             Complex(0.0, 0.0), product_out);
}

// Print progress at most about twenty times per stage and flush, so a long
// stage is visible in a redirected log.
void report_batch_progress(const char *progress_stage, unsigned long long progress_batch,
                           unsigned long long progress_rounds, double progress_start)
{
    const unsigned long long progress_stride =
        std::max<unsigned long long>(1, progress_rounds / 20);
    if ((progress_batch + 1) % progress_stride != 0 && progress_batch + 1 != progress_rounds)
        return;
    // Format in a local stream so std::cout keeps its default float format.
    std::ostringstream progress_line;
    progress_line << "RI streaming " << progress_stage << ": batch " << (progress_batch + 1)
                  << " / " << progress_rounds << ", elapsed " << std::fixed
                  << std::setprecision(1) << (MPI_Wtime() - progress_start) << " s";
    std::cout << progress_line.str() << std::endl;
}

} // namespace

void MolecularLri::add_batched(std::vector<Complex> &matrix,
                                const librpa_int::ArrayDesc &descriptor,
                                double coefficient, bool hartree, bool is_a)
{
    if (hartree)
        add_batched_hartree(matrix, descriptor, coefficient, is_a);
    else
        add_batched_screened(matrix, descriptor, coefficient, is_a);
    malloc_trim(0);
}

void MolecularLri::add_batched_hartree(std::vector<Complex> &matrix,
                                       const librpa_int::ArrayDesc &descriptor,
                                       double coefficient, bool is_a)
{
    // Hartree kernel of this rank's atom pairs (summed over ranks later):
    //   A^H_{ia,jb}(k1,k2) = sum_{P in I, Q in J} C^P_{ia}(k1) V_PQ(q=0) C^Q_{jb}(k2)^*
    //   (B: the same without the conjugation of C^Q).
    // Step 1, once per k2:      D[P][jb] = sum_Q V_PQ(0) C^Q_{jb}(k2)
    // Step 2, per (k1,k2) block: A[jb][ia] = sum_P D[P][jb] C^P_{ia}(k1)
    // The block is stored as (j,b,i,a), as the LibRI on-the-fly kernel returns it.
    const std::size_t nocc = options_.nocc;
    const std::size_t nvirt = options_.nvirt;
    const std::size_t hartree_pair_width = nocc * nvirt;
    const std::size_t hartree_block_limit =
        static_cast<std::size_t>(options_.bse_ri_batch_blocks);
    const std::vector<int> &hartree_row_k = lr_.k1_indices;
    const std::vector<int> &hartree_column_k = lr_.k2_indices;

    // One batch = rows x columns k-pair blocks with rows * columns <= limit.
    // Many k1 rows per batch keep the per-batch work large and reuse D(k2).
    std::size_t hartree_rows_per_batch = 0;
    std::size_t hartree_columns_per_batch = 0;
    unsigned long long hartree_row_groups = 0;
    unsigned long long hartree_batches = 0;
    if (!hartree_row_k.empty() && !hartree_column_k.empty())
    {
        hartree_rows_per_batch = std::min(hartree_row_k.size(), hartree_block_limit);
        hartree_columns_per_batch = std::min(
            hartree_column_k.size(),
            std::max<std::size_t>(1, hartree_block_limit / hartree_rows_per_batch));
        hartree_row_groups =
            (hartree_row_k.size() + hartree_rows_per_batch - 1) / hartree_rows_per_batch;
        const unsigned long long hartree_column_groups =
            (hartree_column_k.size() + hartree_columns_per_batch - 1)
            / hartree_columns_per_batch;
        hartree_batches = hartree_row_groups * hartree_column_groups;
    }
    unsigned long long hartree_rounds = 0;
    MPI_Allreduce(&hartree_batches, &hartree_rounds, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                  descriptor.comm());
    int rank = 0;
    MPI_Comm_rank(descriptor.comm(), &rank);
    if (rank == 0)
        std::cout << "RI streaming: Hartree " << hartree_rows_per_batch << " k1 x "
                  << hartree_columns_per_batch << " k2 blocks per batch on rank 0, "
                  << hartree_rounds << " collective batches" << std::endl;

    const AuxiliaryLayout hartree_row_layout =
        build_auxiliary_layout(lr_.list_I, lr_.Csk_ao_mo, lr_.k_indices);
    const AuxiliaryLayout hartree_column_layout =
        build_auxiliary_layout(lr_.list_J, lr_.Csk_ao_mo, lr_.k_indices);

    std::vector<Complex> hartree_coulomb_q0;
    auto hartree_coulomb_q0_memory = watch_memory("hartree_coulomb_q0", hartree_coulomb_q0);
    std::vector<Complex> hartree_column_coefficients;
    auto hartree_column_coefficients_memory =
        watch_memory("hartree_column_coefficients", hartree_column_coefficients);
    std::vector<Complex> hartree_contracted_columns;
    auto hartree_contracted_columns_memory =
        watch_memory("hartree_contracted_columns", hartree_contracted_columns);
    std::vector<Complex> hartree_row_coefficients;
    auto hartree_row_coefficients_memory =
        watch_memory("hartree_row_coefficients", hartree_row_coefficients);
    KMatrixMap hartree_batch_blocks;
    auto hartree_batch_blocks_memory = watch_memory("hartree_batch_blocks", hartree_batch_blocks);

    const double hartree_start = MPI_Wtime();
    double hartree_seconds_interaction = 0.0;
    double hartree_seconds_coefficients = 0.0;
    double hartree_seconds_gemm = 0.0;
    double hartree_seconds_exchange = 0.0;

    if (hartree_batches > 0)
    {
        // V_PQ(q=0) = sum_R V_PQ(R), built once for the whole stage.
        const double hartree_interaction_start = MPI_Wtime();
        build_dense_interaction_q(lr_.lrik, "Vs_", KPoint{0.0, 0.0, 0.0}, false,
                                  hartree_row_layout, hartree_column_layout,
                                  hartree_coulomb_q0);
        hartree_seconds_interaction += MPI_Wtime() - hartree_interaction_start;
    }
    MemoryTracker::instance().checkpoint();

    long long hartree_cached_column_group = -1;
    const std::size_t hartree_auxiliary_rows = hartree_row_layout.total;
    const std::size_t hartree_auxiliary_columns = hartree_column_layout.total;
    for (unsigned long long hartree_batch = 0; hartree_batch < hartree_rounds; ++hartree_batch)
    {
        hartree_batch_blocks.clear();
        if (hartree_batch < hartree_batches)
        {
            const unsigned long long hartree_column_group = hartree_batch / hartree_row_groups;
            const unsigned long long hartree_row_group = hartree_batch % hartree_row_groups;
            const std::size_t hartree_column_begin =
                hartree_column_group * hartree_columns_per_batch;
            const std::size_t hartree_column_end = std::min(
                hartree_column_begin + hartree_columns_per_batch, hartree_column_k.size());
            const std::size_t hartree_row_begin = hartree_row_group * hartree_rows_per_batch;
            const std::size_t hartree_row_end =
                std::min(hartree_row_begin + hartree_rows_per_batch, hartree_row_k.size());
            const std::size_t hartree_contracted_size =
                hartree_auxiliary_rows * hartree_pair_width;

            if (static_cast<long long>(hartree_column_group) != hartree_cached_column_group)
            {
                hartree_contracted_columns.assign(
                    (hartree_column_end - hartree_column_begin) * hartree_contracted_size,
                    Complex(0.0, 0.0));
                for (std::size_t hartree_column = hartree_column_begin;
                     hartree_column < hartree_column_end; ++hartree_column)
                {
                    const int hartree_k2 = hartree_column_k[hartree_column];
                    const double hartree_coefficients_start = MPI_Wtime();
                    // C^Q_{jb}(k2) (conjugated for A), rows over list_J
                    build_stacked_mo_coefficients(
                        lr_.Csk_ao_mo, lr_.map_psi, hartree_column_layout,
                        hartree_k2, hartree_k2, "O", "V", nocc, nvirt, is_a,
                        hartree_column_coefficients);
                    hartree_seconds_coefficients += MPI_Wtime() - hartree_coefficients_start;
                    const double hartree_gemm_start = MPI_Wtime();
                    // D[P][jb] = sum_Q V_PQ(0) C^Q_{jb}(k2)
                    row_major_gemm('N', 'N', hartree_auxiliary_rows, hartree_pair_width,
                                   hartree_auxiliary_columns, hartree_coulomb_q0.data(),
                                   hartree_column_coefficients.data(),
                                   hartree_contracted_columns.data()
                                       + (hartree_column - hartree_column_begin)
                                             * hartree_contracted_size);
                    hartree_seconds_gemm += MPI_Wtime() - hartree_gemm_start;
                }
                hartree_cached_column_group = static_cast<long long>(hartree_column_group);
            }

            for (std::size_t hartree_row = hartree_row_begin; hartree_row < hartree_row_end;
                 ++hartree_row)
            {
                const int hartree_k1 = hartree_row_k[hartree_row];
                const double hartree_coefficients_start = MPI_Wtime();
                // C^P_{ia}(k1), rows over list_I
                build_stacked_mo_coefficients(
                    lr_.Csk_ao_mo, lr_.map_psi, hartree_row_layout,
                    hartree_k1, hartree_k1, "O", "V", nocc, nvirt, false,
                    hartree_row_coefficients);
                hartree_seconds_coefficients += MPI_Wtime() - hartree_coefficients_start;
                for (std::size_t hartree_column = hartree_column_begin;
                     hartree_column < hartree_column_end; ++hartree_column)
                {
                    const int hartree_k2 = hartree_column_k[hartree_column];
                    RI::Tensor<Complex> hartree_block({nocc, nvirt, nocc, nvirt});
                    const double hartree_gemm_start = MPI_Wtime();
                    // A[jb][ia] = sum_P D[P][jb] C^P_{ia}(k1)
                    row_major_gemm('T', 'N', hartree_pair_width, hartree_pair_width,
                                   hartree_auxiliary_rows,
                                   hartree_contracted_columns.data()
                                       + (hartree_column - hartree_column_begin)
                                             * hartree_contracted_size,
                                   hartree_row_coefficients.data(), hartree_block.ptr());
                    hartree_seconds_gemm += MPI_Wtime() - hartree_gemm_start;
                    hartree_batch_blocks[hartree_k1][hartree_k2] = std::move(hartree_block);
                }
            }
        }
        MemoryTracker::instance().checkpoint();
        // Empty ranks participate too. Every rank uses the same global rounds.
        const double hartree_exchange_start = MPI_Wtime();
        transform_k_2dlocal(matrix, hartree_batch_blocks, descriptor, nk_, pair_dimension_,
                            coefficient, nk_);
        hartree_seconds_exchange += MPI_Wtime() - hartree_exchange_start;
        if (rank == 0)
            report_batch_progress("Hartree", hartree_batch, hartree_rounds, hartree_start);
    }
    hartree_batch_blocks.clear();
    if (rank == 0)
        std::cout << "RI streaming Hartree seconds on rank 0: V(q=0) "
                  << hartree_seconds_interaction << ", MO coefficients "
                  << hartree_seconds_coefficients << ", gemm " << hartree_seconds_gemm
                  << ", block exchange " << hartree_seconds_exchange << std::endl;
}

void MolecularLri::add_batched_screened(std::vector<Complex> &matrix,
                                        const librpa_int::ArrayDesc &descriptor,
                                        double coefficient, bool is_a)
{
    // Screened kernel of this rank's atom pairs (summed over ranks later):
    //   A: A^W_{ia,jb}(k1,k2) = sum_{P in I, Q in J} C^P_{j*i}[k2,k1] W_PQ(q) C^Q_{b a*}[k2,k1]
    //   B: B^W_{ia,jb}(k1,k2) = sum_{P in I, Q in J} C^P_{b*i}[k2,k1] W_PQ(q) C^Q_{j a*}[k2,k1]
    // with q the canonical fine-mesh transfer of (k1,k2), as in LibRI
    // cal_cvc_mo_k_onthefly. Per k pair:
    //   D[P][(m3 m4)] = sum_Q W_PQ(q) C^Q_{m3 m4}
    //   T[(m1 m2)][(m3 m4)] = sum_P C^P_{m1 m2} D[P][(m3 m4)]
    // and T is reordered into the (j,b,i,a) block layout.
    const std::size_t nocc = options_.nocc;
    const std::size_t nvirt = options_.nvirt;
    const std::size_t screened_block_limit =
        static_cast<std::size_t>(options_.bse_ri_batch_blocks);
    const std::vector<std::string> screened_mo_types = is_a
        ? std::vector<std::string>{"O", "O", "V", "V"}
        : std::vector<std::string>{"V", "O", "O", "V"};
    const std::size_t screened_left_width =
        mo_type_count(screened_mo_types[0], nocc, nvirt)
        * mo_type_count(screened_mo_types[1], nocc, nvirt);
    const std::size_t screened_right_width =
        mo_type_count(screened_mo_types[2], nocc, nvirt)
        * mo_type_count(screened_mo_types[3], nocc, nvirt);

    // k pairs in q order, so each W(q) is assembled once.
    std::vector<KPoint> screened_pair_q;
    auto screened_pair_q_memory = watch_memory("screened_pair_q", screened_pair_q);
    std::vector<std::pair<int, int>> screened_pair_k;
    auto screened_pair_k_memory = watch_memory("screened_pair_k", screened_pair_k);
    for (const auto &[screened_q, screened_q_pairs] : lr_.q2kpair)
        for (const auto &screened_k_pair : screened_q_pairs)
        {
            screened_pair_q.push_back(screened_q);
            screened_pair_k.push_back(screened_k_pair);
        }
    const unsigned long long screened_batches =
        (screened_pair_k.size() + screened_block_limit - 1) / screened_block_limit;
    unsigned long long screened_rounds = 0;
    MPI_Allreduce(&screened_batches, &screened_rounds, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                  descriptor.comm());
    int rank = 0;
    MPI_Comm_rank(descriptor.comm(), &rank);
    if (rank == 0)
        std::cout << "RI streaming: screened max " << screened_block_limit
                  << " k-pair blocks per batch, " << lr_.q2kpair.size()
                  << " q on rank 0, " << screened_rounds << " collective batches"
                  << std::endl;

    const AuxiliaryLayout screened_row_layout =
        build_auxiliary_layout(lr_.list_I, lr_.Csk_ao_mo, lr_.k_indices);
    const AuxiliaryLayout screened_column_layout =
        build_auxiliary_layout(lr_.list_J, lr_.Csk_ao_mo, lr_.k_indices);
    const std::size_t screened_auxiliary_rows = screened_row_layout.total;
    const std::size_t screened_auxiliary_columns = screened_column_layout.total;

    std::vector<Complex> screened_interaction_q;
    auto screened_interaction_q_memory =
        watch_memory("screened_interaction_q", screened_interaction_q);
    std::vector<Complex> screened_left_coefficients;
    auto screened_left_coefficients_memory =
        watch_memory("screened_left_coefficients", screened_left_coefficients);
    std::vector<Complex> screened_right_coefficients;
    auto screened_right_coefficients_memory =
        watch_memory("screened_right_coefficients", screened_right_coefficients);
    std::vector<Complex> screened_contracted_right;
    auto screened_contracted_right_memory =
        watch_memory("screened_contracted_right", screened_contracted_right);
    std::vector<Complex> screened_pair_product;
    auto screened_pair_product_memory =
        watch_memory("screened_pair_product", screened_pair_product);
    KMatrixMap screened_batch_blocks;
    auto screened_batch_blocks_memory =
        watch_memory("screened_batch_blocks", screened_batch_blocks);

    const double screened_start = MPI_Wtime();
    double screened_seconds_interaction = 0.0;
    double screened_seconds_coefficients = 0.0;
    double screened_seconds_gemm = 0.0;
    double screened_seconds_exchange = 0.0;
    bool screened_has_cached_q = false;
    KPoint screened_cached_q{0.0, 0.0, 0.0};

    for (unsigned long long screened_batch = 0; screened_batch < screened_rounds;
         ++screened_batch)
    {
        screened_batch_blocks.clear();
        if (screened_batch < screened_batches)
        {
            const std::size_t screened_pair_begin = screened_batch * screened_block_limit;
            const std::size_t screened_pair_end =
                std::min(screened_pair_begin + screened_block_limit, screened_pair_k.size());
            for (std::size_t screened_pair = screened_pair_begin;
                 screened_pair < screened_pair_end; ++screened_pair)
            {
                const KPoint &screened_q = screened_pair_q[screened_pair];
                const int screened_k1 = screened_pair_k[screened_pair].first;
                const int screened_k2 = screened_pair_k[screened_pair].second;
                if (!screened_has_cached_q || screened_q != screened_cached_q)
                {
                    const double screened_interaction_start = MPI_Wtime();
                    // W_PQ(q) = sum_R exp(i 2 pi q.R) W_PQ(R) (+ boundary correction)
                    build_dense_interaction_q(lr_.lrik, "Ws_", screened_q, true,
                                              screened_row_layout, screened_column_layout,
                                              screened_interaction_q);
                    screened_seconds_interaction += MPI_Wtime() - screened_interaction_start;
                    screened_cached_q = screened_q;
                    screened_has_cached_q = true;
                }

                const double screened_coefficients_start = MPI_Wtime();
                // A: C^P_{j*i}[k2,k1]; B: C^P_{b*i}[k2,k1] (rows over list_I)
                build_stacked_mo_coefficients(
                    lr_.Csk_ao_mo, lr_.map_psi, screened_row_layout, screened_k2, screened_k1,
                    screened_mo_types[0], screened_mo_types[1], nocc, nvirt, true,
                    screened_left_coefficients);
                // A: C^Q_{b a*}[k2,k1]; B: C^Q_{j a*}[k2,k1] (rows over list_J)
                build_stacked_mo_coefficients(
                    lr_.Csk_ao_mo, lr_.map_psi, screened_column_layout, screened_k2,
                    screened_k1, screened_mo_types[2], screened_mo_types[3], nocc, nvirt,
                    false, screened_right_coefficients);
                screened_seconds_coefficients += MPI_Wtime() - screened_coefficients_start;

                const double screened_gemm_start = MPI_Wtime();
                screened_contracted_right.resize(screened_auxiliary_rows * screened_right_width);
                screened_pair_product.resize(screened_left_width * screened_right_width);
                // D[P][(m3 m4)] = sum_Q W_PQ(q) C^Q_{m3 m4}
                row_major_gemm('N', 'N', screened_auxiliary_rows, screened_right_width,
                               screened_auxiliary_columns, screened_interaction_q.data(),
                               screened_right_coefficients.data(),
                               screened_contracted_right.data());
                // T[(m1 m2)][(m3 m4)] = sum_P C^P_{m1 m2} D[P][(m3 m4)]
                row_major_gemm('T', 'N', screened_left_width, screened_right_width,
                               screened_auxiliary_rows, screened_left_coefficients.data(),
                               screened_contracted_right.data(), screened_pair_product.data());
                screened_seconds_gemm += MPI_Wtime() - screened_gemm_start;

                RI::Tensor<Complex> screened_block({nocc, nvirt, nocc, nvirt});
                Complex *screened_block_values = screened_block.ptr();
                const Complex *screened_product_values = screened_pair_product.data();
                // Reorder T into the (j,b,i,a) block:
                //   A: block(j,b,i,a) = T[(j,i)][(b,a)]
                //   B: block(j,b,i,a) = T[(b,i)][(j,a)]
                const long long screened_occupied_count = static_cast<long long>(nocc);
#pragma omp parallel for schedule(static)
                for (long long screened_j = 0; screened_j < screened_occupied_count; ++screened_j)
                    for (std::size_t screened_b = 0; screened_b < nvirt; ++screened_b)
                        for (std::size_t screened_i = 0; screened_i < nocc; ++screened_i)
                        {
                            const std::size_t screened_j_index =
                                static_cast<std::size_t>(screened_j);
                            Complex *screened_block_row =
                                screened_block_values
                                + ((screened_j_index * nvirt + screened_b) * nocc + screened_i)
                                      * nvirt;
                            const Complex *screened_product_row = is_a
                                ? screened_product_values
                                      + (screened_j_index * nocc + screened_i)
                                            * screened_right_width
                                      + screened_b * nvirt
                                : screened_product_values
                                      + (screened_b * nocc + screened_i)
                                            * screened_right_width
                                      + screened_j_index * nvirt;
                            for (std::size_t screened_a = 0; screened_a < nvirt; ++screened_a)
                                screened_block_row[screened_a] = screened_product_row[screened_a];
                        }
                RI::LRI_Cal_Aux::add_Ds(std::move(screened_block),
                                        screened_batch_blocks[screened_k1][screened_k2]);
            }
        }
        MemoryTracker::instance().checkpoint();
        // Empty ranks participate too. Every rank uses the same global rounds.
        const double screened_exchange_start = MPI_Wtime();
        transform_k_2dlocal(matrix, screened_batch_blocks, descriptor, nk_, pair_dimension_,
                            coefficient, nk_);
        screened_seconds_exchange += MPI_Wtime() - screened_exchange_start;
        if (rank == 0)
            report_batch_progress("screened", screened_batch, screened_rounds, screened_start);
    }
    screened_batch_blocks.clear();
    if (rank == 0)
        std::cout << "RI streaming screened seconds on rank 0: W(q) "
                  << screened_seconds_interaction << ", MO coefficients "
                  << screened_seconds_coefficients << ", gemm " << screened_seconds_gemm
                  << ", block exchange " << screened_seconds_exchange << std::endl;
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
