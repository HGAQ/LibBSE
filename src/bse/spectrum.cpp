#include "spectrum.h"
#include "interface/librpa_api.h"

#include <RI/ri/Cell_Nearest.h>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;

namespace libbse
{
namespace
{

Complex phase(const librpa_int::Vector3_Order<double> &k,
              const librpa_int::Vector3_Order<int> &r, double sign)
{
    const double argument = sign * 2.0 * PARAM.constants.pi
                            * (k.x * r.x + k.y * r.y + k.z * r.z);
    return std::exp(Complex(0.0, argument));
}

struct PairPartition
{
    int first = 0;
    int count = 0;
};

PairPartition pair_partition(int dimension, int mpi_size, int rank)
{
    const int base = dimension / mpi_size;
    const int remainder = dimension % mpi_size;
    return {rank * base + std::min(rank, remainder),
            base + (rank < remainder ? 1 : 0)};
}

int pair_owner(int pair, int dimension, int mpi_size)
{
    const int base = dimension / mpi_size;
    const int remainder = dimension % mpi_size;
    const int larger_end = (base + 1) * remainder;
    if (pair < larger_end) return pair / (base + 1);
    if (base == 0)
        throw std::logic_error("invalid owner for velocity-matrix pair");
    return remainder + (pair - larger_end) / base;
}

std::size_t velocity_index(int direction, int local_pair, int local_pairs)
{
    return static_cast<std::size_t>(direction) * local_pairs + local_pair;
}

double mean_squared(const std::array<Complex, 3> &dipole)
{
    return (std::norm(dipole[0]) + std::norm(dipole[1])
            + std::norm(dipole[2])) / 3.0;
}

void write_dipoles(const fs::path &file, const std::vector<double> &energies,
                   const std::vector<std::array<Complex, 3>> &dipoles,
                   const std::vector<double> &mean_squared_dipoles)
{
    std::ofstream output(file);
    if (!output) throw std::runtime_error("cannot write " + file.string());
    output << "Transition dipole moment (a.u.)\n";
    output << std::setw(6) << "State" << std::setw(13) << "Energy (eV)"
           << std::setw(15) << "x" << std::setw(23) << "|x|^2"
           << std::setw(19) << "y" << std::setw(23) << "|y|^2"
           << std::setw(19) << "z" << std::setw(23) << "|z|^2"
           << std::setw(13) << "average" << '\n';
    for (std::size_t state = 0; state < energies.size(); ++state)
    {
        output << std::setw(4) << state << std::setw(13) << std::setprecision(6)
               << energies[state] * PARAM.constants.ry_to_ev
               << std::setw(29) << dipoles[state][0]
               << std::setw(13) << std::norm(dipoles[state][0])
               << std::setw(29) << dipoles[state][1]
               << std::setw(13) << std::norm(dipoles[state][1])
               << std::setw(29) << dipoles[state][2]
               << std::setw(13) << std::norm(dipoles[state][2])
               << std::setw(13) << mean_squared_dipoles[state] << '\n';
    }
}

void validate_velocity_mo(const InputParameters &options,
                          const FineVelocityMo &velocity_mo)
{
    const int dimension = velocity_mo.nk * options.nocc * options.nvirt;
    if (velocity_mo.nbands != options.nocc + options.nvirt
        || velocity_mo.first_pair < 0 || velocity_mo.local_pairs < 0
        || velocity_mo.first_pair + velocity_mo.local_pairs > dimension
        || velocity_mo.values.size()
               != static_cast<std::size_t>(3) * velocity_mo.local_pairs
        || velocity_mo.gaps_ha.size()
               != static_cast<std::size_t>(velocity_mo.local_pairs))
        throw std::invalid_argument("invalid fine-grid velocity_mo shape");
    for (const double gap : velocity_mo.gaps_ha)
        if (std::abs(gap) <= PARAM.constants.zero_gap_tolerance)
            throw std::runtime_error(
                "zero KS gap in velocity-gauge spectrum");
}

void validate_velocity_inputs(
    int nstates, const InputParameters &options,
    const FineVelocityMo &velocity_mo,
    const std::vector<Complex> &amplitudes_x,
    const std::vector<Complex> *amplitudes_y)
{
    const int dimension = velocity_mo.nk * options.nocc * options.nvirt;
    if (nstates < 0
        || amplitudes_x.size() != static_cast<std::size_t>(nstates) * dimension
        || (amplitudes_y != nullptr && amplitudes_y->size() != amplitudes_x.size())
        || velocity_mo.first_pair != 0
        || velocity_mo.local_pairs != dimension)
        throw std::invalid_argument("invalid excitation amplitudes for velocity gauge");
    validate_velocity_mo(options, velocity_mo);
}

void validate_distributed_velocity_inputs(
    MPI_Comm comm, const InputParameters &options,
    const FineVelocityMo &velocity_mo,
    const DistributedAmplitudes &amplitudes_x,
    const DistributedAmplitudes *amplitudes_y)
{
    int rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpi_size);
    const int dimension = velocity_mo.nk * options.nocc * options.nvirt;
    const int base = dimension / mpi_size;
    const int remainder = dimension % mpi_size;
    const int expected_first = rank * base + std::min(rank, remainder);
    const int expected_count = base + (rank < remainder ? 1 : 0);
    const auto shape_matches = [&](const DistributedAmplitudes &amplitudes)
    {
        return amplitudes.dimension == dimension
               && amplitudes.nstates > 0
               && amplitudes.first_pair == expected_first
               && amplitudes.local_pairs == expected_count
               && amplitudes.values.size()
                      == static_cast<std::size_t>(amplitudes.nstates)
                             * amplitudes.local_pairs;
    };
    if (!shape_matches(amplitudes_x)
        || (amplitudes_y != nullptr
            && (!shape_matches(*amplitudes_y)
                || amplitudes_y->nstates != amplitudes_x.nstates)))
        throw std::invalid_argument(
            "invalid distributed excitation amplitudes for velocity gauge");
    validate_velocity_mo(options, velocity_mo);
    if (velocity_mo.first_pair != expected_first
        || velocity_mo.local_pairs != expected_count)
        throw std::invalid_argument(
            "velocity_mo and excitation-amplitude ownership do not match");
}

std::array<Complex, 3> transition_dipole_slice(
    int state, int first_pair, int pair_stride,
    const InputParameters &options, const FineVelocityMo &velocity_mo,
    const std::vector<Complex> &amplitudes_x,
    const std::vector<Complex> *amplitudes_y)
{
    const int pair_dimension = options.nocc * options.nvirt;
    const int dimension = velocity_mo.nk * pair_dimension;
    std::array<Complex, 3> result{};
    for (int pair = first_pair; pair < dimension; pair += pair_stride)
    {
        const int velocity_pair = pair - velocity_mo.first_pair;
        if (velocity_pair < 0 || velocity_pair >= velocity_mo.local_pairs)
            throw std::invalid_argument(
                "serial velocity_mo does not cover all excitation pairs");
        const double gap = velocity_mo.gaps_ha[velocity_pair];
        if (std::abs(gap) <= PARAM.constants.zero_gap_tolerance)
            throw std::runtime_error("zero KS gap in velocity-gauge spectrum");
        const auto amplitude_index = static_cast<std::size_t>(state) * dimension + pair;
        for (int direction = 0; direction < 3; ++direction)
        {
            const Complex v = velocity_mo.values[velocity_index(
                direction, velocity_pair, velocity_mo.local_pairs)];
            result[direction] += v * amplitudes_x[amplitude_index] / gap;
            if (amplitudes_y != nullptr)
                result[direction] -= std::conj(v)
                                     * (*amplitudes_y)[amplitude_index] / gap;
        }
    }
    const Complex prefactor(0.0, std::sqrt(2.0));
    for (Complex &value : result) value *= prefactor;
    return result;
}

std::array<Complex, 3> distributed_transition_dipole(
    int state, const InputParameters &options,
    const FineVelocityMo &velocity_mo,
    const DistributedAmplitudes &amplitudes_x,
    const DistributedAmplitudes *amplitudes_y)
{
    std::array<Complex, 3> result{};
    for (int local_index = 0; local_index < amplitudes_x.local_pairs;
         ++local_index)
    {
        const double gap = velocity_mo.gaps_ha[local_index];
        const Complex x = amplitudes_x(state, local_index);
        const Complex y = amplitudes_y == nullptr
                              ? Complex{}
                              : (*amplitudes_y)(state, local_index);
        for (int direction = 0; direction < 3; ++direction)
        {
            const Complex v = velocity_mo.values[velocity_index(
                direction, local_index, velocity_mo.local_pairs)];
            result[direction] += v * x / gap;
            if (amplitudes_y != nullptr)
                result[direction] -= std::conj(v) * y / gap;
        }
    }
    const Complex prefactor(0.0, std::sqrt(2.0));
    for (Complex &value : result) value *= prefactor;
    return result;
}

template <typename T>
void broadcast_vector(std::vector<T> &values, MPI_Datatype datatype,
                      int count, MPI_Comm comm)
{
    if (count < 0
        || static_cast<std::size_t>(count) > values.max_size())
        throw std::overflow_error("invalid MPI broadcast vector size");
    values.resize(static_cast<std::size_t>(count));
    MPI_Bcast(values.data(), count, datatype, 0, comm);
}

} // namespace

std::array<Complex, 3> velocity_gauge_transition_dipole(
    int state, const InputParameters &options, const FineVelocityMo &velocity_mo,
    const std::vector<Complex> &amplitudes_x,
    const std::vector<Complex> *amplitudes_y)
{
    const int pair_dimension = options.nocc * options.nvirt;
    const int dimension = velocity_mo.nk * pair_dimension;
    if (state < 0 || state >= static_cast<int>(amplitudes_x.size() / dimension))
        throw std::invalid_argument("invalid excitation state for velocity gauge");
    validate_velocity_inputs(static_cast<int>(amplitudes_x.size() / dimension),
                             options, velocity_mo, amplitudes_x, amplitudes_y);
    return transition_dipole_slice(state, 0, 1, options, velocity_mo,
                                   amplitudes_x, amplitudes_y);
}

std::vector<std::array<Complex, 3>> velocity_gauge_transition_dipoles_mpi(
    MPI_Comm comm, const InputParameters &options,
    const FineVelocityMo &velocity_mo,
    const DistributedAmplitudes &amplitudes_x,
    const DistributedAmplitudes *amplitudes_y)
{
    validate_distributed_velocity_inputs(
        comm, options, velocity_mo, amplitudes_x, amplitudes_y);
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    const int nstates = amplitudes_x.nstates;
    std::vector<std::array<Complex, 3>> local(static_cast<std::size_t>(nstates));
    for (int state = 0; state < nstates; ++state)
        local[state] = distributed_transition_dipole(
            state, options, velocity_mo, amplitudes_x, amplitudes_y);

    std::vector<std::array<Complex, 3>> result;
    if (rank == 0) result.resize(static_cast<std::size_t>(nstates));
    MPI_Reduce(local.data(), rank == 0 ? result.data() : nullptr,
               3 * nstates, MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, comm);
    return result;
}

FineVelocityMo prepare_fine_velocity_mo(
    const InputParameters &options, const QuasiparticleBands &qp,
    const std::shared_ptr<librpa_int::Dataset> &dataset)
{
    int rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(dataset->comm_h.comm, &rank);
    MPI_Comm_size(dataset->comm_h.comm, &mpi_size);
    const int nao = dataset->mf.get_n_aos();
    const int coarse_nk = dataset->mf.get_n_kpoints();
    const int fine_nk = dataset->mf_band.get_n_kpoints();
    const int pair_dimension = options.nocc * options.nvirt;
    const int dimension = fine_nk * pair_dimension;
    if (dataset->velocity_matrix.size() != 1
        || static_cast<int>(dataset->velocity_matrix[0].size()) != coarse_nk)
        throw std::runtime_error("coarse-grid velocity matrix was not loaded");

    const int last_band = qp.ncore + options.nocc + options.nvirt;
    const auto &eigenvalues = dataset->mf_band.get_eigenvals()[0];
    if (eigenvalues.nr != fine_nk || eigenvalues.nc < last_band)
        throw std::runtime_error(
            "fine-grid eigenvalues do not contain the selected BSE bands");

    const PairPartition partition = pair_partition(dimension, mpi_size, rank);
    FineVelocityMo result;
    result.nk = fine_nk;
    result.nbands = options.nocc + options.nvirt;
    result.first_pair = partition.first;
    result.local_pairs = partition.count;
    result.values.assign(static_cast<std::size_t>(3) * partition.count,
                         Complex{});
    result.gaps_ha.resize(static_cast<std::size_t>(partition.count));
    for (int local_pair = 0; local_pair < partition.count; ++local_pair)
    {
        const int pair = partition.first + local_pair;
        const int ik = pair / pair_dimension;
        const int pair_in_k = pair % pair_dimension;
        const int i = pair_in_k / options.nvirt;
        const int a = pair_in_k % options.nvirt;
        result.gaps_ha[local_pair]
            = eigenvalues(ik, qp.ncore + options.nocc + a)
              - eigenvalues(ik, qp.ncore + i);
    }

    bool grids_match = coarse_nk == fine_nk;
    for (int ik = 0; grids_match && ik < coarse_nk; ++ik)
    {
        const auto &coarse = dataset->pbc.kfrac_list[ik];
        const auto &fine = dataset->kfrac_band_list[ik];
        grids_match = std::abs(coarse.x - fine.x) <= PARAM.constants.kpoint_tolerance
                      && std::abs(coarse.y - fine.y) <= PARAM.constants.kpoint_tolerance
                      && std::abs(coarse.z - fine.z) <= PARAM.constants.kpoint_tolerance;
    }
    if (grids_match)
    {
        for (int local_pair = 0; local_pair < partition.count; ++local_pair)
        {
            const int pair = partition.first + local_pair;
            const int ik = pair / pair_dimension;
            const int pair_in_k = pair % pair_dimension;
            const int i = pair_in_k / options.nvirt;
            const int a = pair_in_k % options.nvirt;
            for (int direction = 0; direction < 3; ++direction)
            {
                const auto &matrix = dataset->velocity_matrix[0][ik][direction];
                if (matrix.nr < last_band || matrix.nc < last_band)
                    throw std::runtime_error(
                        "velocity_mo does not contain the selected BSE bands");
                // velocity_mo stores <column|v|row>.
                result.values[velocity_index(
                    direction, local_pair, partition.count)]
                    = matrix(qp.ncore + options.nocc + a, qp.ncore + i);
            }
        }
        return result;
    }

    if (dataset->mf.get_n_states() != nao)
        throw std::runtime_error("velocity interpolation requires a complete square coarse KS basis");
    if (static_cast<int>(dataset->pbc.Rlist.size()) != coarse_nk)
        throw std::runtime_error("coarse k/R grids are incompatible for velocity interpolation");

    std::vector<std::array<librpa_int::ComplexMatrix, 3>> velocity_ao_k(
        static_cast<std::size_t>(coarse_nk));
    for (int ik = 0; ik < coarse_nk; ++ik)
    {
        const auto *wavefunctions = dataset->mf.find_wfc(0, 0, ik);
        if (wavefunctions == nullptr || wavefunctions->nr != nao || wavefunctions->nc != nao)
            throw std::runtime_error("coarse-grid KS wavefunction is missing or incomplete");
        const auto inverse_wfc = LibRPA_API::inverse(*wavefunctions);
        const auto inverse_bra = LibRPA_API::conjugate(inverse_wfc);
        const auto inverse_ket = LibRPA_API::transpose(inverse_wfc);
        for (int direction = 0; direction < 3; ++direction)
        {
            // velocity_mo[p,q] as <q|v|p> (and its
            // spectrum code consequently addresses the virtual,occupied
            // element).  Convert that storage convention back to the usual
            // <p|v|q> matrix before undoing the AO-to-MO transformation.
            const auto velocity_mo = LibRPA_API::transpose(
                dataset->velocity_matrix[0][ik][direction]);
            velocity_ao_k[ik][direction]
                = inverse_bra * velocity_mo * inverse_ket;
        }
    }

    std::vector<std::array<librpa_int::ComplexMatrix, 3>> velocity_ao_r(
        dataset->pbc.Rlist.size());
    for (std::size_t ir = 0; ir < dataset->pbc.Rlist.size(); ++ir)
    {
        for (int direction = 0; direction < 3; ++direction)
            velocity_ao_r[ir][direction].create(nao, nao);
        for (int ik = 0; ik < coarse_nk; ++ik)
        {
            const Complex factor = phase(dataset->pbc.kfrac_list[ik],
                                         dataset->pbc.Rlist[ir], -1.0)
                                   / static_cast<double>(coarse_nk);
            for (int direction = 0; direction < 3; ++direction)
                LibRPA_API::scale_accumulate(factor, velocity_ao_k[ik][direction],
                                             velocity_ao_r[ir][direction]);
        }
    }
    velocity_ao_k.clear();

    // The inverse FFT returns one representative of each coarse-grid BvK
    // equivalence class.  The physically local representative depends on the
    // two atoms carrying an AO matrix element; this is the same nearest-cell
    // remapping used by ABACUS/LibRI for real-space RI tensors.
    std::map<int, std::array<double, 3>> atom_positions;
    for (const auto &[iat, position] : dataset->atoms.coords)
        atom_positions[static_cast<int>(iat)] = {position.x, position.y, position.z};
    RI::Cell_Nearest<int, int, 3, double, 3> nearest;
    nearest.init(atom_positions, dataset->pbc.latvec_array, dataset->pbc.period_array);
    std::vector<std::vector<std::vector<Cell>>> nearest_cells(
        dataset->pbc.Rlist.size(),
        std::vector<std::vector<Cell>>(
            dataset->atoms.size(), std::vector<Cell>(dataset->atoms.size())));
    for (std::size_t ir = 0; ir < dataset->pbc.Rlist.size(); ++ir)
    {
        const Cell canonical{dataset->pbc.Rlist[ir].x,
                             dataset->pbc.Rlist[ir].y,
                             dataset->pbc.Rlist[ir].z};
        for (int iat = 0; iat < static_cast<int>(dataset->atoms.size()); ++iat)
            for (int jat = 0; jat < static_cast<int>(dataset->atoms.size()); ++jat)
            {
                double distance = 0.0;
                nearest_cells[ir][iat][jat]
                    = nearest.cell_nearest_direction(iat, jat, canonical, distance);
            }
    }

    std::vector<int> local_sources(static_cast<std::size_t>(fine_nk), mpi_size);
    for (int ik = 0; ik < fine_nk; ++ik)
        if (dataset->mf_band.find_wfc(0, 0, ik) != nullptr)
            local_sources[ik] = rank;
    std::vector<int> sources(static_cast<std::size_t>(fine_nk), mpi_size);
    MPI_Allreduce(local_sources.data(), sources.data(), fine_nk, MPI_INT,
                  MPI_MIN, dataset->comm_h.comm);
    for (int ik = 0; ik < fine_nk; ++ik)
        if (sources[ik] == mpi_size)
            throw std::runtime_error(
                "fine-grid KS wavefunction is absent on every MPI rank");

    std::vector<int> send_pair_counts(static_cast<std::size_t>(mpi_size), 0);
    for (int ik = 0; ik < fine_nk; ++ik)
        if (sources[ik] == rank)
            for (int pair_in_k = 0; pair_in_k < pair_dimension; ++pair_in_k)
            {
                const int pair = ik * pair_dimension + pair_in_k;
                ++send_pair_counts[pair_owner(pair, dimension, mpi_size)];
            }
    std::vector<int> receive_pair_counts(static_cast<std::size_t>(mpi_size), 0);
    MPI_Alltoall(send_pair_counts.data(), 1, MPI_INT,
                 receive_pair_counts.data(), 1, MPI_INT,
                 dataset->comm_h.comm);

    std::vector<int> send_pair_offsets(static_cast<std::size_t>(mpi_size), 0);
    std::vector<int> receive_pair_offsets(static_cast<std::size_t>(mpi_size), 0);
    for (int process = 1; process < mpi_size; ++process)
    {
        send_pair_offsets[process] = send_pair_offsets[process - 1]
                                     + send_pair_counts[process - 1];
        receive_pair_offsets[process] = receive_pair_offsets[process - 1]
                                        + receive_pair_counts[process - 1];
    }
    const int total_send = std::accumulate(
        send_pair_counts.begin(), send_pair_counts.end(), 0);
    const int total_receive = std::accumulate(
        receive_pair_counts.begin(), receive_pair_counts.end(), 0);
    if (total_receive != partition.count)
        throw std::runtime_error(
            "incomplete distributed velocity_mo ownership");
    if (total_send > std::numeric_limits<int>::max() / 3
        || total_receive > std::numeric_limits<int>::max() / 3)
        throw std::overflow_error(
            "distributed velocity_mo exchange exceeds the MPI count limit");

    std::vector<int> send_pairs(static_cast<std::size_t>(total_send));
    std::vector<Complex> send_values(static_cast<std::size_t>(3) * total_send);
    std::vector<int> send_cursor = send_pair_offsets;
    for (int ik = 0; ik < fine_nk; ++ik)
    {
        if (sources[ik] != rank) continue;
        const auto *wavefunctions = dataset->mf_band.find_wfc(0, 0, ik);
        if (wavefunctions->nr != dataset->mf_band.get_n_states()
            || wavefunctions->nc != nao)
            throw std::runtime_error("fine-grid KS wavefunction is incomplete");

        std::vector<int> pair_slots(static_cast<std::size_t>(pair_dimension));
        for (int pair_in_k = 0; pair_in_k < pair_dimension; ++pair_in_k)
        {
            const int pair = ik * pair_dimension + pair_in_k;
            const int destination = pair_owner(pair, dimension, mpi_size);
            const int slot = send_cursor[destination]++;
            send_pairs[slot] = pair;
            pair_slots[pair_in_k] = slot;
        }
        for (int direction = 0; direction < 3; ++direction)
        {
            librpa_int::ComplexMatrix velocity_ao(nao, nao);
            for (std::size_t ir = 0; ir < dataset->pbc.Rlist.size(); ++ir)
                for (int row = 0; row < nao; ++row)
                    for (int column = 0; column < nao; ++column)
                    {
                        const int iat = dataset->basis_wfc.get_i_atom(row);
                        const int jat = dataset->basis_wfc.get_i_atom(column);
                        const auto &cell = nearest_cells[ir][iat][jat];
                        const librpa_int::Vector3_Order<int> r(cell[0], cell[1], cell[2]);
                        velocity_ao(row, column)
                            += phase(dataset->kfrac_band_list[ik], r, 1.0)
                               * velocity_ao_r[ir][direction](row, column);
                    }
            const auto velocity_mo = LibRPA_API::conjugate(*wavefunctions) * velocity_ao
                                     * LibRPA_API::transpose(*wavefunctions);
            for (int i = 0; i < options.nocc; ++i)
                for (int a = 0; a < options.nvirt; ++a)
                {
                    const int pair_in_k = i * options.nvirt + a;
                    const int slot = pair_slots[pair_in_k];
                    send_values[static_cast<std::size_t>(3) * slot + direction]
                        = velocity_mo(qp.ncore + i,
                                      qp.ncore + options.nocc + a);
                }
        }
    }
    for (int process = 0; process < mpi_size; ++process)
        if (send_cursor[process]
            != send_pair_offsets[process] + send_pair_counts[process])
            throw std::runtime_error(
                "inconsistent distributed velocity_mo packing");

    std::vector<int> receive_pairs(static_cast<std::size_t>(total_receive));
    std::vector<Complex> receive_values(
        static_cast<std::size_t>(3) * total_receive);
    MPI_Alltoallv(send_pairs.data(), send_pair_counts.data(),
                  send_pair_offsets.data(), MPI_INT,
                  receive_pairs.data(), receive_pair_counts.data(),
                  receive_pair_offsets.data(), MPI_INT,
                  dataset->comm_h.comm);
    std::vector<int> send_value_counts(static_cast<std::size_t>(mpi_size));
    std::vector<int> receive_value_counts(static_cast<std::size_t>(mpi_size));
    std::vector<int> send_value_offsets(static_cast<std::size_t>(mpi_size));
    std::vector<int> receive_value_offsets(static_cast<std::size_t>(mpi_size));
    for (int process = 0; process < mpi_size; ++process)
    {
        send_value_counts[process] = 3 * send_pair_counts[process];
        receive_value_counts[process] = 3 * receive_pair_counts[process];
        send_value_offsets[process] = 3 * send_pair_offsets[process];
        receive_value_offsets[process] = 3 * receive_pair_offsets[process];
    }
    MPI_Alltoallv(send_values.data(), send_value_counts.data(),
                  send_value_offsets.data(), MPI_C_DOUBLE_COMPLEX,
                  receive_values.data(), receive_value_counts.data(),
                  receive_value_offsets.data(), MPI_C_DOUBLE_COMPLEX,
                  dataset->comm_h.comm);

    std::vector<unsigned char> assigned(
        static_cast<std::size_t>(partition.count), 0);
    for (int received = 0; received < total_receive; ++received)
    {
        const int local_pair = receive_pairs[received] - partition.first;
        if (local_pair < 0 || local_pair >= partition.count
            || assigned[local_pair] != 0)
            throw std::runtime_error(
                "invalid distributed velocity_mo pair ownership");
        assigned[local_pair] = 1;
        for (int direction = 0; direction < 3; ++direction)
            result.values[velocity_index(
                direction, local_pair, partition.count)]
                = receive_values[static_cast<std::size_t>(3) * received
                                 + direction];
    }
    return result;
}

void write_velocity_gauge_outputs(
    const InputParameters &options,
    const librpa_int::Dataset &dataset,
    const FineVelocityMo &velocity,
    const std::vector<double> &energies_ry,
    const DistributedAmplitudes &amplitudes_x,
    const DistributedAmplitudes *amplitudes_y,
    const std::string &spin_type,
    const std::string &solution_type)
{
    int rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(dataset.comm_h.comm, &rank);
    MPI_Comm_size(dataset.comm_h.comm, &mpi_size);
    int nstates = rank == 0 ? static_cast<int>(energies_ry.size()) : 0;
    MPI_Bcast(&nstates, 1, MPI_INT, 0, dataset.comm_h.comm);
    const int pair_dimension = options.nocc * options.nvirt;
    const int dimension = velocity.nk * pair_dimension;
    if (amplitudes_x.dimension != dimension
        || amplitudes_x.nstates != nstates
        || (amplitudes_y != nullptr
            && (amplitudes_y->dimension != dimension
                || amplitudes_y->nstates != nstates)))
        throw std::invalid_argument(
            "excitation energies and distributed amplitudes do not match");

    std::vector<double> all_energies = energies_ry;
    broadcast_vector(all_energies, MPI_DOUBLE, nstates, dataset.comm_h.comm);

    auto dipoles = velocity_gauge_transition_dipoles_mpi(
        dataset.comm_h.comm, options, velocity,
        amplitudes_x, amplitudes_y);
    std::vector<double> local_weight1(static_cast<std::size_t>(velocity.nk), 0.0);
    std::vector<double> local_weight2(static_cast<std::size_t>(velocity.nk), 0.0);
    std::vector<long long> local_contribution_indices;
    std::vector<Complex> local_contribution_values;
    for (int state = 0; state < nstates; ++state)
        for (int local_index = 0;
             local_index < amplitudes_x.local_pairs; ++local_index)
        {
            const int pair = amplitudes_x.first_pair + local_index;
            const int ik = pair / pair_dimension;
            const Complex x = amplitudes_x(state, local_index);
            const Complex y = amplitudes_y == nullptr
                                  ? Complex{}
                                  : (*amplitudes_y)(state, local_index);
            local_weight1[ik] += std::norm(x) + std::norm(y);
            if (std::abs(x) > 0.3)
            {
                local_contribution_indices.push_back(
                    static_cast<long long>(state) * dimension + pair);
                local_contribution_values.push_back(x);
            }
            const double gap = velocity.gaps_ha[local_index];
            if (std::abs(gap) <= PARAM.constants.zero_gap_tolerance)
                throw std::runtime_error("zero KS gap in velocity-gauge spectrum");
            for (int direction = 0; direction < 3; ++direction)
            {
                const Complex v = velocity.values[velocity_index(
                    direction, local_index, velocity.local_pairs)];
                local_weight2[ik] += 2.0 * std::norm(v * x / gap)
                                     + 2.0 * std::norm(std::conj(v) * y / gap);
            }
        }
    std::vector<double> weight1(static_cast<std::size_t>(velocity.nk), 0.0);
    std::vector<double> weight2(static_cast<std::size_t>(velocity.nk), 0.0);
    MPI_Reduce(local_weight1.data(), rank == 0 ? weight1.data() : nullptr,
               velocity.nk, MPI_DOUBLE, MPI_SUM, 0, dataset.comm_h.comm);
    MPI_Reduce(local_weight2.data(), rank == 0 ? weight2.data() : nullptr,
               velocity.nk, MPI_DOUBLE, MPI_SUM, 0, dataset.comm_h.comm);

    const int local_count_valid = local_contribution_indices.size()
                                           <= static_cast<std::size_t>(
                                               std::numeric_limits<int>::max())
                                       ? 1
                                       : 0;
    int all_counts_valid = 0;
    MPI_Allreduce(&local_count_valid, &all_counts_valid, 1, MPI_INT,
                  MPI_MIN, dataset.comm_h.comm);
    if (!all_counts_valid)
        throw std::overflow_error(
            "local transition analysis exceeds the MPI count limit");
    const int local_contribution_count =
        static_cast<int>(local_contribution_indices.size());
    std::vector<int> contribution_counts;
    if (rank == 0) contribution_counts.resize(static_cast<std::size_t>(mpi_size));
    MPI_Gather(&local_contribution_count, 1, MPI_INT,
               rank == 0 ? contribution_counts.data() : nullptr,
               1, MPI_INT, 0, dataset.comm_h.comm);
    std::vector<int> contribution_offsets;
    std::vector<long long> contribution_indices;
    std::vector<Complex> contribution_values;
    int total_count_valid = 1;
    if (rank == 0)
    {
        contribution_offsets.resize(static_cast<std::size_t>(mpi_size), 0);
        long long total = 0;
        for (int process = 0; process < mpi_size; ++process)
        {
            if (total > std::numeric_limits<int>::max())
                total_count_valid = 0;
            if (total_count_valid)
                contribution_offsets[process] = static_cast<int>(total);
            total += contribution_counts[process];
        }
        if (total > std::numeric_limits<int>::max())
            total_count_valid = 0;
        if (total_count_valid)
        {
            contribution_indices.resize(static_cast<std::size_t>(total));
            contribution_values.resize(static_cast<std::size_t>(total));
        }
    }
    MPI_Bcast(&total_count_valid, 1, MPI_INT, 0, dataset.comm_h.comm);
    if (!total_count_valid)
        throw std::overflow_error(
            "transition analysis exceeds the MPI count limit");
    MPI_Gatherv(local_contribution_indices.data(), local_contribution_count,
                MPI_LONG_LONG,
                rank == 0 ? contribution_indices.data() : nullptr,
                rank == 0 ? contribution_counts.data() : nullptr,
                rank == 0 ? contribution_offsets.data() : nullptr,
                MPI_LONG_LONG, 0, dataset.comm_h.comm);
    MPI_Gatherv(local_contribution_values.data(), local_contribution_count,
                MPI_C_DOUBLE_COMPLEX,
                rank == 0 ? contribution_values.data() : nullptr,
                rank == 0 ? contribution_counts.data() : nullptr,
                rank == 0 ? contribution_offsets.data() : nullptr,
                MPI_C_DOUBLE_COMPLEX, 0, dataset.comm_h.comm);
    if (rank != 0) return;

    std::vector<double> means(static_cast<std::size_t>(nstates));
    std::vector<double> oscillator_strengths(static_cast<std::size_t>(nstates));
    for (int state = 0; state < nstates; ++state)
    {
        means[state] = mean_squared(dipoles[state]);
        oscillator_strengths[state] = 2.0 * all_energies[state] * means[state];
    }
    const double oscillator_sum = std::accumulate(
        oscillator_strengths.begin(), oscillator_strengths.end(), 0.0)
                                  / (4.0 * velocity.nk * options.nocc);
    const std::string label = spin_type + "_" + solution_type;
    std::cout << "Total oscillator strength (" << label << ") = "
              << std::setprecision(10) << oscillator_sum << '\n';

    const fs::path output_dir(options.output_dir);
    if (spin_type != "triplet")
        write_dipoles(output_dir / ("trans_dipole_" + label + ".dat"),
                      all_energies, dipoles, means);

    std::ofstream analysis(output_dir / ("trans_analysis_" + label + ".dat"));
    std::ofstream kweight(output_dir / ("trans_kweight_" + label + ".dat"));
    if (!analysis || !kweight) throw std::runtime_error("cannot write transition analysis");
    analysis << "==================================================================== \n"
             << std::setw(40) << label << '\n'
             << "==================================================================== \n"
             << std::setw(8) << "State" << std::setw(30) << "Excitation Energy (Ry, eV)"
             << std::setw(90) << "Transition dipole x, y, z (a.u.)"
             << std::setw(30) << "Oscillator strength(a.u.)" << '\n'
             << "------------------------------------------------------------------------------------ \n";
    for (int state = 0; state < nstates; ++state)
        analysis << std::setw(8) << state << std::setw(15) << std::setprecision(6)
                 << all_energies[state] << std::setw(15)
                 << all_energies[state] * PARAM.constants.ry_to_ev
                 << std::setprecision(4) << std::setw(30) << dipoles[state][0]
                 << std::setw(30) << dipoles[state][1] << std::setw(30) << dipoles[state][2]
                 << std::setprecision(6) << std::setw(30) << oscillator_strengths[state] << '\n';
    analysis << "------------------------------------------------------------------------------------ \n"
             << std::setw(8) << "State" << std::setw(20) << "Occupied orbital"
             << std::setw(20) << "Virtual orbital" << std::setw(30) << "Excitation amplitude"
             << std::setw(30) << "Excitation rate" << std::setw(10) << "k-point" << '\n'
             << "------------------------------------------------------------------------------------ \n";
    std::vector<std::vector<std::pair<int, Complex>>> contributions_by_state(
        static_cast<std::size_t>(nstates));
    for (std::size_t index = 0; index < contribution_indices.size(); ++index)
    {
        const int state = static_cast<int>(contribution_indices[index]
                                           / dimension);
        const int pair = static_cast<int>(contribution_indices[index]
                                          % dimension);
        if (state < 0 || state >= nstates || pair < 0 || pair >= dimension)
            throw std::runtime_error(
                "invalid distributed transition-analysis index");
        contributions_by_state[state].emplace_back(
            pair, contribution_values[index]);
    }
    for (int state = 0; state < nstates; ++state)
    {
        auto &contributions = contributions_by_state[state];
        std::sort(contributions.begin(), contributions.end(),
                  [](const auto &left, const auto &right)
                  { return std::abs(left.second) > std::abs(right.second); });
        for (auto iterator = contributions.begin(); iterator != contributions.end(); ++iterator)
        {
            const int ik = iterator->first / pair_dimension;
            const int local_pair = iterator->first % pair_dimension;
            const int i = local_pair / options.nvirt;
            const int a = local_pair % options.nvirt;
            analysis << std::setw(8)
                     << (iterator == contributions.begin() ? std::to_string(state) : " ")
                     << std::setw(20) << i + 1 << std::setw(20) << options.nocc + a + 1
                     << std::setw(30) << iterator->second
                     << std::setw(30) << std::norm(iterator->second)
                     << std::setw(10) << ik + 1 << '\n';
        }
    }

    kweight << "# Sum of exciton contribution of k-point.\n"
            << "# weight1(k) = sum_{state,spin,occ,virt} |X(state,spin,k,occ,virt)|^2+|Y(state,spin,k,occ,virt)|^2.\n"
            << "# weight2(k) = sum_{state,spin,direction,occ,virt} |td * X|^2 + |td *Y|^2, where td index as (spin,dir,k,occ,virt).\n"
            << "# \n"
            << "k-point" << std::setw(10) << "kx" << std::setw(12) << "ky"
            << std::setw(12) << "kz" << std::setw(12) << "weight1"
            << std::setw(12) << "weight2" << '\n'
            << std::fixed << std::setprecision(5);
    for (int ik = 0; ik < velocity.nk; ++ik)
    {
        const auto &k = dataset.kfrac_band_list[ik];
        kweight << std::setw(5) << ik + 1 << std::setw(12) << k.x
                << std::setw(12) << k.y << std::setw(12) << k.z
                << std::setw(12) << weight1[ik] << std::setw(12) << weight2[ik] << '\n';
    }
}

} // namespace libbse
