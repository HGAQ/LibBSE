#pragma once

#include "parameter/parameter.h"

#include <mpi.h>

#include <filesystem>

namespace libbse
{

/** Detect the producer recorded in basis_out and resolve automatic options. */
void resolve_input_format(InputParameters &options);

/**
 * Build a non-destructive input view for LibRPA's public file reader.
 *
 * The view aliases the FHI-aims files, maps coulomb_cut_* to the historical
 * LibRPA Coulomb name, and aliases a canonical velocity_matrix created by
 * tools/aims_mommat_to_velocity.py. Velocity is mandatory.
 */
std::filesystem::path prepare_fhi_aims_reader_view(
    MPI_Comm comm, const InputParameters &options);

} // namespace libbse
