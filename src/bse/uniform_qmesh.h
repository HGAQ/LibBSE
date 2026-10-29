#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <set>
#include <vector>

namespace libbse
{

// Return the dimensions of a complete uniform mesh in fractional coordinates.
// Shifted meshes are allowed. Return {0, 0, 0} if the mesh is invalid.
inline std::array<int, 3> infer_uniform_qmesh(
    const std::vector<std::array<double, 3>> &qmesh_kpoints)
{
    constexpr double qmesh_tolerance = 1.e-10;
    std::array<int, 3> qmesh_shape{};
    std::array<double, 3> qmesh_origin{};
    if (qmesh_kpoints.empty()) return {};

    for (int axis = 0; axis < 3; ++axis)
    {
        // Fold equivalent coordinates into [0, 1) before sorting.
        std::vector<double> qmesh_coordinates;
        for (const auto &qmesh_kpoint : qmesh_kpoints)
        {
            if (!std::isfinite(qmesh_kpoint[axis])) return {};
            double qmesh_coordinate = qmesh_kpoint[axis]
                                      - std::floor(qmesh_kpoint[axis]);
            if (qmesh_coordinate > 1.0 - qmesh_tolerance)
                qmesh_coordinate = 0.0;
            qmesh_coordinates.push_back(qmesh_coordinate);
        }
        std::sort(qmesh_coordinates.begin(), qmesh_coordinates.end());

        std::vector<double> qmesh_unique_coordinates;
        for (const double qmesh_coordinate : qmesh_coordinates)
        {
            if (qmesh_unique_coordinates.empty()
                || qmesh_coordinate - qmesh_unique_coordinates.back()
                       > qmesh_tolerance)
            {
                qmesh_unique_coordinates.push_back(qmesh_coordinate);
            }
        }

        const int qmesh_axis_size = static_cast<int>(qmesh_unique_coordinates.size());
        qmesh_shape[axis] = qmesh_axis_size;
        qmesh_origin[axis] = qmesh_unique_coordinates.front();
        for (int index = 0; index < qmesh_axis_size; ++index)
        {
            const double qmesh_expected_offset = static_cast<double>(index)
                                                 / qmesh_axis_size;
            const double qmesh_actual_offset = qmesh_unique_coordinates[index]
                                               - qmesh_origin[axis];
            if (std::abs(qmesh_actual_offset - qmesh_expected_offset)
                > qmesh_tolerance)
                return {};
        }
    }

    const std::size_t qmesh_size = static_cast<std::size_t>(qmesh_shape[0])
                                  * qmesh_shape[1] * qmesh_shape[2];
    if (qmesh_size != qmesh_kpoints.size()) return {};

    // Every grid site must occur exactly once, including periodic equivalents.
    std::set<std::array<int, 3>> qmesh_occupied;
    for (const auto &qmesh_kpoint : qmesh_kpoints)
    {
        std::array<int, 3> qmesh_index{};
        for (int axis = 0; axis < 3; ++axis)
        {
            const int qmesh_axis_size = qmesh_shape[axis];
            const double qmesh_scaled = (qmesh_kpoint[axis] - qmesh_origin[axis])
                                        * qmesh_axis_size;
            const long long qmesh_rounded = std::llround(qmesh_scaled);
            if (std::abs(qmesh_scaled - qmesh_rounded)
                > qmesh_tolerance * qmesh_axis_size)
                return {};

            long long qmesh_wrapped_index = qmesh_rounded % qmesh_axis_size;
            if (qmesh_wrapped_index < 0) qmesh_wrapped_index += qmesh_axis_size;
            qmesh_index[axis] = static_cast<int>(qmesh_wrapped_index);
        }
        qmesh_occupied.insert(qmesh_index);
    }
    if (qmesh_occupied.size() != qmesh_kpoints.size()) return {};
    return qmesh_shape;
}

// Use one floating-point representation for each periodic transfer on the
// inferred k mesh, removing roundoff duplicates without using the coarse mesh.
inline std::array<double, 3> canonical_uniform_q(
    const std::array<double, 3> &qmesh_transfer,
    const std::array<int, 3> &qmesh_shape)
{
    if (qmesh_shape[0] == 0) return qmesh_transfer;

    std::array<double, 3> qmesh_canonical{};
    for (int axis = 0; axis < 3; ++axis)
    {
        const int qmesh_axis_size = qmesh_shape[axis];
        const long long qmesh_index = std::llround(qmesh_transfer[axis]
                                                  * qmesh_axis_size);
        long long qmesh_wrapped_index = qmesh_index % qmesh_axis_size;
        if (qmesh_wrapped_index < 0) qmesh_wrapped_index += qmesh_axis_size;
        qmesh_canonical[axis] = static_cast<double>(qmesh_wrapped_index)
                                / qmesh_axis_size;
    }
    return qmesh_canonical;
}

} // namespace libbse
