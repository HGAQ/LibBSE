#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <vector>

namespace libbse {
// Recognize complete, uniformly spaced Cartesian products in fractional
// reciprocal coordinates. Shifted meshes are allowed; arbitrary paths are not.
inline std::array<int,3> infer_uniform_qmesh(
    const std::vector<std::array<double,3>> &qmesh_kpoints)
{
    constexpr double qmesh_tolerance = 1.e-10;
    std::array<int,3> qmesh_shape{};
    std::array<double,3> qmesh_origin{};
    if (qmesh_kpoints.empty()) return {};
    for (int qmesh_axis=0; qmesh_axis<3; ++qmesh_axis) {
        std::vector<double> qmesh_coordinates;
        for (const auto &qmesh_kpoint:qmesh_kpoints) {
            if (!std::isfinite(qmesh_kpoint[qmesh_axis])) return {};
            double qmesh_coordinate=qmesh_kpoint[qmesh_axis]-std::floor(qmesh_kpoint[qmesh_axis]);
            if (qmesh_coordinate>1.-qmesh_tolerance) qmesh_coordinate=0.;
            qmesh_coordinates.push_back(qmesh_coordinate);
        }
        std::sort(qmesh_coordinates.begin(),qmesh_coordinates.end());
        std::vector<double> qmesh_unique_coordinates;
        for (const double qmesh_coordinate:qmesh_coordinates)
            if (qmesh_unique_coordinates.empty() || qmesh_coordinate-qmesh_unique_coordinates.back()>qmesh_tolerance)
                qmesh_unique_coordinates.push_back(qmesh_coordinate);
        qmesh_shape[qmesh_axis]=static_cast<int>(qmesh_unique_coordinates.size());
        qmesh_origin[qmesh_axis]=qmesh_unique_coordinates.front();
        for (int qmesh_index=0; qmesh_index<qmesh_shape[qmesh_axis]; ++qmesh_index)
            if (std::abs(qmesh_unique_coordinates[qmesh_index]-qmesh_origin[qmesh_axis]
                         -double(qmesh_index)/qmesh_shape[qmesh_axis])>qmesh_tolerance)
                return {};
    }
    if (std::size_t(qmesh_shape[0])*qmesh_shape[1]*qmesh_shape[2]!=qmesh_kpoints.size()) return {};
    std::set<std::array<int,3>> qmesh_occupied;
    for (const auto &qmesh_kpoint:qmesh_kpoints) {
        std::array<int,3> qmesh_index{};
        for (int qmesh_axis=0; qmesh_axis<3; ++qmesh_axis) {
            const double qmesh_scaled=(qmesh_kpoint[qmesh_axis]-qmesh_origin[qmesh_axis])*qmesh_shape[qmesh_axis];
            const long long qmesh_rounded=std::llround(qmesh_scaled);
            if (std::abs(qmesh_scaled-qmesh_rounded)>qmesh_tolerance*qmesh_shape[qmesh_axis]) return {};
            qmesh_index[qmesh_axis]=int((qmesh_rounded%qmesh_shape[qmesh_axis]+qmesh_shape[qmesh_axis])%qmesh_shape[qmesh_axis]);
        }
        qmesh_occupied.insert(qmesh_index);
    }
    if (qmesh_occupied.size()!=qmesh_kpoints.size()) return {};
    return qmesh_shape;
}

// Use one exactly identical floating representation per integer transfer.
// This removes roundoff duplicates, without rounding onto the coarse mesh.
inline std::array<double,3> canonical_uniform_q(
    const std::array<double,3> &qmesh_transfer,
    const std::array<int,3> &qmesh_shape)
{
    if (qmesh_shape[0]==0) return qmesh_transfer;
    std::array<double,3> qmesh_canonical{};
    for (int qmesh_axis=0; qmesh_axis<3; ++qmesh_axis) {
        const long long qmesh_index=std::llround(qmesh_transfer[qmesh_axis]*qmesh_shape[qmesh_axis]);
        qmesh_canonical[qmesh_axis]=double((qmesh_index%qmesh_shape[qmesh_axis]+qmesh_shape[qmesh_axis])%qmesh_shape[qmesh_axis])/qmesh_shape[qmesh_axis];
    }
    return qmesh_canonical;
}
}
