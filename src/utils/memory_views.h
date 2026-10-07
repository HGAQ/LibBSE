#pragma once
#include "memory.h"
#include "bse/bse_types.h"
#include "bse/spectrum.h"
#include "interface/librpa_api.h"

namespace libbse {
template<class T>
void visit_memory(MemoryVisitor &v, const RI::Tensor<T> &x, const std::string &name) {
    if (x.data && x.data->size()) v.add(x.ptr(), x.data->size() * sizeof(T), name);
}
template<class T>
void visit_memory(MemoryVisitor &v, const librpa_int::matrix_m<T> &x, const std::string &name) {
    const auto data = x.sptr();
    if (data && data->size()) v.add(x.ptr(), data->size() * sizeof(T), name);
}
inline void visit_memory(MemoryVisitor &v, const librpa_int::ComplexMatrix &x, const std::string &name) {
    v.add(x.c, static_cast<std::uint64_t>(x.nr) * x.nc * sizeof(Complex), name);
}
inline void visit_memory(MemoryVisitor &v, const librpa_int::matrix &x, const std::string &name) {
    v.add(x.c, static_cast<std::uint64_t>(x.nr) * x.nc * sizeof(double), name);
}
inline void visit_memory(MemoryVisitor &v, const librpa_int::MeanField &x, const std::string &name) {
    visit_memory(v, x.get_eigenvectors(), name + ".wavefunctions");
    visit_memory(v, x.get_eigenvals(), name + ".energies");
    visit_memory(v, x.get_weight(), name + ".occupations");
}
inline void visit_memory(MemoryVisitor &v, const librpa_int::Dataset &x, const std::string &name) {
    visit_memory(v, x.mf, name + ".SCF");
    visit_memory(v, x.mf_band, name + ".band");
    visit_memory(v, x.velocity_matrix, name + ".velocity");
    visit_memory(v, x.cs_data.data_IJR, name + ".Cs");
    visit_memory(v, x.cs_data.data_libri, name + ".Cs_LibRI");
    visit_memory(v, x.vq, name + ".Vq");
    visit_memory(v, x.vq_cut, name + ".Vq_cut");
    visit_memory(v, x.vq_block_loc, name + ".Vq_local");
    visit_memory(v, x.vq_cut_block_loc, name + ".Vq_cut_local");
}
inline void visit_memory(MemoryVisitor &v, const QuasiparticleBands &x, const std::string &name) {
    visit_memory(v, x.energies_ry, name + ".energies");
}
inline void visit_memory(MemoryVisitor &v, const EigenSolution &x, const std::string &name) {
    visit_memory(v, x.energies_ry, name + ".energies");
    visit_memory(v, x.vectors_local, name + ".eigenvectors");
}
inline void visit_memory(MemoryVisitor &v, const DistributedAmplitudes &x, const std::string &name) {
    visit_memory(v, x.values, name + ".amplitudes");
}
inline void visit_memory(MemoryVisitor &v, const IpaSolution &x, const std::string &name) {
    visit_memory(v, x.energies, name + ".energies");
    visit_memory(v, x.amplitudes, name + ".amplitudes");
}
inline void visit_memory(MemoryVisitor &v, const ChannelSolution &x, const std::string &name) {
    visit_memory(v, x.energies, name + ".energies");
    visit_memory(v, x.amplitudes_x, name + ".amplitudes_x");
    visit_memory(v, x.amplitudes_y, name + ".amplitudes_y");
}
inline void visit_memory(MemoryVisitor &v, const FineVelocityMo &x, const std::string &name) {
    visit_memory(v, x.values, name + ".velocity");
    visit_memory(v, x.gaps_ha, name + ".gaps");
}
} // namespace libbse
