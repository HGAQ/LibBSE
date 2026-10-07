#include "memory.h"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace libbse {
bool mpi_is_active() noexcept {
    int initialized = 0, finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized) MPI_Finalized(&finalized);
    return initialized && !finalized;
}
bool memory_output_root(MPI_Comm comm) noexcept {
    if (!mpi_is_active()) return true;
    int rank = 0;
    MPI_Comm_rank(comm == MPI_COMM_NULL ? MPI_COMM_WORLD : comm, &rank);
    return rank == 0;
}
void MemoryVisitor::add(const void *address, std::uint64_t bytes, const std::string &name) {
    if (!address || bytes == 0) return;
    auto &entry = buffers[address];
    if (bytes > entry.bytes) entry = {bytes, name};
}
MemoryTracker &MemoryTracker::instance() {
    static MemoryTracker tracker;
    return tracker;
}
std::size_t MemoryTracker::attach(Observer observer) {
    const auto id = ++next_;
    observers_.emplace(id, std::move(observer));
    checkpoint();
    return id;
}
void MemoryTracker::detach(std::size_t id) noexcept {
    checkpoint(); // Includes any last resize, while the owner is still alive.
    observers_.erase(id);
    checkpoint();
}
MemorySnapshot MemoryTracker::snapshot() {
    MemoryVisitor visitor;
    for (const auto &entry : observers_) entry.second(visitor);
    std::uint64_t current = 0, growth = 0;
    std::string cause = "tracked arrays";
    for (const auto &[ptr, buffer] : visitor.buffers) {
        if (buffer.bytes > std::numeric_limits<std::uint64_t>::max() - current)
            throw std::overflow_error("tracked memory total overflow");
        current += buffer.bytes;
        const auto old = previous_.find(ptr);
        const auto before = old == previous_.end() ? 0 : old->second.bytes;
        if (buffer.bytes > before && buffer.bytes - before > growth) {
            growth = buffer.bytes - before;
            cause = buffer.name;
        }
        if (buffer.bytes > state_.largest) {
            state_.largest = buffer.bytes;
            state_.largest_name = buffer.name;
        }
    }
    state_.current = current;
    if (current > state_.peak) { state_.peak = current; state_.peak_name = cause; }
    previous_ = std::move(visitor.buffers);
    return state_;
}
void MemoryTracker::checkpoint() noexcept {
    try { snapshot(); } catch (...) { state_.complete = false; }
}
void MemoryTracker::reset() {
    if (!observers_.empty()) throw std::logic_error("cannot reset live memory registrations");
    previous_.clear(); state_ = {}; next_ = 0;
}
namespace {
struct Wire {
    unsigned long long current = 0, peak = 0, largest = 0;
    int complete = 1;
    char peak_name[256]{}, largest_name[256]{};
};
std::vector<Wire> collect(const MemorySnapshot &s, MPI_Comm comm) {
    Wire local{};
    local.current = s.current; local.peak = s.peak; local.largest = s.largest;
    local.complete = s.complete;
    std::strncpy(local.peak_name, s.peak_name.c_str(), sizeof(local.peak_name) - 1);
    std::strncpy(local.largest_name, s.largest_name.c_str(), sizeof(local.largest_name) - 1);
    if (!mpi_is_active() || comm == MPI_COMM_NULL) return {local};
    int size = 1, rank = 0;
    MPI_Comm_size(comm, &size); MPI_Comm_rank(comm, &rank);
    std::vector<Wire> all(rank == 0 ? size : 0);
    MPI_Gather(&local, sizeof(Wire), MPI_BYTE, all.data(), sizeof(Wire), MPI_BYTE, 0, comm);
    return all;
}
struct FormatGuard {
    std::ostream &out;
    std::ios::fmtflags flags;
    std::streamsize precision;
    explicit FormatGuard(std::ostream &s) : out(s), flags(s.flags()), precision(s.precision()) {}
    ~FormatGuard() { out.flags(flags); out.precision(precision); }
};
void statistics(std::ostream &out, const std::vector<Wire> &all, int field) {
    const auto value = [field](const Wire &x) { return field == 0 ? x.current : field == 1 ? x.peak : x.largest; };
    std::size_t lo = 0, hi = 0; long double sum = 0;
    for (std::size_t i = 0; i < all.size(); ++i) {
        if (value(all[i]) < value(all[lo])) lo = i;
        if (value(all[i]) > value(all[hi])) hi = i;
        sum += value(all[i]);
    }
    out << std::right << std::fixed << std::setprecision(3);
    out << "  |   Minimum: " << std::setw(12) << value(all[lo]) / 1.e6 << " MB (";
    if (field == 2) {
        out << all[lo].largest_name << " on rank " << lo;
    } else {
        out << "on rank " << lo;
        if (field == 1) out << " after allocating " << all[lo].peak_name;
    }
    out << ")\n";

    out << "  |   Maximum: " << std::setw(12) << value(all[hi]) / 1.e6 << " MB (";
    if (field == 2) {
        out << all[hi].largest_name << " on rank " << hi;
    } else {
        out << "on rank " << hi;
        if (field == 1) out << " after allocating " << all[hi].peak_name;
    }
    out << ")\n";
    out << "  |   Average: " << std::setw(12) << static_cast<double>(sum / all.size() / 1.e6L) << " MB\n";
}
} // namespace
void MemoryTracker::report_current(std::ostream &out, MPI_Comm comm) {
    checkpoint(); const auto all = collect(state_, comm);
    if (!memory_output_root(comm)) return;
    FormatGuard guard(out);
    out << "  | Current value for overall tracked memory usage:\n";
    statistics(out, all, 0);
    if (std::any_of(all.begin(), all.end(), [](const Wire &x) { return !x.complete; }))
        out << "  | Memory accounting incomplete: an observation failed.\n";
}
void MemoryTracker::report_final(std::ostream &out, MPI_Comm comm) {
    checkpoint(); const auto all = collect(state_, comm);
    if (!memory_output_root(comm)) return;
    FormatGuard guard(out);
    long double residual = 0;
    for (const auto &x : all) residual += x.current;
    out << "Partial memory accounting:\n"
        << "  | Residual value for overall tracked memory usage across ranks: "
        << std::fixed << std::setprecision(6) << static_cast<double>(residual / 1.e3L)
        << " MB (should be 0.000 MB)\n"
        << "  | Peak values for overall tracked memory usage:\n";
    statistics(out, all, 1);
    out << "  | Largest tracked array allocation:\n";
    statistics(out, all, 2);
    out << "  Note: These values currently only include a subset of arrays which are explicitly tracked.\n"
        << "        The \"true\" memory usage will be greater.\n"
        << "        MB = 1000000 bytes; shared payloads are counted once per task.\n"
        << "        Peaks are observed at explicit checkpoints, not allocator/RSS high-water marks.\n";
    if (std::any_of(all.begin(), all.end(), [](const Wire &x) { return !x.complete; }))
        out << "  | Memory accounting incomplete: an observation failed.\n";
}
} // namespace libbse
