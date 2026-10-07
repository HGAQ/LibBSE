#pragma once

#include <mpi.h>
#include <array>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <map>
#include <unordered_map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace libbse {

struct MemoryBuffer {
    std::uint64_t bytes = 0;
    std::string name;
};

// Counts payload storage only. Aliases sharing the same allocation are counted
// once. Container nodes, allocator overhead and external workspaces are omitted.
class MemoryVisitor {
public:
    void add(const void *address, std::uint64_t bytes, const std::string &name);
    std::map<const void *, MemoryBuffer> buffers;
};

struct MemorySnapshot {
    std::uint64_t current = 0, peak = 0, largest = 0;
    std::string peak_name = "none", largest_name = "none";
    bool complete = true;
};

// The control thread observes registered owners at explicit allocation/resize
// checkpoints and scope boundaries. Peaks are observed payload peaks, not RSS
// or a malloc interception. Observers must not outlive the objects they visit.
class MemoryTracker {
public:
    using Observer = std::function<void(MemoryVisitor &)>;
    static MemoryTracker &instance();
    std::size_t attach(Observer observer);
    void detach(std::size_t id) noexcept;
    MemorySnapshot snapshot();
    void checkpoint() noexcept;
    void report_current(std::ostream &, MPI_Comm comm = MPI_COMM_NULL);
    void report_final(std::ostream &, MPI_Comm comm = MPI_COMM_NULL);
    void reset(); // Tests only; refuses to discard live registrations.
private:
    std::map<std::size_t, Observer> observers_;
    std::map<const void *, MemoryBuffer> previous_;
    MemorySnapshot state_;
    std::size_t next_ = 0;
};

class MemoryWatch {
public:
    explicit MemoryWatch(MemoryTracker::Observer observer)
        : id_(MemoryTracker::instance().attach(std::move(observer))) {}
    ~MemoryWatch() { MemoryTracker::instance().detach(id_); }
    MemoryWatch(const MemoryWatch &) = delete;
    MemoryWatch &operator=(const MemoryWatch &) = delete;
private:
    std::size_t id_;
};

// Scalar fields have no separately owned storage. ADL supplies payload views
// for RI tensors, LibRPA matrices and LibBSE result objects in memory_views.h.
template<class T, std::enable_if_t<std::is_trivially_copyable_v<T>, int> = 0>
void visit_memory(MemoryVisitor &, const T &, const std::string &) {}

template<class A, class B>
void visit_memory(MemoryVisitor &v, const std::pair<A, B> &x, const std::string &name) {
    visit_memory(v, x.first, name); visit_memory(v, x.second, name);
}

template<class T, class A>
void visit_memory(MemoryVisitor &v, const std::vector<T, A> &x, const std::string &name);
template<class K, class T, class C, class A>
void visit_memory(MemoryVisitor &v, const std::map<K, T, C, A> &x, const std::string &name);
template<class T, std::size_t N>
void visit_memory(MemoryVisitor &v, const std::array<T, N> &x, const std::string &name);
template<class T>
void visit_memory(MemoryVisitor &v, const std::shared_ptr<T> &x, const std::string &name);

template<class T, class A>
void visit_memory(MemoryVisitor &v, const std::vector<T, A> &x, const std::string &name) {
    static_assert(!std::is_same_v<T, bool>, "packed vector<bool> needs an explicit view");
    v.add(x.data(), static_cast<std::uint64_t>(x.capacity()) * sizeof(T), name);
    if constexpr (!std::is_trivially_copyable_v<T>)
        for (const auto &item : x) visit_memory(v, item, name);
}
template<class K, class T, class C, class A>
void visit_memory(MemoryVisitor &v, const std::map<K, T, C, A> &x, const std::string &name) {
    for (const auto &item : x) visit_memory(v, item.second, name);
}
template<class T, std::size_t N>
void visit_memory(MemoryVisitor &v, const std::array<T, N> &x, const std::string &name) {
    for (const auto &item : x) visit_memory(v, item, name);
}
template<class T>
void visit_memory(MemoryVisitor &v, const std::shared_ptr<T> &x, const std::string &name) {
    if (x) visit_memory(v, *x, name);
}
template<class K, class T, class H, class E, class A>
void visit_memory(MemoryVisitor &v, const std::unordered_map<K, T, H, E, A> &x, const std::string &name) {
    for (const auto &item : x) visit_memory(v, item.second, name);
}
template<class T>
MemoryWatch watch_memory(const std::string &name, const T &object) {
    return MemoryWatch([&object, name](MemoryVisitor &v) { visit_memory(v, object, name); });
}
template<class T> MemoryWatch watch_memory(const std::string &, const T &&) = delete;

bool mpi_is_active() noexcept;
bool memory_output_root(MPI_Comm comm = MPI_COMM_NULL) noexcept;
} // namespace libbse
