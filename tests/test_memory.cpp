#include "utils/memory_views.h"
#include "utils/profiler.h"
#include <mpi.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool x, const char *why) { if (!x) throw std::runtime_error(why); }
}
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    try {
        auto &tracker = libbse::MemoryTracker::instance();
        tracker.reset();
        {
            std::vector<double> first(128 * (rank + 1));
            auto first_watch = libbse::watch_memory("first", first);
            const auto bytes = first.capacity() * sizeof(double);
            require(tracker.snapshot().current == bytes, "initial payload not counted");
            {
                auto alias = libbse::watch_memory("alias", first);
                require(tracker.snapshot().current == bytes, "alias counted twice");
            }
            first.reserve(first.capacity() * 2);
            const auto grown = first.capacity() * sizeof(double);
            require(tracker.snapshot().current == grown, "resize/capacity not refreshed");
            first.clear();
            require(tracker.snapshot().current == grown, "clear falsely released capacity");
            std::vector<double> moved = std::move(first);
            auto moved_watch = libbse::watch_memory("moved", moved);
            require(tracker.snapshot().current == grown, "move lost or duplicated payload");
            std::vector<double>().swap(moved);
            require(tracker.snapshot().current == 0, "released storage is still counted");
            require(tracker.snapshot().peak == grown, "peak disappeared after release");
        }
        require(tracker.snapshot().current == 0, "RAII registrations did not balance");
        tracker.reset();
        {
            RI::Tensor<libbse::Complex> tensor({5,7});
            auto a = libbse::watch_memory("tensor", tensor);
            RI::Tensor<libbse::Complex> shared = tensor;
            auto b = libbse::watch_memory("shared", shared);
            require(tracker.snapshot().current == 35 * sizeof(libbse::Complex), "shared RI tensor duplicated");
        }
        tracker.reset();
        // Distinct rank totals exercise true MPI extrema, average and labels.
        {
            std::vector<unsigned char> small(1000 * (rank + 1));
            auto a = libbse::watch_memory("small", small);
            std::vector<unsigned char> large(2000 * (rank + 1));
            auto b = libbse::watch_memory("large", large);
            const auto state = tracker.snapshot();
            require(state.peak == small.capacity() + large.capacity(), "coexisting arrays not in peak");
            require(state.largest == large.capacity() && state.largest_name == "large", "wrong largest allocation");
            require(state.peak_name == "large", "wrong allocation triggering peak");
            std::ostringstream current;
            current << std::scientific << std::setprecision(9);
            const auto flags = current.flags(); const auto precision = current.precision();
            tracker.report_current(current, MPI_COMM_WORLD);
            require(current.flags() == flags && current.precision() == precision, "report changed caller format");
            if (rank == 0) {
                require(current.str().find("0.003 MB (on task 0)") != std::string::npos, "wrong MPI minimum");
                std::ostringstream maximum; maximum << std::fixed << std::setprecision(3) << .003 * size << " MB (on task " << size-1 << ')';
                require(current.str().find(maximum.str()) != std::string::npos, "wrong MPI maximum/rank");
                const auto average_pos = current.str().find("Average:");
                require(average_pos != std::string::npos, "missing MPI average");
                std::istringstream average(current.str().substr(average_pos + 8));
                double displayed = -1.; average >> displayed;
                require(std::abs(displayed - .003*(size+1)/2) <= .0005000001,
                        "wrong MPI average (after 3-decimal rounding)");
            } else require(current.str().empty(), "nonroot emitted report");
            std::ostringstream live;
            tracker.report_final(live, MPI_COMM_WORLD);
            if (rank == 0) {
                std::ostringstream residual; residual << std::fixed << std::setprecision(6) << .003*size*(size+1)/2 << " MB (should";
                require(live.str().find(residual.str()) != std::string::npos, "live residual was hidden");
            }
        }
        std::ostringstream final;
        tracker.report_final(final, MPI_COMM_WORLD);
        if (rank == 0) {
            require(final.str().find("0.000000 MB (should be 0.000000 MB)") != std::string::npos, "final residual not zero");
            require(final.str().find("after allocating large") != std::string::npos, "peak provenance lost");
            std::cout << final.str();
        }
        libbse::Profiler profiler;
        if (rank == 0) { libbse::ScopedTimer local(profiler, "root_only"); }
        // A rank-local throw must not introduce a collective in the destructor.
        if (rank == size - 1) {
            try {
                libbse::ScopedTimer unwind(profiler, "unwind", "Unwind", MPI_COMM_WORLD);
                throw std::runtime_error("expected");
            } catch (const std::runtime_error &) {}
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Comm sub;
        MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &sub);
        { libbse::ScopedTimer timer(profiler, "sub", "Subcommunicator stage", sub); }
        MPI_Comm_free(&sub);
        require(tracker.snapshot().current == 0, "test left tracked storage");
    } catch (const std::exception &e) {
        std::cerr << "memory test on rank " << rank << ": " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    // Do not call MPI_Comm_rank after finalization.
    libbse::Profiler profiler;
    { libbse::ScopedTimer after(profiler, "after_finalize"); }
    return 0;
}
