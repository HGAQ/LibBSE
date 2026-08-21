#pragma once

#include <chrono>
#include <cstddef>
#include <ctime>
#include <iosfwd>
#include <string>
#include <vector>

namespace libbse
{

//! Hierarchical CPU and wall-clock profiler for the LibBSE control thread.
class Profiler
{
  public:
    void start(const std::string &name, const std::string &note = {});
    void stop(const std::string &name) noexcept;
    void terminate() noexcept;
    void reset() noexcept;

    std::size_t get_num_timers() const noexcept;
    std::size_t get_call_count(const std::string &name) const noexcept;
    double get_cpu_time_last(const std::string &name) const noexcept;
    double get_wall_time_last(const std::string &name) const noexcept;
    std::string get_profile_string() const;
    void display(std::ostream &output) const;

  private:
    using WallClock = std::chrono::steady_clock;
    static constexpr std::size_t no_parent = static_cast<std::size_t>(-1);

    struct Timer
    {
        std::string name;
        std::string note;
        std::size_t parent = no_parent;
        std::size_t calls = 0;
        std::clock_t cpu_start = 0;
        WallClock::time_point wall_start{};
        double cpu_time = 0.0;
        double wall_time = 0.0;
        double cpu_time_last = 0.0;
        double wall_time_last = 0.0;
        bool running = false;
    };

    std::size_t find_child(std::size_t parent, const std::string &name) const noexcept;
    const Timer *find_timer(const std::string &name) const noexcept;
    void append_timer(std::ostream &output, std::size_t index, int level) const;

    std::vector<Timer> timers_;
    std::vector<std::size_t> active_;
};

//! Exception-safe start/stop pair for a profiler entry.
class ScopedTimer
{
  public:
    ScopedTimer(Profiler &profiler, std::string name, std::string note = {});
    ~ScopedTimer();

    ScopedTimer(const ScopedTimer &) = delete;
    ScopedTimer &operator=(const ScopedTimer &) = delete;

  private:
    Profiler *profiler_;
    std::string name_;
};

namespace global
{
extern Profiler profiler;
}

} // namespace libbse
