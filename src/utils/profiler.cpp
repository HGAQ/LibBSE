#include "profiler.h"

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <utility>

namespace libbse
{

std::size_t Profiler::find_child(std::size_t parent,
                                 const std::string &name) const noexcept
{
    for (std::size_t index = 0; index < timers_.size(); ++index)
        if (timers_[index].parent == parent && timers_[index].name == name)
            return index;
    return timers_.size();
}

const Profiler::Timer *Profiler::find_timer(const std::string &name) const noexcept
{
    const auto timer = std::find_if(
        timers_.begin(), timers_.end(),
        [&name](const Timer &entry) { return entry.name == name; });
    return timer == timers_.end() ? nullptr : &*timer;
}

void Profiler::start(const std::string &name, const std::string &note)
{
    const std::size_t parent = active_.empty() ? no_parent : active_.back();
    std::size_t index = find_child(parent, name);
    if (index == timers_.size())
    {
        timers_.push_back({});
        index = timers_.size() - 1;
        timers_[index].name = name;
        timers_[index].note = note;
        timers_[index].parent = parent;
    }

    Timer &timer = timers_[index];
    if (timer.running) stop(name);
    ++timer.calls;
    timer.cpu_start = std::clock();
    timer.wall_start = WallClock::now();
    timer.cpu_time_last = 0.0;
    timer.wall_time_last = 0.0;
    timer.running = true;
    active_.push_back(index);
}

void Profiler::stop(const std::string &name) noexcept
{
    if (active_.empty()) return;
    const std::size_t index = active_.back();
    Timer &timer = timers_[index];
    if (timer.name != name) return;

    timer.cpu_time_last = static_cast<double>(std::clock() - timer.cpu_start)
                          / CLOCKS_PER_SEC;
    timer.wall_time_last = std::chrono::duration<double>(WallClock::now()
                                                         - timer.wall_start)
                               .count();
    timer.cpu_time += timer.cpu_time_last;
    timer.wall_time += timer.wall_time_last;
    timer.running = false;
    active_.pop_back();
}

void Profiler::terminate() noexcept
{
    while (!active_.empty()) stop(timers_[active_.back()].name);
}

void Profiler::reset() noexcept
{
    terminate();
    timers_.clear();
    active_.clear();
}

std::size_t Profiler::get_num_timers() const noexcept
{
    return timers_.size();
}

std::size_t Profiler::get_call_count(const std::string &name) const noexcept
{
    const Timer *timer = find_timer(name);
    return timer == nullptr ? 0 : timer->calls;
}

double Profiler::get_cpu_time_last(const std::string &name) const noexcept
{
    const Timer *timer = find_timer(name);
    return timer == nullptr ? -1.0 : timer->cpu_time_last;
}

double Profiler::get_wall_time_last(const std::string &name) const noexcept
{
    const Timer *timer = find_timer(name);
    return timer == nullptr ? -1.0 : timer->wall_time_last;
}

void Profiler::append_timer(std::ostream &output, std::size_t index,
                            int level) const
{
    const Timer &timer = timers_[index];
    const std::string indent(static_cast<std::size_t>(level), ' ');
    const std::string label = indent + (timer.note.empty() ? timer.name : timer.note);
    output << std::left << std::setw(49) << label << ' '
           << std::setw(12) << timer.calls << ' '
           << std::setw(18) << std::fixed << std::setprecision(4)
           << timer.cpu_time << ' '
           << std::setw(18) << timer.wall_time << '\n';
    for (std::size_t child = 0; child < timers_.size(); ++child)
        if (timers_[child].parent == index) append_timer(output, child, level + 1);
}

std::string Profiler::get_profile_string() const
{
    std::ostringstream output;
    output << "LibBSE timing profile\n"
           << std::left << std::setw(49) << "Entry" << ' '
           << std::setw(12) << "#calls" << ' '
           << std::setw(18) << "CPU time (s)" << ' '
           << std::setw(18) << "Wall time (s)" << '\n'
           << std::string(100, '-') << '\n';
    for (std::size_t index = 0; index < timers_.size(); ++index)
        if (timers_[index].parent == no_parent) append_timer(output, index, 0);
    return output.str();
}

void Profiler::display(std::ostream &output) const
{
    output << get_profile_string();
}

ScopedTimer::ScopedTimer(Profiler &profiler, std::string name, std::string note)
    : profiler_(&profiler), name_(std::move(name))
{
    profiler_->start(name_, note);
}

ScopedTimer::~ScopedTimer()
{
    if (profiler_ != nullptr) profiler_->stop(name_);
}

namespace global
{
Profiler profiler;
}

} // namespace libbse
