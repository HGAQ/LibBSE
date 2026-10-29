#include "utils/profiler.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

void exercise_exception_path(libbse::Profiler &profiler)
{
    libbse::ScopedTimer timer(profiler, "exception_scope",
                              "Exception-safe scope");
    throw std::runtime_error("expected test exception");
}

} // namespace

int main()
{
    try
    {
        libbse::Profiler profiler;
        profiler.start("outer", "Outer operation");
        {
            libbse::ScopedTimer timer(profiler, "inner", "Inner operation");
            volatile double value = 0.0;
            for (int i = 0; i < 10000; ++i) value += i;
            (void)value;
        }
        profiler.stop("outer");

        profiler.start("outer", "Outer operation");
        profiler.stop("outer");
        require(profiler.get_num_timers() == 2,
                "profiler did not preserve its timer hierarchy");
        require(profiler.get_call_count("outer") == 2,
                "profiler did not accumulate repeated calls");
        require(profiler.get_call_count("inner") == 1,
                "profiler recorded an incorrect child call count");
        require(profiler.get_cpu_time_last("outer") >= 0.0,
                "profiler returned an invalid CPU time");
        require(profiler.get_wall_time_last("inner") >= 0.0,
                "profiler returned an invalid wall time");

        try
        {
            exercise_exception_path(profiler);
        }
        catch (const std::runtime_error &)
        {
        }
        profiler.start("after_exception", "After exception");
        profiler.stop("after_exception");
        require(profiler.get_num_timers() == 4,
                "ScopedTimer did not close the exception-path timer");

        const std::string report = profiler.get_profile_string();
        require(report.find("LibBSE timing profile") != std::string::npos,
                "profile report is missing its title");
        require(report.find("#calls") != std::string::npos,
                "profile report is missing the call-count column");
        require(report.find("CPU time (s)") != std::string::npos,
                "profile report is missing the CPU-time column");
        require(report.find("Wall time (s)") != std::string::npos,
                "profile report is missing the wall-time column");
        require(report.find(" Inner operation") != std::string::npos,
                "profile report does not indent child timers");
    }
    catch (const std::exception &error)
    {
        std::cerr << "test_profiler failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
