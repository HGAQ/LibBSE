#include "progress.h"

#include <chrono>
#include <iomanip>
#include <iostream>

namespace libbse
{
namespace
{

const auto program_start = std::chrono::steady_clock::now();

} // namespace

void done(const std::string &description, MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank != 0) return;

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - program_start).count();
    const auto flags = std::cout.flags();
    const auto precision = std::cout.precision();
    std::cout << " DONE(" << std::setw(10) << std::setprecision(6)
              << std::defaultfloat << elapsed << " SEC) : " << description
              << '\n';
    std::cout.flags(flags);
    std::cout.precision(precision);
}

} // namespace libbse
