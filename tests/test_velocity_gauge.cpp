#include "bse/spectrum.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{

bool close(const libbse::Complex &actual, const libbse::Complex &expected,
           double tolerance = 1.0e-13)
{
    return std::abs(actual - expected) <= tolerance;
}

std::size_t velocity_index(int direction)
{
    return static_cast<std::size_t>(direction);
}

} // namespace

int main()
{
    try
    {
        libbse::InputParameters options;
        options.nocc = 1;
        options.nvirt = 1;

        libbse::FineVelocityMo velocity;
        velocity.nk = 1;
        velocity.nbands = 2;
        velocity.first_pair = 0;
        velocity.local_pairs = 1;
        velocity.values.assign(3, libbse::Complex{});
        velocity.gaps_ha = {0.5};

        // v_x = 0.5 gives v_x/(e_a-e_i) = 1.  The spin-adapted
        // velocity-gauge prefactor is i*sqrt(2).
        velocity.values[velocity_index(0)] = 0.5;
        const std::vector<libbse::Complex> x{1.0};
        const auto tda = libbse::velocity_gauge_transition_dipole(
            0, options, velocity, x, nullptr);
        if (!close(tda[0], {0.0, std::sqrt(2.0)})
            || !close(tda[1], {}) || !close(tda[2], {}))
            throw std::runtime_error("TDA velocity-gauge contraction is incorrect");

        // The full-BSE Y term is -conj(v)Y/gap.
        const std::vector<libbse::Complex> y{0.25};
        const auto full = libbse::velocity_gauge_transition_dipole(
            0, options, velocity, x, &y);
        if (!close(full[0], {0.0, 0.75 * std::sqrt(2.0)}))
            throw std::runtime_error("full-BSE velocity-gauge Y contraction is incorrect");

        // An imaginary matrix element checks that the full-BSE term uses the
        // Hermitian conjugate rather than v itself.
        velocity.values[velocity_index(0)] = {0.0, 0.5};
        const auto complex_full = libbse::velocity_gauge_transition_dipole(
            0, options, velocity, x, &y);
        if (!close(complex_full[0], {-1.25 * std::sqrt(2.0), 0.0}))
            throw std::runtime_error("velocity-gauge conjugation is incorrect");

        bool rejected_zero_gap = false;
        velocity.gaps_ha[0] = 0.0;
        try
        {
            (void)libbse::velocity_gauge_transition_dipole(
                0, options, velocity, x, nullptr);
        }
        catch (const std::runtime_error &)
        {
            rejected_zero_gap = true;
        }
        if (!rejected_zero_gap)
            throw std::runtime_error("zero KS gap was not rejected");
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
