#pragma once

#include <complex>
#include <cstddef>

namespace libbse
{

inline bool retain_screened_k_pair(int screened_k1, int screened_k2)
{
    if (screened_k1 == screened_k2) return true;
    // Alternate orientation rather than assigning all work to the upper half.
    const bool screened_even_pair = (screened_k1 + screened_k2) % 2 == 0;
    return (screened_k1 < screened_k2) == screened_even_pair;
}

inline void conjugate_screened_k_block(
    const std::complex<double> *screened_forward_block,
    std::complex<double> *screened_reverse_block,
    std::size_t screened_transition_count)
{
    // Column-major A(k2,k1) = A(k1,k2)^H; input and output must not alias.
    for (std::size_t screened_row = 0;
         screened_row < screened_transition_count; ++screened_row)
        for (std::size_t screened_column = 0;
             screened_column < screened_transition_count; ++screened_column)
            screened_reverse_block[
                screened_row + screened_column * screened_transition_count]
                = std::conj(screened_forward_block[
                    screened_column + screened_row * screened_transition_count]);
}

} // namespace libbse
