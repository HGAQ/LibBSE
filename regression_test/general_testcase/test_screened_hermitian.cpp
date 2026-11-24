#include "bse/screened_hermitian.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
using ScreenedComplex = std::complex<double>;

ScreenedComplex screened_coefficient(int screened_aux, int screened_state)
{
    return {std::sin(0.7 * (screened_aux + 1) * (screened_state + 1)),
            std::cos(0.4 * (screened_aux + 2) * (screened_state + 1))};
}

ScreenedComplex screened_interaction(int screened_p, int screened_q)
{
    return {1.0 / (1.0 + std::abs(screened_p - screened_q)),
            0.1 * (screened_p - screened_q)};
}

void check_reconstruction(int screened_nk, int screened_width, int screened_ranks)
{
    const int screened_naux = 5;
    const int screened_dimension = screened_nk * screened_width;
    std::vector<ScreenedComplex> screened_reference(screened_dimension * screened_dimension);
    std::vector<ScreenedComplex> screened_reconstructed(screened_reference.size());
    // Independent full C^H W C reference; W is Hermitian and complex.
    for (int screened_row = 0; screened_row < screened_dimension; ++screened_row)
        for (int screened_column = 0; screened_column < screened_dimension; ++screened_column)
            for (int screened_p = 0; screened_p < screened_naux; ++screened_p)
                for (int screened_q = 0; screened_q < screened_naux; ++screened_q)
                    screened_reference[screened_row + screened_column * screened_dimension]
                        += std::conj(screened_coefficient(screened_p, screened_row))
                           * screened_interaction(screened_p, screened_q)
                           * screened_coefficient(screened_q, screened_column);

    int screened_retained = 0;
    for (int screened_k1 = 0; screened_k1 < screened_nk; ++screened_k1)
        for (int screened_k2 = 0; screened_k2 < screened_nk; ++screened_k2)
        {
            if (!libbse::retain_screened_k_pair(screened_k1, screened_k2)) continue;
            ++screened_retained;
            // Partition only rows of W: rank-local contributions are generally
            // NOT Hermitian. Also exercise ranks with no auxiliary rows.
            for (int screened_rank = 0; screened_rank < screened_ranks; ++screened_rank)
            {
                std::vector<ScreenedComplex> screened_forward(screened_width * screened_width);
                std::vector<ScreenedComplex> screened_reverse(screened_forward.size());
                for (int screened_i = 0; screened_i < screened_width; ++screened_i)
                    for (int screened_j = 0; screened_j < screened_width; ++screened_j)
                        for (int screened_p = screened_rank;
                             screened_p < screened_naux; screened_p += screened_ranks)
                            for (int screened_q = 0; screened_q < screened_naux; ++screened_q)
                                screened_forward[screened_i + screened_j * screened_width]
                                    += std::conj(screened_coefficient(
                                           screened_p, screened_k1 * screened_width + screened_i))
                                       * screened_interaction(screened_p, screened_q)
                                       * screened_coefficient(
                                           screened_q, screened_k2 * screened_width + screened_j);
                libbse::conjugate_screened_k_block(
                    screened_forward.data(), screened_reverse.data(), screened_width);
                for (int screened_i = 0; screened_i < screened_width; ++screened_i)
                    for (int screened_j = 0; screened_j < screened_width; ++screened_j)
                    {
                        screened_reconstructed[screened_k1 * screened_width + screened_i
                            + (screened_k2 * screened_width + screened_j) * screened_dimension]
                            += screened_forward[screened_i + screened_j * screened_width];
                        if (screened_k1 != screened_k2)
                            screened_reconstructed[screened_k2 * screened_width + screened_i
                                + (screened_k1 * screened_width + screened_j) * screened_dimension]
                                += screened_reverse[screened_i + screened_j * screened_width];
                    }
            }
        }
    if (screened_retained != screened_nk * (screened_nk + 1) / 2)
        throw std::runtime_error("incorrect retained k-pair count");
    for (std::size_t screened_index = 0; screened_index < screened_reference.size(); ++screened_index)
        if (std::abs(screened_reference[screened_index] - screened_reconstructed[screened_index]) > 1.e-12)
            throw std::runtime_error("partial atom-pair reconstruction disagrees with full C^H W C");
}
} // namespace

int main()
{
    for (int screened_nk : {1, 2, 5, 8})
        for (int screened_width : {1, 3, 6})
            for (int screened_ranks : {1, 2, 4, 8})
                check_reconstruction(screened_nk, screened_width, screened_ranks);
    std::cout << "Screened Hermitian reconstruction: 48 cases passed\n";
}
