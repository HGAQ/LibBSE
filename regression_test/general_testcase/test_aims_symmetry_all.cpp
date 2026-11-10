#include "io/aims_symmetry_all.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool passed, const char *message) {
    if (!passed) throw std::runtime_error(message);
}
void check_rotation() {
    using Complex = std::complex<double>;
    // Exchange the atoms and give one a Bloch phase. A transpose instead of
    // an adjoint, or conjugation without rotation, gives a different answer.
    const std::vector<Complex> representative_w{{2,0},{1,2},{1,-2},{3,0}};
    const std::vector<Complex> symmetry_delta{{0,0},{0,1},{1,0},{0,0}};
    const std::vector<Complex> expected_w{{3,0},{2,1},{2,-1},{2,0}};
    const std::vector<Complex> expected_antiunitary_w{{3,0},{-2,1},{-2,-1},{2,0}};
    const auto reconstructed_w = libbse::reconstruct_aims_symmetry_w(representative_w, symmetry_delta, 2);
    const auto antiunitary_w = libbse::reconstruct_aims_symmetry_w(representative_w, symmetry_delta, 2, true);
    for (std::size_t entry = 0; entry < 4; ++entry) {
        require(std::abs(reconstructed_w[entry]-expected_w[entry]) < 1.e-14, "wrong spatial rotation");
        require(std::abs(antiunitary_w[entry]-expected_antiunitary_w[entry]) < 1.e-14, "wrong antiunitary rotation");
    }
}
void check_reader(const std::filesystem::path &directory) {
    const auto manifest_path = directory / "aims_symmetry_all.dat";
    const auto delta_path = directory / "delta.dat";
    {
        std::ofstream manifest(manifest_path);
        manifest << "AIMS_SYMMETRY_ALL 1\n2 1\nFULL_W D_W_DAGGER\n"
                 << "8 4 2 0 0.3333333333333333 0 0 0 0.3333333333333333 0 delta.dat\n";
        std::ofstream delta(delta_path);
        delta << "AIMS_DELTA 1\n2 8 4\n1 1 0 0\n1 2 0 1\n2 1 1 0\n2 2 0 0\n";
    }
    const auto symmetry_metadata = libbse::read_aims_symmetry_all(manifest_path, 2, 1);
    const auto symmetry_delta = libbse::read_aims_symmetry_delta(symmetry_metadata.q_mappings.at(0), 2);
    require(symmetry_delta.at(1) == std::complex<double>(0,1), "Delta storage order wrong");
    bool rejected = false;
    try { libbse::read_aims_symmetry_all(manifest_path, 3, 1); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "mismatched auxiliary basis accepted");
    {
        std::ofstream delta(delta_path);
        delta << "AIMS_DELTA 1\n2 8 4\n1 1 0 0\n1 2 0 2\n2 1 1 0\n2 2 0 0\n";
    }
    rejected = false;
    try { libbse::read_aims_symmetry_delta(symmetry_metadata.q_mappings.at(0), 2); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "nonunitary Delta accepted");
    {
        std::ofstream delta(delta_path);
        delta << "AIMS_DELTA 1\n2 8 4\n1 1 1 0\n1 1 1 0\n2 1 0 0\n2 2 1 0\n";
    }
    rejected = false;
    try { libbse::read_aims_symmetry_delta(symmetry_metadata.q_mappings.at(0), 2); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "duplicate Delta entry accepted");
    {
        std::ofstream delta(delta_path);
        delta << "AIMS_DELTA 1\n2 8 4\n1 1 1 0\n";
    }
    rejected = false;
    try { libbse::read_aims_symmetry_delta(symmetry_metadata.q_mappings.at(0), 2); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "truncated Delta accepted");
}
} // namespace

int main() {
    const auto test_directory = std::filesystem::temp_directory_path() /
        ("libbse-aims-symmetry-" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(test_directory);
    try {
        check_rotation();
        check_reader(test_directory);
        std::filesystem::remove_all(test_directory);
        std::cout << "aims symmetry rotation and malformed-input checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        std::filesystem::remove_all(test_directory);
        return 1;
    }
}
