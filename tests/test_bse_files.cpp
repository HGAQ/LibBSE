#include "io/bse_files.h"

#include <librpa_file_reader.hpp>
#include <mpi.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;

namespace
{

void write_wc(const fs::path &file, int iat, int jat, double value)
{
    std::ofstream output(file);
    if (!output) throw std::runtime_error("cannot create Wc test file");
    output << "%%MatrixMarket matrix coordinate complex general\n"
           << "% atom pair " << iat << ' ' << jat << " R = (0 0 0)\n"
           << "1 1 1\n"
           << "1 1 " << value << " 0\n";
}

void test_coarse_qp_reader(const fs::path &input_dir)
{
    std::ofstream output(input_dir / "energy_qp");
    if (!output) throw std::runtime_error("cannot create energy_qp test file");
    output << "  state     occ_num        e_gs(Ha)        e_qp(Ha)\n"
           << "--------------------------------------------------\n"
           << "  K_point    1 :           0.0000          0.0000          0.0000\n"
           << "--------------------------------------------------\n"
           << "       1    2.0000   -1.1000000000E+00   -1.0000000000E+00\n"
           << "       2    2.0000   -3.0000000000E-01   -2.0000000000E-01\n"
           << "       3    0.0000    2.0000000000E-01    3.0000000000E-01\n"
           << "--------------------------------------------------\n\n"
           << "  K_point    2 :           0.1429          0.0000          0.0000\n"
           << "--------------------------------------------------\n"
           << "       1    2.0000   -1.0000000000E+00   -9.0000000000E-01\n"
           << "       2    2.0000   -2.0000000000E-01   -1.0000000000E-01\n"
           << "       3    0.0000    2.0000000000E-01    2.5000000000E-01\n"
           << "--------------------------------------------------\n";
    output.close();

    librpa_int::Dataset dataset(MPI_COMM_WORLD);
    dataset.mf_band = librpa_int::MeanField(1, 2, 3, 1);
    dataset.kfrac_band_list = {{0.0, 0.0, 0.0}, {1.0 / 7.0, 0.0, 0.0}};

    libbse::InputParameters options;
    options.input_dir = input_dir.string();
    options.nocc = 1;
    options.nvirt = 1;
    options.bse_use_fine_kgrid = 0;
    const auto qp = libbse::read_qp_bands(options, dataset);
    const std::vector<double> expected{-0.4, 0.6, -0.2, 0.5};
    if (qp.ncore != 1 || qp.nk != 2 || qp.nbands != 2
        || qp.energies_ry.size() != expected.size())
        throw std::runtime_error("coarse-grid quasiparticle dimensions are incorrect");
    for (std::size_t i = 0; i != expected.size(); ++i)
        if (std::abs(qp.energies_ry[i] - expected[i]) > 1.0e-13)
            throw std::runtime_error("coarse-grid quasiparticle energy is incorrect");
    if (std::abs(qp.direct_gap_ry - 0.7) > 1.0e-13
        || std::abs(qp.indirect_gap_ry - 0.7) > 1.0e-13)
        throw std::runtime_error("coarse-grid quasiparticle gap is incorrect");
}

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int status = 0;
    const fs::path root = fs::temp_directory_path()
                          / ("libbse_wc_local_"
                             + std::to_string(static_cast<long long>(::getpid())));
    try
    {
        const fs::path input_dir = root / "input";
        const fs::path wc_dir = root / "librpa.d";
        fs::create_directories(input_dir);
        fs::create_directories(wc_dir);

        test_coarse_qp_reader(input_dir);

        libbse::TensorMap<libbse::Complex> bare;
        const libbse::Cell cell{0, 0, 0};
        for (int iat = 0; iat < 2; ++iat)
            for (int jat = 0; jat < 2; ++jat)
            {
                RI::Tensor<libbse::Complex> tensor({1, 1});
                tensor(0, 0) = 1.0;
                bare[iat][{jat, cell}] = std::move(tensor);
            }

        // Only this local LibRI atom pair is present. Any attempt to read a
        // nonlocal pair therefore makes the test fail with a missing file.
        write_wc(wc_dir / "Wc_Mu_0_Nu_1_iR_0_ifreq_0.mtx", 0, 1, 2.0);

        libbse::InputParameters options;
        options.input_dir = input_dir.string();
        const auto screened = libbse::read_screened_interaction(
            options, bare, 1, std::vector<int>{0}, std::vector<int>{1});
        if (screened.size() != 1 || screened.at(0).size() != 1
            || std::abs(screened.at(0).at({1, cell})(0, 0)
                        - libbse::Complex(3.0, 0.0)) > 1.0e-13)
            throw std::runtime_error(
                "local-pair screened interaction is incorrect");

        fs::remove_all(root);
    }
    catch (const std::exception &error)
    {
        fs::remove_all(root);
        std::cerr << error.what() << '\n';
        status = 1;
    }
    MPI_Finalize();
    return status;
}
