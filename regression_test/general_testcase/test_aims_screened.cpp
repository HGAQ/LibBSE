#include "io/aims_screened.h"
#include "io/full_w_binary.h"
#include <cstdint>
#include "interface/librpa_api.h"
#include <../src/core/epsilon.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unistd.h>

namespace fs=std::filesystem;
// A three-point trigonometric polynomial is reconstructed exactly at arbitrary
// fine q. Split each q over two producer ranks and put the lowest frequency at
// index 2 to test assembly, node selection, conjugation, sign, and 1/Nq.
void fixture(const fs::path &root) {
    for (int iq=0; iq<2; ++iq) for (int rank=0; rank<2; ++rank) {
        std::ofstream out(root/("periodic_gw_w_q_"+std::to_string(iq+1)+"_rank_"+std::to_string(rank)+".dat"));
        const double q=iq/3.0, angle=2*libbse::PARAM.constants.pi*q;
        out << std::setprecision(17)
            << "# quantity: w\n# n_basbas n_freq q_index: 2 2 " << iq+1
            << "\n# q_fractional: " << q << " 0 0\n"
            << "# Original RI auxiliary basis; atomic units; indices start at 1.\n";
        for (int iw=1; iw<=2; ++iw) for (int col=0; col<2; ++col) {
            const int row=rank;
            libbse::Complex x;
            if (row==col) x=row==0 ? 2+std::cos(angle) : 4;
            else x=libbse::Complex(0,(row==0 ? 1:-1)*std::sin(angle));
            if (iw==1) x*=99;
            out << iw << ' ' << (iw==1 ? 0.9:0.001) << ' ' << row+1 << ' ' << col+1 << ' '
                << x.real() << ' ' << x.imag() << '\n';
        }
    }
}
// Same analytic W as the text fixture, but with the complete static q mesh.
void write_binary_w_fixture(const fs::path &w_root) {
    for (int w_iq=0; w_iq<3; ++w_iq) {
        std::ofstream w_output(w_root/("full_w_q_"+std::to_string(w_iq)+".bin"), std::ios::binary);
        const std::int32_t w_dimensions[4]={2,3,1,1};
        const double w_q_frequency[4]={w_iq/3.,0.,0.,0.};
        const double w_angle=2*libbse::PARAM.constants.pi*w_q_frequency[0];
        const libbse::Complex w_auxiliary_q[4]={2+std::cos(w_angle),
            libbse::Complex(0,std::sin(w_angle)),libbse::Complex(0,-std::sin(w_angle)),4};
        w_output.write("LIBBSE_FULL_W_V1",16);
        w_output.write(reinterpret_cast<const char*>(w_dimensions),sizeof(w_dimensions));
        w_output.write(reinterpret_cast<const char*>(w_q_frequency),sizeof(w_q_frequency));
        w_output.write(reinterpret_cast<const char*>(w_auxiliary_q),sizeof(w_auxiliary_q));
        if (!w_output) throw std::runtime_error("cannot write binary W fixture");
    }
}
// Explicit spatial transformations must reproduce the same W(R) as q/-q.
void write_symmetry_fixture(const fs::path &symmetry_root) {
    std::ofstream symmetry_map(symmetry_root/"aims_symmetry_all.dat");
    symmetry_map << "AIMS_SYMMETRY_ALL 1\n2 3\nFULL_W D_W_DAGGER\n" << std::setprecision(17);
    for (int target_id=1; target_id<=3; ++target_id) {
        const int representative_id = target_id==3 ? 2 : target_id;
        const std::string delta_filename="delta_"+std::to_string(target_id)+".dat";
        symmetry_map << target_id << ' ' << representative_id << " 1 0 " << (target_id-1)/3.
                     << " 0 0 " << (representative_id-1)/3. << " 0 0 " << delta_filename << '\n';
        std::ofstream delta_output(symmetry_root/delta_filename);
        delta_output << "AIMS_DELTA 1\n2 " << target_id << ' ' << representative_id << "\n1 1 "
                     << (target_id==3 ? -1 : 1) << " 0\n1 2 0 0\n2 1 0 0\n2 2 1 0\n";
    }
}

// Convert the multi-frequency fixture to the producer's single-node layout.
// Add an empty producer block to exercise header-only input on the fast path.
void write_single_frequency_fixture(const fs::path &aims_root)
{
    fixture(aims_root);
    for (int aims_iq = 1; aims_iq <= 2; ++aims_iq)
    {
        for (int aims_rank = 0; aims_rank < 2; ++aims_rank)
        {
            const auto aims_path = aims_root / ("periodic_gw_w_q_" +
                std::to_string(aims_iq) + "_rank_" + std::to_string(aims_rank) + ".dat");
            std::ifstream aims_input(aims_path);
            std::string aims_single_text;
            std::string aims_line;
            while (std::getline(aims_input, aims_line))
            {
                if (aims_line.rfind("# n_basbas", 0) == 0)
                {
                    aims_single_text += "# n_basbas n_freq q_index: 2 1 " +
                                        std::to_string(aims_iq) + "\n";
                }
                else if (aims_line.rfind("2 ", 0) == 0)
                {
                    aims_single_text += "1" + aims_line.substr(1) + "\n";
                }
                else if (aims_line.rfind("#", 0) == 0)
                {
                    aims_single_text += aims_line + "\n";
                }
            }
            aims_input.close();
            std::ofstream aims_output(aims_path);
            aims_output << aims_single_text;
        }
        std::ofstream aims_empty(aims_root / ("periodic_gw_w_q_" +
                                std::to_string(aims_iq) + "_rank_9.dat"));
        aims_empty << "# quantity: w\n# n_basbas n_freq q_index: 2 1 " << aims_iq
                   << "\n# q_fractional: " << std::setprecision(17) << (aims_iq - 1) / 3.0
                   << " 0 0\n# Original RI auxiliary basis; atomic units; indices start at 1.\n";
    }
}

// LibBSE's OpenMP W(q) -> W(R) transform must reproduce LibRPA FT_Wc_q2R
// (no symmetry context) element by element: non-orthogonal cell, 3x2x1 BvK
// mesh, two atoms with 2 and 3 auxiliary functions, and stars containing an
// inverse member (q_bz != q) so the conj(W(q)) branch is exercised.
void test_screened_fourier_matches_librpa()
{
    librpa_int::Dataset fourier_test_dataset(MPI_COMM_WORLD);
    fourier_test_dataset.basis_aux = librpa_int::AtomicBasis(std::vector<std::size_t>{2, 3});
    fourier_test_dataset.pbc.set_latvec({1.0, 0.1, 0.0, 0.2, 1.1, 0.0, 0.0, 0.3, 1.4});
    fourier_test_dataset.pbc.set_period(3, 2, 1);

    // Stored q points (arbitrary values in LibRPA's reciprocal units) and stars.
    const std::vector<librpa_int::Vector3_Order<double>> fourier_test_q_points = {
        {0.0, 0.0, 0.0}, {0.31, 0.0, 0.0}, {0.0, 0.43, 0.05}, {0.27, 0.19, 0.0}};
    for (const auto &fourier_test_q : fourier_test_q_points)
    {
        std::vector<librpa_int::Vector3_Order<double>> fourier_test_star = {fourier_test_q};
        if (fourier_test_q.x + fourier_test_q.y + fourier_test_q.z > 0.0)
        {
            // -q is the inverse star member: contributes conj(W(q)).
            fourier_test_star.push_back(
                librpa_int::Vector3_Order<double>(-fourier_test_q.x, -fourier_test_q.y,
                                                  -fourier_test_q.z));
        }
        fourier_test_dataset.pbc.map_irk_ks[fourier_test_q] = fourier_test_star;
    }

    // Deterministic complex W(q) blocks for all four atom pairs.
    LibRPA_API::ScreenedQBlocks fourier_test_w_q;
    for (int iat = 0; iat < 2; ++iat)
    {
        for (int jat = 0; jat < 2; ++jat)
        {
            const int n_mu = static_cast<int>(fourier_test_dataset.basis_aux[iat]);
            const int n_nu = static_cast<int>(fourier_test_dataset.basis_aux[jat]);
            for (std::size_t iq = 0; iq < fourier_test_q_points.size(); ++iq)
            {
                librpa_int::Matz fourier_test_block(n_mu, n_nu, librpa_int::MAJOR::ROW);
                for (int i = 0; i < n_mu; ++i)
                {
                    for (int j = 0; j < n_nu; ++j)
                    {
                        fourier_test_block(i, j) = libbse::Complex(
                            std::sin(1.0 + i + 2.0 * j + 3.0 * iq + 5.0 * iat + 7.0 * jat),
                            std::cos(0.5 + 2.0 * i + j + iq + 3.0 * iat - jat));
                    }
                }
                fourier_test_w_q[iat][jat][fourier_test_q_points[iq]] = fourier_test_block;
            }
        }
    }

    // Reference: LibRPA FT_Wc_q2R, as previously called by LibBSE.
    const auto fourier_test_reference = librpa_int::FT_Wc_q2R(
        fourier_test_dataset.comm_h, fourier_test_dataset.basis_aux,
        fourier_test_dataset.symmetry_context, fourier_test_w_q, fourier_test_dataset.tfg,
        fourier_test_dataset.pbc, fourier_test_dataset.pbc.Rlist, true, "", false);
    const auto fourier_test_candidate = LibRPA_API::transform_screened_q_to_r(
        fourier_test_dataset, fourier_test_dataset.pbc, fourier_test_w_q);

    std::size_t fourier_test_compared_blocks = 0;
    for (const auto &[iat, fourier_test_j_blocks] : fourier_test_reference)
    {
        for (const auto &[jat, fourier_test_r_blocks] : fourier_test_j_blocks)
        {
            for (const auto &[r, fourier_test_reference_block] : fourier_test_r_blocks)
            {
                const auto &fourier_test_candidate_block = fourier_test_candidate.at(
                    static_cast<int>(iat)).at({static_cast<int>(jat), {r.x, r.y, r.z}});
                for (int i = 0; i < fourier_test_reference_block.nr(); ++i)
                {
                    for (int j = 0; j < fourier_test_reference_block.nc(); ++j)
                    {
                        if (std::abs(fourier_test_candidate_block(i, j) -
                                     fourier_test_reference_block(i, j)) > 1.e-14)
                        {
                            throw std::runtime_error(
                                "OpenMP W(q)->W(R) differs from LibRPA FT_Wc_q2R");
                        }
                    }
                }
                ++fourier_test_compared_blocks;
            }
        }
    }
    // 4 atom pairs x 6 cells, and no extra blocks in the candidate.
    std::size_t fourier_test_candidate_blocks = 0;
    for (const auto &fourier_test_candidate_row : fourier_test_candidate)
    {
        fourier_test_candidate_blocks += fourier_test_candidate_row.second.size();
    }
    if (fourier_test_compared_blocks != 24 || fourier_test_candidate_blocks != 24)
    {
        throw std::runtime_error("W(q)->W(R) block count differs from 4 pairs x 6 cells");
    }
}

int main(int argc,char **argv) {
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    long id=rank==0 ? getpid():0; MPI_Bcast(&id,1,MPI_LONG,0,MPI_COMM_WORLD);
    const auto root=fs::temp_directory_path()/("libbse_aims_w_"+std::to_string(id));
    try {
        LibRPA_API::initialize();
        test_screened_fourier_matches_librpa();
        if (rank==0) { fs::create_directories(root); fixture(root); write_binary_w_fixture(root); }
        MPI_Barrier(MPI_COMM_WORLD);
        {
            librpa_int::Dataset dataset(MPI_COMM_WORLD);
            dataset.basis_aux=librpa_int::AtomicBasis(std::vector<std::size_t>{1,1});
            dataset.pbc.set_latvec({1,0,0,0,1,0,0,0,1});
            dataset.pbc.set_period(3,1,1);
            dataset.mf_band=librpa_int::MeanField(1,4,2,2);
            libbse::InputParameters options;
            options.input_format="fhi_aims"; options.screened_dir=root.string();
            std::vector<int> rows;
            for (int i=0;i<2;++i) if (i%size==rank) rows.push_back(i);
            const auto wr=libbse::read_aims_screened_interaction(options,dataset,rows,{0,1});
            for (int i:rows) for (int j=0;j<2;++j) for (int iq=0;iq<4;++iq) {
                const double q=iq/4.0, angle=2*libbse::PARAM.constants.pi*q;
                libbse::Complex x=0;
                for (const auto &r:dataset.pbc.Rlist)
                    x+=std::exp(libbse::Complex(0,angle*r.x))*wr.at(i).at({j,{r.x,r.y,r.z}})(0,0);
                const libbse::Complex expected=i==j ? libbse::Complex(i==0 ? 2+std::cos(angle):4)
                    : libbse::Complex(0,(i==0 ? 1:-1)*std::sin(angle));
                if (std::abs(x-expected)>1.e-12) throw std::runtime_error("3->4 Fourier interpolation or W normalization failed");
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0)
            {
                write_single_frequency_fixture(root);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            const auto aims_single_wr =
                libbse::read_aims_screened_interaction(options, dataset, rows, {0,1});
            for (int aims_iatom : rows)
            {
                for (int aims_jatom = 0; aims_jatom < 2; ++aims_jatom)
                {
                    for (const auto &aims_cell : dataset.pbc.Rlist)
                    {
                        const auto aims_pair_key = std::make_pair(aims_jatom,
                            std::array<int,3>{aims_cell.x, aims_cell.y, aims_cell.z});
                        if (std::abs(aims_single_wr.at(aims_iatom).at(aims_pair_key)(0,0) -
                                     wr.at(aims_iatom).at(aims_pair_key)(0,0)) > 1.e-12)
                        {
                            throw std::runtime_error("single-frequency fast path changed W(R)");
                        }
                    }
                }
            }
            // q=1/3 is owned by rank 1 in multi-rank batches. Its parse errors
            // must be propagated to idle ranks as well as the rank-zero reader.
            const std::vector<std::string> aims_invalid_rows = {
                "1 0.001 1 1 3 0", // duplicate
                "1 0.001 0 1 3 0", // invalid auxiliary index
                "1 0.001 1 1 nan 0", // non-finite value
                "1 0.002 1 1 3 0", // inconsistent physical frequency
                "2 0.001 1 1 3 0", // illegal single-node index
                "1 0.001 1x 1 3 0", // malformed token
                "1 0.001 1 1 1e9999 0"}; // overflow
            for (const auto &aims_invalid_row : aims_invalid_rows)
            {
                if (rank == 0)
                {
                    write_single_frequency_fixture(root);
                    std::ofstream aims_bad_file(root / "periodic_gw_w_q_2_rank_1.dat",
                                                std::ios::app);
                    aims_bad_file << aims_invalid_row << '\n';
                }
                MPI_Barrier(MPI_COMM_WORLD);
                int aims_rejected = 0;
                try
                {
                    (void)libbse::read_aims_screened_interaction(options, dataset, rows, {0,1});
                }
                catch (const std::runtime_error &)
                {
                    aims_rejected = 1;
                }
                int aims_rejected_count = 0;
                MPI_Allreduce(&aims_rejected, &aims_rejected_count, 1, MPI_INT, MPI_SUM,
                              MPI_COMM_WORLD);
                if (aims_rejected_count != size)
                {
                    throw std::runtime_error("reader error was not rejected on every rank");
                }
            }
            if (rank == 0)
            {
                write_single_frequency_fixture(root);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            const auto w_binary_r=libbse::read_full_w_binary(options,dataset,rows,{0,1});
            for (int w_iatom:rows) for (int w_jatom=0; w_jatom<2; ++w_jatom)
                for (const auto &w_cell:dataset.pbc.Rlist) {
                    const auto w_pair_key=std::make_pair(w_jatom,std::array<int,3>{w_cell.x,w_cell.y,w_cell.z});
                    if (std::abs(w_binary_r.at(w_iatom).at(w_pair_key)(0,0)-wr.at(w_iatom).at(w_pair_key)(0,0))>1.e-12)
                        throw std::runtime_error("binary and text full W Fourier transforms differ");
                }
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) write_symmetry_fixture(root);
            MPI_Barrier(MPI_COMM_WORLD);
            const auto symmetry_wr=libbse::read_aims_screened_interaction(options,dataset,rows,{0,1});
            for (int symmetry_iatom:rows) for (int symmetry_jatom=0; symmetry_jatom<2; ++symmetry_jatom)
                for (const auto &symmetry_cell:dataset.pbc.Rlist) {
                    const auto symmetry_pair_key=std::make_pair(symmetry_jatom,std::array<int,3>{symmetry_cell.x,symmetry_cell.y,symmetry_cell.z});
                    if (std::abs(symmetry_wr.at(symmetry_iatom).at(symmetry_pair_key)(0,0)-wr.at(symmetry_iatom).at(symmetry_pair_key)(0,0))>1.e-12)
                        throw std::runtime_error("Delta restoration changed the full W Fourier transform");
                }
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) fs::remove(root/"aims_symmetry_all.dat");
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) fs::remove(root/"full_w_q_2.bin");
            MPI_Barrier(MPI_COMM_WORLD);
            bool w_missing_rejected=false;
            try { (void)libbse::read_full_w_binary(options,dataset,rows,{0,1}); }
            catch (const std::runtime_error &) { w_missing_rejected=true; }
            if (!w_missing_rejected) throw std::runtime_error("incomplete binary W mesh was accepted");
            // Remove one producer block: missing values must not silently turn
            // into zeros in the BSE kernel. All ranks see the same invalid input.
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) fs::remove(root/"periodic_gw_w_q_1_rank_1.dat");
            MPI_Barrier(MPI_COMM_WORLD);
            bool rejected=false;
            try { (void)libbse::read_aims_screened_interaction(options,dataset,rows,{0,1}); }
            catch (const std::runtime_error &) { rejected=true; }
            if (!rejected) throw std::runtime_error("incomplete rank blocks were accepted");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank==0) fs::remove_all(root);
        LibRPA_API::finalize();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();
}
