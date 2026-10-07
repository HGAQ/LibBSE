#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace libbse
{

struct Constants
{
    static constexpr double pi = 3.141592653589793238462643383279502884;
    static constexpr double ry_to_ev = 13.605693122994;
    static constexpr double ha_to_ry = 2.0;
    static constexpr double cs_threshold = 1.0e-12;
    static constexpr double coulomb_threshold = 1.0e-12;
    static constexpr double kpoint_tolerance = 1.0e-10;
    static constexpr double band_file_kpoint_tolerance = 1.0e-6;
    static constexpr double energy_qp_kpoint_tolerance = 5.1e-5;
    static constexpr double occupation_tolerance = 0.1;
    static constexpr double matrix_symmetry_threshold = 1.0e-6;
    static constexpr double output_zero_tolerance = 1.0e-10;
    static constexpr double zero_gap_tolerance = 1.0e-14;
    static constexpr const char *input_filename = "libbse.in";
};

struct InputParameters
{
    // Names exposed in libbse.in
    std::string input_dir;
    std::string output_dir = "libbse.d";
    std::string input_format = "auto";
    std::string qp_data;
    std::string qp_format = "auto";
    std::string screened_dir;
    // librpa_wc: Wc(R); fhi_aims_w: full W(q,iw).
    // chi0 paths use full V inside epsilon and cut V outside, as in LibRPA GW.
    std::string screened_format = "librpa_wc";
    std::string chi0_coulomb_metric = "full";
    bool chi0_headwing = true;
    bool out_screening_matrices = false;
    // Positive plasma energy enables a one-shot effective kernel. Eb is computed
    // from the lowest STATIC excitation and the minimum direct QP transition.
    // Other states are eigenstates of this same target-exciton kernel.
    double bse_plasma_energy_ev = 0.0;
    // Storage/scheduling controls only: do not change the transition space.
    bool bse_memory_optimized = true;
    int bse_ri_batch_blocks = 64; // maximum returned k-pair blocks/rank; 0 = legacy
    int bse_nstates = -1;
    int nocc = 4;
    int nvirt = 4;
    std::string bse_solver = "elpa";
    std::vector<std::string> bse_spin_types{"singlet", "triplet"};
    int bse_continue = 0;
    std::string bse_tda = "both";
    bool bse_ri_hartree = true;
    int bse_use_fine_kgrid = 1;
    int bse_q_approx_mode = 0;
    bool out_bse_ab = false;
    // Disable only text eigenvector output; all states are still solved and
    // contracted for spectra. Existing spectrum-only restarts need these files.
    bool out_bse_eigenvectors = true;
    std::string abs_gauge = "velocity";
    std::string wavefunction_gauge = "auto";
    double spectrum_broadening_ev = 0.10;
    double spectrum_energy_step_ev = 0.01;
    double spectrum_energy_min_ev = 0.0;
    // A negative maximum selects max(excitation energy) + 5*broadening.
    double spectrum_energy_max_ev = -1.0;

    bool solve_tda() const noexcept;
    bool solve_full() const noexcept;
    bool spectrum_only() const noexcept;
    bool has_spin_type(const std::string &spin_type) const noexcept;
    bool requires_hartree() const noexcept;
    bool requires_screened() const noexcept;
    bool ipa_only() const noexcept;
};

struct InteractionCoefficients
{
    double hartree = 0.0;
    double screened = 0.0;
};

InteractionCoefficients interaction_coefficients(const std::string &spin_type);

class Parameter
{
  public:
    InputParameters inp; // store all input parameters read from libbse.in
    const Constants constants{};

    void read(const std::filesystem::path &filename = Constants::input_filename);// then prase 
    void parse(const std::string &contents,
               const std::filesystem::path &base_directory = ".");
    void print(std::ostream &output) const;

  private:
    void validate_and_resolve(const std::filesystem::path &base_directory); // add rule that check the input parameters and resolve the relative path to absolute path
};

extern Parameter PARAM;

} // namespace libbse
