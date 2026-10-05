#include "parameter/parameter.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

void require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

void test_whitespace_separated_parameters()
{
    libbse::Parameter parameter;
    parameter.parse(R"(
INPUT_PARAMETERS
input_dir relative/OUT.librpa
output_dir results/test_bse
input_format fhi_aims
qp_data ../gw
qp_format energy_qp
screened_dir ../screening
bse_nstates 8
nocc 2
nvirt 3
bse_solver spectrum
bse_spin_types singlet triplet rpa ipa
bse_tda full
bse_ri_hartree 1
bse_use_fine_kgrid 1
bse_q_approx_mode 0
out_bse_ab 0
out_bse_eigenvectors false
abs_gauge velocity
wavefunction_gauge native
spectrum_broadening_ev 0.15
spectrum_energy_step_ev 0.02
spectrum_energy_min_ev 0.5
spectrum_energy_max_ev 8.0
)", "/tmp/libbse-parameter-base");

    require(parameter.inp.bse_nstates == 8, "bse_nstates was not parsed");
    require(!parameter.inp.out_bse_eigenvectors, "eigenvector output was not disabled");
    require(parameter.inp.spectrum_only(), "bse_solver was not parsed");
    require(parameter.inp.nocc == 2 && parameter.inp.nvirt == 3,
            "BSE band counts were not parsed");
    require(!parameter.inp.solve_tda() && parameter.inp.solve_full(),
            "bse_tda full was not interpreted correctly");
    require(parameter.inp.bse_spin_types
                == std::vector<std::string>({"singlet", "triplet", "rpa", "ipa"}),
            "bse_spin_types list was not parsed");
    require(parameter.inp.input_dir
                == "/tmp/libbse-parameter-base/relative/OUT.librpa",
            "relative input_dir was not resolved against libbse.in");
    require(parameter.inp.output_dir
                == "/tmp/libbse-parameter-base/results/test_bse",
            "relative output_dir was not resolved against libbse.in");
    require(parameter.inp.input_format == "fhi_aims"
                && parameter.inp.qp_format == "energy_qp",
            "FHI-aims format selectors were not parsed");
    require(parameter.inp.qp_data == "/tmp/gw"
                && parameter.inp.screened_dir == "/tmp/screening",
            "FHI-aims data paths were not resolved against libbse.in");
    require(parameter.inp.wavefunction_gauge == "native"
                && parameter.inp.spectrum_broadening_ev == 0.15
                && parameter.inp.spectrum_energy_step_ev == 0.02
                && parameter.inp.spectrum_energy_min_ev == 0.5
                && parameter.inp.spectrum_energy_max_ev == 8.0,
            "FHI-aims spectrum/gauge options were not parsed");
}

void test_librpa_style_assignments_and_last_value_wins()
{
    libbse::Parameter parameter;
    parameter.parse(R"(
input_dir = first
input_dir = second # the last assignment must win
bse_tda = tda
bse_ri_hartree = true
out_bse_ab = false
)", "/tmp");

    require(parameter.inp.input_dir == "/tmp/second",
            "key=value parsing or last-value semantics failed");
    require(parameter.inp.output_dir == "/tmp/libbse.d",
            "default output_dir was not resolved against libbse.in");
    require(parameter.inp.out_bse_eigenvectors, "legacy eigenvector-output default changed");
    require(parameter.inp.qp_data == "/tmp/second"
                && parameter.inp.screened_dir == "/tmp/librpa.d",
            "default QP or screened-interaction path is incorrect");
    require(parameter.inp.solve_tda() && !parameter.inp.solve_full(),
            "bse_tda tda was not interpreted correctly");
    require(parameter.inp.bse_spin_types
                == std::vector<std::string>({"singlet", "triplet"}),
            "bse_spin_types defaults were not applied");
}

void test_coarse_kgrid_mode()
{
    libbse::Parameter parameter;
    parameter.parse(
        "input_dir data\nbse_use_fine_kgrid 0\n",
        "/tmp");
    require(parameter.inp.bse_use_fine_kgrid == 0,
            "coarse-k-grid mode was not accepted");
}

void test_invalid_or_unsupported_parameters_are_rejected()
{
    const auto rejected = [](const std::string &contents)
    {
        try
        {
            libbse::Parameter parameter;
            parameter.parse("input_dir data\n" + contents, "/tmp");
        }
        catch (const std::invalid_argument &)
        {
            return true;
        }
        return false;
    };

    require(rejected("screened_format librpa_chi0\nbse_plasma_energy_ev 15\nbse_tda tda\nbse_spin_types ipa\n"), "IPA silently accepted an effective screening model");
    require(rejected("bse_plasma_energy_ev nan\n"), "nonfinite plasma energy was accepted");
    require(rejected("bse_plasma_energy_ev 15\nbse_tda tda\n"), "dynamic model accepted a W-only input");
    require(rejected("screened_format librpa_chi0\nchi0_coulomb_metric single_cut\nchi0_headwing true\n"), "head/wing accepted a cut-only dielectric metric");
    require(rejected("abs_gauge length\n"), "length gauge was not rejected");
    require(rejected("bse_spin_types unsupported\n"),
            "unsupported spin type was not rejected");
    require(rejected("bse_spin_types singlet singlet\n"),
            "duplicate spin types were not rejected");
    require(rejected("bse_spin_types ipa\nbse_tda full\n"),
            "full-BSE pure IPA was not rejected");
    require(rejected("bse_spin_types rpa\nbse_ri_hartree 0\n"),
            "RPA without the available LibRI Hartree path was not rejected");
    require(rejected("unknown_parameter 1\n"), "unknown parameter was not rejected");
    require(rejected("bse_nstates zero\n"), "malformed integer was not rejected");
    require(rejected("suffix old_output\n"), "obsolete suffix parameter was not rejected");
    require(rejected("read_file_dir old_input\n"),
            "obsolete read_file_dir parameter was not rejected");
    require(rejected("bse_use_fine_kgrid 2\n"),
            "unsupported fine-k-grid mode was not rejected");
    require(rejected("input_format unknown\n"),
            "unsupported input format was not rejected");
    require(rejected("qp_format unknown\n"),
            "unsupported QP format was not rejected");
    require(rejected("wavefunction_gauge random\n"),
            "unsupported wavefunction gauge was not rejected");
    require(rejected("bse_compute_spectrum 0\n"),
            "removed velocity-optional mode was not rejected");
    require(rejected("qp_format aims_gw\n"),
            "FHI-aims quasiparticle format was not rejected");
    require(rejected("spectrum_broadening_ev 0\n"),
            "zero spectrum broadening was not rejected");
    require(rejected("spectrum_energy_step_ev word\n"),
            "malformed spectrum step was not rejected");
    require(rejected("spectrum_broadening_ev nan\n"),
            "non-finite spectrum broadening was not rejected");
}

} // namespace

int main()
{
    try
    {
        test_whitespace_separated_parameters();
        test_librpa_style_assignments_and_last_value_wins();
        test_coarse_kgrid_mode();
        test_invalid_or_unsupported_parameters_are_rejected();
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
