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
abs_gauge velocity
)", "/tmp/libbse-parameter-base");

    require(parameter.inp.bse_nstates == 8, "bse_nstates was not parsed");
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
