#include "parameter.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace libbse
{
namespace
{

std::string trim(std::string value)
{
    const auto first = std::find_if_not(value.begin(), value.end(),
                                        [](unsigned char c) { return std::isspace(c); });
    const auto last = std::find_if_not(value.rbegin(), value.rend(),
                                       [](unsigned char c) { return std::isspace(c); }).base();
    return first < last ? std::string(first, last) : std::string{};
}

std::string lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

int parse_integer(const std::string &key, const std::string &value)
{
    std::size_t consumed = 0;
    int result = 0;
    try
    {
        result = std::stoi(value, &consumed);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument("invalid integer for " + key + ": " + value);
    }
    if (!trim(value.substr(consumed)).empty())
        throw std::invalid_argument("invalid integer for " + key + ": " + value);
    return result;
}

double parse_double(const std::string &key, const std::string &value)
{
    std::size_t consumed = 0;
    double result = 0.0;
    try
    {
        result = std::stod(value, &consumed);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument("invalid real value for " + key + ": "
                                    + value);
    }
    if (!trim(value.substr(consumed)).empty())
        throw std::invalid_argument("invalid real value for " + key + ": "
                                    + value);
    return result;
}

bool parse_boolean(const std::string &key, const std::string &value)
{
    const std::string normalized = lower(trim(value));
    if (normalized == "1" || normalized == "true" || normalized == "t"
        || normalized == ".true." || normalized == "yes")
        return true;
    if (normalized == "0" || normalized == "false" || normalized == "f"
        || normalized == ".false." || normalized == "no")
        return false;
    throw std::invalid_argument("invalid boolean for " + key + ": " + value);
}

std::vector<std::string> parse_string_list(std::string value)
{
    std::replace(value.begin(), value.end(), ',', ' ');
    std::istringstream input(value);
    std::vector<std::string> result;
    std::string item;
    while (input >> item) result.push_back(lower(item));
    return result;
}

std::map<std::string, std::string> parse_assignments(const std::string &contents)
{
    std::map<std::string, std::string> assignments;
    std::istringstream input(contents);
    std::string line;
    int line_number = 0;
    while (std::getline(input, line))
    {
        ++line_number;
        const auto hash = line.find('#');
        const auto bang = line.find('!');
        const auto comment = std::min(hash, bang);
        if (comment != std::string::npos) line.erase(comment);
        line = trim(line);
        if (line.empty() || lower(line) == "input_parameters") continue;

        const auto separator = line.find_first_of("= \t");
        const std::string key = lower(trim(line.substr(0, separator)));
        std::string value = separator == std::string::npos
                                ? std::string{}
                                : trim(line.substr(separator));
        if (!value.empty() && value.front() == '=') value = trim(value.substr(1));
        if (key.empty() || value.empty())
            throw std::invalid_argument("invalid libbse.in assignment at line "
                                        + std::to_string(line_number));
        assignments[key] = value;
    }
    return assignments;
}

} // namespace

Parameter PARAM;

bool InputParameters::solve_tda() const noexcept
{
    return bse_tda == "tda" || bse_tda == "both";
}

bool InputParameters::solve_full() const noexcept
{
    return bse_tda == "full" || bse_tda == "both";
}

bool InputParameters::spectrum_only() const noexcept
{
    return bse_solver == "spectrum";
}

bool InputParameters::has_spin_type(const std::string &spin_type) const noexcept
{
    return std::find(bse_spin_types.begin(), bse_spin_types.end(), spin_type)
           != bse_spin_types.end();
}

bool InputParameters::requires_hartree() const noexcept
{
    return has_spin_type("singlet") || has_spin_type("rpa");
}

bool InputParameters::requires_screened() const noexcept
{
    return has_spin_type("singlet") || has_spin_type("triplet");
}

bool InputParameters::ipa_only() const noexcept
{
    return bse_spin_types.size() == 1 && bse_spin_types.front() == "ipa";
}

InteractionCoefficients interaction_coefficients(const std::string &spin_type)
{
    if (spin_type == "singlet") return {2.0, -1.0};
    if (spin_type == "triplet") return {0.0, -1.0};
    if (spin_type == "rpa") return {2.0, 0.0};
    if (spin_type == "ipa") return {0.0, 0.0};
    throw std::invalid_argument("unsupported BSE spin type: " + spin_type);
}

void Parameter::read(const fs::path &filename)
{
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("cannot open LibBSE input file: " + filename.string());
    std::ostringstream contents;
    contents << input.rdbuf();
    const fs::path absolute = fs::absolute(filename);
    parse(contents.str(), absolute.parent_path());
}

void Parameter::parse(const std::string &contents, const fs::path &base_directory)
{
    inp = InputParameters{};
    auto assignments = parse_assignments(contents);

    const auto take = [&](const char *name) -> std::string
    {
        const auto iter = assignments.find(name);
        if (iter == assignments.end()) return {};
        std::string value = iter->second;
        assignments.erase(iter);
        return value;
    };
    if (auto value = take("input_dir"); !value.empty()) inp.input_dir = value;
    if (auto value = take("output_dir"); !value.empty()) inp.output_dir = value;
    if (auto value = take("input_format"); !value.empty())
        inp.input_format = lower(value);
    if (auto value = take("qp_data"); !value.empty()) inp.qp_data = value;
    if (auto value = take("qp_format"); !value.empty())
        inp.qp_format = lower(value);
    if (auto value = take("screened_format"); !value.empty())
        inp.screened_format = lower(value);
    if (auto value = take("chi0_coulomb_metric"); !value.empty()) inp.chi0_coulomb_metric = lower(value);
    if (auto value = take("out_screening_matrices"); !value.empty()) inp.out_screening_matrices = parse_boolean("out_screening_matrices", value);
    if (auto value = take("chi0_headwing"); !value.empty()) inp.chi0_headwing = parse_boolean("chi0_headwing", value);
    if (auto value = take("bse_plasma_energy_ev"); !value.empty()) inp.bse_plasma_energy_ev = parse_double("bse_plasma_energy_ev", value);
    if (auto value = take("screened_dir"); !value.empty())
        inp.screened_dir = value;
    if (auto value = take("bse_nstates"); !value.empty())
        inp.bse_nstates = parse_integer("bse_nstates", value);
    if (auto value = take("nocc"); !value.empty()) inp.nocc = parse_integer("nocc", value);
    if (auto value = take("nvirt"); !value.empty()) inp.nvirt = parse_integer("nvirt", value);
    if (auto value = take("bse_solver"); !value.empty()) inp.bse_solver = lower(value);
    if (auto value = take("bse_spin_types"); !value.empty())
        inp.bse_spin_types = parse_string_list(value);
    if (auto value = take("bse_continue"); !value.empty())
        inp.bse_continue = parse_integer("bse_continue", value);
    if (auto value = take("bse_tda"); !value.empty()) inp.bse_tda = lower(value);
    if (auto value = take("bse_ri_hartree"); !value.empty())
        inp.bse_ri_hartree = parse_boolean("bse_ri_hartree", value);
    if (auto value = take("bse_use_fine_kgrid"); !value.empty())
        inp.bse_use_fine_kgrid = parse_integer("bse_use_fine_kgrid", value);
    if (auto value = take("bse_q_approx_mode"); !value.empty())
        inp.bse_q_approx_mode = parse_integer("bse_q_approx_mode", value);
    if (auto value = take("out_bse_ab"); !value.empty())
        inp.out_bse_ab = parse_boolean("out_bse_ab", value);
    if (auto value = take("abs_gauge"); !value.empty()) inp.abs_gauge = lower(value);
    if (auto value = take("wavefunction_gauge"); !value.empty())
        inp.wavefunction_gauge = lower(value);
    if (auto value = take("spectrum_broadening_ev"); !value.empty())
        inp.spectrum_broadening_ev
            = parse_double("spectrum_broadening_ev", value);
    if (auto value = take("spectrum_energy_step_ev"); !value.empty())
        inp.spectrum_energy_step_ev
            = parse_double("spectrum_energy_step_ev", value);
    if (auto value = take("spectrum_energy_min_ev"); !value.empty())
        inp.spectrum_energy_min_ev
            = parse_double("spectrum_energy_min_ev", value);
    if (auto value = take("spectrum_energy_max_ev"); !value.empty())
        inp.spectrum_energy_max_ev
            = parse_double("spectrum_energy_max_ev", value);

    if (!assignments.empty())
        throw std::invalid_argument("unknown LibBSE input parameter: "
                                    + assignments.begin()->first);
    validate_and_resolve(base_directory);
}

void Parameter::validate_and_resolve(const fs::path &base_directory)
{
    inp.input_dir = trim(inp.input_dir);
    inp.output_dir = trim(inp.output_dir);
    inp.input_format = lower(trim(inp.input_format));
    inp.qp_data = trim(inp.qp_data);
    inp.qp_format = lower(trim(inp.qp_format));
    inp.screened_dir = trim(inp.screened_dir);
    inp.screened_format = lower(trim(inp.screened_format));
    if (inp.screened_format != "librpa_wc" && inp.screened_format != "fhi_aims_w"
        && inp.screened_format != "fhi_aims_chi0" && inp.screened_format != "librpa_chi0")
        throw std::invalid_argument("screened_format must be librpa_wc, fhi_aims_w, fhi_aims_chi0 or librpa_chi0");
    if (inp.chi0_coulomb_metric != "full" && inp.chi0_coulomb_metric != "single_cut")
        throw std::invalid_argument("chi0_coulomb_metric must be full or single_cut");
    if (inp.screened_format.find("chi0") != std::string::npos && inp.chi0_headwing && inp.chi0_coulomb_metric != "full")
        throw std::invalid_argument("LibRPA head/wing requires chi0_coulomb_metric full");
    if (!std::isfinite(inp.bse_plasma_energy_ev) || inp.bse_plasma_energy_ev < 0)
        throw std::invalid_argument("bse_plasma_energy_ev must be finite and nonnegative");
    if (inp.bse_plasma_energy_ev > 0 && (inp.screened_format.find("chi0") == std::string::npos
        || inp.bse_tda != "tda" || inp.bse_solver != "elpa"))
        throw std::invalid_argument("effective dynamical BSE requires a chi0 input, bse_tda tda and bse_solver elpa");
    inp.bse_solver = lower(trim(inp.bse_solver));
    for (std::string &spin_type : inp.bse_spin_types)
        spin_type = lower(trim(spin_type));
    inp.bse_tda = lower(trim(inp.bse_tda));
    inp.abs_gauge = lower(trim(inp.abs_gauge));
    inp.wavefunction_gauge = lower(trim(inp.wavefunction_gauge));

    if (inp.input_dir.empty())
        throw std::invalid_argument("input_dir is required in libbse.in");
    if (inp.output_dir.empty())
        throw std::invalid_argument("output_dir must not be empty");
    if (inp.input_format != "auto" && inp.input_format != "librpa"
        && inp.input_format != "fhi_aims")
        throw std::invalid_argument(
            "input_format must be auto, librpa, or fhi_aims");
    if (inp.qp_format != "auto" && inp.qp_format != "energy_qp"
        && inp.qp_format != "fine_band")
        throw std::invalid_argument(
            "qp_format must be auto, energy_qp, or fine_band");
    if (inp.nocc <= 0 || inp.nvirt <= 0
        || inp.bse_nstates == 0 || inp.bse_nstates < -1)
        throw std::invalid_argument("invalid nocc, nvirt, or bse_nstates");
    if (inp.bse_solver != "elpa" && inp.bse_solver != "spectrum")
        throw std::invalid_argument("bse_solver must be elpa or spectrum");
    if (inp.bse_tda != "tda" && inp.bse_tda != "full" && inp.bse_tda != "both")
        throw std::invalid_argument("bse_tda must be tda, full, or both");
    if (inp.bse_spin_types.empty())
        throw std::invalid_argument("bse_spin_types must not be empty");
    std::set<std::string> unique_spin_types;
    for (const std::string &spin_type : inp.bse_spin_types)
    {
        (void)interaction_coefficients(spin_type);
        if (!unique_spin_types.insert(spin_type).second)
            throw std::invalid_argument("duplicate BSE spin type: " + spin_type);
    }
    if (inp.bse_plasma_energy_ev > 0 && !inp.requires_screened())
        throw std::invalid_argument("effective dynamical BSE requires a singlet or triplet screened channel");
    if (inp.ipa_only() && inp.bse_tda != "tda")
        throw std::invalid_argument("IPA requires bse_tda tda");
    if (inp.bse_continue != 0)
        throw std::invalid_argument("this LibBSE path currently requires bse_continue 0");
    if (!inp.bse_ri_hartree && inp.requires_hartree())
        throw std::invalid_argument(
            "singlet and RPA channels require bse_ri_hartree 1 in LibBSE");
    if (inp.bse_use_fine_kgrid != 0 && inp.bse_use_fine_kgrid != 1)
        throw std::invalid_argument("bse_use_fine_kgrid must be 0 or 1");
    if (inp.bse_q_approx_mode != 0)
        throw std::invalid_argument("this LibBSE path currently requires bse_q_approx_mode 0");
    if (inp.out_bse_ab)
        throw std::invalid_argument("out_bse_ab is not implemented in LibBSE");
    if (inp.abs_gauge != "velocity")
        throw std::invalid_argument("LibBSE supports only abs_gauge velocity");
    if (inp.wavefunction_gauge != "auto"
        && inp.wavefunction_gauge != "native"
        && inp.wavefunction_gauge != "first_k")
        throw std::invalid_argument(
            "wavefunction_gauge must be auto, native, or first_k");
    if (!std::isfinite(inp.spectrum_broadening_ev)
        || !std::isfinite(inp.spectrum_energy_step_ev)
        || !std::isfinite(inp.spectrum_energy_min_ev)
        || !std::isfinite(inp.spectrum_energy_max_ev)
        || inp.spectrum_broadening_ev <= 0.0
        || inp.spectrum_energy_step_ev <= 0.0
        || inp.spectrum_energy_min_ev < 0.0
        || (inp.spectrum_energy_max_ev >= 0.0
            && inp.spectrum_energy_max_ev < inp.spectrum_energy_min_ev))
        throw std::invalid_argument(
            "invalid optical-spectrum energy grid or broadening");

    fs::path input_path(inp.input_dir);
    if (input_path.is_relative()) input_path = base_directory / input_path;
    inp.input_dir = fs::absolute(input_path).lexically_normal().string();

    fs::path output_path(inp.output_dir);
    if (output_path.is_relative()) output_path = base_directory / output_path;
    inp.output_dir = fs::absolute(output_path).lexically_normal().string();

    fs::path qp_path(inp.qp_data.empty() ? inp.input_dir : inp.qp_data);
    if (qp_path.is_relative()) qp_path = base_directory / qp_path;
    inp.qp_data = fs::absolute(qp_path).lexically_normal().string();

    fs::path screened_path = inp.screened_dir.empty()
        ? fs::path(inp.input_dir).parent_path() / "librpa.d"
        : fs::path(inp.screened_dir);
    if (screened_path.is_relative()) screened_path = base_directory / screened_path;
    inp.screened_dir = fs::absolute(screened_path).lexically_normal().string();
}

void Parameter::print(std::ostream &output) const
{
    output << "LibBSE input parameters\n"
           << "  input_dir: " << inp.input_dir << '\n'
           << "  output_dir: " << inp.output_dir << '\n'
           << "  input_format: " << inp.input_format << '\n'
           << "  qp_data: " << inp.qp_data << '\n'
           << "  qp_format: " << inp.qp_format << '\n'
           << "  screened_format: " << inp.screened_format << '\n'
           << "  chi0_coulomb_metric: " << inp.chi0_coulomb_metric << '\n'
           << "  chi0_headwing: " << inp.chi0_headwing << '\n'
           << "  bse_plasma_energy_ev: " << inp.bse_plasma_energy_ev << '\n'
           << "  screened_dir: " << inp.screened_dir << '\n'
           << "  bse_nstates: " << inp.bse_nstates << '\n'
           << "  nocc: " << inp.nocc << '\n'
           << "  nvirt: " << inp.nvirt << '\n'
           << "  bse_solver: " << inp.bse_solver << '\n'
           << "  bse_spin_types:";
    for (const std::string &spin_type : inp.bse_spin_types)
        output << ' ' << spin_type;
    output << '\n'
           << "  bse_tda: " << inp.bse_tda << '\n'
           << "  bse_ri_hartree: " << inp.bse_ri_hartree << '\n'
           << "  bse_use_fine_kgrid: " << inp.bse_use_fine_kgrid << '\n'
           << "  bse_q_approx_mode: " << inp.bse_q_approx_mode << '\n'
           << "  abs_gauge: " << inp.abs_gauge << '\n'
           << "  wavefunction_gauge: " << inp.wavefunction_gauge << '\n'
           << "  spectrum_broadening_ev: "
           << inp.spectrum_broadening_ev << '\n'
           << "  spectrum_energy_step_ev: "
           << inp.spectrum_energy_step_ev << '\n'
           << "  spectrum_energy_min_ev: "
           << inp.spectrum_energy_min_ev << '\n'
           << "  spectrum_energy_max_ev: "
           << inp.spectrum_energy_max_ev << '\n';
}

} // namespace libbse
