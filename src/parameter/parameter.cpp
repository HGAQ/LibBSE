#include "parameter.h"

#include <algorithm>
#include <cctype>
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

    if (!assignments.empty())
        throw std::invalid_argument("unknown LibBSE input parameter: "
                                    + assignments.begin()->first);
    validate_and_resolve(base_directory);
}

void Parameter::validate_and_resolve(const fs::path &base_directory)
{
    inp.input_dir = trim(inp.input_dir);
    inp.output_dir = trim(inp.output_dir);
    inp.bse_solver = lower(trim(inp.bse_solver));
    for (std::string &spin_type : inp.bse_spin_types)
        spin_type = lower(trim(spin_type));
    inp.bse_tda = lower(trim(inp.bse_tda));
    inp.abs_gauge = lower(trim(inp.abs_gauge));

    if (inp.input_dir.empty())
        throw std::invalid_argument("input_dir is required in libbse.in");
    if (inp.output_dir.empty())
        throw std::invalid_argument("output_dir must not be empty");
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

    fs::path input_path(inp.input_dir);
    if (input_path.is_relative()) input_path = base_directory / input_path;
    inp.input_dir = fs::absolute(input_path).lexically_normal().string();

    fs::path output_path(inp.output_dir);
    if (output_path.is_relative()) output_path = base_directory / output_path;
    inp.output_dir = fs::absolute(output_path).lexically_normal().string();
}

void Parameter::print(std::ostream &output) const
{
    output << "LibBSE input parameters\n"
           << "  input_dir: " << inp.input_dir << '\n'
           << "  output_dir: " << inp.output_dir << '\n'
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
           << "  abs_gauge: " << inp.abs_gauge << '\n';
}

} // namespace libbse
