#include "fhi_aims_adapter.h"

#include <fstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace libbse
{
namespace
{

bool basis_identifies_aims(const fs::path &input_dir)
{
    std::ifstream input(input_dir / "basis_out");
    int atoms = 0;
    int nao = 0;
    int naux = 0;
    std::string producer;
    return input && (input >> atoms >> nao >> naux >> producer)
           && producer == "aims";
}

void create_source_link(const fs::path &source, const fs::path &target)
{
    if (fs::exists(target) || fs::is_symlink(target))
    {
        if (fs::is_symlink(target)
            && fs::weakly_canonical(target) == fs::weakly_canonical(source))
            return;
        throw std::runtime_error("FHI-aims adapter target already exists: "
                                 + target.string());
    }
    fs::create_symlink(fs::absolute(source), target);
}

void validate_adapter_source(const fs::path &view, const fs::path &source)
{
    const fs::path marker = view / ".source_directory";
    if (fs::exists(marker))
    {
        std::ifstream input(marker);
        std::string recorded;
        std::getline(input, recorded);
        if (fs::path(recorded) != fs::weakly_canonical(source))
            throw std::runtime_error(
                "the existing FHI-aims reader view belongs to another input directory: "
                + recorded);
        return;
    }
    std::ofstream output(marker);
    if (!output)
        throw std::runtime_error("cannot create FHI-aims adapter marker");
    output << fs::weakly_canonical(source).string() << '\n';
}

std::string collective_error(MPI_Comm comm, const std::string &root_error)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    int length = rank == 0 ? static_cast<int>(root_error.size()) : 0;
    MPI_Bcast(&length, 1, MPI_INT, 0, comm);
    std::string error(static_cast<std::size_t>(length), '\0');
    if (rank == 0) error = root_error;
    if (length != 0) MPI_Bcast(error.data(), length, MPI_CHAR, 0, comm);
    return error;
}

void install_velocity(const fs::path &source, const fs::path &view)
{
    const fs::path source_velocity = source / "velocity_matrix";
    const fs::path view_velocity = view / "velocity_matrix";
    if (fs::exists(view_velocity) || fs::is_symlink(view_velocity))
        fs::remove(view_velocity);

    if (!fs::is_regular_file(source_velocity))
        throw std::runtime_error(
            "FHI-aims input has no canonical velocity_matrix; run "
            "tools/aims_mommat_to_velocity.py AIMS_EXPORT_DIR first");
    create_source_link(source_velocity, view_velocity);
}

} // namespace

void resolve_input_format(InputParameters &options)
{
    if (options.input_format == "auto")
        options.input_format = basis_identifies_aims(options.input_dir)
                                   ? "fhi_aims" : "librpa";
    if (options.input_format == "fhi_aims"
        && !basis_identifies_aims(options.input_dir))
        throw std::runtime_error(
            "input_format fhi_aims requires an aims producer tag in basis_out");

    if (options.wavefunction_gauge == "auto")
    {
        // RI coefficients, KS coefficients, and momentum elements from
        // FHI-aims share one native band gauge.  An independent overlap gauge
        // would rotate degeneracies inconsistently with the optical operator.
        options.wavefunction_gauge
            = options.input_format == "fhi_aims" ? "native" : "first_k";
    }
}

fs::path prepare_fhi_aims_reader_view(MPI_Comm comm,
                                      const InputParameters &options)
{
    const fs::path source(options.input_dir);
    const fs::path view = fs::path(options.output_dir)
                          / "fhi_aims_reader_view";
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    std::string root_error;
    if (rank == 0)
    {
        try
        {
            fs::create_directories(view);
            validate_adapter_source(view, source);
            for (const auto &entry : fs::directory_iterator(source))
            {
                if (!entry.is_regular_file()) continue;
                const std::string name = entry.path().filename().string();
                // velocity_matrix is installed with explicit validation below.
                if (name == "velocity_matrix") continue;
                const fs::path target = view / entry.path().filename();
                if (!fs::exists(target) && !fs::is_symlink(target))
                    create_source_link(entry.path(), target);
            }
            for (const auto &entry : fs::directory_iterator(source))
            {
                const std::string name = entry.path().filename().string();
                constexpr const char *prefix = "coulomb_cut_";
                if (!entry.is_regular_file() || name.rfind(prefix, 0) != 0)
                    continue;
                const fs::path target
                    = view / ("coulomb_unshrinked_cut_"
                              + name.substr(std::char_traits<char>::length(prefix)));
                if (!fs::exists(target) && !fs::is_symlink(target))
                    create_source_link(entry.path(), target);
            }
            //before run, use tools/aims_mommat_to_velocity.py to convert 
            install_velocity(source, view);
        }
        catch (const std::exception &error)
        {
            root_error = error.what();
        }
    }
    const std::string error = collective_error(comm, root_error);
    if (!error.empty()) throw std::runtime_error(error);
    MPI_Barrier(comm);
    return view;
}

} // namespace libbse
