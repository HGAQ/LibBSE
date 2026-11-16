#pragma once
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace libbse
{
// Parse whitespace-delimited aims fields without constructing a stream per row.
// No locale changes are made: callers use the same process numeric locale as before.
inline void aims_skip_space(const char *&aims_cursor)
{
    while (std::isspace(static_cast<unsigned char>(*aims_cursor)))
    {
        ++aims_cursor;
    }
}

inline void aims_check_token_end(const char *aims_end)
{
    if (*aims_end != '\0' && !std::isspace(static_cast<unsigned char>(*aims_end)))
    {
        throw std::runtime_error("invalid aims numeric token suffix");
    }
}

inline int aims_read_integer(const char *&aims_cursor)
{
    aims_skip_space(aims_cursor);
    char *aims_end = nullptr;
    errno = 0;
    const long aims_integer = std::strtol(aims_cursor, &aims_end, 10);
    if (aims_end == aims_cursor || errno == ERANGE ||
        aims_integer < std::numeric_limits<int>::min() ||
        aims_integer > std::numeric_limits<int>::max())
    {
        throw std::runtime_error("invalid or overflowing aims integer");
    }
    aims_check_token_end(aims_end);
    aims_cursor = aims_end;
    return static_cast<int>(aims_integer);
}

inline double aims_read_real(const char *&aims_cursor)
{
    aims_skip_space(aims_cursor);
    const char *aims_unsigned = aims_cursor;
    if (*aims_unsigned == '+' || *aims_unsigned == '-')
    {
        ++aims_unsigned;
    }
    if (aims_unsigned[0] == '0' && (aims_unsigned[1] == 'x' || aims_unsigned[1] == 'X'))
    {
        throw std::runtime_error("hexadecimal aims real is not supported");
    }
    char *aims_end = nullptr;
    errno = 0;
    const double aims_real = std::strtod(aims_cursor, &aims_end);
    // strtod may set ERANGE for finite subnormals/underflow to zero, which the
    // former stream reader accepted. Reject overflow and non-finite values.
    if (aims_end == aims_cursor || !std::isfinite(aims_real))
    {
        throw std::runtime_error("invalid or non-finite aims real");
    }
    aims_check_token_end(aims_end);
    aims_cursor = aims_end;
    return aims_real;
}
} // namespace libbse
