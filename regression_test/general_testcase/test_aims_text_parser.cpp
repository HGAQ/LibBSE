#include "io/aims_text_parser.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void expect_aims_real_failure(const std::string &aims_token)
{
    const char *aims_cursor = aims_token.c_str();
    bool aims_rejected = false;
    try
    {
        libbse::aims_read_real(aims_cursor);
    }
    catch (const std::runtime_error &)
    {
        aims_rejected = true;
    }
    if (!aims_rejected)
    {
        throw std::runtime_error("accepted invalid real: " + aims_token);
    }
}

int main()
{
    try
    {
        const std::vector<std::string> aims_real_tokens = {
            "6.8497162612874427E-004", "-2.3824208220425406E-005",
            "0", "-0.0", "+1.25e100", "1e-310", "1e-9999"};
        for (const auto &aims_token : aims_real_tokens)
        {
            const char *aims_cursor = aims_token.c_str();
            const double aims_parsed_real = libbse::aims_read_real(aims_cursor);
            double aims_stream_real = 0.;
            std::istringstream aims_stream(aims_token);
            aims_stream >> aims_stream_real;
            if (aims_stream.fail() || aims_parsed_real != aims_stream_real ||
                std::signbit(aims_parsed_real) != std::signbit(aims_stream_real))
            {
                throw std::runtime_error("real conversion differs from stream: " + aims_token);
            }
        }
        for (const auto &aims_token : {"", "abc", "1x", "NaN", "Inf", "1e9999", "1.2D-3"})
        {
            expect_aims_real_failure(aims_token);
        }
        for (const auto &aims_token : {"", "2x", "999999999999999999999999", "2147483648"})
        {
            const char *aims_cursor = aims_token;
            bool aims_rejected = false;
            try
            {
                libbse::aims_read_integer(aims_cursor);
            }
            catch (const std::runtime_error &)
            {
                aims_rejected = true;
            }
            if (!aims_rejected)
            {
                throw std::runtime_error("accepted invalid integer");
            }
        }
        const char *aims_cursor = " \t1 6.8497162612874427E-004 694 262 -2.3e-5 +0\r";
        if (libbse::aims_read_integer(aims_cursor) != 1 ||
            libbse::aims_read_real(aims_cursor) != 6.8497162612874427E-004 ||
            libbse::aims_read_integer(aims_cursor) != 694 ||
            libbse::aims_read_integer(aims_cursor) != 262 ||
            libbse::aims_read_real(aims_cursor) != -2.3e-5 ||
            libbse::aims_read_real(aims_cursor) != 0.)
        {
            throw std::runtime_error("mixed row conversion failed");
        }
    }
    catch (const std::exception &aims_error)
    {
        std::cerr << aims_error.what() << '\n';
        return 1;
    }
    std::cout << "aims numeric parser checks passed\n";
    return 0;
}
