/*
    LibBSE_Chi_0.cpp: 
    This is the part trying to use our own routines to calculate 
    chi_0 from Cs_data and KS_eigenvector data. 
*/
#include "LibBSE_Chi_0.h"
#include "../math/Tensor.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <valarray>
#include <vector>

namespace LibBSE{

    int convert_LRI_to_RI_coeff(const Enviroment& Enviro, MpiComm Comm){
        
    }

    int calculate_chi0_BSE(const Enviroment& Enviro, MpiComm Comm){
        LibBSE_printf_root(Comm, "Start calculating chi0...\n");
        try{
            //1. convert the LRI into RI-coeff.
            // C^μ(0)_i(k),j(0) = \sum_R e^{ikR}  \tilde{C}^μ(0)_i(R),j(0)
            // Try to implement a naive mult since this is not the bottleneck.
            // TODO: implement a more efficient way like matrix flattening
            //convert_LRI_to_RI_coeff(Enviro, Comm);
            LibBSE_printf_root(Comm,
                "Converted LRI to k-space RI coeff: atom-pair blocks=%zu\n",
                114);
            // \tilde{C}^μ_{i,j} (k + q, k) = \tilde{C}^μ(0)j(0),i(− k − q) + \tilde{C}^μ(0)i(0),j(k). 

            //2. MO RI-coeff from AO RI-coeff (Cs_data)
            // C^μ_mnσ(k + q, k) = \sum_{i,j} c^∗_imσ(k+q) * c_jnσ(k) * \tilde{C}^μ_ijσ(k+q,k). 

            //

            //
        }
        catch (const std::exception& e){
            LibBSE_printf_root(Comm, "LibBSE[Chi0_BSE]: %s\n", e.what());
            return 1;
        }
        return 0;
    }
}
