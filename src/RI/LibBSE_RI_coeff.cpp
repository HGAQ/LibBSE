/*
    LibBSE_RI_M.cpp: 
    This is the part trying to use our own routines to calculate 
    M_mat from Cs_data and KS_eigenvector data. 
*/
#include "LibBSE_RI_coeff.h"
#include "../math/Tensor.h"
#include "../interface/Blas_Interface.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <valarray>


namespace LibBSE
{

    int get_RI_k_coeff(Enviroment& Envir, MpiComm Comm, int q_point){
        
        return 0;
    }

    int calculate_RI_coeff(Enviroment& Envir, MpiComm Comm){
        LibBSE_printf_root(Comm, "Start calculating RI_coeff...\n");
        try{
            //Now, we calculate by q index and store 4-indice mat. We wont store full RI mat 
            // s.t. it need recalculate in W mat and V mat process.
            for(int curr_q_point = 0; curr_q_point < Envir.n_k_point; curr_q_point++){
                //3. D^\mu_m,n(k+q,k) = \sum_i \tilde{M}^\mu_i,m(k+q)^ * c_i,n(k)
                //   C^\mu_m,n(k+q,k) = D^\mu_m,n(k+q,k) + D^\mu*_n,m(k+q,k)
                get_RI_k_coeff(Envir, Comm, curr_q_point);
            }
            //
            return 0;
            //
        }
        catch (const std::exception& e){
            LibBSE_printf_root(Comm, "LibBSE[Chi0_BSE]: %s\n", e.what());
            return 1;
        }
        return 0;
    }

} // namespace LibBSE

