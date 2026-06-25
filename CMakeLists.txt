cmake_minimum_required(VERSION 3.16)

project(LibBSE LANGUAGES CXX)

find_package(MPI REQUIRED)
# Use sequential MKL BLAS with GCC OpenMP.  Linking MKL's Intel-threaded BLAS
# together with GCC OpenMP pulls both libiomp5 and libgomp into one executable.
set(BLA_VENDOR Intel10_64lp_seq)
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(LibBSE
    driver/main.cpp
    src/RI/Chi_0.cpp
    src/RI/LibBSE_RI_M.cpp
    src/RI/LibBSE_RI_coeff.cpp
    src/io/Read_aims_output.cpp
    src/io/LibBSE_io.cpp
    src/mpi/LibBSE_mpi.cpp
)

# Keep headers local to this project and link MPI through CMake's imported target.
target_include_directories(LibBSE PRIVATE     
    src
    thirdparty/LibRI/include
    thirdparty/LibComm/include
    thirdparty/cereal-1.3.0/include
)
target_compile_features(LibBSE PRIVATE cxx_std_17)
target_link_libraries(LibBSE PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})

# Basic MPI smoke test with the sample FHI-aims output.
enable_testing()
set(LIBBSE_TEST_DATA "${CMAKE_SOURCE_DIR}/../2MGO")
if(EXISTS "${LIBBSE_TEST_DATA}/band_out")
    add_test(NAME LibBSE_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:LibBSE> ${LIBBSE_TEST_DATA}
    )
endif()
