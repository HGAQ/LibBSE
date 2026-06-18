cmake_minimum_required(VERSION 3.16)

project(LibBSE LANGUAGES CXX)

find_package(MPI REQUIRED)

add_executable(LibBSE
    driver/main.cpp
    src/io/LibBSE_io.cpp
    src/mpi/LibBSE_mpi.cpp
)

# Keep headers local to this project and link MPI through CMake's imported target.
target_include_directories(LibBSE PRIVATE include)
target_compile_features(LibBSE PRIVATE cxx_std_11)
target_link_libraries(LibBSE PRIVATE MPI::MPI_CXX)

# Basic MPI smoke test for CTest.
enable_testing()
add_test(NAME LibBSE_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:LibBSE> .
)
