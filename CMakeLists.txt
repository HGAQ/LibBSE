cmake_minimum_required(VERSION 3.16)

project(LibBSE LANGUAGES CXX Fortran)
set(CMAKE_CXX_EXTENSIONS OFF)
# GCC's GNU mode exposes the C99 imaginary-unit macro `I`, which collides with
# long-standing LibRPA atom-index names.  Request strict C++17 for this whole
# source tree, including the embedded LibRPA build.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)

set(CEREAL_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../LibRPA/thirdparty/cereal-1.3.0/include"
    CACHE PATH "Path to the cereal include directory")
set(LIBRPA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../LibRPA/include"
    CACHE PATH "Path to the LibRPA include directory")
set(LIBRI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/LibRI/include"
    CACHE PATH "Path to the LibRI include directory")
set(LIBCOMM_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../LibRPA/thirdparty/LibComm/include"
    CACHE PATH "Path to the LibComm include directory")
set(EXTERNAL_ELPA_DIR "/opt/elpa_2025.06.001-install" CACHE PATH
    "ELPA installation prefix")

foreach(required_dir IN ITEMS
    CEREAL_INCLUDE_DIR
    LIBRPA_INCLUDE_DIR
    LIBRI_INCLUDE_DIR
    LIBCOMM_INCLUDE_DIR
    EXTERNAL_ELPA_DIR)
  if(NOT IS_DIRECTORY "${${required_dir}}")
    message(FATAL_ERROR "${required_dir} does not exist: ${${required_dir}}")
  endif()
endforeach()

# LibRPA is built from the checkout containing the requested public include
# directory. LibBSE itself consumes only headers reachable from that directory.
get_filename_component(LIBRPA_SOURCE_ROOT "${LIBRPA_INCLUDE_DIR}/.." ABSOLUTE)
if(NOT EXISTS "${LIBRPA_SOURCE_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR
    "LIBRPA_INCLUDE_DIR must belong to a LibRPA source checkout: ${LIBRPA_INCLUDE_DIR}")
endif()

set(LIBRPA_ENABLE_DRIVER OFF CACHE BOOL "" FORCE)
set(LIBRPA_ENABLE_TEST OFF CACHE BOOL "" FORCE)
set(LIBRPA_ENABLE_CPP_TEST OFF CACHE BOOL "" FORCE)
set(LIBRPA_ENABLE_FORTRAN_BIND OFF CACHE BOOL "" FORCE)
set(LIBRPA_ENABLE_FORTRAN_TEST OFF CACHE BOOL "" FORCE)
set(LIBRPA_USE_LIBRI ON CACHE BOOL "" FORCE)
set(LIBRPA_USE_EXTERNAL_ELPA ON CACHE BOOL "" FORCE)
set(LIBRPA_USE_BUNDLED_ELPA OFF CACHE BOOL "" FORCE)

add_subdirectory("${LIBRPA_SOURCE_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/librpa")

add_library(libbse_core
  src/bse/bse_calculation.cpp
  src/bse/distributed_amplitudes.cpp
  src/bse/elpa_solver.cpp
  src/bse/matrix_checks.cpp
  src/bse/molecular_lri.cpp
  src/bse/spectrum.cpp
  src/interface/librpa_api.cpp
  src/io/aims_screened.cpp
  src/io/chi0_screening.cpp
  src/io/bse_files.cpp
  src/io/fhi_aims_adapter.cpp
  src/parameter/parameter.cpp
  src/utils/profiler.cpp
  src/utils/progress.cpp
)

target_compile_features(libbse_core PUBLIC cxx_std_17)
target_compile_definitions(libbse_core PRIVATE HAVE_SKEWSYMMETRIC)
target_include_directories(libbse_core
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

find_package(OpenMP REQUIRED COMPONENTS CXX)
target_link_libraries(libbse_core PUBLIC librpa_file_reader OpenMP::OpenMP_CXX)

add_executable(LibBSE driver/main.cpp)
target_link_libraries(LibBSE PRIVATE libbse_core)


option(LIBBSE_BUILD_TESTING "Build LibBSE tests" ON)
if(LIBBSE_BUILD_TESTING)
  enable_testing()

  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_test(NAME LibBSE_reject_command_line
    COMMAND $<TARGET_FILE:LibBSE> unexpected_argument)
  set_tests_properties(LibBSE_reject_command_line PROPERTIES WILL_FAIL TRUE)

  add_executable(test_velocity_gauge tests/test_velocity_gauge.cpp)
  target_link_libraries(test_velocity_gauge PRIVATE libbse_core)
  add_test(NAME LibBSE_velocity_gauge COMMAND test_velocity_gauge)

  add_executable(test_parameter tests/test_parameter.cpp)
  target_link_libraries(test_parameter PRIVATE libbse_core)
  add_test(NAME LibBSE_parameter COMMAND test_parameter)

  add_executable(test_profiler
    tests/test_profiler.cpp
    src/utils/profiler.cpp)
  target_compile_features(test_profiler PRIVATE cxx_std_17)
  target_include_directories(test_profiler PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME LibBSE_profiler COMMAND test_profiler)

  add_executable(test_progress tests/test_progress.cpp)
  target_link_libraries(test_progress PRIVATE libbse_core)
  add_test(NAME LibBSE_progress_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:test_progress>)

  add_executable(test_spectrum_mpi tests/test_spectrum_mpi.cpp)
  target_link_libraries(test_spectrum_mpi PRIVATE libbse_core)
  add_test(NAME LibBSE_spectrum_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:test_spectrum_mpi>)
  set_tests_properties(LibBSE_spectrum_mpi PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=2")

  add_executable(test_chi0_screening tests/test_chi0_screening.cpp)
  target_link_libraries(test_chi0_screening PRIVATE libbse_core)
  add_test(NAME LibBSE_chi0_screening_serial COMMAND test_chi0_screening)
  add_test(NAME LibBSE_chi0_screening_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_chi0_screening>)
  set_tests_properties(LibBSE_chi0_screening_serial LibBSE_chi0_screening_mpi
    PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")

  add_executable(test_aims_screened tests/test_aims_screened.cpp)
  target_link_libraries(test_aims_screened PRIVATE libbse_core)
  add_test(NAME LibBSE_aims_screened_serial COMMAND test_aims_screened)
  add_test(NAME LibBSE_aims_screened_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_aims_screened>)
  set_tests_properties(LibBSE_aims_screened_serial LibBSE_aims_screened_mpi
    PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")

  add_executable(test_bse_files tests/test_bse_files.cpp)
  target_link_libraries(test_bse_files PRIVATE libbse_core)
  add_test(NAME LibBSE_bse_files COMMAND test_bse_files)

  add_test(NAME LibBSE_aims_w_converter
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_aims_w_to_librpa.py)

  add_test(NAME LibBSE_aims_mommat_converter
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_aims_mommat_to_velocity.py)

  add_executable(test_molecular_lri_comm
    tests/test_molecular_lri_comm.cpp)
  target_link_libraries(test_molecular_lri_comm PRIVATE libbse_core)
  add_test(NAME LibBSE_molecular_lri_comm_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:test_molecular_lri_comm>)
  set_tests_properties(LibBSE_molecular_lri_comm_mpi PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=2")

  add_executable(test_elpa_solver tests/test_elpa_solver.cpp)
  target_link_libraries(test_elpa_solver PRIVATE libbse_core)
  add_test(NAME LibBSE_elpa_solver_serial COMMAND test_elpa_solver)
  add_test(NAME LibBSE_elpa_solver_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:test_elpa_solver>)
  set_tests_properties(LibBSE_elpa_solver_serial LibBSE_elpa_solver_mpi PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=2")
endif()
