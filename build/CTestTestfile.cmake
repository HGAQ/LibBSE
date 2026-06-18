# CMake generated Testfile for 
# Source directory: /nas/longleaf/home/lsr/5_LibBSE/LibBSE
# Build directory: /nas/longleaf/home/lsr/5_LibBSE/LibBSE/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(LibBSE_mpi "/nas/sycamore/apps/mvapich/4.0/intel_2024.2.1/bin/mpiexec" "-n" "2" "/nas/longleaf/home/lsr/5_LibBSE/LibBSE/build/LibBSE" ".")
set_tests_properties(LibBSE_mpi PROPERTIES  _BACKTRACE_TRIPLES "/nas/longleaf/home/lsr/5_LibBSE/LibBSE/CMakeLists.txt;20;add_test;/nas/longleaf/home/lsr/5_LibBSE/LibBSE/CMakeLists.txt;0;")
