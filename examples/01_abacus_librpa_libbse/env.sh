#!/usr/bin/env bash
# Source from run.sh. Override any value by exporting it before running.
if [[ ${LOAD_MODULES:-1} == 1 ]]; then
    module purge
    module load gcc/15.2.0 openmpi/5.0.9/gcc_15.2.0 aocl/5.2.0
    module load hdf5/2.1.1/gcc_15.2.0
fi
export ABACUS_EXE=${ABACUS_EXE:-/work/users/l/s/lsr/b_BSE/abacus-develop/build/abacus_std_para}
export AIMS_EXPORT_EXE=${AIMS_EXPORT_EXE:-/work/users/l/s/lsr/FHIaims/FHIaims/build_stable_HDF/aims.x}
export AIMS_GW_EXE=${AIMS_GW_EXE:-/work/users/l/s/lsr/FHIaims/FHIaims/build_stable/aims.x}
export LIBRPA_EXE=${LIBRPA_EXE:-/work/users/l/s/lsr/LibBSE/1_LibBSE/LibRPA/build_chi0/chi0_main.exe}
export LIBBSE_EXE=${LIBBSE_EXE:-/work/users/l/s/lsr/LibBSE/1_LibBSE/LibBSE/build_external_libri/LibBSE}
export MPIEXEC=${MPIEXEC:-mpirun}
export PYTHON=${PYTHON:-python3}
export NPROCS=${NPROCS:-${SLURM_NTASKS:-4}}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS BLIS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
export OMPI_MCA_coll=${OMPI_MCA_coll:-^hcoll}
export LD_LIBRARY_PATH=/work/users/l/s/lsr/b_BSE/elpa-2024.05.001/build/install/lib:/nas/sycamore/apps/aocl/5.2.0/lib_LP64:/nas/sycamore/apps/scalapack/2.2.2_gcc_15.2.0/lib:${LD_LIBRARY_PATH:-}
