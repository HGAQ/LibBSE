# LibBSE
## How to build LibBSE

```
cd ~/5_LibBSE &&
cmake -S ./LibBSE -B ./LibBSE/build &&
cmake --build ./LibBSE/build -j4
```

Requirement: `MPI`, `Blas`, `LibRI`(?) and also `LibRI`'s requirement.

## How to run LibBSE

1.  Run `output librpa` in FHI-aims (or other equivalent software) to get basic DFT result. Including:

`band_out`, `coulomb_cut`, `coulomb_mat`, `Cs_data`, `dielecfunc_out`, `KS_eigenvector`, `stru_out`, `vxc_out`

Detailed format is included in LibRPA Docs.

2.  Run GW calculation in FHI-aims (or other equivalent method) to get quasi-particle energy `energy_qp`.


3.  You are ready to go! Put them in the same directory and run

```
mpirun -np 4 ./LibBSE/build/LibBSE ./3MGO
```

