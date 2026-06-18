# LibBSE
## How to build LibBSE

```
cd ~/5_LibBSE
cmake -S ./LibBSE -B ./LibBSE/build 
cmake --build ./LibBSE/build -j4
```

## How to run LibBSE

```
mpirun -np 2 ./LibBSE/build/LibBSE ./2MGO >> ./LibBSE.out
```