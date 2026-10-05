# LibBSE 使用教程

LibBSE 用来计算周期性材料的电子–空穴激发和光学跃迁。程序读取 Kohn–Sham（KS）电子结构、GW 准粒子能量和屏蔽相互作用，构造并求解 Bethe–Salpeter 方程（BSE），得到激发能、激子振幅、跃迁偶极矩、振子强度及展宽光谱。例如，研究 Si 的光吸收时，先用 DFT 得到波函数，再用 GW 修正单粒子能量，最后用 BSE 加入电子–空穴相互作用，比较独立粒子跃迁与激子激发。

第 1 节介绍计算方法和数据流程，第 2–3 节介绍依赖安装与程序编译，第 4 节说明运行环境，第 5 节列出 `libbse.in` 的全部选项，第 6 节介绍 8 个 Si 算例，第 7–8 节说明结果检查和常见问题。上游程序的说明限于生成 LibBSE 输入所需的配置。

## 1. 程序功能与计算流程

### 1.1 BSE 本征问题

在 Tamm–Dancoff 近似（TDA）下，程序求解

```text
Σ(v'c'k') A(vck,v'c'k') X_S(v'c'k') = Ω_S X_S(vck)
A = ΔE_QP + K
ΔE_QP(vck) = E_QP(c,k) − E_QP(v,k)
```

`v` 表示选中的占据带，`c` 表示选中的空带，`k` 是 BSE 网格点；`Ω_S` 是第 S 个中性激发的能量，`X_S` 是它在各电子–空穴跃迁上的振幅。矩阵维数为 `N = Nk × nocc × nvirt`。例子 01 的 8 个 k 点、4 条占据带、4 条空带产生 128×128 的 TDA 矩阵。增大网格或能带窗口会增加矩阵维数、内存及对角化工作量；减少输出态数不等于减少输入跃迁空间。

| 通道 | 共振矩阵 A | 含义 |
| --- | --- | --- |
| `singlet` | `ΔE_QP + 2V − W` | 单重态；含裸相互作用项与屏蔽吸引项，模板研究的光学通道 |
| `triplet` | `ΔE_QP − W` | 三重态；从单重态基态出发的电偶极振子强度为零 |
| `rpa` | `ΔE_QP + 2V` | 此处的 RPA 激发通道，不是运行 LibRPA 的相关能任务 |
| `ipa` | `ΔE_QP` | 独立粒子近似，用于观察去掉电子–空穴耦合后的跃迁 |

完整 BSE 还包含共振与反共振的耦合矩阵 B，求解 X/Y 两部分振幅。TDA 忽略这一耦合，因此更适合作为入门流程；其准确性需要针对研究体系检验。动态例子则从静态结果构造一次有效动态核，再求解相同跃迁空间。

`V` 是裸 Coulomb 相互作用，`W` 是屏蔽后的完整相互作用，`Wc = W − V` 只是相关部分；`χ₀` 是独立粒子响应，仍需与 Coulomb 矩阵组合得到 W。`screened_format` 指定屏蔽数据的类型及其处理方式，输入文件须符合相应格式。

### 1.2 数据流程

```text
结构、基组、k 网格
        │
        ▼
ABACUS / FHI-aims（DFT 与接口导出）
        ├── KS 能量/波函数、RI 系数、Coulomb、速度/动量 ─────────┐
        ▼                                                   │
LibRPA / FHI-aims（GW）                                      │
        └── QP 能量 + Wc / W / χ₀ ────────────────────────────┤
                                                            ▼
                                             LibBSE → 激发能、振幅、光谱
```

RI 是 resolution of identity：用辅助基展开轨道乘积，降低相互作用矩阵构造的成本。LibRI 负责相关张量收缩，LibRPA 提供数据读取和响应/屏蔽等基础功能，ScaLAPACK 与 ELPA 负责分布式矩阵计算和本征问题。MPI 分配进程间工作，OpenMP 分配一个进程内的线程工作。

LibBSE 读取已有数据，不负责执行 DFT 或 GW。完整计算流程由 `examples/*/run.sh` 组织，依次调用上游程序、数据转换脚本和 LibBSE。

### 1.3 目录结构与算例

本地目录布局为：

```text
/work/users/l/s/lsr/LibBSE/
├── 1_LibBSE/
│   ├── LibBSE/                 # 当前项目；examples/ 是本教程的输入来源
│   ├── LibRPA/                 # 带公共 include/ 的完整源码
│   └── LibRI/LibRI/include/    # 当前使用的外部 LibRI 头文件
├── 3_LibBSE_template/          # 8 个干净模板
└── 3_LibBSE_templatetest/      # 已完成的计算、日志和验证记录
```

ABACUS 或 FHI-aims 产生 KS 波函数、RI 系数、Coulomb 和速度/动量数据；LibRPA 或 FHI-aims GW 产生准粒子（QP）能量与屏蔽数据；LibBSE 读取这些数据，构造 BSE 矩阵、调用 ELPA 并输出激发能和光谱。

| 例子 | DFT → GW | 屏蔽输入 `screened_format` | 粗网格 → BSE 网格 | BSE |
| --- | --- | --- | --- | --- |
| 01 | ABACUS → LibRPA | `librpa_wc` | 2×2×2 | 静态 TDA |
| 02 | FHI-aims → LibRPA | `librpa_wc` | 2×2×2 | 静态 TDA |
| 03 | FHI-aims → FHI-aims | `fhi_aims_w` | 2×2×2 | 静态 TDA |
| 04 | FHI-aims → FHI-aims | `fhi_aims_chi0` | 2×2×2 | 静态 TDA |
| 05 | FHI-aims → FHI-aims | `fhi_aims_chi0` | 2×2×2 | 有效动态 TDA |
| 06 | ABACUS → LibRPA | `librpa_chi0` | 2×2×2 | 有效动态 TDA |
| 07 | ABACUS → LibRPA | `librpa_wc` | 3×3×3 → 4×4×4 | 静态 TDA |
| 08 | FHI-aims → FHI-aims | `fhi_aims_w` | 3×3×3 → 4×4×4 | 静态 TDA |

建议先完成 01 或 02，确认环境与输出，再依次运行其他例子。每个例子都能独立生成相应的电子结构数据，不需要复制前一个例子的计算结果。

## 2. 配置编译和运行环境

### 2.1 软件依赖

编译依赖与数据生成程序分别列于下表。ABACUS 和 FHI-aims 用于生成物理输入，无需链接到 LibBSE。HDF5 用于 FHI-aims 的动量导出和转换，当前 LibBSE 的 CMake 不直接查找 HDF5。

| 软件包 | 在本工作流中的用途 | 安装位置与检查方法 |
| --- | --- | --- |
| GCC（C/C++/Fortran） | C++ 编译 LibBSE/LibRPA；Fortran 编译 GreenX、数学库 | 集群 `gcc/15.2.0`；检查 `gcc`、`g++`、`gfortran` |
| CMake、Make | 生成构建规则并调用编译器 | `cmake --version`、`make --version`；本地参考构建用 CMake 3.31.8 |
| OpenMPI | MPI 头文件、库和编译包装器/启动器 | `openmpi/5.0.9/gcc_15.2.0`；检查 `mpicc`、`mpicxx`、`mpifort`、`mpirun` |
| OpenMP | 单个 MPI 进程内部的线程并行 | GCC 自带支持，不需单独下载；CMake 应报告找到 OpenMP |
| BLAS/LAPACK | 本地稠密矩阵乘法、分解、本征求解基础例程 | 本地 `/usr/lib64/libopenblas.so`；OpenBLAS 同时提供这两组接口 |
| ScaLAPACK/BLACS | 跨 MPI 进程分布的矩阵操作和通信 | 本地 AOCL 的 `lib_LP64/libscalapack.so`；必须与所用 MPI 兼容 |
| ELPA | TDA 和完整 BSE 的分布式本征值求解 | 本地安装或按 2.3 节编译；确认 `libelpa_openmp.so` 与 skew 接口 |
| LibRPA | 输入 reader、GW/响应/屏蔽基础库；另有独立 GW 驱动 | 同级完整源码，必须同时有 `include/` 和 `CMakeLists.txt` |
| LibRI | 局域 RI 张量收缩 | 当前用 `LibRI/LibRI/include`；作为头文件依赖参与编译 |
| LibComm | LibRPA/LibRI 所需的通信组件 | 使用 `LibRPA/thirdparty/LibComm/include` |
| cereal | C++ 对象/张量序列化 | 使用 `LibRPA/thirdparty/cereal-1.3.0/include`，不需单独编译 |
| GreenX | minimax 时间/频率网格等功能 | 使用 `LibRPA/thirdparty/greenX`，随 LibRPA 构建，不必手动安装 |
| Python、NumPy、h5py | 数据转换与数值检查；h5py 读取动量 HDF5 | 按 2.2 节导入检查或安装到虚拟环境 |
| ABACUS / FHI-aims | 产生 KS、RI、Coulomb、QP 等物理输入 | 第 4 节指定已有兼容版本；仅运行对应路线时需要 |

本地完整源码已经提供 LibRI、LibComm、cereal 和 GreenX。这些组件不应仅凭同名用任意版本替换，尤其当前 LibBSE 使用了本地 LibRPA 的公共 API。迁移机器时先携带这一组兼容源码及模板，再替换编译器、MPI、数学库和路径。

- CMake ≥ 3.16、支持 C++17 的编译器、Fortran 编译器、MPI 和 OpenMP。
- BLAS、LAPACK、ScaLAPACK；本地参考配置使用 LP64 数学库。
- 启用反对称本征值求解器的 ELPA；当前项目在链接时需要相关接口，即使本教程只运行 TDA，也应使用兼容的 ELPA 安装。
- 完整 LibRPA 源码，以及 LibRI、LibComm、cereal 头文件。GreenX 随 LibRPA 源码参与编译，因此仍需 Fortran。
- Python 3、NumPy、h5py；FHI-aims 动量转换需要后两者，ABACUS 插值脚本需要 NumPy。
- 运行例子还需要带相应导出扩展的 ABACUS/FHI-aims 可执行文件。模板使用了本地接口扩展，不能仅凭程序同名就假定普通上游版本兼容。

LibBSE 的 CMake 会把 LibRPA 作为子项目编译，所以 `LIBRPA_INCLUDE_DIR` 必须属于完整源码树，不能只指向安装后孤立的头文件目录。LibBSE 编译会关闭该子项目的独立驱动；例子 01、02、06、07 所需的 `chi0_main.exe` 要另外编译。

### 2.2 集群环境配置

在支持 `module` 的 Bash 会话中执行；后续命令沿用这些变量：

```bash
export LIBBSE_SRC=/work/users/l/s/lsr/LibBSE/1_LibBSE/LibBSE
export LIBRPA_SRC=/work/users/l/s/lsr/LibBSE/1_LibBSE/LibRPA
export LIBRI_INC=/work/users/l/s/lsr/LibBSE/1_LibBSE/LibRI/LibRI/include
export ELPA_PREFIX=/work/users/l/s/lsr/b_BSE/elpa-2024.05.001/build/install
export SCALAPACK_DIR=/nas/sycamore/apps/aocl/5.2.0/lib_LP64
export BLAS_LIB=/usr/lib64/libopenblas.so
export SCALAPACK_LIB="$SCALAPACK_DIR/libscalapack.so"

module purge
module load gcc/15.2.0 openmpi/5.0.9/gcc_15.2.0 aocl/5.2.0
module load hdf5/2.1.1/gcc_15.2.0
export LD_LIBRARY_PATH="$ELPA_PREFIX/lib:$SCALAPACK_DIR:/nas/sycamore/apps/scalapack/2.2.2_gcc_15.2.0/lib:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_coll='^hcoll'

command -v cmake make gcc g++ gfortran mpicc mpicxx mpifort mpirun python3
cmake --version
mpicxx --showme:command
mpifort --showme:command
python3 -c 'import numpy, h5py; print(numpy.__version__, h5py.__version__)'
test -f "$LIBRPA_SRC/CMakeLists.txt"
test -f "$LIBRI_INC/RI/ri/RI_Tools.h"
test -d "$LIBRPA_SRC/thirdparty/LibComm/include"
test -f "$LIBRPA_SRC/thirdparty/cereal-1.3.0/include/cereal/cereal.hpp"
test -f "$LIBRPA_SRC/thirdparty/greenX/CMakeLists.txt"
test -f "$BLAS_LIB"
test -f "$SCALAPACK_LIB"
test -f "$ELPA_PREFIX/lib/libelpa_openmp.so"
```

缺少 Python 包时，可在相应的虚拟环境安装，并在后续将 `PYTHON` 指向它：

```bash
python3 -m venv "$HOME/.venvs/libbse"
source "$HOME/.venvs/libbse/bin/activate"
python -m pip install numpy h5py
export PYTHON="$HOME/.venvs/libbse/bin/python"
"$PYTHON" -c 'import numpy, h5py; print("Python dependencies ready")'
```

不要混用不同 MPI 实现或不兼容的编译器运行时。更换编译器、MPI 或数学库后，使用新的 build 目录。下面显式指定本地参考构建实际找到的 OpenBLAS 和 ScaLAPACK；其他机器应换成兼容的库路径，不要把 LP64 与 ILP64 库混在一起。

### 2.3 数学库的源码编译

已有第 2.2 节所列兼容安装时，可直接进行第 3 节的程序编译。本节给出使用本地现有依赖源码、安装到个人目录的另一条路线。GCC、OpenMPI、CMake 和 Make 仍由 module 提供；不用管理员权限，也不覆盖系统库。先加载 2.2 节的 GCC/OpenMPI 环境。本地 ScaLAPACK 源码使用旧版 CMake 最低版本声明，此路线采用本地 CMake 3.x。

建立新的构建目录，指定源码位置：

```bash
export DEP_WORK=$(mktemp -d /work/users/l/s/lsr/LibBSE/deps_tutorial.XXXXXX)
export DEP_PREFIX="$DEP_WORK/install"
export OPENBLAS_SRC=/work/users/l/s/lsr/b_BSE/OpenBLAS
export SCALAPACK_SRC=/work/users/l/s/lsr/b_BSE/scalapack
export ELPA_SRC=/work/users/l/s/lsr/b_BSE/elpa-2024.05.001
mkdir -p "$DEP_PREFIX"
test -f "$OPENBLAS_SRC/CMakeLists.txt"
test -f "$SCALAPACK_SRC/CMakeLists.txt"
test -x "$ELPA_SRC/configure"
```

这些命令假定本地完整依赖源码存在；在新机器上应先放置对应源码，再修改三个 `*_SRC` 路径。不要把原有 build 目录当作源码包迁移。

**步骤 1：编译 OpenBLAS。**

```bash
cmake -S "$OPENBLAS_SRC" -B "$DEP_WORK/openblas-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran \
  -DCMAKE_INSTALL_PREFIX="$DEP_PREFIX/openblas" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF \
  -DBUILD_WITHOUT_LAPACK=OFF \
  -DINTERFACE64=OFF -DDYNAMIC_ARCH=ON -DUSE_OPENMP=ON \
  -DBUILD_TESTING=OFF
cmake --build "$DEP_WORK/openblas-build" -j 8
cmake --install "$DEP_WORK/openblas-build"
export BLAS_LIB="$DEP_PREFIX/openblas/lib/libopenblas.so"
test -f "$BLAS_LIB"
```

`BUILD_WITHOUT_LAPACK=OFF` 保留 LAPACK；若只编 BLAS，后续矩阵分解会缺符号。`INTERFACE64=OFF` 使用 32 位整数接口（LP64），并不表示只能处理 32 位浮点数。`DYNAMIC_ARCH=ON` 编入多个受支持 CPU 内核并在运行时选择，避免只针对登录节点优化。`USE_OPENMP=ON` 选择 OpenMP 线程后端；`BUILD_SHARED_LIBS=ON` 生成 `.so`。此配置关闭依赖库的测试套件。安装完成后，运行第 3.3 节的 LibBSE 单元测试，检查数学库与程序的兼容性。

**步骤 2：编译 ScaLAPACK。**

```bash
cmake -S "$SCALAPACK_SRC" -B "$DEP_WORK/scalapack-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=mpicc -DCMAKE_Fortran_COMPILER=mpifort \
  -DMPI_C_COMPILER=mpicc -DMPI_Fortran_COMPILER=mpifort \
  -DCMAKE_INSTALL_PREFIX="$DEP_PREFIX/scalapack" \
  -DBUILD_SHARED_LIBS=ON \
  -DBLAS_LIBRARIES="$BLAS_LIB" -DLAPACK_LIBRARIES="$BLAS_LIB"
cmake --build "$DEP_WORK/scalapack-build" -j 8
cmake --install "$DEP_WORK/scalapack-build"
export SCALAPACK_DIR="$DEP_PREFIX/scalapack/lib"
export SCALAPACK_LIB="$SCALAPACK_DIR/libscalapack.so"
export LD_LIBRARY_PATH="$DEP_PREFIX/openblas/lib:$SCALAPACK_DIR:${LD_LIBRARY_PATH:-}"
test -f "$SCALAPACK_LIB"
ldd "$SCALAPACK_LIB"
```

ScaLAPACK 使用 MPI 进行分布式矩阵计算，OpenBLAS 提供进程内的 BLAS/LAPACK 运算。`mpicc`、`mpifort` 是包装器，会给底层编译器添加当前 MPI 的头文件和链接选项。`ldd` 中 MPI 和 OpenBLAS 应指向预期安装，不应出现 `not found`。

**步骤 3：编译 ELPA。**

```bash
export ELPA_PREFIX="$DEP_PREFIX/elpa"
mkdir -p "$DEP_WORK/elpa-build"
(
  cd "$DEP_WORK/elpa-build" || exit 1
  "$ELPA_SRC/configure" \
    --prefix="$ELPA_PREFIX" \
    --with-mpi=yes --enable-openmp --enable-skew-symmetric-support \
    --disable-avx512 \
    CC=mpicc CXX=mpicxx FC=mpifort \
    CFLAGS='-O2' CXXFLAGS='-O2' FCFLAGS='-O2' \
    LDFLAGS="-Wl,-rpath,$SCALAPACK_DIR -Wl,-rpath,$DEP_PREFIX/openblas/lib" \
    LIBS="$SCALAPACK_LIB $BLAS_LIB -lm" || exit 1
  make -j 8 && make install
)
export LD_LIBRARY_PATH="$ELPA_PREFIX/lib:$DEP_PREFIX/openblas/lib:$SCALAPACK_DIR:${LD_LIBRARY_PATH:-}"
test -f "$ELPA_PREFIX/lib/libelpa_openmp.so"
nm -D "$ELPA_PREFIX/lib/libelpa_openmp.so" | rg elpa_skew_eigenvectors
ldd "$ELPA_PREFIX/lib/libelpa_openmp.so"
```

`--prefix` 指定安装目录；`LIBS` 提供链接的 ScaLAPACK、BLAS/LAPACK 和数学库；`LDFLAGS` 中的 rpath 使库能找到这一套依赖。`--disable-avx512` 是本地示例的 x86 兼容性选择，不能保证覆盖任意机器的全部 CPU 特征；跨架构时应重新配置 ELPA。完整 BSE 会调用 `elpa_skew_eigenvectors`，因此最后检查导出符号。本地发布源码已有 `configure`，不需要先运行 `autoreconf`；若使用未生成 `configure` 的开发源码，还需它要求的 Autoconf/Automake/Libtool 构建工具。

任一步出现错误就先停在该步，检查 CMake 输出或 ELPA 的 `config.log`。完成此路线后，第 3 节会通过 `BLAS_LIB`、`SCALAPACK_LIB`、`ELPA_PREFIX` 使用这套新依赖。第 4 节运行副本的 `env.sh` 中，也要将原有硬编码的 ELPA/AOCL 库路径改成上述三个安装目录，并设置 `LOAD_MODULES=0` 保留已配置环境。

### 2.4 头文件依赖与 GreenX

`LibRI`、`LibComm`、`cereal` 在当前路径中作为头文件依赖使用，不需要为它们单独执行 `make install`。正确的 include 根目录下应分别能看到 `RI/`、`Comm/` 和 `cereal/`。LibBSE 的 CMake 会检查目录存在，LibRPA 的查找模块进一步检查对应头文件。

GreenX 是编译型依赖。LibRPA 默认通过 `add_subdirectory(thirdparty/greenX)` 构建它，所以看到 Fortran 编译输出是正常现象。不要因为 LibBSE 主体是 C++ 就去掉 Fortran 编译器，也不要为这一条路线设置 `LIBRPA_USE_EXTERNAL_GREENX=ON`。仅装一个外部 `librpa` 库文件不够：本项目还需要完整源码与公共头文件。

## 3. 编译 LibBSE 和独立 LibRPA

### 3.1 编译 LibBSE

使用独立的 `build_tutorial` 目录，避免覆盖现有参考构建：

```bash
cmake -S "$LIBBSE_SRC" -B "$LIBBSE_SRC/build_tutorial" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_Fortran_COMPILER=mpifort \
  -DCMAKE_CXX_EXTENSIONS=OFF \
  -DCEREAL_INCLUDE_DIR="$LIBRPA_SRC/thirdparty/cereal-1.3.0/include" \
  -DLIBRPA_INCLUDE_DIR="$LIBRPA_SRC/include" \
  -DLIBRI_INCLUDE_DIR="$LIBRI_INC" \
  -DLIBCOMM_INCLUDE_DIR="$LIBRPA_SRC/thirdparty/LibComm/include" \
  -DEXTERNAL_ELPA_DIR="$ELPA_PREFIX" \
  -DBLAS_LIBRARIES="$BLAS_LIB" \
  -DLAPACK_LIBRARIES="$BLAS_LIB" \
  -DScaLAPACK_LIBRARY="$SCALAPACK_LIB" \
  -DLIBBSE_BUILD_TESTING=ON \
  -DPython3_EXECUTABLE="$(command -v "${PYTHON:-python3}")"
cmake --build "$LIBBSE_SRC/build_tutorial" -j 8
export LIBBSE_EXE="$LIBBSE_SRC/build_tutorial/LibBSE"
test -x "$LIBBSE_EXE"
ldd "$LIBBSE_EXE"
```

确认 `ldd` 没有 `not found`。这里 `-j 8` 是编译并行度，需符合当前节点的资源限制。若 shell 中残留其他工具链的 `MKLROOT`，应先清理该环境；LibRPA 的 CMake 检测到它时会选择 MKL 分支。

这一步分为“配置”和“编译”：第一次 `cmake` 只查找依赖并生成构建规则，最后应显示 `Configuring done`、`Generating done`；`cmake --build` 才产生目标程序，成功后应存在 `build_tutorial/LibBSE`。配置失败时不要继续执行编译命令。

| 命令部分 | 实际作用 |
| --- | --- |
| `-S`、`-B` | 分别指定源码目录和构建目录；所有缓存和目标文件放在后者 |
| `-DCMAKE_BUILD_TYPE=Release` | 使用优化构建；排查源码问题才考虑 Debug |
| `CMAKE_CXX_COMPILER`、`CMAKE_Fortran_COMPILER` | 指定同一 MPI 环境中的 C++/Fortran 包装器 |
| `CMAKE_CXX_EXTENSIONS=OFF` | 禁止 GNU C++ 方言；LibBSE 还显式添加严格 C++17 |
| 四个 `*_INCLUDE_DIR` | 指向 LibRPA、LibRI、LibComm、cereal 的 include 根目录，不是库文件 |
| `EXTERNAL_ELPA_DIR` | ELPA 的安装前缀，下面应有 include 和 lib；不是 ELPA 源码目录 |
| `BLAS_LIBRARIES`、`LAPACK_LIBRARIES` | 指定所用数学库；这里两者都由同一个 OpenBLAS 提供 |
| `ScaLAPACK_LIBRARY` | 指定准确的库文件；注意变量名大小写与单数形式 |
| `LIBBSE_BUILD_TESTING=ON` | 编译并注册 LibBSE 测试程序，首次安装应保留 |
| `Python3_EXECUTABLE` | 供 CTest 转换器测试使用的 Python；应与已安装 NumPy/h5py 的解释器一致 |

`-D变量=值` 表示写入 CMake 配置缓存，不是传给 `libbse.in` 的物理参数。编译完成后无需 `sudo make install`，本教程直接用绝对路径运行生成的程序。

### 3.2 编译独立 LibRPA 驱动

```bash
cmake -S "$LIBRPA_SRC" -B "$LIBRPA_SRC/build_tutorial" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_Fortran_COMPILER=mpifort \
  -DCMAKE_CXX_EXTENSIONS=OFF \
  -DCMAKE_CXX_FLAGS=-std=c++17 \
  -DCEREAL_INCLUDE_DIR="$LIBRPA_SRC/thirdparty/cereal-1.3.0/include" \
  -DLIBCOMM_INCLUDE_DIR="$LIBRPA_SRC/thirdparty/LibComm/include" \
  -DLIBRI_INCLUDE_DIR="$LIBRI_INC" \
  -DLIBRPA_USE_LIBRI=ON \
  -DLIBRPA_USE_EXTERNAL_ELPA=ON \
  -DLIBRPA_USE_BUNDLED_ELPA=OFF \
  -DEXTERNAL_ELPA_DIR="$ELPA_PREFIX" \
  -DBLAS_LIBRARIES="$BLAS_LIB" \
  -DLAPACK_LIBRARIES="$BLAS_LIB" \
  -DScaLAPACK_LIBRARY="$SCALAPACK_LIB" \
  -DLIBRPA_ENABLE_DRIVER=ON \
  -DLIBRPA_ENABLE_TEST=OFF \
  -DLIBRPA_ENABLE_FORTRAN_BIND=OFF
cmake --build "$LIBRPA_SRC/build_tutorial" --target rpa_exe -j 8
export LIBRPA_EXE="$LIBRPA_SRC/build_tutorial/chi0_main.exe"
test -x "$LIBRPA_EXE"
ldd "$LIBRPA_EXE"
```

严格 C++17 可避免 GNU 扩展中的宏 `I` 与源码变量名冲突。上述两个构建使用相同的 LibRI 和数学库，但分别生成相应的构建产物。`LIBRPA_ENABLE_DRIVER=ON` 打开独立驱动；其 CMake 目标名为 `rpa_exe`，生成的文件名却是 `chi0_main.exe`。`LIBRPA_USE_EXTERNAL_ELPA=ON` 和 `LIBRPA_USE_BUNDLED_ELPA=OFF` 选择第 2 节安装的 ELPA；`LIBRPA_USE_LIBRI=ON` 开启模板采用的张量收缩路径。`LIBRPA_ENABLE_TEST=OFF`、`LIBRPA_ENABLE_FORTRAN_BIND=OFF` 省去独立 LibRPA 的测试和对外 Fortran 绑定，不会取消 GreenX 所需的 Fortran 编译。只做 03、04、05、08 时，可以跳过独立驱动的编译；LibBSE 内部仍依赖 LibRPA。

### 3.3 运行单元测试

在允许启动 MPI 的计算节点/已有资源分配中运行：

```bash
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export MKL_NUM_THREADS=1
ctest --test-dir "$LIBBSE_SRC/build_tutorial" --output-on-failure
```

测试包含双进程 MPI、ELPA、输入参数、文件读取和 Python 转换器检查，通过标准为 CTest 全部通过。测试内容见 [tests/README.md](../tests/README.md)。若 MPI 在登录节点无法启动，应在计算资源中重试。

## 4. 运行准备

### 4.1 建立计算目录

从仓库 `examples` 创建一个新的运行目录；不要在已有验证结果中直接重新运行：

```bash
export TUTORIAL_RUN_ROOT=$(mktemp -d /work/users/l/s/lsr/LibBSE/tutorial_runs.XXXXXX)
for src in "$LIBBSE_SRC"/examples/0[1-8]_*; do
  cp -a "$src" "$TUTORIAL_RUN_ROOT/"
done
printf '%s\n' "$TUTORIAL_RUN_ROOT"
```

保存打印出的路径。重新登录后需要重新设置 `LIBBSE_SRC`、`LIBBSE_EXE`、`LIBRPA_EXE` 和 `TUTORIAL_RUN_ROOT` 等变量。也可以从 `3_LibBSE_template` 复制相应目录；不要从含有旧输出的 `3_LibBSE_templatetest` 复制整套运行目录作为干净起点。

### 4.2 可执行文件与并行设置

在提交作业的同一会话中执行：

```bash
export LIBBSE_EXE="$LIBBSE_SRC/build_tutorial/LibBSE"
export LIBRPA_EXE="$LIBRPA_SRC/build_tutorial/chi0_main.exe"
export ABACUS_EXE=/work/users/l/s/lsr/b_BSE/abacus-develop/build/abacus_std_para
export AIMS_EXPORT_EXE=/work/users/l/s/lsr/FHIaims/FHIaims/build_stable_HDF/aims.x
export AIMS_GW_EXE=/work/users/l/s/lsr/FHIaims/FHIaims/build_stable/aims.x
export MPIEXEC=mpirun
export PYTHON=${PYTHON:-python3}
export OMP_NUM_THREADS=1
export AIMS_EXPORT_THREADS=4
unset NPROCS
```

每个例子的 `env.sh` 会保留已导出的程序路径，否则回落到本地旧构建 `build_external_libri/LibBSE` 和 `build_chi0/chi0_main.exe`。因此要测试刚编译的版本，必须设置上面的两个变量，或修改运行副本的 `env.sh`。

默认资源为 `inter` 分区、1 节点、4 MPI 进程、每进程 1 CPU、64 GB 内存、2 小时；按所在机器修改 `run.sh` 的 `#SBATCH` 行。不设置 `NPROCS` 时，脚本采用 `SLURM_NTASKS`，没有 Slurm 变量则用 4。FHI-aims 的导出阶段采用 1 个 MPI 进程和默认 4 个线程，在节点临时目录运行后复制回来；确保分配的 CPU 足够。

如果自己准备好了环境，可以 `export LOAD_MODULES=0` 跳过自动 module 加载。迁移机器时还应修改 `env.sh` 中固定的 `LD_LIBRARY_PATH`；`LOAD_MODULES=0` 不影响该路径设置。脚本中的 `--bind-to none` 是 OpenMPI 参数，换 MPI 实现时也要调整。

### 4.3 输入检查与作业提交

检查全部算例的输入文件与运行环境：

```bash
for d in "$TUTORIAL_RUN_ROOT"/0[1-8]_*; do
  (cd "$d" && bash run.sh --check) || break
done
```

检查通过后显示 `Input files and environment checks passed; no calculation started.`。该检查确认输入文件、可执行文件路径和 Python 依赖，不执行计算或验证程序接口。

后续必须进入对应例子目录再 `sbatch run.sh`，因为脚本在 Slurm 下使用 `SLURM_SUBMIT_DIR`。在已有计算资源内，也可以用 `bash run.sh` 前台运行。每个脚本拒绝覆盖已存在的关键输出；重新做完整流程时另建干净副本。

## 5. `libbse.in` 选项

本节覆盖当前 [parameter.h](../src/parameter/parameter.h) 和 [parameter.cpp](../src/parameter/parameter.cpp) 接受的全部 28 个输入选项。下面的“默认”指省略参数时源码采用的值；模板中的显式设置可以覆盖默认值。未定义的参数会触发错误。`librpa.in`、ABACUS 和 FHI-aims 使用各自独立的输入文件。

### 5.1 输入语法

程序从**当前计算目录**读取固定文件名 `libbse.in`，运行方式是 `mpirun -np 4 "$LIBBSE_EXE"`，程序不接受命令行输入文件参数。支持 `key value` 和 `key = value`，`#`、`!` 后是注释；同一选项出现多次，最后一次生效。参数名和枚举值不区分大小写，路径区分大小写。布尔值可用 `1/0`、`true/false`、`t/f`、`.true./.false.`、`yes/no`。

```text
input_dir       OUT.librpa      # 相对于 libbse.in 所在目录
qp_data         energy_qp       # 不会自动拼接到 input_dir 后面
qp_format       energy_qp
bse_spin_types  singlet         # 多通道可写 singlet triplet
bse_tda         tda
```

相对路径以 `libbse.in` 所在目录为基准，与可执行文件的位置无关。程序可保留在源码树的 build 目录中，并通过绝对路径从计算目录调用。

<a id="bse-inputs"></a>

### 5.2 数据输入与输出

#### `input_dir`：公共电子结构数据目录

**类型：** 字符串（目录）。**默认值：** 无，必填。

指向包含结构、基组、KS 本征值/本征矢、RI 系数和 Coulomb/速度数据的目录。准粒子能量和屏蔽数据分别由 `qp_data` 和 `screened_dir` 指定。ABACUS 静态路径用 `OUT.librpa`，FHI-aims 用 `aims_export`，ABACUS χ₀ 路径用 `OUT.librpa_chi0`。更换此目录时必须同时检查 QP、屏蔽和 RI 是否仍来自匹配的结构、基组与网格。

#### `input_format`：公共数据格式

**类型：** 字符串。**默认值：** `auto`。**取值：** `auto`、`librpa`、`fhi_aims`。

`auto` 根据 `basis_out` 的 producer 标记选择：`aims` 对应 FHI-aims，其余采用 LibRPA 文件约定。`librpa` 用于标准 LibRPA/ABACUS 导出；`fhi_aims` 要求 aims 标记，并在输出目录下建立 `fhi_aims_reader_view` 兼容链接视图。建立该视图前须完成动量转换，生成 `velocity_matrix`。各例显式指定公共数据格式；GW 程序及其输出由准粒子和屏蔽数据选项独立指定。

#### `output_dir`：输出目录

**类型：** 字符串（目录）。**默认值：** `libbse.d`。

保存能量、分布式振幅、光谱和相关诊断文件，必要时由程序创建。`bse_solver spectrum` 也从此目录读取已有结果。不同参数的计算应使用不同的输出目录。光谱重算前须将已有能量和振幅文件放入指定目录。模板统一用 `libbse.d`，便于脚本检查输出。

#### `qp_data`：准粒子能量路径

**类型：** 字符串（文件或目录）。**默认值：** `input_dir`。

可指向文件或目录。在 `qp_format energy_qp` 下，文件路径直接使用，目录则读取其下的 `energy_qp`；`fine_band` 则读取指定文件或目录下的 `GW_band_spin_1.dat`。独立 LibRPA 在工作目录输出 QP 能量，因此算例 01/02/06 显式写 `energy_qp`；03–05 写 `aims_gw/energy_qp`；07 写 `energy_qp_fine`；08 写转换后的 `aims_gw/energy_qp`。省略此项时，程序在 `input_dir` 内查找准粒子能量。

#### `qp_format`：准粒子能量格式

**类型：** 字符串。**默认值：** `auto`。**取值：** `auto`、`energy_qp`、`fine_band`。

`energy_qp` 是模板统一采用的块格式，内含 k 点、占据数、KS/QP 能量，输入能量单位为 Hartree，程序内部转换为 Ry。该格式支持粗网格和细网格数据。`fine_band` 保留给旧式 LibRPA `GW_band_spin_1.dat` 细网格表。

`auto` 优先识别目录下的 `energy_qp` 或恰好名为 `energy_qp` 的文件；找不到时仅在 `bse_use_fine_kgrid=1` 下回退到 `fine_band`，否则报错。因此例子 07 的自定义文件名 `energy_qp_fine` 必须显式配 `qp_format energy_qp`。本程序不直接解析 FHI-aims 的 `GW_band*.out`；例子 08 先调用转换脚本。QP 内容中的 k 点坐标必须覆盖实际 BSE 网格，网格转换须在生成 QP 数据时完成。

<a id="bse-screening"></a>

### 5.3 屏蔽相互作用

#### `screened_dir`：屏蔽数据目录

**类型：** 字符串（目录）。**默认值：** `input_dir` 的父目录下的 `librpa.d`。

文件内容由 `screened_format` 决定。singlet/triplet 需要屏蔽输入；只有 IPA/RPA 的静态计算不需要读取 W。03–05/08 的数据实际在 `aims_gw`，06 在 `librpa.d/chi0_static`，必须覆盖默认目录。

#### `screened_format`：屏蔽数据格式

**类型：** 字符串。**默认值：** `librpa_wc`。**取值：** `librpa_wc`、`fhi_aims_w`、`fhi_aims_chi0`、`librpa_chi0`。

各格式对应的输入文件和处理方式如下：

| 取值 | 预期数据 | 处理方式 | 对应例子 |
| --- | --- | --- | --- |
| `librpa_wc` | 实空间 `Wc_Mu_*_Nu_*_iR_*_ifreq_0.mtx` | 读取最低频率索引的 Wc，再加裸 cut Coulomb 得到 W | 01、02、07 |
| `fhi_aims_w` | `periodic_gw_w_q_*_rank_*.dat` | 读取 full W，选择最低绝对虚频，不再加 V | 03、08 |
| `fhi_aims_chi0` | `periodic_gw_chi0_q_*_rank_*.dat` | 读取 χ₀，使用匹配辅助基的 Coulomb 重建 W | 04、05 |
| `librpa_chi0` | `chi0_rf.info` 及 `Chi0_Mu_*_ifreq_0.mtx` 等数据 | 读取 LibRPA χ₀ 并重建 W；06 使用额外的严格零频样本 | 06 |

`input_format fhi_aims` 可以与 `screened_format librpa_wc` 共存：例子 02 的 KS 来自 FHI-aims，屏蔽却来自独立 LibRPA。这两项描述不同的数据来源。索引 0 不自动表示物理频率为零；01/02/07 使用最低虚频的静态近似，04/05 使用 FHI-aims 最低有限虚频，06 才专门导出了 ω=0 的 χ₀。

#### `chi0_coulomb_metric`：χ₀ 屏蔽重建的 Coulomb 约定

**类型：** 字符串。**默认值：** `full`。**取值：** `full`、`single_cut`。**适用条件：** χ₀ 输入。

仅 χ₀ 重建路径使用。在 `full` 下，忽略 Gamma 解析替换的简写公式为：

```text
ε = I − sqrt(V_full) χ₀ sqrt(V_full)
W = sqrt(V_cut) ε^(-1) sqrt(V_cut)
```

矩阵平方根和求逆在完整辅助基空间执行，不是逐元素开方，也不能对每个原子块单独求逆。full V 用于介电屏蔽内部，cut V 处理外部相互作用。`single_cut` 把上式内部的 V_full 也换成 V_cut，改变的是屏蔽约定，适用于不同 Coulomb 约定的对照计算，其结果通常与生产程序的 GW 屏蔽不同。04–06 选 `full`，以保持所采用的 LibRPA 屏蔽约定。

#### `chi0_headwing`：χ₀ 的解析 head/wing 修正

**类型：** 布尔值。**默认值：** `true`。**适用条件：** χ₀ 输入；启用时要求 `chi0_coulomb_metric full`。

head 是长波极限的奇异通道，wing 是它与其余通道的耦合。在 χ₀ 路径开启此项时，LibBSE 调用 LibRPA 的解析 head/wing 和 Gamma 小胞平均，并让解析修正使用与读入 χ₀ 一致的物理频率。它要求 `chi0_coulomb_metric full`；与 `single_cut` 同时开启会报错。关闭可做不含该修正的对照，但一般会改变结果。它不会重新处理已读入的原生 W 或 Wc。

#### `out_screening_matrices`：屏蔽诊断矩阵输出

**类型：** 布尔值。**默认值：** `false`。**适用条件：** χ₀ 输入。

χ₀ 路径设置为 true 时，rank 0 在 `output_dir` 下写 `screening_q_<index>.bin`，保存粗 q 网格、插值前的矩阵。当前记录依次包含原生 int32 维数、4 个 float64（分数 q 坐标与 Hartree 单位的频率）、按行排列的 complex128 `sqrt(V_cut)` 和 `ε⁻¹`。它用于检查 χ₀→W 的中间过程，不是激发能或普通文本数据，不应使用 `np.loadtxt` 读取。模板省略它并沿用 false，避免无必要的矩阵 I/O；输出内容与 `out_bse_ab` 所指的 BSE 矩阵不同。

<a id="bse-space"></a>

### 5.4 跃迁空间与 k 网格

#### `nocc`：活跃占据带数

**类型：** 整数。**默认值：** `4`。**取值范围：** 正整数。

从 QP 数据覆盖的占据态中选取最靠近带顶的 `nocc` 条带参与 BSE。它不是体系总电子数，也不是全电子程序的所有占据带数。Si 全电子 FHI-aims 有深芯态，模板取价带 11–14，对应 `nocc 4`；赝势 ABACUS 对应价带 1–4。若 QP 或 KS 数据未覆盖所需窗口，会在读取时失败。

#### `nvirt`：活跃空带数

**类型：** 整数。**默认值：** `4`。**取值范围：** 正整数。

取所选占据窗口之后最前面的 `nvirt` 条空带。ABACUS 模板选 4，FHI-aims 模板选 3，与各自的能带窗口一致。增加它需要上游先导出更多 KS/QP 带；BSE 无法补算缺失波函数。若在简并组中间截断，即使不同运行的能带能量一致，波函数基底旋转仍可能改变截断空间中的结果，例子 08 就展示了这一限制。

#### `bse_nstates`：激发态数

**类型：** 整数。**默认值：** `-1`。**取值范围：** `-1` 或不超过跃迁空间维数的正整数。

`-1` 表示全部 `N = Nk × nocc × nvirt` 个激发态；正整数指定最低的若干激发态。0、低于 −1 或大于矩阵维数的值无效。此参数控制最终保留的激发态数，矩阵维数仍由 `nocc`、`nvirt` 和 k 点数决定。减少激发态数会截断高能光谱贡献。8 个算例均取 −1，保留完整激发谱以供比较。

#### `bse_use_fine_kgrid`：细网格选择

**类型：** 整数。**默认值：** `1`。**取值：** `0`、`1`。

0 使用 SCF 规则网格及其 KS 波函数，QP 要映射到同一网格；1 使用独立的 band/细网格，包括 `band_kpath_info` 和细网格本征值/本征矢。当粗细网格不同时，RI/屏蔽仍来自粗网格的实空间表示，速度算符也要通过局域表示与细网格 KS 波函数重建。

01–06 没有第二套网格，所以显式设 0；若遗漏，默认 1 会要求不存在的 band 数据。07–08 设 1，粗网格 27 点、BSE 网格 64 点。细网格 KS 波函数和 QP 能量须由上游计算提供。

<a id="bse-solver"></a>

### 5.5 求解器与计算通道

#### `bse_solver`：求解模式

**类型：** 字符串。**默认值：** `elpa`。**取值：** `elpa`、`spectrum`。

`elpa` 构造矩阵、求本征态并完成光学分析；纯 IPA TDA 有直接生成独立跃迁的专门路径。`spectrum` 跳过矩阵构造和本征求解，从 `output_dir` 读取已有能量及各 rank 振幅，重新计算光学量，适合只改展宽/光谱网格。必须保留匹配的输入与振幅文件，并用与生成振幅时相同的 MPI 进程数。例子全部从头求解，因此选 `elpa`。动态模式的正等离子体能量不能与 `spectrum` 联用。

#### `bse_spin_types`：计算通道

**类型：** 字符串列表。**默认值：** `singlet triplet`。**取值：** `singlet`、`triplet`、`rpa`、`ipa` 的非空、无重复组合。

可从 `singlet`、`triplet`、`rpa`、`ipa` 选择一个或多个，以空白或逗号分隔，不能为空或重复。各通道矩阵见第 1.1 节。singlet/triplet 需要 W；singlet/rpa 需要裸相互作用项；只有 IPA 时不构造相互作用矩阵，且要求 `bse_tda tda`。8 个模板采用 `singlet`，用于 Si 的单重光学激发及回归比较。增加计算通道后，应分别检查各通道的输出文件与态数。

#### `bse_tda`：TDA 与完整 BSE

**类型：** 字符串。**默认值：** `both`。**取值：** `tda`、`full`、`both`。

`tda` 只求 A；`full` 求含 A/B 耦合的完整问题；`both` 对每个请求通道分别执行二者。TDA 输出 `Excitation_Energy_<type>.dat` 和 `Excitation_Amplitude_<type>_<rank>.dat`；完整 BSE 输出带 `full` 标记的能量及 X/Y 振幅。8 个例子统一选择 `tda`，控制问题规模并使参考值处于同一近似；05/06 的有效动态模型也明确要求 TDA。省略这一项会运行默认的两套计算，无法直接按教程结果文件验收。

#### `bse_ri_hartree`：裸相互作用项

**类型：** 布尔值。**默认值：** `true`。**限制：** singlet 和 rpa 通道必须启用。

控制 RI 构造中的 Hartree/交换项。singlet 和 rpa 通道要求取 true；仅计算 triplet 或 IPA 时允许取 false。该参数不控制屏蔽相互作用 W。各算例均计算 singlet，因此设为 1。

<a id="bse-dynamic"></a>

### 5.6 有效动态核

#### `bse_plasma_energy_ev`：等离子体能量

**类型：** 实数。**默认值：** `0`。**单位：** eV。**取值范围：** 有限非负数。

正数开启一次有效动态修正，当前必须同时满足：χ₀ 输入、`bse_tda tda`、`bse_solver elpa`，且至少包含 singlet 或 triplet 屏蔽通道。首先求静态最低激发能 `Ω₀`，从当前活跃窗口的最小直接 QP gap `Eg_direct` 得到 `Eb = Eg_direct − Ω₀`，再用这一束缚能构造同一个有效核并求其全部请求态。所有请求态使用同一有效核，不进行逐态自洽或时域传播。

若 d 是静态逆介电矩阵的本征值，P 是所输入的等离子体能量，模型在介电本征通道使用 `f(d) = 1 − P(1−d)/(P + Eb·sqrt(1−d))`。程序转换能量单位后求值；束缚能必须非负，逆介电本征值须在 [0,1] 内（允许微小浮点误差）。`Eb=0` 是代码允许的边界，此时退回静态核；模板的两个动态例子都具有正束缚能。等离子体能量是与材料有关的模型参数，需根据研究体系确定。

05/06 选 15 eV 演示动态修正，其他例子省略并保持 0。开启后标准激发能与光谱保存有效结果，额外文件保留静态解和位移。对比静态与动态时应先固定结构、KS/QP、辅助基和 χ₀ 处理，仅改变本项。

<a id="bse-optics"></a>

### 5.7 波函数规范与光谱

#### `abs_gauge`：光学规范

**类型：** 字符串。**默认值：** `velocity`。**取值：** 当前仅支持 `velocity`。

使用速度/动量矩阵与 KS 跃迁能隙构造跃迁偶极矩，再与激子振幅收缩。`length` 等其他值会被拒绝。当前光学分析总会执行，因此所有例子都需要有效的 `velocity_matrix`；不能通过省略本选项跳过速度数据。

#### `wavefunction_gauge`：波函数相位规范

**类型：** 字符串。**默认值：** `auto`。**取值：** `auto`、`native`、`first_k`。

`auto` 对 FHI-aims 解析为 `native`，对其他标准 LibRPA 数据解析为 `first_k`。`native` 保持生产程序原有且与 RI 一致的规范；`first_k` 按每条带与第一个 k 点同带的重叠进行标量相位对齐，是旧 ABACUS 路径沿用的约定。它与 `abs_gauge` 不同：前者控制波函数相位，后者控制光学算符形式。

模板对 ABACUS 显式用 `first_k`，FHI-aims 显式用 `native`，以匹配各自数据链。简并子空间中的波函数可发生酉旋转，逐带标量相位对齐无法消除这种自由度，因此 `first_k` 不适用于一般的简并子空间规范变换。

#### `spectrum_broadening_ev`：Lorentz 展宽半宽

**类型：** 实数。**默认值：** `0.10`。**单位：** eV。**取值范围：** 有限正数。

将离散跃迁展宽为 Lorentz 线型。该参数为半高半宽 HWHM，半高全宽 FWHM 为其两倍。0.10 eV 对应 FWHM 0.20 eV。改变它会改变峰的宽度与峰高，不改变已经求得的激发能，也不能替代 k 网格收敛。

#### `spectrum_energy_step_ev`：光谱能量步长

**类型：** 实数。**默认值：** `0.01`。**单位：** eV。**取值范围：** 有限正数。

指定输出光谱的能量采样间隔，不影响 BSE 本征问题的求解精度。模板步长是半宽的十分之一，便于分辨线型；过大的步长可能跳过尖峰。

#### `spectrum_energy_min_ev`：光谱能量下限

**类型：** 实数。**默认值：** `0.0`。**单位：** eV。**取值范围：** 有限非负数。

限定输出横轴起点。模板从 0 eV 开始，即使最低激发能大于 0，Lorentz 尾部在低能端仍可非零。

#### `spectrum_energy_max_ev`：光谱能量上限

**类型：** 实数。**默认值：** `-1.0`。**单位：** eV。**取值范围：** 有限实数；负值采用自动上限。

任意有限负数表示自动取“最高保留激发能 + 5 个展宽半宽”；非负数指定上限，且不能小于下限，单位 eV。8 个例子显式设 12.0，以便在同一 0–12 eV 网格比较，和 0.01 步长一起生成 1201 个数据点。它不限制求哪些激发态，12 eV 以上的态仍可能通过展宽尾部贡献。

<a id="bse-reserved"></a>

### 5.8 保留选项

#### `bse_continue`：续算选项（保留）

**类型：** 整数。**默认值：** `0`。**取值：** 当前仅支持 `0`。

保留的旧接口，非零会报错。当前未实现矩阵构造的断点续算；已有本征态的光谱重算使用 `bse_solver spectrum`。模板省略它即使用 0，失败后的整条工作流要按实际成功阶段重新安排。

#### `bse_q_approx_mode`：动量转移近似（保留）

**类型：** 整数。**默认值：** `0`。**取值：** 当前仅支持 `0`。

使用当前实现的显式动量转移/实空间相位路径。非零的历史近似模式没有实现，会报错。双网格例子仍保持 0；插值由 `bse_use_fine_kgrid` 和相应输入控制。

#### `out_bse_ab`：BSE 矩阵输出（保留）

**类型：** 布尔值。**默认值：** `false`。**取值：** 当前仅支持 `false`。

原始分布式 BSE A/B 矩阵输出尚未实现，true 会被拒绝。正常运行仍会写出本征值、分布式振幅和光学结果。若检查 χ₀ 重建过程，可使用 `out_screening_matrices`；它输出的是屏蔽中间矩阵，不是 BSE 的 A/B。

### 5.9 参数组合与适用条件

| 计算类型 | 参数与数据要求 |
| --- | --- |
| 单网格静态 singlet TDA | `bse_use_fine_kgrid 0`、`bse_tda tda`、`bse_spin_types singlet`、`bse_ri_hartree 1`；QP 与 SCF 网格一致 |
| 细网格 BSE | `bse_use_fine_kgrid 1`；完整细网格 KS/band 数据；同网格 QP；匹配的粗网格 RI/屏蔽 |
| χ₀ 加解析 head/wing | `screened_format` 为 χ₀ 类型，`chi0_coulomb_metric full`、`chi0_headwing true` |
| χ₀ 的 single-cut 对照 | `chi0_coulomb_metric single_cut` 必须配 `chi0_headwing false` |
| 有效动态 BSE | 正的 `bse_plasma_energy_ev`、χ₀、TDA、ELPA、屏蔽通道，以及运行时满足束缚/介电条件 |
| 纯 IPA | `bse_spin_types ipa`、`bse_tda tda`；仍需 KS/QP 和速度数据 |
| 只重算光谱 | `bse_solver spectrum`、完整旧结果、相同 MPI 进程数和物理输入；正动态参数不能保留 |

第 6 节结合算例说明上述参数的选择。输入文件中省略的选项采用默认值。

## 6. Si 算例

本节从 Si 的一次静态光学激发计算开始，随后分别改变电子结构来源、屏蔽输入和 BSE 网格。例子 01、02 通过 LibRPA 获得 GW 数据；例子 03、04 改用 FHI-aims 的原生 W 和 χ₀；例子 05、06 在 χ₀ 的基础上加入有效动态修正；最后两个例子介绍粗网格相互作用与细网格波函数的配合使用。第一次使用时，建议先完整做完 01 或 02，熟悉每一步产生的文件，再继续后面的计算。

以下命令在第 4 节建立的运行副本中执行。每个例子的 `run.sh` 会从头完成所需的 DFT、GW 和 BSE 步骤，后一个例子不依赖前一个例子的输出。文中的输入片段均已包含在相应文件中，阅读时可对照查看，无需重复追加；转换程序也由 `run.sh` 自动调用。作业结束后，先按第 7.1 节确认退出状态，再执行本节的结果分析命令。

本节使用小网格和有限能带窗口，以便完成整套流程。列出的数值取自 `3_LibBSE_templatetest` 中已经完成的计算，可用于检查文件读取、矩阵维数和输出单位；它们不是 Si 的收敛结果。

<a id="example-common"></a>

### 6.1 例子 01：Si 的静态激发能与光谱

本例用 ABACUS 计算 Si 的基态和波函数，用 LibRPA 计算 GW 准粒子能量及屏蔽相互作用，最后求解 singlet TDA 方程。完成后，我们将从输出中读出最低激发能，并比较低能激发态的振子强度，理解激发能与光谱峰之间的区别。完整输入见 [例子 01](../examples/01_abacus_librpa_libbse/README.md)。

#### 6.1.1 提交计算

进入运行目录，检查程序路径和输入文件，然后提交作业：

```bash
cd "$TUTORIAL_RUN_ROOT/01_abacus_librpa_libbse"
bash run.sh --check
sbatch --export=ALL run.sh
```

保存 `sbatch` 返回的 job ID。计算运行期间，可以依次阅读 `INPUT`、`librpa.in` 和 `libbse.in`。这三个文件分别控制 DFT、GW 和 BSE；`run.sh` 按这一顺序调用程序，并将日志写入 `abacus.out`、`librpa.out` 和 `libbse.out`。下面沿着文件的生成顺序说明各阶段的设置。

#### 6.1.2 生成波函数和 GW 数据

`STRU` 给出 Si 原胞的结构以及所用赝势、轨道和辅助基文件，`KPT` 采用 2×2×2 网格。ABACUS 的 `INPUT` 中，`nbands 44` 为响应和 GW 计算准备占据态与空态；`rpa 1` 开启接口导出，`rpa_outdir ./OUT.librpa/` 指定导出目录，`rpa_out_vel 1` 同时输出后面计算光谱所需的速度矩阵。本例在 SCF 阶段生成这些数据，不另做 NSCF。

SCF 完成后，`OUT.scf/vxc_out.dat` 保存 KS 基中的交换关联势矩阵元，`OUT.librpa` 保存 KS 能量、波函数、RI 系数、Coulomb 矩阵及 `velocity_matrix`。接下来的 LibRPA 读取这些文件。打开 `librpa.in`，注意以下几项：

```text
task = g0w0
input_dir = OUT.librpa
fn_vxc_scf = ../OUT.scf/vxc_out.dat
output_dir = librpa.d
i_state_low = 0
i_state_high = 8
output_energy_qp = t
output_wc_rf = t
ifreq_output_wc_end = 1
```

`task = g0w0` 指定一次 GW 计算。这里 `fn_vxc_scf` 相对于 LibRPA 的 `input_dir` 解析，因此需要先从 `OUT.librpa` 返回上一级，再进入 `OUT.scf`。准粒子区间采用从 0 开始、右端不包含的编号，`0` 到 `8` 对应前 8 条能带，覆盖后面 BSE 要用的 4 条占据带和 4 条空带。`output_energy_qp` 将准粒子能量写入工作目录的 `energy_qp`；`output_wc_rf` 和 `ifreq_output_wc_end` 将所需的最低频率 Wc 块写入 `librpa.d/Wc_Mu_*_ifreq_0.mtx`。这两类输出分别提供 BSE 的单粒子能量差和屏蔽相互作用。

本例保留 `use_shrink_abfs = t`，使用 ABACUS 导出的辅助基压缩变换。当前构建的压缩 Wc 多进程导出有停滞记录，因此 `run.sh` 仅在 LibRPA 阶段采用一个 MPI 进程，并将分配的 CPU 用作 OpenMP 线程；ABACUS 和 LibBSE 仍使用 `NPROCS` 个进程。后面的例子 06 会介绍另一种在统一小辅助基中生成 χ₀ 的方法。

#### 6.1.3 设置 BSE 输入与跃迁空间

现在查看 `libbse.in` 的数据路径部分：

```text
input_dir                  OUT.librpa
input_format               librpa
qp_data                    energy_qp
qp_format                  energy_qp
screened_format            librpa_wc
screened_dir               librpa.d
output_dir                 libbse.d
```

三个输入位置对应刚才产生的三组数据：`input_dir` 指向电子结构和 RI 数据，`qp_data` 指向准粒子能量，`screened_dir` 指向屏蔽矩阵。它们不必位于同一目录，路径规则见 [5.2 节](#bse-inputs)。ABACUS 的接口文件遵循 LibRPA 约定，因此 `input_format` 选择 `librpa`。GW 输出的是相关部分 Wc，而 BSE 核需要完整 W；`screened_format librpa_wc` 会按 [5.3 节](#bse-screening) 的规则补上裸 Coulomb 相互作用。`output_dir` 则将本次 BSE 结果集中写入 `libbse.d`。

下面几项决定 BSE 矩阵的大小和求解方式：

```text
nocc                       4
nvirt                      4
bse_use_fine_kgrid          0
bse_nstates                -1
bse_solver                 elpa
bse_spin_types             singlet
bse_tda                    tda
bse_ri_hartree             1
```

Si 赝势计算有 4 条占据价带，本例全部纳入，并选取最低 4 条空带。根据 [5.4 节](#bse-space)，`nocc`、`nvirt` 和 k 点数共同决定电子–空穴跃迁空间；`bse_use_fine_kgrid 0` 表示直接使用上述 2×2×2 网格，因此 TDA 矩阵维数为 `4×4×8 = 128`。这里的 8 条活跃能带与 ABACUS 的 `nbands 44` 用途不同：前者决定最终 BSE 空间，后者还要为上游响应和自能计算提供空态。增大 `nvirt` 前，应先确认 `energy_qp` 覆盖新增的能带。

`bse_solver elpa` 构造并对角化 BSE 矩阵，`bse_nstates -1` 保存全部 128 个激发态。我们研究单重光学激发，所以选用 `bse_spin_types singlet`；该通道的核包含 `2V−W`，需要用 `bse_ri_hartree 1` 保留裸相互作用项。`bse_tda tda` 只求解共振部分，覆盖默认的 `both`，使本例得到一套 TDA 结果。各通道与求解模式的区别见 [5.5 节](#bse-solver)。

#### 6.1.4 设置光谱

`libbse.in` 的最后几项控制相位约定和光谱输出：

```text
abs_gauge                  velocity
wavefunction_gauge         first_k
spectrum_broadening_ev      0.10
spectrum_energy_step_ev     0.01
spectrum_energy_min_ev      0.0
spectrum_energy_max_ev      12.0
```

`abs_gauge velocity` 使用速度矩阵计算跃迁强度，这就是前面开启 `rpa_out_vel` 的原因。`wavefunction_gauge first_k` 与这套 ABACUS 数据的相位约定配合使用；下一例换用 FHI-aims 后会改为 `native`，两者的含义见 [5.7 节](#bse-optics)。

离散激发态经过展宽后形成光谱。这里 `spectrum_broadening_ev 0.10` 是 Lorentz 半宽，单位为 eV；`spectrum_energy_step_ev 0.01` 使每个半宽内有 10 个采样间隔。上下限取 0 和 12 eV，输出包括两个端点，共 1201 个能量点。这几个参数改变的是谱线展宽和采样，不改变已经求得的激发能。12 eV 只指定输出范围，并不保证当前能带窗口足以描述整个范围内的光谱。

后续 7 个例子沿用 `output_dir libbse.d`、`qp_format energy_qp`、ELPA、singlet、TDA、全部本征态，以及这一组光谱参数。输入中没有写出的选项采用第 5 节列出的默认值：本例 `bse_plasma_energy_ev` 为 0，做静态计算；`out_screening_matrices` 为 false，不输出屏蔽诊断矩阵；`bse_continue`、`bse_q_approx_mode` 和 `out_bse_ab` 分别保持 0、0 和 false。后面只展示与所讨论步骤有关的片段，完整设置仍以各例的 `libbse.in` 为准。

#### 6.1.5 读取激发能与振子强度

作业正常结束后，先检查 GW 是否覆盖全部 k 点，再查看 BSE 的低能态：

```bash
rg -c 'K_point' energy_qp
head -n 8 libbse.d/oscillator_strength_singlet_tda.dat
```

第一条命令应得到 8，表示 QP 文件有 8 个 k 点块。第二条命令显示列名和前 6 个激发态。为便于阅读，将保存结果的态编号、能量及各向同性振子强度摘录如下：

```text
state    energy_eV       f_isotropic
0        2.969877172     9.963282e-08
1        2.969879664     1.099604e-07
2        2.969881544     1.424189e-07
3        3.023003391     6.923023e+01
4        3.023006230     6.923032e+01
5        3.023009656     6.923045e+01
```

最低激发能约为 2.970 eV，但前三个态的振子强度很小；约 3.023 eV 的下一组态具有明显更大的光学权重。因此，读光谱时不能直接把最低本征值称为吸收峰。展宽后峰的位置和形状由邻近各态共同决定；`spectrum_singlet_tda.dat` 中最后一列给出各向同性的展宽振子强度密度，单位为 eV⁻¹。

还应检查激发态总数。`Excitation_Energy_singlet.dat` 以 Ry 保存能量，文件行数不等于态数；下面按数值个数计数，并转换最低能量的单位：

```bash
"$PYTHON" - <<'PY'
import numpy as np
energy = np.loadtxt('libbse.d/Excitation_Energy_singlet.dat').reshape(-1)
print('Number of states:', energy.size)
print(f'Lowest excitation: {energy.min() * 13.605693122994:.9f} eV')
PY
```

预期输出为 128 个态和约 2.969877172 eV，与振子强度文件的第一行一致。完成这些检查后，再运行第 7.2 节的完整数值检查，确认全谱有限、能量排序及输出维数。到这里，一次从 DFT 到 BSE 光谱的计算就完成了。

### 6.2 例子 02：使用 FHI-aims 波函数

本例保留 LibRPA 的 GW 计算，将基态数据改由 FHI-aims 提供。通过这一计算，可以区分 `input_format` 与 `screened_format` 的作用：前者指定电子结构数据接口，后者指定屏蔽矩阵的格式，它们可以来自不同程序。完整输入见 [例子 02](../examples/02_fhiaims_librpa_libbse/README.md)。

```bash
cd "$TUTORIAL_RUN_ROOT/02_fhiaims_librpa_libbse"
bash run.sh --check
sbatch --export=ALL run.sh
```

先查看 `control.in`。`output librpa bse` 使 FHI-aims 导出 BSE 所需的 KS、RI 和 Coulomb 数据；动量矩阵由 `compute_momentummatrix` 输出到 `mommat.h5`。`run.sh` 将这一阶段放在 `aims_export` 目录，并自动调用 `aims_mommat_to_velocity.py`，把 HDF5 中的动量数据转换为 `aims_export/velocity_matrix`。因而后面的 `abs_gauge velocity` 仍然成立。这里需要第 2 节安装的 NumPy 和 h5py；不能仅将 `mommat.h5` 改名来代替转换。

接着阅读 `libbse.in`：

```text
input_dir                  aims_export
input_format               fhi_aims
qp_data                    energy_qp
screened_dir               librpa.d
screened_format            librpa_wc
nocc                       4
nvirt                      3
bse_use_fine_kgrid          0
wavefunction_gauge         native
```

`input_format fhi_aims` 启用 FHI-aims 数据接口。KS 波函数和 RI 数据使用同一套原生相位约定，所以 `wavefunction_gauge` 改为 `native`，保留其内部一致性，见 [5.7 节](#bse-optics)。GW 仍由独立 LibRPA 完成，因此准粒子文件仍是根目录的 `energy_qp`，屏蔽仍从 `librpa.d` 按 `librpa_wc` 读取。更换 DFT 程序不要求把 Wc 改成其他屏蔽格式。

FHI-aims 是全电子计算，不能照搬例子 01 的绝对能带编号。本例的 4 条活跃占据带是第 11–14 带，3 条空带是第 15–17 带。`librpa.in` 中 `i_state_low = 10`、`i_state_high = 18` 按从 0 开始的半开区间写出第 11–18 带的 QP 能量，覆盖所需窗口。这里保留 3 条空带是为了复现这套算例；其收敛性仍需进一步检验。

计算结束后，确认 `aims_export/velocity_matrix` 已生成，检查 QP 网格及激发态：

```bash
test -s aims_export/velocity_matrix
rg -c 'K_point' energy_qp
head -n 5 libbse.d/oscillator_strength_singlet_tda.dat
```

QP 应覆盖 8 点；按上一例的 Python 命令计数，应得到 `4×3×8 = 96` 个激发态，保存结果的最低能量为 2.863575829 eV。它与例子 01 不同，除了电子结构程序和基组不同，本例的空带窗口也变了，因此两者不能作为只改变一个参数的收敛对照。

### 6.3 例子 03：直接读取 FHI-aims 的屏蔽相互作用 W

前两例都由 LibRPA 提供 GW 数据。本例改由 FHI-aims 同时计算准粒子能量和完整屏蔽相互作用 W，LibBSE 直接读取其输出。重点是理解完整 W 与相关部分 Wc 的区别，以及两个 FHI-aims 阶段分别准备什么数据。完整输入见 [例子 03](../examples/03_fhiaims_fhiaims_libbse_Wmatrix/README.md)。

```bash
cd "$TUTORIAL_RUN_ROOT/03_fhiaims_fhiaims_libbse_Wmatrix"
bash run.sh --check
sbatch --export=ALL run.sh
```

运行脚本先以 `control.in` 在 `aims_export` 生成 KS、RI 和速度数据，再以 `control_gw.in` 在 `aims_gw` 做原生 GW。需要分成两次 FHI-aims 运行，是因为本地接口的 `output librpa bse` 不能与原生 W/χ₀ 导出合用。两套输入的结构、基组和粗网格必须一致；本例也保持相同的 16 点频率设置。整个流程无需执行独立 LibRPA 驱动。

打开 `control_gw.in`，可以看到原生 GW 的输出设置：

```text
read_write_qpe w
output gw_regular_kgrid
output k_eigenvalue 8
periodic_gw_output_w .true.
```

这些设置为 2×2×2 规则网格准备 QP 和 W 文件。尤其要保留 `output k_eigenvalue 8`，否则本地版本可能只写出 Gamma 点的准粒子数据，无法覆盖 BSE 所用的 8 个 k 点。与之对应，`libbse.in` 中的数据路径为：

```text
input_dir                  aims_export
input_format               fhi_aims
qp_data                    aims_gw/energy_qp
qp_format                  energy_qp
screened_dir               aims_gw
screened_format            fhi_aims_w
```

`qp_format energy_qp` 描述文件的组织方式，并不限定它必须由 LibRPA 产生；这里读入的是 FHI-aims 的 QP。`screened_format fhi_aims_w` 则读取 `periodic_gw_w_q_*_rank_*.dat` 中的完整 W，按 [5.3 节](#bse-screening) 选择绝对值最小的虚频作为静态近似。由于文件已包含裸相互作用，不能再按 Wc 的方式补一次 V。

其余物理设置与例子 02 相同：`nocc 4`、`nvirt 3` 保留第 11–17 带窗口，`bse_use_fine_kgrid 0` 使用单一 8 点网格，`wavefunction_gauge native` 保持 FHI-aims 数据的相位约定。公共的 singlet、TDA 和光谱设置见 [6.1 节](#example-common)。

计算结束后，先看准粒子文件是否覆盖整个网格：

```bash
rg -c 'K_point' aims_gw/energy_qp
head -n 5 libbse.d/oscillator_strength_singlet_tda.dat
```

应得到 8 个 QP 块和 96 个 BSE 激发态，最低激发能约为 2.868235487 eV。例子 02 的对应值为 2.863575829 eV；这两个数接近，但它们来自不同的 GW 实现和屏蔽数据处理，不能据此要求全谱逐项相等。下一例将保持这套 FHI-aims 输入，改为让 LibBSE 从 χ₀ 构造 W。

### 6.4 例子 04：从 χ₀ 构造静态屏蔽

本例考察另一种屏蔽输入：FHI-aims 只导出独立粒子响应 χ₀，由 LibBSE 将其与 Coulomb 矩阵组合成 W。这样可以明确选择构造介电矩阵所用的 Coulomb 度量，以及 Gamma 点的 head/wing 处理。完整输入见 [例子 04](../examples/04_fhiaims_fhiaims_libbse_chi0/README.md)。

```bash
cd "$TUTORIAL_RUN_ROOT/04_fhiaims_fhiaims_libbse_chi0"
bash run.sh --check
sbatch --export=ALL run.sh
```

输入仍分为 `control.in` 和 `control_gw.in` 两部分，运行顺序与例子 03 相同。区别在于 `control_gw.in` 使用 `periodic_gw_output_chi0 .true.`，生成 `aims_gw/periodic_gw_chi0_q_*_rank_*.dat`。在 `libbse.in` 中，与这一变化对应的是：

```text
screened_dir               aims_gw
screened_format            fhi_aims_chi0
chi0_coulomb_metric         full
chi0_headwing              true
```

根据 [5.3 节](#bse-screening)，χ₀ 还不是屏蔽相互作用。`chi0_coulomb_metric full` 使程序用未截断的 full V 构造介电矩阵，再在外侧使用 cut V 得到 W；`chi0_headwing true` 在 Gamma 点加入 LibRPA 的解析 head/wing 修正。这两个选项应配套理解：这里的解析修正要求 full 度量，不能在保留 `true` 的同时随意改成 `single_cut`。

本例省略 `bse_plasma_energy_ev`，采用默认值 0，因此仍求解静态 BSE。程序选取原生 χ₀ 文件中最低的有限虚频点近似静态响应，并非额外计算严格零频 χ₀。`input_dir aims_export`、`qp_data aims_gw/energy_qp`、`nocc 4`、`nvirt 3`、`bse_use_fine_kgrid 0` 和 `wavefunction_gauge native` 沿用上一例，使电子结构来源、活跃空间和 k 网格保持相同。

作业完成后，读取低能激发态：

```bash
head -n 5 libbse.d/oscillator_strength_singlet_tda.dat
```

应仍有 96 个态，最低激发能约为 3.012523260 eV，比例子 03 的 2.868235487 eV 高约 0.144288 eV。这里改变了屏蔽构造方法，尤其 Gamma 点处理与直接读取原生 W 不同，因此这个差值不代表同一矩阵在两种存储格式之间转换的误差。若需要查看重建过程，可在新的运行副本中设置 `out_screening_matrices true`，检查第 5.3 节介绍的屏蔽中间矩阵；它不会输出 BSE 的 A/B 矩阵。

本例的结果还将作为下一例动态修正的静态参照。比较时应保留 χ₀ 的来源、Coulomb 度量和 head/wing 设置，这样才能把能量变化归因于动态核。

### 6.5 例子 05：比较静态与有效动态激发能

本例在上一例的 χ₀ 屏蔽上加入一次有效动态修正。计算目标是比较修正前后的最低激发能，并从输出中检查程序采用的束缚能。完整输入见 [例子 05](../examples/05_fhiaims_fhiaims_libbse_dynamics/README.md)。

```bash
cd "$TUTORIAL_RUN_ROOT/05_fhiaims_fhiaims_libbse_dynamics"
bash run.sh --check
sbatch --export=ALL run.sh
```

对照 04 和 05 的 `libbse.in`，可以看到电子结构路径、QP、屏蔽来源、4 条占据带和 3 条空带均保持不变。动态计算所需的组合为：

```text
screened_format            fhi_aims_chi0
chi0_coulomb_metric         full
chi0_headwing              true
bse_solver                 elpa
bse_spin_types             singlet
bse_tda                    tda
bse_plasma_energy_ev       15
```

新增的物理输入是 `bse_plasma_energy_ev 15`。这一正值启用 [5.6 节](#bse-dynamic) 的有效动态模型，要求 χ₀ 输入、ELPA、TDA 和包含屏蔽作用的通道。因此，不能在例子 03 的原生 W 输入上只追加这一行就得到同样的计算。15 eV 是本算例采用的模型参数；换成其他材料时，需要另行确定。

LibBSE 先求解静态 TDA，再用最小直接 QP 带隙与最低静态激发能之差确定束缚能，构造一次有效动态核，随后重新求解。`bse_use_fine_kgrid 0`、`nocc 4` 和 `nvirt 3` 使两次求解都在同一 96 维空间中进行。模型对束缚能和介电谱有适用条件，详见第 5.6 节；本计算没有进行实时传播或逐态自洽迭代。

作业结束后，先读动态对照文件，而不只看标准激发能文件：

```bash
head -n 8 libbse.d/dynamical_singlet.dat
```

保存结果的文件头和第一条数据为：

```text
# One-shot spectral effective BSE; all energies in eV
# plasma 15
# direct_QP_gap 3.550986200949546
# binding_used 0.5384629404753166
# state static effective shift
1 3.012523260474229 2.83861833612226 -0.1739049243519689
```

`static` 列给出约 3.012523260 eV，与例子 04 的最低激发能一致。`binding_used` 满足 `3.550986201 − 3.012523260 = 0.538462940 eV`，其中第一项为最小直接 QP 带隙。`effective` 列降到约 2.838618336 eV，`shift` 是两者之差，即 −0.173904924 eV，约为 −173.905 meV。读这个文件时不需要再做 Ry 到 eV 的换算。

再查看 `libbse.d/oscillator_strength_singlet_tda.dat`，其第一条能量应对应 `effective`，标准光谱也由有效动态本征态计算。单独保存的 `static_excitation_singlet.dat` 则以 Ry 给出静态本征值，用来检查静态参照。动态文件的态编号从 1 开始，振子强度文件从 0 开始；比较时应按行序和能量对应，不能直接把两个编号当成相同格式。

### 6.6 例子 06：使用 LibRPA 的零频 χ₀

这一例回到 ABACUS 和 LibRPA 数据来源，生成严格零频的 χ₀ 后计算有效动态 BSE。它同时说明一个数据准备要求：χ₀、RI 系数和 Coulomb 矩阵必须采用相同的辅助基。完整输入见 [例子 06](../examples/06_abacus_librpa_libbse_dynamics/README.md)。

```bash
cd "$TUTORIAL_RUN_ROOT/06_abacus_librpa_libbse_dynamics"
bash run.sh --check
sbatch --export=ALL run.sh
```

先阅读 `run.sh` 中 ABACUS 之后的处理步骤。ABACUS 导出的 `Cs_data_*` 和 `Cs_shrinked_data_*` 分属压缩前后的辅助基，不能与另一种维数的 Coulomb 矩阵混用。脚本调用 `prepare_chi0_input.py OUT.librpa OUT.librpa_chi0`，在 `OUT.librpa_chi0` 中选取相容的小辅助基 RI、full V 和 cut V 文件，建立相对链接，并保留原始 KS 和速度数据。文件映射记录在该目录的 `input_view.json` 中，矩阵数值本身不被修改。

因此，本例的 LibRPA 和 LibBSE 都将 `input_dir` 设为 `OUT.librpa_chi0`。打开 `librpa.in`，检查：

```text
input_dir = OUT.librpa_chi0
use_shrink_abfs = f
use_shrink_chi = f
output_wc_rf = f
output_chi0_rf = f
output_chi0_static = t
```

这里关闭进一步压缩，是因为输入已经选用了统一的小辅助基。`output_chi0_static = t` 额外计算并输出严格零频的响应，文件写入 `librpa.d/chi0_static`；这一步与 GW 虚频积分网格上的常规 χ₀ 输出不同。对应的 `libbse.in` 设置为：

```text
input_dir                  OUT.librpa_chi0
input_format               librpa
qp_data                    energy_qp
screened_dir               librpa.d/chi0_static
screened_format            librpa_chi0
chi0_coulomb_metric         full
chi0_headwing              true
bse_plasma_energy_ev       15
nocc                       4
nvirt                      4
bse_use_fine_kgrid          0
wavefunction_gauge         first_k
```

`screened_format librpa_chi0` 按 [5.3 节](#bse-screening) 读取零频响应并重建 W；`full` 和 `true` 保持与例子 05 相同的度量及解析修正约定。KS 数据来自 ABACUS，所以相位选项仍为 `first_k`。BSE 使用 8 个 k 点和 4×4 条占据–空带组合，静态与有效动态求解均有 128 个态。动态参数仍取 15 eV，公共的 singlet、TDA、ELPA 设置满足其求解要求。

计算结束后，先检查辅助基记录及零频响应的元数据，再查看动态结果：

```bash
cat OUT.librpa_chi0/input_view.json
test -s librpa.d/chi0_static/chi0_rf.info
head -n 8 libbse.d/dynamical_singlet.dat
```

`input_view.json` 应说明使用 ABACUS 的小辅助基，并记录未修改矩阵值。零频目录还应包含 `Chi0_Mu_*_ifreq_0.mtx`。动态文件第一态的静态能量约为 2.961446239 eV，有效能量约为 2.838264540 eV，位移为 −0.123181699 eV；文件头给出的直接 QP 带隙约为 3.302330146 eV，对应束缚能约为 0.340883907 eV。

这个静态值与例子 01 的 2.969877172 eV 已经不同，因为两例还改变了辅助基处理和屏蔽生成方式。评估本例的动态修正应比较同一个 `dynamical_singlet.dat` 中的 `static` 与 `effective`，不能把例子 01 与本例的最终能量差全部算作动态效应。

### 6.7 例子 07：在细 k 网格上求解 BSE

前面的例子都在同一个网格上准备电子结构、屏蔽和 BSE 跃迁空间。本例改用 3×3×3 粗网格计算 RI 与屏蔽，在 4×4×4 细网格上准备 KS 波函数和 QP 能量，最后求解细网格 BSE。这样可以增加电子–空穴跃迁的 k 点采样，而不在细网格上重新计算屏蔽。完整输入见 [例子 07](../examples/07_abacus_librpa_libbse_interpolate_333_444/README.md)。

```bash
cd "$TUTORIAL_RUN_ROOT/07_abacus_librpa_libbse_interpolate_333_444"
bash run.sh --check
sbatch --export=ALL run.sh
```

#### 6.7.1 准备两套网格的数据

目录中有 `INPUT_scf`、`KPT_scf` 和 `INPUT_nscf`、`KPT_nscf` 两套 ABACUS 输入。脚本先将 SCF 输入复制为 `INPUT` 和 `KPT`，在 3×3×3 网格上得到自洽电荷密度，并导出粗网格 RI 等数据到 `OUT.librpa`。随后切换到 NSCF 输入，读取 `OUT.scf` 的电荷密度，在 4×4×4 网格上生成细网格波函数，输出到 `OUT.nscf`。

接下来，脚本用 `preprocess_abacus_for_librpa_band.py -i OUT.nscf -o OUT.librpa` 将 NSCF 结果整理成细网格 band 文件。LibRPA 使用粗网格响应计算屏蔽，同时在 64 个细网格点计算自能，写出 `GW_band_spin_1.dat` 和 `KS_band_spin_1.dat`。最后调用 `band_to_energy_qp.py librpa . energy_qp_fine`，生成后面 BSE 需要的细网格 QP 文件。这些步骤都包含在 `run.sh` 中，不需要手工重复执行。

这里的 band 数据覆盖整个规则网格，不是通常画能带图所用的少数高对称路径。BSE 需要每个细网格 k 点的波函数及准粒子能量；只准备一条能带路径不能代替 4×4×4 网格。

#### 6.7.2 选择细网格输入

打开 `libbse.in`，与例子 01 对照以下几项：

```text
input_dir                  OUT.librpa
input_format               librpa
qp_data                    energy_qp_fine
qp_format                  energy_qp
screened_dir               librpa.d
screened_format            librpa_wc
nocc                       4
nvirt                      4
bse_use_fine_kgrid          1
bse_nstates                -1
wavefunction_gauge         first_k
```

`bse_use_fine_kgrid 1` 启用 [5.4 节](#bse-space) 的细网格路径：使用细网格 KS 数据和 QP 能量，并在粗网格得到的实空间相互作用表示上构造 BSE。它不会自动生成 NSCF 波函数或补齐 QP，因此必须先完成上一小节的数据准备。

`qp_data energy_qp_fine` 特别重要。工作目录中还可能存在粗网格的 `energy_qp`，文件名看似合适，内容却只有 27 个 k 点，不能用于 64 点 BSE。细网格转换结果仍采用 `energy_qp` 的内容格式，所以设置 `qp_format energy_qp`；这一显式设置也避免了自定义文件名触发 `auto` 的其他解析分支，详见 [5.2 节](#bse-inputs)。

`screened_format librpa_wc` 仍读取粗网格 Wc，动态参数保持默认 0。我们保留 4 条占据带和 4 条空带，将 BSE 点数增加到 64，因此 TDA 维数变为 `4×4×64 = 1024`；`bse_nstates -1` 将保存全部态。与例子 01 相比，矩阵维数增加了 8 倍，矩阵元素数增加到 64 倍，不能按 k 点数的增长直接估计对角化成本。LibRPA 的压缩和单 MPI 进程设置沿用例子 01。

#### 6.7.3 检查细网格和结果

作业结束后，先核对输入覆盖，再看 BSE 能量：

```bash
head -n 5 OUT.librpa/band_kpath_info
rg -c 'K_point' energy_qp_fine
head -n 5 libbse.d/oscillator_strength_singlet_tda.dat
```

`band_kpath_info` 第一行应为 `44 44 1 64`，最后的 64 表示细网格点数；后面列出 k 点坐标，最初几个点的坐标间隔为 0.25。QP 文件也应有 64 个 `K_point` 块。按例子 01 的计数方法读取激发能，应得到 1024 个态，最低能量约为 3.080997007 eV。

光谱仍采用 0–12 eV、0.01 eV 步长和 0.10 eV 半宽，因此光谱行数仍为 1201，并不随激发态数量增大。比较本例与例子 01 时，粗网格从 2×2×2 改成了 3×3×3，最终 BSE 网格也改成了 4×4×4；能量差包含两种变化。若要单独检验细网格插值的收敛，应固定粗网格与屏蔽，逐步加密细网格，并为每个新网格重新生成匹配的 KS 和 QP 数据。

### 6.8 例子 08：FHI-aims 的双网格计算

最后一个例子用 FHI-aims 完成同样的 3×3×3 → 4×4×4 计算，并直接读取原生 W。它把例子 03 的屏蔽接口与例子 07 的双网格思路结合起来，同时说明细网格下能带简并对活跃空间选择的影响。完整输入见 [例子 08](../examples/08_fhiaims_fhiaims_libbse_interpolate_333_444/README.md)。

```bash
cd "$TUTORIAL_RUN_ROOT/08_fhiaims_fhiaims_libbse_interpolate_333_444"
bash run.sh --check
sbatch --export=ALL run.sh
```

#### 6.8.1 生成细网格波函数与准粒子能量

`control.in` 和 `control_gw.in` 的 `k_grid` 都保持 3×3×3，分别完成接口导出和原生 GW。两套 control 还包含 `bands_444.in` 记录的 16 段 `output band`；每段 4 个点，合计覆盖无偏移 4×4×4 网格的全部 64 点。这里借助 band 输出准备规则网格数据，所以两套文件中的坐标、点数和顺序必须一致。

导出阶段生成细网格 KS 数据及动量矩阵，脚本随后转换速度矩阵。原生 GW 阶段生成 `GW_band1001.out` 至 `GW_band1016.out`，脚本用 `band_to_energy_qp.py aims aims_gw aims_gw/energy_qp` 将它们转换为统一 QP 文件。本例仍把文件命名为 `energy_qp`，但其内容已从例子 03 的 8 点变为 64 点，不能只凭文件名判断网格。

转换文件的 `e_gs` 列以 QP 能量占位，LibBSE 使用的 KS 数据来自独立导出目录。因而不能用该文件的 `e_qp−e_gs` 估计 GW 修正；需要分析修正时，应回到原始 KS/GW band 数据。共同的化学势能量平移会在跃迁能量差中抵消。

#### 6.8.2 设置 BSE 网格和能带窗口

本例的关键输入为：

```text
input_dir                  aims_export
input_format               fhi_aims
qp_data                    aims_gw/energy_qp
qp_format                  energy_qp
screened_dir               aims_gw
screened_format            fhi_aims_w
nocc                       4
nvirt                      3
bse_use_fine_kgrid          1
bse_nstates                -1
wavefunction_gauge         native
```

`fhi_aims_w` 继续直接使用完整 W，选择最低绝对虚频作静态近似，不额外加 V。`bse_use_fine_kgrid 1` 使用 64 点 KS/QP 数据，粗网格 RI 和屏蔽仍对应 3×3×3。`wavefunction_gauge native` 保留 FHI-aims 波函数与 RI 的相位约定；它的选择由电子结构数据决定，不因启用细网格而改变。

`nocc 4` 和 `nvirt 3` 沿用第 11–17 带窗口，所以矩阵维数为 `4×3×64 = 768`。它比上一例的 1024 小，是因为空带数为 3，而不是遗漏了细网格点。与例子 02 一样，这一窗口用于复现已有输入，正式计算需要检验能带数收敛。

#### 6.8.3 分析结果与简并能带

计算结束后，核对 QP 点数并查看低能态：

```bash
rg -c 'K_point' aims_gw/energy_qp
head -n 5 libbse.d/oscillator_strength_singlet_tda.dat
```

QP 应有 64 个 k 点块，BSE 应有 768 个态，保存结果的最低激发能为 3.074078960 eV。第 7.2 节的数值检查还会验证能量与振子强度文件一致，以及光谱是否包含预期的 1201 个采样点。

本例还有一个值得检查的能带窗口问题。在部分细网格 k 点，第 17、18 带属于简并子空间，而 `nvirt 3` 只保留到第 17 带。独立重新求解 KS 方程时，简并子空间中的本征矢可以发生旋转；保留其中一部分就可能改变截断后的 BSE 空间，即使 KS 和 QP 能量看起来一致。`wavefunction_gauge native` 保持已有数据的规范，但不能补回被能带窗口排除的态。

已保存的独立重算相对旧基准，最低态差为 0.277189 meV，全谱最大差为 7.791300 meV。使用旧电子结构数据和当前程序可以复现旧结果，而新 KS 配合旧 QP/W 又复现新结果，受控对照见 [说明文件](../../../3_LibBSE_templatetest/reference_checks/README.md)。因此，这一算例应结合维数、网格覆盖及数值一致性判断是否完成，不能以逐位复现旧谱作为唯一标准。后续研究应扩大窗口以包含完整的简并组，同时确认 KS/QP 数据覆盖新增能带，再建立相应的收敛结果。

### 6.9 参数收敛的起点

完成上述算例后，可以从所需数据路线的干净副本开始收敛研究。先选定要观察的量，例如最低几个激发能、某一能量范围内的谱峰位置或积分强度，再一次改变一类参数。第 7.3 节的数值用于复现既定输入，不能代替这一过程。

对 BSE 能带窗口，应逐步增加 `nocc` 和 `nvirt`，同时保持目标窗口内的 KS/QP 数据完整。`bse_nstates` 控制保存的激发态数，减小它不会缩小由占据带、空带和 k 点组成的矩阵，因此不能用它代替能带窗口收敛。对双网格计算，应分别检验粗网格屏蔽和细网格跃迁空间；每次加密细网格都要准备新的波函数、速度矩阵和 QP 能量，不能只改一个开关。

最后检验光谱采样与展宽。减小 `spectrum_energy_step_ev` 只加密谱线采样，改变 `spectrum_broadening_ev` 只改变展宽；它们都不能补偿过小的能带窗口或过粗的 k 网格。若只研究这两项，可按第 8 节使用已有本征态重算光谱。上游基组、响应空态数和频率网格同样需要收敛，应在相关 DFT/GW 输入中调整，并重新生成受影响的数据。

## 7. 结果检查

### 7.1 作业和日志

在作业目录执行，将示例 job ID 替换成相应的：

```bash
job_id=1234567
squeue -j "$job_id"
sacct -j "$job_id" --format=JobID,State,ExitCode,Elapsed -P
tail -n 20 "workflow-$job_id.out"
tail -n 30 libbse.out
```

应看到最终作业 `COMPLETED`、`ExitCode=0:0`，工作流日志末尾有 `Workflow completed: libbse.d`。任务从 `squeue` 消失不等于成功；仍需看 `sacct`。失败时按阶段查看 `abacus.out/.err`、`abacus_nscf.out/.err`、`aims_export/aims.out/.err`、`aims_gw/aims.out/.err`、`librpa.out/.err` 和 `libbse.out/.err` 中实际存在的文件。

### 7.2 输出文件及单位

8 个例子均设置 singlet、TDA、ELPA、速度规范、全部本征态，输出到 `libbse.d`。

| 文件 | 内容与检查方式 |
| --- | --- |
| `Excitation_Energy_singlet.dat` | 激发能，Ry；按全部数值计数，不能用行数当态数 |
| `Excitation_Amplitude_singlet_<rank>.dat` | 各 MPI rank 的局域振幅；保留全套文件 |
| `oscillator_strength_singlet_tda.dat` | `state energy_eV f_x f_y f_z f_isotropic` |
| `spectrum_singlet_tda.dat` | `energy_eV S_x S_y S_z S_isotropic`，谱密度单位 eV⁻¹ |
| `trans_dipole_singlet_tda.dat` | 跃迁偶极矩 |
| `trans_analysis_singlet_tda.dat`、`trans_kweight_singlet_tda.dat` | 跃迁和 k 点贡献分析 |
| `static_excitation_singlet.dat` | 仅动态例子：静态激发能，Ry |
| `dynamical_singlet.dat` | 仅动态例子：态编号、静态能、有效能、位移，能量单位 eV |

这些 `spectrum` 文件是展宽振子强度密度，不应直接标成介电函数 ε₂。模板采用 Lorentz 半宽 0.10 eV，能量范围 0–12 eV，步长 0.01 eV，因此光谱应有 1201 行数据。最低激发态可能是暗态，最低激发能不必等于吸收峰位置。

可在任一已完成的例子目录运行以下初步数值检查：

```bash
"$PYTHON" - <<'PY'
from pathlib import Path
import numpy as np

p = {}
for line in Path('libbse.in').read_text().splitlines():
    fields = line.split('#')[0].replace('=', ' ').split()
    if fields:
        p[fields[0]] = ' '.join(fields[1:])
nk = 64 if p['bse_use_fine_kgrid'] == '1' else 8
expected = nk * int(p['nocc']) * int(p['nvirt'])
out = Path(p['output_dir'])
energy = np.loadtxt(out/'Excitation_Energy_singlet.dat').reshape(-1) * 13.605693122994
osc = np.loadtxt(out/'oscillator_strength_singlet_tda.dat', ndmin=2)
spec = np.loadtxt(out/'spectrum_singlet_tda.dat', ndmin=2)
assert energy.size == expected
assert osc.shape == (expected, 6) and spec.shape == (1201, 5)
assert all(np.isfinite(a).all() for a in (energy, osc, spec))
assert (energy > 0).all() and (np.diff(energy) >= -1e-6).all()
assert np.allclose(energy, osc[:, 1], atol=1e-6, rtol=0)
assert (osc[:, 2:] >= -1e-10).all() and (spec[:, 1:] >= -1e-10).all()
assert np.allclose(spec[:, 0], np.arange(1201)*0.01)
print(f'Basic numerical checks passed: {expected} states; E1 = {osc[0,1]:.9f} eV')
PY
```

该段只检查常规能量/光谱；完整验收还要检查 QP 网格覆盖、细网格 KS、动态一致性与作业退出状态。预期完整 TDA 维数为 `Nk × nocc × nvirt`。

### 7.3 已完成测试的参考值

以下均来自保存的 [validation_report.md](../../../3_LibBSE_templatetest/validation_report.md)，其最终作业均为 `COMPLETED / 0:0`。05、06 的最低激发能列是有效动态值。

| 例子 | 态数 | 最低激发能 / eV | 历史最终 job ID |
| --- | ---: | ---: | ---: |
| 01 | 128 | 2.969877172 | 4461517 |
| 02 | 96 | 2.863575829 | 4460370 |
| 03 | 96 | 2.868235487 | 4461321 |
| 04 | 96 | 3.012523260 | 4461322 |
| 05 | 96 | 2.838618336 | 4461323 |
| 06 | 128 | 2.838264540 | 4461516 |
| 07 | 1024 | 3.080997007 | 4461518 |
| 08 | 768 | 3.074078960 | 4460601 |

05 的静态最低能量为 3.012523260 eV，动态位移 −173.904924 meV，使用的束缚能为 0.538462940 eV；06 对应值为 2.961446239 eV、−123.181699 meV 和 0.340883907 eV。动态表应满足 `位移 = 有效能量 − 静态能量`，以及 `束缚能 = 最小直接 QP gap − 最低静态激发能`。

这些是指定输入、程序及数值设置的参考值，不能用于证明不同代码/基组间必须精确一致。历史测试中 01、06、07 在修正后处理后复用了各自已成功的 ABACUS 导出；其他最终作业执行完整流程。重试经过保存在 [job_history.json](../../../3_LibBSE_templatetest/job_history.json)。

### 7.4 使用已有完整验证脚本

`3_LibBSE_templatetest/validate_results.py` 以脚本所在目录为根，默认读取该目录保存的作业记账记录；执行后会更新验证报告。下面命令针对历史测试目录，不会自动验证第 4 节新建的副本：

```bash
cd /work/users/l/s/lsr/LibBSE/3_LibBSE_templatetest
python3 validate_results.py
python3 compare_reference.py
```

需要查询 Slurm 更新状态时才使用 `python3 validate_results.py --refresh-jobs`；历史作业若已超出记账保留期限，可使用保存的快照。若要为新的运行目录复用整套验证，应同时适配脚本根目录、参考路径和相应的 job ID 记录，不能沿用历史 job ID 作为新作业成功的证据。

## 8. 常见问题

| 现象 | 检查与处理 |
| --- | --- |
| `LIBRI_INCLUDE_DIR does not exist` | 本地实际路径是 `LibRI/LibRI/include`，不是默认的 `LibRI/include` |
| 找不到 ELPA 或反对称求解符号 | 检查安装前缀、头文件、库版本以及 ELPA 是否包含所需接口 |
| 编译时变量 `I` 引发错误 | 检查独立 LibRPA 是否采用严格 C++17，换新 build 目录重新配置 |
| 动态库 `not found` | 在与运行作业相同的 module 环境中检查 `ldd` 和 `LD_LIBRARY_PATH` |
| 已编译却找不到 `chi0_main.exe` | LibBSE 子项目关闭独立驱动，按 3.2 节另外编译 |
| `--check` 成功但导出关键字不识别 | 核对 ABACUS/FHI-aims 是否为兼容模板扩展的版本 |
| `Existing output` | 用新的干净副本重新运行；脚本没有通用断点续算开关 |
| FHI-aims QP 只有 Gamma | 03–05 保留 `output k_eigenvalue 8`；改好后重新生成 GW 数据 |
| 缺少 `velocity_matrix` | 检查 ABACUS 速度导出，或 FHI-aims 的 `mommat.h5` 与转换器日志 |
| 01/07 在 LibRPA Wc 导出处停滞 | 保留脚本中的单 rank LibRPA 设置；不把该限制外推为所有路径都不能并行 |
| 06 的矩阵维数或屏蔽异常 | 检查 `OUT.librpa_chi0/input_view.json` 及 `use_shrink_abfs = f`，避免混基 |
| 07/08 的 QP 点数不够 | 检查 64 点 band 数据和转换步骤；只改 `bse_use_fine_kgrid` 不会生成细网格输入 |
| 动态模型拒绝计算 | 检查 χ₀、TDA、ELPA、静态束缚态及介电谱条件，不要只增大等离子体能量绕过错误 |

只调整展宽或光谱能量网格时，可在保留原结果的副本中将 `bse_solver` 改为 `spectrum`，使用相同 MPI 进程数读取已有能量和各 rank 振幅，再生成光谱。对于 05/06，还要将 `bse_plasma_energy_ev` 设为 0 以通过光谱重启的参数检查；这一步读取的是已有有效动态本征态，不会重新求静态核。`bse_continue` 不是通用重启开关。改变结构、基组、k 网格或 BSE 能带窗口时，应重新生成与之匹配的数据。

完成这些例子后，可从对应路径的输入副本开始做参数收敛。对 FHI-aims 保留 `wavefunction_gauge native`；改变 `nvirt` 时确认 KS/QP 导出覆盖完整窗口，尤其要纳入完整的简并能带组。
