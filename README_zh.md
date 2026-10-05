# LibBSE

```text
                ██╗     ██╗██╗     ██████╗ ███████╗███████╗
                ██║     ╚═╝██║     ██╔══██╗██╔════╝██╔════╝
                ██║     ██║██████╗ ██████╔╝███████╗█████╗
                ██║     ██║██╔══██╗██╔══██╗╚════██║██╔══╝
                ███████╗██║██████╔╝██████╔╝███████║███████╗
                ╚══════╝╚═╝╚═════╝ ╚═════╝ ╚══════╝╚══════╝
```

[English](README.md)

LibBSE 使用 FHI-aims、ABACUS 或 LibRPA 的输出求解周期性 Bethe--Salpeter
方程。当前计算路径通过 LibRPA 读取数据，使用外部 LibRI 源码构造 BSE
矩阵元，并用 ELPA 对 Tamm--Dancoff 近似（TDA）和完整 BSE 问题进行对角化。
光学分析目前仅支持速度规范。不同数据生产程序对应的公式推导和数据映射见
[`docs/fhi_aims_io_zh.md`](docs/fhi_aims_io_zh.md)。

## 编译 LibBSE

- CMake 3.16 或更高版本，以及支持 C++17 的编译器
- MPI、OpenMP、BLAS、LAPACK 和 ScaLAPACK
- FHI-aims 动量转换器需要 Python 3、NumPy 和 h5py
- 启用了反对称本征值求解器的 ELPA 安装
- 默认使用同级目录中的 LibRPA 源码和 `thirdparty/LibRI`

依赖库头文件路径统一通过 `CEREAL_INCLUDE_DIR`、`LIBRPA_INCLUDE_DIR`、
`LIBRI_INCLUDE_DIR` 和 `LIBCOMM_INCLUDE_DIR` 配置。因为 LibBSE 会把
LibRPA 的 BSE 文件读取 API 作为子项目编译，`LIBRPA_INCLUDE_DIR` 必须指向
LibRPA 源码目录中的 `include`。ELPA 的安装前缀单独通过
`EXTERNAL_ELPA_DIR` 配置。

编译命令示例：

```sh
cd LibBSE
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DEXTERNAL_ELPA_DIR=/work/users/l/s/lsr/b_BSE/elpa-2024.05.001/build/install \
  -DCEREAL_INCLUDE_DIR=../LibRPA/thirdparty/cereal-1.3.0/include \
  -DLIBRPA_INCLUDE_DIR=../LibRPA/include \
  -DLIBRI_INCLUDE_DIR=thirdparty/LibRI/include \
  -DLIBCOMM_INCLUDE_DIR=../LibRPA/thirdparty/LibComm/include
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

## 所需输入

LibBSE 从当前工作目录读取 `libbse.in`。`input_dir` 参数指向 LibRPA
工作流生成的 `OUT.librpa` 目录。当前计算路径使用以下数据：

- `OUT.librpa` 中由 LibRPA 导出的结构、基组、k 点网格、本征值、本征矢、
  速度、Cs 和截断 Coulomb 文件；
- 当 `bse_use_fine_kgrid=0` 时，从 `energy_qp` 读取粗网格准粒子能量；
- 当 `bse_use_fine_kgrid=1` 时，读取 `GW_band_spin_1.dat`、细网格
  `band_kpath_info` 以及 band 本征值和本征矢；
- 同级 `librpa.d` 目录中最低虚频对应的
  `Wc_Mu_*_Nu_*_iR_*_ifreq_0.mtx` 文件，作为静态极限近似。

对于 FHI-aims 直接导出的数据，设置 `input_format=fhi_aims`。先运行
`tools/aims_mommat_to_velocity.py AIMS_EXPORT_DIR`，把 `mommat.h5` 转为标准
`velocity_matrix`；脚本会显式修正 Fortran HDF5 的 k 网格轴序。LibBSE 再通过
非破坏性的兼容视图链接该文件与 `coulomb_cut_*`。所有计算都必须提供速度矩阵。
准粒子能量必须来自独立 LibRPA 步骤写出的 `energy_qp`；LibBSE 不再读取
FHI-aims 的 `GW_band*.out`。

粗网格 `velocity_matrix` 包含 SCF k 点网格上的 `velocity_mo`。如果 BSE
网格与其相同，LibBSE 直接使用这些 MO 矩阵元。对于双网格计算，例如 Si 的
5x5x5 到 6x6x6 计算，程序先把算符变换到局域 AO/实空间表示，再使用细网格
KS 波函数构造 `velocity_mo`。最终的光谱收缩始终使用 `velocity_mo`。

设置 `bse_use_fine_kgrid=0`，可同时使用 SCF k 点、本征矢以及 `energy_qp`
中的准粒子能量。在这种模式下不需要 `band_kpath_info` 和
`band_KS_{eigenvalue,eigenvector}_k_*.txt`。模式 `1` 保留独立的细网格/双网格
计算路径。

## 运行

在计算目录创建 `libbse.in`。完整模板如下：

```text
input_dir               OUT.librpa
output_dir              libbse.d
input_format            auto
qp_data                  OUT.librpa
qp_format                auto
screened_dir             librpa.d
bse_nstates             -1
nocc                    4
nvirt                   4
bse_solver              elpa
bse_spin_types          singlet triplet
bse_continue            0
bse_tda                 both
bse_ri_hartree          1
bse_use_fine_kgrid      1
bse_q_approx_mode       0
out_bse_ab              0
abs_gauge               velocity
wavefunction_gauge       auto
spectrum_broadening_ev   0.10
spectrum_energy_step_ev  0.01
spectrum_energy_min_ev   0.0
spectrum_energy_max_ev  -1
```

然后运行：

```sh
OMP_NUM_THREADS=8 mpirun -n 4 /PATH/build/LibBSE
```

输入同时接受以空白分隔的 `key value` 和 `key = value` 写法。注释以 `#`
或 `!` 开头；同一参数出现多次时，以最后一次为准。省略 `output_dir` 时，
默认值为 `libbse.d`。参数名和枚举值不区分大小写；路径保留原有大小写，
并相对于 `libbse.in` 所在目录解析。

布尔参数的真值可以写成 `1`、`true`、`t`、`.true.` 或 `yes`，假值可以写成
`0`、`false`、`f`、`.false.` 或 `no`。

ABACUS 与 FHI-aims 的完整可运行流程模板见
[`examples/README.md`](examples/README.md)。

## 完整参数说明

### `input_dir`

- **必填，没有默认值。** 指定公共 mean-field/RI 数据集目录。
- 对于 LibRPA 数据集，通常指向 `OUT.librpa`，并直接传给 LibRPA reader。
- 对于 FHI-aims 数据集，`basis_out` 必须包含 `aims` producer 标记；目录中
  还必须包含当前计算所需的结构、k 网格、能带、波函数、RI 和 Coulomb 文件。
- 相对路径以 `libbse.in` 所在目录为基准，而不是以可执行文件所在位置为基准。

### `output_dir`

- **默认值：** `libbse.in` 同级目录下的 `libbse.d`。
- 保存激发能、每个 rank 的局域振幅、光学分析文件、LibRI 日志，以及需要时
  创建的 FHI-aims 兼容视图。
- `bse_solver spectrum` 从该目录读取已有能量和振幅；普通计算会在必要时创建
  该目录。

### `input_format`

- **默认值：** `auto`。
- `auto`：检查 `basis_out` 的 producer 字段；如果是 `aims`，选择
  `fhi_aims`，否则选择 `librpa`。
- `librpa`：按照标准 LibRPA 文件约定直接读取 `input_dir`，不建立 FHI-aims
  兼容视图。
- `fhi_aims`：要求存在 `aims` 标记；在
  `output_dir/fhi_aims_reader_view` 中链接预先转换好的标准
  `velocity_matrix`，并为 `coulomb_cut_*` 建立兼容旧 LibRPA Coulomb 名称的链接。

### `qp_data`

- **默认值：** 解析后的 `input_dir`。
- 独立指定 LibRPA 准粒子能量文件或目录。三阶段流程中，独立 LibRPA 通常在
  工作目录写出 `energy_qp`，因此该参数可以与 mean-field/RI 目录分开。
- 当 `qp_format energy_qp` 时，目录表示 `<qp_data>/energy_qp`，文件路径则
  直接使用。
- 对于细网格路径，目录表示
  `<qp_data>/GW_band_spin_1.dat`，文件路径则直接使用。

### `qp_format`

- **默认值：** `auto`。
- `auto`：如果 `qp_data` 中存在 `energy_qp`，选择它；否则仅当
  `bse_use_fine_kgrid=1` 时使用历史 `GW_band_spin_1.dat` reader。不会回退到
  FHI-aims QP 文件。
- `energy_qp`：读取 LibRPA 粗网格数据块，其中包含占据数、Hartree 单位的
  KS 能量和 QP 能量。
- `fine_band`：读取历史 `GW_band_spin_1.dat` 细网格表。

### `screened_dir`

- **默认值：** `input_dir` 同级的 `librpa.d`，即
  `<parent-of-input_dir>/librpa.d`。
- 目录必须包含 LibRPA 写出的实空间相关相互作用文件
  `Wc_Mu_*_Nu_*_iR_*_ifreq_0.mtx`。
- LibBSE 使用它们构造 `W(R,iw0) = V(R) + Wc(R,iw0)`。`singlet` 和
  `triplet` 通道需要该目录；纯 `rpa` 或 `ipa` 通道不需要。

### `bse_nstates`

- **默认值：** `-1`。
- `-1`：计算或读取完整电子-空穴对空间中的全部本征态。
- 任意正整数：只保留最低的指定数量本征态。
- BSE 维数为 `Nk * nocc * nvirt`；`0`、小于 `-1` 或超过该维数的值会被
  拒绝。

### `nocc`

- **默认值：** `4`；接受任意正整数。
- 表示每个 k 点纳入 BSE 的占据能带数。LibBSE 选择 QP 记录中最后 `nocc`
  个占据态，所以它表示活跃价带窗口，而不是全电子计算的总占据态数。
- 它与 `nvirt` 和当前 k 网格共同决定 BSE 矩阵维数，以及 RI/光学收缩所用
  的能带行。

### `nvirt`

- **默认值：** `4`；接受任意正整数。
- 表示每个 k 点纳入 BSE 的空态数。LibBSE 选择占据 QP 窗口之后最前面的
  `nvirt` 个空态。
- 每条映射后的 QP 记录以及 KS 波函数数据必须覆盖所请求的完整占据态加空态
  窗口。

### `bse_solver`

- **默认值：** `elpa`。
- `elpa`：构造请求的 IPA/BSE 矩阵，求解本征态并写入 `output_dir`，然后
  进行光学分析。
- `spectrum`：跳过 kernel 构造和 ELPA，从 `output_dir` 重新读取所请求的
  能量及每个 rank 的局域振幅，只执行光学分析。重启时应使用与写出振幅文件
  相同的 MPI 进程数。

### `bse_spin_types`

- **默认值：** `singlet triplet`。
- 接受以空白或逗号分隔的列表。列表不能为空，且每种类型不能重复。
- `singlet`：`A = gap + 2V - W`，`B = 2V - W`；同时需要裸相互作用和
  屏蔽相互作用。
- `triplet`：`A = gap - W`，`B = -W`；不包含 Hartree/交换贡献，需要屏蔽
  相互作用。
- `rpa`：`A = gap + 2V`，`B = 2V`；使用裸相互作用，但不读取 Wc。
- `ipa`：`A = gap`，`B = 0`。如果列表中只有 IPA，程序直接构造并排序跃迁，
  不调用 LibRI 或 ELPA，并要求 `bse_tda=tda`。IPA 也可以出现在混合列表中，
  此时使用公共求解路径。

### `bse_continue`

- **默认值和当前唯一支持值：** `0`。
- `0`：正常开始所请求的 kernel/本征值求解流程。
- 任何非零整数都会被拒绝。该参数是保留的旧开关，并不是光谱重启方式；复用
  本征态应设置 `bse_solver=spectrum`。

### `bse_tda`

- **默认值：** `both`。
- `tda`：只求解共振的 Hermitian A 矩阵。输出文件名为
  `Excitation_Energy_<type>.dat` 和
  `Excitation_Amplitude_<type>_<rank>.dat`。
- `full`：只求解耦合 A/B 问题，写出带 `full` 标记的能量以及 X/Y 振幅文件。
- `both`：对每个请求的自旋类型同时执行上述两种计算。
- 纯 IPA 计算只支持 `tda`。

### `bse_ri_hartree`

- **默认值：** true。
- true：允许构造由裸相互作用产生的 Hartree/交换项。当
  `bse_spin_types` 包含 `singlet` 或 `rpa` 时必须开启。
- false：仅当所有请求通道均不需要 Hartree 项时才允许，例如仅计算 triplet
  或仅计算 IPA。该选项不会关闭屏蔽相互作用 W。

### `bse_use_fine_kgrid`

- **默认值：** `1`。
- `0`：使用 SCF k 点和规则网格 KS 本征矢，不读取独立的
  `band_kpath_info` 和 band-path 本征矢文件。QP 行必须映射到 SCF 网格；
  FHI-aims 3x3x3 直接路径使用该模式。
- `1`：使用独立的细/band k 网格及其本征值、本征矢。粗细网格不同时，光学
  分析会通过局域 AO/实空间表示变换速度算符。
- 其他整数都会被拒绝。

### `bse_q_approx_mode`

- **默认值和当前唯一支持值：** `0`。
- `0`：使用已经实现的 LibRI 显式动量转移/实空间相位路径，不选择其他
  q 近似分支。
- 非零的旧模式尚未实现，会被拒绝。

### `out_bse_ab`

- **默认值和当前唯一支持值：** false。
- false：写出本征值、分布式振幅以及请求的光谱分析，但不输出原始分布式
  A/B 矩阵。
- true 尚未实现，会被明确拒绝而不是静默忽略。

### `abs_gauge`

- **默认值和当前唯一支持值：** `velocity`。
- `velocity`：利用动量/速度矩阵和 KS 跃迁能隙计算跃迁偶极矩与振子强度。
- 长度规范及其他值目前不受支持，会被拒绝。

### `spectrum_broadening_ev`、`spectrum_energy_step_ev`、
`spectrum_energy_min_ev`、`spectrum_energy_max_ev`

- 控制每个 `spectrum_*.dat` 文件的 Lorentz 半宽和能量网格。默认值依次为
  0.10 eV、0.01 eV 和 0 eV。
- 最大能量为负数（默认 `-1`）时，自动取最高激发能加五个半宽。半宽和步长
  必须为正数。
- 光学分析现在总是执行。已经删除的 `bse_compute_spectrum` 会被拒绝；缺少
  速度/动量矩阵的输入也会立即报错。

### `wavefunction_gauge`

- **默认值：** `auto`。
- `auto`：对 FHI-aims 解析为 `native`，对其他 LibRPA 数据集解析为
  `first_k`。
- `native`：在数据生产程序原有且彼此一致的能带规范中使用 KS 和 RI 系数。
  这是 FHI-aims 的物理适当选择，也适用于简并点。
- `first_k`：采用历史标量相位约定，把每条能带按照它与第一个 k 点同一能带
  的重叠进行对齐。该模式保留给使用此约定生成的数据；一般情况下不能安全地
  用它固定简并子空间的规范。

初始化后，rank 0 会输出 MPI 进程数、请求/实际提供的 MPI 线程级别以及
OpenMP 线程配置。程序结束时还会输出分层的 LibBSE 计时表，其中包括文件
读取、相互作用准备、LibRI 矩阵构造、ELPA 求解、速度矩阵准备和光谱分析的
调用次数、CPU 时间与墙钟时间。主要阶段结束时会输出
`DONE(elapsed SEC) : description` 标记。

每次组装 A 矩阵后，LibBSE 检查 `||A-A^H||_F`；每次组装 B 矩阵后检查
`||B-B^T||_F`。检查直接作用于现有的二维块循环分布，并使用 ScaLAPACK 和
MPI 归约；超过 `1e-6` 阈值时会报告警告。

LibRI 的 k 空间块通过 `transform_k_2dlocal` 算法直接发送到负责对应二维
矩阵块的进程，不会在每个 rank 上组装一份完整 BSE 矩阵。ELPA 求解后，
本征矢不经过 root gather，而是重新分布，使每个 rank 仅保存其连续电子-空穴
对区间上的全部目标态。

`FineVelocityMo` 只为相同的局域电子-空穴对区间保存三个速度分量和 KS gap；
双网格数据从细网格波函数的 owner 直接发送到振幅 owner。速度规范收缩和
k 点权重仅使用局域数据，然后归约到 rank 0。只有较小的最终跃迁数据表以串行
方式写出。Wc 输入同样只读取局域 LibRI `list_I x list_J` 原子对。

不同通道的矩阵系数为：

- singlet：`A = gap + 2V - W`，`B = 2V - W`；
- triplet：`A = gap - W`，`B = -W`；
- RPA：`A = gap + 2V`，`B = 2V`；
- IPA：`A = gap`，`B = 0`。

纯 IPA 计算要求 `bse_tda=tda`，程序会直接构造并排序独立粒子态，不使用
LibRI 或 ELPA。其他通道使用 LibRI 和 ELPA。每个请求的 `<type>` 会写出：

- `Excitation_Energy_<type>.dat`
- `Excitation_Amplitude_<type>_<rank>.dat`
- `Excitation_Energy_full_<type>.dat`
- `Excitation_Amplitude_full_{X,Y}_<type>_<rank>.dat`
- `trans_dipole_<type>_{tda,full}.dat`
- `oscillator_strength_<type>_{tda,full}.dat`
- `spectrum_<type>_{tda,full}.dat`
- `trans_analysis_<type>_{tda,full}.dat`
- `trans_kweight_<type>_{tda,full}.dat`

`spectrum` 文件给出 Lorentz 展宽后的 x/y/z 方向及各向同性振子强度密度。
振子强度无量纲，谱密度单位为 eV^-1。输出的是每个原胞的强度：相干跃迁偶极
在强度中包含 k 网格归一化因子 `1/Nk`。
triplet 也会写出同样的文件，但根据从 singlet 基态出发的电偶极自旋选择定则，
其中的偶极矩与振子强度均为零。
每个 MPI rank 独立读写自己的振幅文件。光谱重启计算应使用与生成这些文件的
ELPA 计算相同的 MPI 进程数。

单元测试清单见 [`tests/README.md`](tests/README.md)。
