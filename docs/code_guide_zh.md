# LibBSE 代码上手与修改指南

这份文档面向第一次接手 LibBSE 的开发者：先把程序跑起来，再知道一个数从哪里读入、经过什么公式、在哪些进程上计算、最后写到哪里。它按当前源码讲解，不把输入关键字的名字当作功能已经实现的证据。

分析日期：2026-09-14。LibBSE 基准提交：91693d23387acfae67ff5b6a6a78d75b1a5f0c3b。本次只增加开发文档，不修改数值算法或构建配置。相邻 LibRPA、LibRI 的实现也已核对；它们是独立源码目录，会随各自版本变化。

配套索引：

- [逐函数参考](code_functions_zh.md)：本项目 C++、Python 函数的签名、源码行号、作用、调用线索与并行语句。
- [逐文件清单](code_files_zh.md)：逐项列出分析时 Git 管理的全部文件，包括第三方源码、测试、图片和缓存文件，说明归属及修改入口。
- [第三方接口与实现导航](code_external_zh.md)：LibRPA、LibRI、LibComm、ELPA、BLAS/LAPACK/ScaLAPACK、GreenX 等的调用链。

阅读顺序建议：第 1—4 节建立整体印象，第 5—9 节理解数值与并行，再按第 12 节找要改的文件，最后查逐函数参考。代码片段是开发用法示意；涉及 MPI 的调用必须放在已初始化且通信域一致的环境里。

## 阅读目录

- [1. 程序到底负责什么](#section-1)
- [2. 目录地图与构建关系](#section-2)
- [3. 第一次配置、运行和验证](#section-3)
- [4. 输入参数逐项解释](#section-4)
- [5. 数据结构、下标和单位](#section-5)
- [6. 主流程和重要分支](#section-6)
- [7. 数学公式对应代码](#section-7)
- [8. 光谱从哪里来](#section-8)
- [9. 并行到底如何设置、如何分工](#section-9)
- [10. 文件协议和边界条件](#section-10)
- [11. 文件与函数的开发用法](#section-11)
- [12. 按修改目标找入口](#section-12)
- [13. 测试如何证明修改正确](#section-13)
- [14. 性能、内存和排错](#section-14)
- [15. 覆盖范围与阅读证据](#section-15)

<a id="section-1"></a>

## 1. 程序到底负责什么

LibBSE 接收已经算好的单粒子数据和屏蔽相互作用，构造电子—空穴哈密顿量，求激发能和激发态，再计算光学跃迁。可以把一个激发态理解为许多“从占据带跳到空带”的小跃迁叠加在一起。

~~~text
ABACUS 或 FHI-aims
    └─ KS 能带、波函数、基组、结构、RI 系数、裸库仑、速度/动量
独立运行的 LibRPA
    └─ QP 能量 energy_qp 或 GW_band_spin_1.dat、Wc(R,iw0)
LibBSE
    ├─ 读取和统一数据约定
    ├─ LibRI 计算相互作用核
    ├─ ELPA 求 TDA / full-BSE 激发态
    └─ 速度规范光谱和跃迁分析
~~~

LibBSE 虽然链接 rpa_lib，但这条主流程没有调用 GW/χ0 计算入口来现场生成 QP 和 Wc。它调用的是读取、数据结构、傅里叶变换及线性代数接口。运行前必须准备相应数据。

正式入口是 [driver/main.cpp](../driver/main.cpp) 的 main。可执行文件名区分大小写，为 LibBSE。它不接受命令行参数，固定读取**当前运行目录的 libbse.in**。输入文件中的相对路径以该输入文件所在目录解析。

<a id="section-2"></a>

## 2. 目录地图与构建关系

| 位置 | 负责的事 | 开发时怎么用 |
|---|---|---|
| CMakeLists.txt | 引入 LibRPA，创建核心库、程序和 CTest | 加源文件、改依赖、加测试 |
| driver/main.cpp | MPI 生命周期、参数、读取数据、调用总流程 | 加全局初始化/退出行为 |
| src/parameter/ | 输入参数、默认值、合法性检查、常量 | 加参数的第一站 |
| src/interface/ | LibRPA 和线性代数包装 | 外部 API 变更优先收口于此 |
| src/io/ | QP/Wc 读取、RI 转换、FHI-aims 适配 | 改文件格式、单位、带窗口 |
| src/bse/ | 核矩阵、求解、振幅分布、光谱 | 改物理算法 |
| src/utils/ | 计时和进度信息 | 定位耗时、补日志 |
| tests/ | 小规模确定性测试 | 找最小调用示例 |
| tools/ | 动量转换、RI 系数调试比对 | 处理上游数据，不参加 C++ 主程序编译 |
| examples/ | ABACUS/FHI-aims 完整生产链模板 | 参考目录与输入组织 |
| docs/ | 接口说明与本指南 | 查数据约定及开发入口 |
| thirdparty/ | 本仓库保存的依赖副本 | 先确认是否被构建选中 |
| build* | 编译产物、CMake 缓存 | 查实际编译命令；不要当源文件修改 |

核心库 libbse_core 编译 12 个 .cpp：bse_calculation、distributed_amplitudes、elpa_solver、matrix_checks、molecular_lri、spectrum、librpa_api、bse_files、fhi_aims_adapter、parameter、profiler、progress。LibBSE 额外编译 driver/main.cpp，再链接核心库。

~~~text
LibBSE → libbse_core → rpa_lib
                         ├─ 外部 ELPA
                         ├─ BLAS/LAPACK/ScaLAPACK + BLACS
                         ├─ MPI/OpenMP 运行库
                         └─ LibGXMiniMax / GXCommon
头文件模板：LibRI → LibComm → cereal
离线 Python 转换：h5py → HDF5，NumPy
~~~

### 2.1 先认清实际生效的依赖副本

顶层 CMake 默认 LIBRI_INCLUDE_DIR=../LibRI/include；在当前磁盘布局中，这个默认路径不存在。现有 build_external_libri 缓存实际用 ../LibRI/LibRI/include。因此第一次配置需要显式填写。

本次读取该构建目录的 CMakeCache.txt、CMakeFiles/libbse_core.dir/flags.make 和 CMakeFiles/LibBSE.dir/link.txt 得到：

| 项目 | 当前构建记录 |
|---|---|
| LibRPA | ../LibRPA/include，由其所在源码目录构建 |
| LibRI | ../LibRI/LibRI/include |
| LibComm | ../LibRPA/thirdparty/LibComm/include |
| cereal | ../LibRPA/thirdparty/cereal-1.3.0/include |
| ELPA | /work/users/l/s/lsr/b_BSE/elpa-2024.05.001/build/install，libelpa_openmp.so |
| MPI 编译器 | OpenMPI 5.0.9 / GCC 15.2.0 下的 mpicxx、mpifort |
| BLAS/LAPACK | /usr/lib64/libopenblas.so |
| ScaLAPACK | /nas/sycamore/apps/aocl/5.2.0/lib_LP64/libscalapack.so |
| GreenX | LibRPA 子构建中的 libLibGXMiniMax.a、libGXCommon.a |

这是**缓存和链接命令的快照**，不保证所有 build* 目录相同，也不是重新编译后的验证报告。修改模板前查 flags.make 的 -I 顺序；修改二进制库前查 link.txt，运行时再查 ldd。改本仓库 thirdparty/LibRI，当前这份构建通常不会因此改变。

<a id="section-3"></a>

## 3. 第一次配置、运行和验证

下面在仓库根目录执行；依赖路径按本机快照填写，环境模块应提供同一套编译器/MPI/数学库。

~~~bash
cmake -S . -B build-dev \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_Fortran_COMPILER=mpifort \
  -DLIBRPA_INCLUDE_DIR="$PWD/../LibRPA/include" \
  -DLIBRI_INCLUDE_DIR="$PWD/../LibRI/LibRI/include" \
  -DLIBCOMM_INCLUDE_DIR="$PWD/../LibRPA/thirdparty/LibComm/include" \
  -DCEREAL_INCLUDE_DIR="$PWD/../LibRPA/thirdparty/cereal-1.3.0/include" \
  -DEXTERNAL_ELPA_DIR=/work/users/l/s/lsr/b_BSE/elpa-2024.05.001/build/install \
  -DSCALAPACK_DIR=/nas/sycamore/apps/aocl/5.2.0/lib_LP64 \
  -DLIBBSE_BUILD_TESTING=ON
cmake --build build-dev -j 4
ctest --test-dir build-dev --output-on-failure
~~~

-j 只是同时编译任务数。程序运行 MPI 数由 mpirun -np 或作业系统决定。

粗网格 singlet TDA 的 libbse.in 示例：

~~~text
input_dir              ./export
output_dir             ./libbse.d
input_format           auto
qp_data                ./energy_qp
qp_format              energy_qp
screened_dir           ./librpa.d
nocc                   4
nvirt                  4
bse_nstates            -1
bse_solver             elpa
bse_spin_types         singlet
bse_tda                tda
bse_use_fine_kgrid      0
abs_gauge              velocity
wavefunction_gauge     auto
spectrum_broadening_ev 0.10
spectrum_energy_step_ev 0.01
~~~

在装有该文件和真实数据的运行目录启动：

~~~bash
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
mpirun -np 4 /绝对路径/LibBSE/build-dev/LibBSE > libbse.out 2> libbse.err
~~~

若只改展宽或输出能量网格，已有振幅可用时设 bse_solver spectrum。仍需 KS、速度和 QP 文件，并保持原 MPI 数、带窗口、k 顺序、波函数规范和求解类型。bse_continue 不是这条重启路径的开关，它只允许 0。

<a id="section-4"></a>

## 4. 输入参数逐项解释

解析器接受 key value 和 key = value；#、! 后是注释；可有 INPUT_PARAMETERS 标题；键和枚举不区分大小写；重复键以最后一次为准；未知键报错。路径不是 shell 表达式，文件内容中的 $HOME 不会由解析器展开。

| 参数 | 默认值 | 真正作用及限制 |
|---|---|---|
| input_dir | 无，必填 | LibRPA 风格导出目录 |
| output_dir | libbse.d | 结果、rank 日志、aims 读取视图 |
| input_format | auto | auto/librpa/fhi_aims；auto 看 basis_out 的 aims 标记 |
| qp_data | 解析后为 input_dir | 可为文件或目录；energy_qp 常在独立 LibRPA 运行目录，需显式设置 |
| qp_format | auto | energy_qp/fine_band；auto 优先发现 energy_qp |
| screened_dir | input_dir 父目录下 librpa.d | Wc 文件目录 |
| bse_nstates | -1 | 全部正激发态；否则必须正数且不超过 D |
| nocc | 4 | 选取靠近带隙的占据态数量 |
| nvirt | 4 | 选取靠近带隙的空态数量 |
| bse_solver | elpa | elpa 求解；spectrum 重算光谱 |
| bse_spin_types | singlet triplet | 还可 rpa ipa；逗号或空格分隔，不可重复 |
| bse_continue | 0 | 非零报错 |
| bse_tda | both | tda/full/both |
| bse_ri_hartree | true | singlet/rpa 要求 true；不是 RI 数据读取总开关 |
| bse_use_fine_kgrid | 1 | 1 读独立 band 数据；0 使用 SCF 网格 |
| bse_q_approx_mode | 0 | 非零报错；当前按 k 差生成 q |
| out_bse_ab | false | true 报错，未实现 A/B 输出 |
| abs_gauge | velocity | 仅支持 velocity |
| wavefunction_gauge | auto | aims 为 native，其它为 first_k；可显式指定 |
| spectrum_broadening_ev | 0.10 | Lorentz 半高半宽 γ，必须正数 |
| spectrum_energy_step_ev | 0.01 | 网格步长，必须正数 |
| spectrum_energy_min_ev | 0.0 | 不可为负 |
| spectrum_energy_max_ev | -1.0 | 负数取最大激发能与下界中较大者，再加 5γ |

纯 IPA（列表只有 ipa）要求 bse_tda tda，走单跃迁快速路径。混合列表中的 IPA 不满足 ipa_only()，会随通用矩阵流程处理。不能把纯 IPA 的省内存行为推广到混合列表。

Constants 的主要阈值：RI/库仑筛选均 1e-12；k 比较 1e-10；fine-band QP 坐标容差 1e-6；energy_qp 坐标容差 5.1e-5；占据判断 0.1；矩阵逐元素对称误差 1e-6；写振幅置零阈值 1e-10；KS 零能隙阈值 1e-14。这些不是输入关键字，修改需要编译。

<a id="section-5"></a>

## 5. 数据结构、下标和单位

### 5.1 先记住三个维度

设 BSE 网格有 \(N_k\) 个点，占据带数 \(N_o\)，空带数 \(N_v\)。

$$
P=N_oN_v,\qquad D=N_kP,\qquad p(k,i,a)=kP+iN_v+a.
$$

全部从 0 开始。p 是哈密顿量行列编号。逆映射为 k=p/P，i=(p%P)/nvirt，a=p%nvirt。例如 nocc=2,nvirt=3，每个 k 有 6 对；(k=1,i=0,a=2) 对应 pair 8。

QP 窗口先存 nocc 个占据态，再存 nvirt 个空态；原始 KS 波函数可能有核态，取行使用 qp.ncore + selected_band。不要直接拿 QP 文件行号访问 KS 波函数。

### 5.2 核心类型

| 类型 | 内容 | 存储/用法 |
|---|---|---|
| Complex | std::complex<double> | 共轭位置有物理意义 |
| Cell / KPoint | 三个整数位移 / 三个分数坐标 | 相位包含 2π |
| AtomCell | 第二原子编号和 R | 与外层第一原子组成键 |
| TensorMap<T> | map<I,map<(J,R),RI::Tensor<T>>> | 原子块稀疏组织 |
| QuasiparticleBands | 窗口、核偏移、QP 能量、带隙 | energies_ry[k*nbands+b] |
| EigenSolution | 激发能、局部块循环本征向量 | 列主序；full 仍是 2D 行列的分布 |
| DistributedAmplitudes | D、态数、本地 pair 起点/数量、振幅 | values[state*local_pairs+local_pair] |
| FineVelocityMo | 本地 pair 的速度三分量及 KS gap | values[direction*local_pairs+local_pair] |
| OscillatorStrength | 能量 eV、方向强度及平均 | ABACUS Ry 约定 |
| SpectrumPoint | 网格能量、方向密度及平均 | 密度单位 eV⁻¹ |
| MatrixCheckResult | 通过标志、差范数、和范数、相对误差 | passed 看逐元素绝对阈值 |

DistributedAmplitudes::operator() 返回元素引用，const 重载只读，均不检查边界。一般 RI::Tensor 的布局与 ScaLAPACK 不同。LibRI 返回 CVC 张量按 (j,b,i,a) 排布，其缓冲在 transform_k_2dlocal 中按 row + column*P 解释，这个约定必须整体维护。

### 5.3 单位边界

| 数据 | 单位/转换 |
|---|---|
| energy_qp 的 QP 列 | Ha，读入乘 2 得 Ry |
| GW_band_spin_1.dat 能量 | eV，除 13.605693122994 得 Ry |
| qp.energies_ry、A/B、激发能文件 | Ry |
| LibRI 相互作用结果 | 按 Ha 约定，装配乘 2/Nk |
| Dataset KS 能量、gaps_ha | Ha |
| canonical velocity_matrix 文件 | eV·Å，LibRPA reader 转回原子单位 |
| 光谱/跃迁表能量显示 | Ry 乘 13.605693122994 得 eV |

常见错误：v/gap 用 Ry gap 差一个 2；装配后的核再除 Nk 会重复归一化；直接把 Wc 当 W 会少裸库仑。

<a id="section-6"></a>

## 6. 主流程和重要分支

### 6.1 main 生命周期

1. MPI_Init_thread 请求 MPI_THREAD_FUNNELED，检查提供级别。
2. 启动总计时，PARAM.read()，resolve_input_format()。
3. LibRPA_API::initialize() 初始化 LibRPA 全局状态及其 ELPA 环境。
4. ReaderOptions：spectrum/纯 IPA 不读 RI；细网格开关控制 band 数据。
5. LibRPA_API::read_dataset() 返回共享 Dataset。
6. 所有 rank 调 run_bse()。
7. 先销毁 Dataset，再 finalize LibRPA，最后 MPI_Finalize。

main 捕获标准异常后返回 1，但没有普遍的跨 rank 异常同步或 MPI_Abort 兜底。若一个 rank 抛异常、其它 rank 进入 collective，仍可能挂住。aims adapter 的 collective_error 展示了“root 捕获、广播错误、所有 rank 一起失败”的局部做法。

### 6.2 run_bse 公共前置步骤

创建结果目录 → read_qp_bands → 报告带隙 → apply_wavefunction_gauge → 计算 D 和态数。规范处理在分支之前，因此 spectrum restart 也要求规范与旧振幅一致。

### 6.3 spectrum 分支

准备本地速度 → root 读能量，各 rank 读自己的振幅 → write_velocity_gauge_outputs。TDA 的 Y 指针为空；full 同时读 X/Y。仅提供激发能和振幅不够，读取视图、KS 和 QP 数据仍要存在。

### 6.4 纯 IPA 分支

solve_ipa_tda 枚举 QP 跃迁能差、排序、选最低 nstates 个。每态只有一个 pair 振幅为 1，各 rank 只填自己拥有的位置。随后写结果和算光谱。它不调用 ELPA，不构造 D×D 矩阵。简并能差排序没有额外物理规范保证，宜比较简并子空间或合并强度。

### 6.5 通用相互作用分支

~~~text
build_bare_coulomb                 V(q)→V(R)，清空 vq_cut
MolecularLri 构造                  设置原子/k 任务分配及 q 映射
read_screened_interaction          读局部 Wc，形成 W=V+Wc
convert_lri_coefficients           转 complex，清空原 cs_data
remap_to_nearest_bvk_cell          Cs、V、W 重选最近周期胞
MolecularLri::initialize           安装 Cs/V/W、准备波函数、预变换 C'
ArrayDesc::init                   D×D 块循环分布
add_hartree_a / add_screened_a     按通道需要算公共 A 核
add_hartree_b / add_screened_b     full 时按需要算公共 B 核
release_interactions              释放 LibRI 中 V/W
每通道组合 A/B → 检查 → ELPA → 振幅重分布 → 写文件
prepare_fine_velocity_mo → 每通道输出光谱
~~~

requires_hartree/requires_screened 控制核收缩；**通用路径前面的 V/Wc 读取没有同样按这两个条件跳过**。例如只求 rpa 仍执行 read_screened_interaction。不能由系数为零推断文件不需要。

公共核只构造一次，再按通道系数组合。tda_results/full_results 保存分布式振幅直到统一算光谱，多通道和 both 模式的峰值内存包含这些结果。

<a id="section-7"></a>

## 7. 数学公式对应代码

公式按所检查实现写出。复数 B 核不宜用“类似 A”代替。

### 7.1 能隙和核

$$
\Delta_p^{QP}=E^{QP}_{a,k}-E^{QP}_{i,k}.
$$

add_qp_diagonal 在本 rank 拥有的对角元素写入它。assemble_channel_matrix 清零、写对角、累加核。H/W 表示已乘 2/Nk 的矩阵：

$$
A=\operatorname{diag}(\Delta^{QP})+c_HH_A+c_WW_A,\qquad
B=c_HH_B+c_WW_B.
$$

| 通道 | c_H | c_W |
|---|---:|---:|
| singlet | 2 | -1 |
| triplet | 0 | -1 |
| rpa | 2 | 0 |
| ipa | 0 | 0 |

系数来自 interaction_coefficients。screened_* 存相互作用收缩结果，吸引负号由 c_W=-1 加入，不在 reader 加负号。

### 7.2 LRI：把四轨道积分拆成小张量

AO 是原子轨道，MO 是能带轨道，辅助基近似轨道乘积。Cs[I][(J,R)] 的形状为“辅助基、第一 AO、第二 AO”。LibRI 先做：

$$
C^\mu_{st}(k)=\sum_R C^\mu_{st}(R)e^{+i2\pi k\cdot R},\qquad
C'^\mu_{sm}(k)=\sum_t C^\mu_{st}(k)c_{mt}(k).
$$

对应 LRI_k::cal_Csk_ao_mo。Csk_ao_mo 保存一个 AO 指标已变成 MO 的中间量，供核重复使用。

定义 Cs_ao_mo_to_Cs_mo 实际计算的两个对象：

$$
L^\mu_{mn}(k_1,k_2)=\sum_s[
c^*_{ms}(k_1)C'^\mu_{sn}(k_2)+C'^{\mu*}_{sm}(k_1)c_{ns}(k_2)],
$$

$$
R^\mu_{mn}(k_1,k_2)=\sum_s[
c_{ms}(k_1)C'^{\mu*}_{sn}(k_2)+C'^\mu_{sm}(k_1)c^*_{ns}(k_2)].
$$

is_left_conj=true 选 L，false 选 R；O 选 [0,nocc)，V 选 [nocc,nocc+nvirt)。两项来自局部 RI 两端贡献，不能随手删去一项。

输出矩阵行是 (k1,i,a)，列是 (k2,j,b)，q=(k2-k1) mod 1。当前外部 LibRI 收缩为：

$$
\widetilde H_A=\sum_{\mu\nu}R^\mu_{ia}(k_1,k_1)V_{\mu\nu}(0)L^\nu_{jb}(k_2,k_2),
$$

$$
\widetilde H_B=\sum_{\mu\nu}R^\mu_{ia}(k_1,k_1)V_{\mu\nu}(0)R^\nu_{jb}(k_2,k_2),
$$

$$
\widetilde W_A=\sum_{\mu\nu}L^\mu_{ji}(k_2,k_1)W_{\mu\nu}(q)R^\nu_{ba}(k_2,k_1),
$$

$$
\widetilde W_B=\sum_{\mu\nu}L^\mu_{bi}(k_2,k_1)W_{\mu\nu}(q)R^\nu_{ja}(k_2,k_1).
$$

波浪表示 LibRI 原始结果；transform_k_2dlocal 再乘 coefficient * 2/Nk 并累加。辅助指标包含原子及局部辅助基，MPI 装配会相加不同原子任务的局部和。

| LibBSE 方法 | LibRI 方法 | psi_type | is_A |
|---|---|---|---|
| add_hartree_a | cal_cvc_mo_k_hartree_onthefly | O,V,O,V | true |
| add_hartree_b | 同上 | O,V,O,V | false |
| add_screened_a | cal_cvc_mo_k_onthefly | O,O,V,V | true |
| add_screened_b | 同上 | V,O,O,V | false |

### 7.3 Wc、傅里叶符号和最近胞

build_bare_coulomb 调 FT_Vq(..., true, false)，采用 q→R 负号及 1/Ncell 因子。所检查 LibRPA 路径最后取 VR_cplx.real()，LibBSE 再包装成 complex tensor，不能声称保留任意复 V(R)。该路径对非原始 q-star 成员用共轭，源码注明其对称补全能力有限。

$$
W(R,i\omega_0)=V(R)+W_c(R,i\omega_0).
$$

ifreq_0 是频率**索引 0**，对应所用 minimax 网格最低虚频节点，不自动等于严格 ω=0。当前将它用于静态极限近似。

remap_to_nearest_bvk_cell 根据原子坐标、晶格、BvK 周期改 R 的代表，不改数值。move_tensor 遇已有键报错，不静默覆盖。粗网格等价 R 在任意细 q 上相位未必相同，改最近胞策略会影响插值。LibRI set_tensors_map2 还可能周期归类；调相位须继续追外部 LRI.hpp。

### 7.4 波函数规范

native 返回全 1 相位，不改波函数。first_k 对每个带与**列表第一个 k 点**做 AO 系数重叠：

$$
s_n(k)=\sum_\mu c^*_{n\mu}(k)c_{n\mu}(k_0),\quad
p_n(k)=s_n(k)/|s_n(k)|,\quad c_n(k)\leftarrow p_n(k)c_n(k).
$$

k0 不保证 Γ；重叠没有 AO 重叠矩阵 S，不能解释成严格跨 k 波函数内积。零重叠报错。这只能固定带整体相位，不能处理简并子空间任意酉旋转。

同网格速度需要同步：

$$
v_{ia}(k)\leftarrow p_a(k)p_i(k)^*v_{ia}(k).
$$

不同网格用改规范后的细网格波函数投影 AO 算符，自然带入规范。只改其中一边会破坏光谱一致性。

### 7.5 TDA 与 full-BSE

TDA 解 AX=ΩX，A 要 Hermitian。solve_tda_elpa 设置 nev=nstates，调复数 elpa_eigenvectors，输入矩阵作为可覆盖工作区。

full-BSE 定义由测试残差验证：

$$
\mathcal H\binom X Y=\Omega\binom X Y,\qquad
\mathcal H=\begin{pmatrix}A&B\\-B^*&-A^*\end{pmatrix},
\quad A=A^\dagger,\ B=B^T.
$$

它不是普通 Hermitian 矩阵。实现构造：

$$
M=\begin{pmatrix}\Re(A+B)&\Im(A-B)\\-\Im(A+B)&\Re(A-B)\end{pmatrix},
\quad J=\begin{pmatrix}0&I\\-I&0\end{pmatrix}.
$$

elpa_cholesky 给上三角 U，M=UᵀU。两次分布式 GEMM 得 K=UJUᵀ，调用 elpa_skew_eigenvectors，取该接口约定的实谱参数 Ω 与分开存储实/虚部的向量。每列除 √|Ω|，再乘 Uᵀ，最后：

$$
Q=\frac1{\sqrt2}\begin{pmatrix}I&-iI\\-I&-iI\end{pmatrix},
\qquad v=QU^Tz/\sqrt{|\Omega|}.
$$

恢复 X/Y。能量取排序后 D 项中的前 nstates 项；X 是这些列上半行，Y 是下半行。full 仍计算 2D 个 skew 本征向量，小 nstates 不代表省去完整 skew 求解。

这条路径依赖 M 正定，Cholesky 失败需检查数据和物理稳定性。full 规范应为 X†X−Y†Y=I，不能改为普通欧氏归一化。

### 7.6 矩阵检查

check_hermitian 用共轭转置，check_symmetric 用普通转置。各 rank 检查真实本地元素再 Allreduce：

$$
\epsilon=\frac{\|M-M^{H/T}\|_F}{\|M+M^{H/T}\|_F}.
$$

日志输出 ε，但通过条件是**所有元素绝对差不超过 threshold**，不是 ε<threshold。两范数都零时比值为 0，仅分母零时为无穷大。run_bse 失败只打印 WARNING，仍求解，并非强制停止。

<a id="section-8"></a>

## 8. 光谱从哪里来

### 8.1 同网格速度

prepare_fine_velocity_mo 比较粗细 k 数量和逐点坐标，顺序必须一致。相等则直接选速度窗口。Dataset 存储约定为 matrix(row,column)=〈column|v|row〉，因此访问 (a,i) 得 v_ia，再乘规范因子。

### 8.2 不同网格插值

要求粗 KS 带数等于 AO 数，波函数方阵可逆，R 数等于粗 k 数。C 行为 band、列为 AO，输入速度先转置成普通 MO 算符：

$$
V_{AO}(k)=\overline{C(k)^{-1}}\,V_{MO}(k)[C(k)^{-1}]^T,
\qquad
V_{AO}(R)=\frac1{N_k^{coarse}}\sum_k e^{-i2\pi kR}V_{AO}(k).
$$

按 AO 所属原子对找最近胞：

$$
[V_{AO}(k_f)]_{st}=\sum_R e^{+i2\pi k_fR_{I(s)J(t)}^{near}}[V_{AO}(R)]_{st},
\qquad V_{MO}(k_f)=\overline{C(k_f)}V_{AO}(k_f)C(k_f)^T.
$$

inverse 调 LAPACK zgetrf/zgetri，其它乘法用 LibRPA ComplexMatrix。粗 AO/k 和 AO/R 算符目前每 rank 构造；细网格投影按拥有波函数的 source rank 分工，再 Alltoallv 到 pair owner。不能说全程无复制。

### 8.3 跃迁偶极

令 g_p=ε_a,k^KS−ε_i,k^KS，单位 Ha：

$$
d_{S\alpha}=i\sqrt2\sum_p
\frac{v_{p\alpha}X_{Sp}-v_{p\alpha}^*Y_{Sp}}{g_p}.
$$

TDA 的 Y 为零。distributed_transition_dipole 做本地和，velocity_gauge_transition_dipoles_mpi Reduce 到 root。没有共轭 X，full 不能改成简单 v*(X+Y)。

### 8.4 强度、展宽及分析权重

$$
f_{S\alpha}=2\Omega_S^{Ry}|d_{S\alpha}|^2,\qquad
f_S^{iso}=(f_{Sx}+f_{Sy}+f_{Sz})/3.
$$

calculate_oscillator_strengths 验证 kpoint_count 正数，但**每态强度内部不除 Nk**。日志 total 另算 Σf_iso/(4Nk*nocc)。现有 tests/README.md 相关措辞与实现有出入，应以函数和断言为准。

$$
S_\alpha(E)=\sum_S f_{S\alpha}\frac{\gamma}{\pi[(E-E_S)^2+\gamma^2]}.
$$

γ 是半高半宽。输出是强度密度，不是已经包含材料光学转换因子的吸收系数、介电函数或折射率。

trans_kweight 对所有请求态聚合：weight1=Σ(|X|²+|Y|²)；weight2=Σ2(|vX/g|²+|v*Y/g|²)，后者没有不同 pair 的干涉项，不能当总偶极平方。trans_analysis 只收集 abs(X)>0.3 的贡献，按绝对值排序，并非完整振幅。

triplet 按当前无自旋轨道耦合处理设为电偶极禁戒：root 清零偶极和 weight2，仍保留激发能、振幅和 weight1。

<a id="section-9"></a>

## 9. 并行到底如何设置、如何分工

### 9.1 三层设置不要混淆

| 层次 | 设置位置 | 作用 |
|---|---|---|
| MPI rank 数 | mpirun -np / 作业系统任务数 | 不同进程各有独立内存 |
| OpenMP 线程数 | OMP_NUM_THREADS；ELPA 设置 omp_threads | 同一进程内共享内存计算 |
| 数学库内部线程 | OPENBLAS_NUM_THREADS / MKL_NUM_THREADS 等 | GEMM、LAPACK 内部执行 |

main 请求 MPI_THREAD_FUNNELED：进程中只有初始化 MPI 的主线程调用 MPI。它不是 MPI_THREAD_MULTIPLE。即便相邻 LibRPA 的 CMake 为其独立程序选择 MULTIPLE，LibBSE 的 main 仍明确请求 FUNNELED。

ELPA handle 把 omp_get_max_threads() 传给 omp_threads。日志打印最大线程数、可用处理器数和 dynamic 状态；它报告运行库设置，不证明所有源码 pragma 已生效。

**当前构建的具体发现：**build_external_libri 的 libbse_core/LibBSE flags.make 中 CXX_FLAGS 没有 -fopenmp，但链接命令含 libgomp 和 libelpa_openmp。相邻 rpa_lib 把 OpenMP::OpenMP_CXX 放在 PRIVATE 链接依赖中。由这些生成文件判断，不能认定 LibBSE 自身及在其翻译单元实例化的 LibRI 模板已经开启 OpenMP 编译；ELPA 内部线程与此是两件事。本次没有修改构建配置。

若接下来要启用这一层，可在 CMake 顶层显式 find_package(OpenMP REQUIRED)，并将 OpenMP::OpenMP_CXX 链到 libbse_core；是否 PUBLIC 取决于其公开头文件模板是否需要相同编译选项。修改后必须确认实际编译命令出现 -fopenmp，再验证 1/2 线程数值一致。只设置环境变量无法补回编译阶段忽略的 pragma。

### 9.2 LibRI 原子和 k 任务分配

MolecularLri 构造调用：

~~~cpp
lr_.init(kpoints, options.nocc, options.nvirt);
lr_.set_parallel(dataset.comm_h.comm, dataset.atoms.size(),
                 nk, dataset.pbc.period_array);
~~~

外部 LR::set_parallel 调 Distribute_Equally::distribute_atom_and_k_pair，把任务看成 (I,J,k1,k2) 四维。它先比较 nat 与 nk，将较小维度放前，通过 Split_Processes::split_all 拆通信域，再 divide_atoms 取得各维下标列表。

结果为 list_I、list_J、k1_indices、k2_indices；list_IJ 和 k_indices 是各自并集。flag_task_repeatable=false 避免剩余子组重复完整任务。这个划分不等同于下面的 BLACS 二维进程网格。

Wc reader 只读 list_I×list_J；build_wavefunctions 只为 k_indices 装配逐原子的 MO 系数。全局矩阵元素可能收到多个原子任务贡献，后续必须累加。不能用“最后收到一个值覆盖原值”的方式组装。

### 9.3 LibRI 内部 OpenMP（以编译启用为前提）

当前生效外部头文件 LRI_k-cal_cvc_mo.hpp：

| 位置 | 调度方式 | 数据/同步 |
|---|---|---|
| cal_Csk_ao_mo | parallel for，static，collapse(2) | 展开 k×原子；每任务写独立预分配 tensor |
| screened 核 | parallel + for，dynamic,64，collapse(3) | q×μ原子×ν原子；每线程局部结果 map |
| Hartree V(q=0) | for，dynamic | 遍历 ν,R；线程局部 Vq |
| Hartree 核 | for，dynamic,64，collapse(2) | μ原子×k1；内部遍历 ν、k2 |
| 结果合并 | add_Ds_omp_try_map / wait_map | 按外层 key 的 omp_lock_t 保护 |
| Hartree 中间阶段 | barrier → master 清理 Vq 锁 → barrier | 合并完成后才读共享 Vq |

这些收缩的 OpenMP 区域没有直接写 LibBSE 的 ScaLAPACK 矩阵；结束后才调用 transform_k_2dlocal 的 MPI 交换。不能让工作线程自行执行这个集体交换。

若定义 __MKL_RI，库在部分收缩前暂将 MKL 线程设为 1，结束恢复。当前 core 编译定义中没有该宏，也不能据此推断 OpenBLAS 自动单线程；使用外层 OpenMP 时应另检查数学库线程设置，避免每个线程再起一队线程。

### 9.4 求解器二维块循环分布

Dataset 构造时初始化 blacs_h，再 set_square_grid()，按 MPI 数选接近方形网格，默认倾向行数较多、行优先 rank 布局。LibBSE 的输入没有 nprow/npcol 参数；要更改需改 Dataset/BLACS 初始化或增加正式配置传递。

run_bse 设置正方块边长：

| D | mb=nb |
|---|---:|
| D≤500 | 1 |
| 500<D≤1000 | 32 |
| D>1000 | 64 |

source process row/column 都为 0。全局矩阵切成 mb×nb 小块，轮流分给进程网格；不是每个 rank 存一段完整连续行。

ArrayDesc 负责全局/局部转换：

- indx_g2l_r/c：全局行列转本地，不属于此 rank 的进程行/列返回负数。
- indx_l2g_r/c：本地行列转全局。
- g2p_r/c + get_pnum：定位完整元素所属 rank。
- m_loc/n_loc 是真实本地行列数，lld 是列主序 leading dimension。
- 缓冲分配 lld*n_loc，访问 local_row + local_column*lld。检查矩阵时只遍历真实 m_loc，不能把 padding 算入物理范数。

full_descriptor 是 2D×2D，块大小沿用 A/B。把子块复制到 full 不能直接复制 vector：不同全局位置可能属于另一 rank，必须调用 ScaLAPACK redistribute。

### 9.5 LibRI k-block → 块循环矩阵

transform_k_2dlocal 接收 blocks[k1][k2]。每块为 P×P，按 k1 每 64 个批处理：

1. 在 BLACS block 边界上切小矩形，计算 owner。
2. 统计发往每个 rank 的块头数与复数值数，分别 Alltoall。
3. 本 rank 目标直接累加；远端打包 BlockHead 与值，乘 coefficient*2/Nk。
4. 分别 Alltoallv 交换块头和数值。
5. 接收方验证 ownership，然后按本地列主序累加。

BlockHead 含 global_row/global_column/rows/columns，用 offsetof 构造 MPI struct datatype，不假设 C++ struct 无 padding。checked_total 防止 MPI int count 超限。batch 64 是代码常量，当前没有输入参数。

全部 rank 必须执行同样的批次数，即使本 rank 没有任何源块。这里没有 gather 全局 D×D 矩阵到 root。

### 9.6 本征向量 → 连续 pair 分布

求解后每个 rank 保存所有请求态，但只保存自己的 pair 区间。设 MPI 数为 M，D/M 的商为 b、余数为 r，rank t：

$$
\mathrm{count}_t=b+[t<r],\qquad
\mathrm{first}_t=tb+\min(t,r).
$$

例如 D=10,M=3，三个区间为 [0,4)、[4,7)、[7,10)。这是振幅和 FineVelocityMo 共同的分区规则。

redistribute_amplitudes 从源二维块循环矩阵选一个子块，发送目标本地索引和复数值；Alltoall 交换数量，两次 Alltoallv 交换索引/值，检查重复与遗漏。TDA 选 (row_offset=0,column_offset=0)；full 的正能量列从 D 开始，X 行偏移 0，Y 行偏移 D。

注意：这个接口的 offset **从 0 开始**；LibRPA_API::redistribute 的 ScaLAPACK 起点 **从 1 开始**。名字相似、下标约定不同。

### 9.7 光谱 MPI

| 阶段 | collective | 传什么 |
|---|---|---|
| 对齐规范 | Allreduce MIN、Bcast、Allreduce SUM | 波函数 source、首 k 参考、规范相位 |
| 不同网格速度投影 | Allreduce MIN | 每个细 k 选择一个拥有波函数的 source |
| 速度发给 pair owner | Alltoall、两次 Alltoallv | pair 编号和三个复速度 |
| 光谱状态信息 | Bcast | 态数、激发能 |
| 偶极 | Reduce SUM | 每态 3 个复数 |
| k 权重 | 两次 Reduce SUM | 每 k 的两个标量 |
| 大跃迁贡献 | Gather/Gatherv | abs(X)>0.3 的稀疏索引和值 |

全部激发振幅不会集中到 root；但能量、偶极、k 权重和筛选后的分析贡献会集中。源速度矩阵与粗网格插值中间量存在复制，内存评估要逐阶段看。

### 9.8 谁读写文件

每 rank 都读参数、QP；LibRPA reader 关闭 SCF k 波函数分发开关，粗波函数读取存在复制。Wc 由局部原子对分工读取。每 rank 写自己的振幅和 libri_rank_N.log。root 写激发能和光谱分析。

FHI-aims 视图只由 root 创建，广播错误后 Barrier，保证其它 rank 再进入 reader。done() 只在 root 打印时间，**不含 Barrier**，不是全局同步标记。

<a id="section-10"></a>

## 10. 文件协议和边界条件

### 10.1 QP 文件

fine_band 默认文件 GW_band_spin_1.dat，每行：索引、kx ky kz，再重复 occupation energy_eV。历史格式 occupation 含均匀 k 权重，reader 乘 Nk 后再判断占据。

energy_qp 按 K_point index : kx ky kz 找块，读取 state occupation KS_Ha QP_Ha，使用 QP 列。QP 可只含价带/导带窗口，而 KS 含核态，因此 wavefunction_core_offset 优先用 Dataset mean-field occupation 找 KS 窗口偏移。

match_kpoint 按周期距离 |Δ−round(Δ)| 比较，不依赖文件行顺序；重复、缺失、歧义 k 点报错。store_requested_bands 要求占据态在前、空态在后，并检查不同 k 的核态偏移一致。calculate_gaps 在选择窗口内计算 min_k(CBM_k−VBM_k) 和 min_k CBM_k−max_k VBM_k。

### 10.2 Wc 文件

文件名 Wc_Mu_I_Nu_J_iR_N_ifreq_0.mtx，I/J/iR 从 0 开始。真正 R 从注释括号中读取，不把 iR 当三个坐标。支持代码预期的复数稀疏坐标记录：

~~~text
%%MatrixMarket matrix coordinate complex general
% R = (0 0 0)
2 2 2
1 1 -0.10 0.00
2 2 -0.20 0.00
~~~

条目行列从 1 开始，内部减 1。检查 R 存在于裸库仑表、形状一致及行列范围；读取 nnz 项后加 V。这个 reader 不是完整通用 MatrixMarket 解析器：不要默认支持 real、array、对称压缩补全或所有注释排版。

### 10.3 激发态和重启格式

| 文件 | 内容/写入者 |
|---|---|
| Excitation_Energy_SPIN.dat | TDA 能量，Ry，root |
| Excitation_Energy_full_SPIN.dat | full 正激发能，Ry，root |
| Excitation_Amplitude_SPIN_RANK.dat | TDA，每 rank 本地 pair |
| Excitation_Amplitude_full_X_SPIN_RANK.dat | full X |
| Excitation_Amplitude_full_Y_SPIN_RANK.dat | full Y |
| trans_dipole_SPIN_tda/full.dat | 偶极、模平方、平均、eV，root |
| oscillator_strength_SPIN_tda/full.dat | 方向/平均强度，root |
| spectrum_SPIN_tda/full.dat | 展宽密度，root |
| trans_analysis_SPIN_tda/full.dat | 能量与大贡献列表，root |
| trans_kweight_SPIN_tda/full.dat | 累积 k 权重，root |

表中 tda/full 表示选择其一，文件名不包含斜线。振幅每行一个态，列为本 rank 连续 pair；复数用 C++ 流格式 (re,im)，科学计数 8 位小数，小于等于置零阈值直接写 0。文件没有记录 MPI 数/窗口/规范等元数据。reader 检查元素个数，但相同个数不保证来源兼容；改变这些设置需要重算或实现带元数据的重分布协议。

### 10.4 FHI-aims 转换与视图

resolve_input_format 看 basis_out 开头的 atoms nao naux producer，producer 为 aims 时选择 aims，auto 规范选择 native。

prepare_fhi_aims_reader_view 在 output_dir/fhi_aims_reader_view 创建普通文件的绝对路径符号链接，并将 coulomb_cut_* 映射为 coulomb_unshrinked_cut_*。marker .source_directory 记录源目录，防止把不同数据源混在同一视图。它不重新计算 RI，不把 aims 自己的 QP 输出变成 LibRPA QP。

必须先运行：

~~~bash
python3 tools/aims_mommat_to_velocity.py /路径/aims_export
~~~

工具使用 NumPy/h5py，读 band_out 与 mommat.h5。Energy_window 要在带范围内，当前只支持非自旋极化。h5py 看到的三网格轴为 kz,ky,kx，先反转，再按文件记录的 k 索引排列。上三角按原始打包顺序恢复，从 gradient 得 velocity=-i*gradient；补 Hermitian 另一半，对角取实数，乘 Ha_to_eV*Bohr_to_Angstrom。

输出 binary v1：7 个小端 int32 头（marker=-12345680，dtype=29，nk,nspin,nbands,nao,3）；每 k 表项为 int32 的 1-based k 索引和 int64 字节偏移；数据为 complex128，按 k/spin/direction/band/band 顺序。窗口外填零，不能据此认为窗口外速度真实为零。先写同目录临时文件，fsync 后 os.replace；拒绝覆盖符号链接。

<a id="section-11"></a>

## 11. 文件与函数的开发用法

完整签名和逐函数说明见[函数参考](code_functions_zh.md)。这里给出跨模块的最小使用关系。

### 11.1 参数模块

~~~cpp
libbse::Parameter p;
p.parse("input_dir ./export\nbse_tda tda\n", "/一个运行目录");
const auto &opts = p.inp;
if (opts.solve_tda()) { /* 进入 TDA */ }
~~~

parse 每次重置默认值，再解析，不是在原参数上追加。read 则从文件读取并设置 base_directory。print 不保证打印所有保留字段；添加参数需同时维护结构、解析、验证、输出和测试。

### 11.2 矩阵与求解

~~~cpp
librpa_int::ArrayDesc desc(dataset->blacs_h);
desc.init(D, D, block, block, 0, 0);
std::vector<libbse::Complex> a(
    static_cast<std::size_t>(desc.lld()) * desc.n_loc(), libbse::Complex{});
// 仅填属于本 rank 的矩阵元素。
auto check = libbse::check_hermitian(a, desc, 1e-6);
auto solution = libbse::solve_tda_elpa(a, desc, nstates);
auto x = libbse::redistribute_amplitudes(
    dataset->comm_h.comm, solution.vectors_local, desc,
    0, 0, D, nstates);
~~~

这是调用关系示意，不能用全零 A 代替真实输入验证材料结果。solve_full_elpa 会清空并 shrink A/B；调用后不能继续依赖其原数据。EigenSolution 的本地向量尺寸按 descriptor，不是 nstates*D 的全局 dense 数组。

### 11.3 光谱接口

~~~cpp
auto phases = libbse::apply_wavefunction_gauge(*dataset, opts, qp);
auto vel = libbse::prepare_fine_velocity_mo(opts, qp, dataset, phases);
auto dipoles = libbse::velocity_gauge_transition_dipoles_mpi(
    dataset->comm_h.comm, opts, vel, x, nullptr);
~~~

所有 rank 调，只有 root 得到非空 dipoles。serial velocity_gauge_transition_dipole 要求完整 pair 范围，不适合直接传某个 rank 的局部数据。正常生产流程用 write_velocity_gauge_outputs 自动处理广播、归约和写文件。

### 11.4 计时与进度

~~~cpp
{
    libbse::ScopedTimer timer(libbse::global::profiler,
                             "my_stage", "My calculation stage");
    // 新计算阶段
}
libbse::done("my calculation stage", dataset->comm_h.comm);
~~~

ScopedTimer 析构停止计时，异常展开也生效。Profiler 按父子关系存 timer；stop 只匹配栈顶，错误顺序不会自动跳过嵌套层。CPU 时间是本进程 std::clock，wall 是 steady_clock；root 打印不是 MPI 最慢 rank 统计。Profiler 不提供线程安全保证，应放主控制线程。

### 11.5 RI 调试工具

compare_ri_coeff_debug.py 只比较实部，主要为 3MGO 数据写成。默认 AO 数 9,8，辅助基数 43,42，不能用于其它材料而不改参数：

~~~bash
python3 tools/compare_ri_coeff_debug.py \
  --case-dir /路径/参考数据 \
  --mine-dir /路径/本地调试数据 \
  --basis-per-atom 9,8 --aux-per-atom 43,42 \
  --max-n-basis-sp 9 --reference-format csc
~~~

ELSI 列打包 col=(basis2−1)*max_n_basis_sp+basis1，basis1 在参考侧可能为原子内编号；工具映射到全局 AO。比较准则 |mine−ref|≤atol+rtol*|ref|。缺失 CSC 显式项在可表示区域按隐式零处理；参考无法表示和参考有而本地无单独统计。

--infer-aux-permutation 按同 k、同 AO 对、同原子的实数值推断辅助基重排；residual 行只是补齐一一对应的猜测。--infer-o-basis-pair-relocation 再诊断目标原子 AO 对位置。推断模式退出 0 不等于映射物理正确，更没有写回或修复生产数据。

<a id="section-12"></a>

## 12. 按修改目标找入口

| 要改什么 | 修改入口 | 必须一起检查 |
|---|---|---|
| 新增参数 | parameter.h/.cpp | 默认值、take、validate、print、test_parameter |
| 加 QP 格式 | bse_files.cpp | QpRecord 单位/占据尺度、k 匹配、核态偏移 |
| 改 W 频率或格式 | read_screened_interaction | Wc 与 W 定义、文件命名、R、单位、上游输出 |
| 加一个通道 | interaction_coefficients、requires_* | 参数合法性、是否需要各核、光学自旋规则 |
| 改 Hartree/screened 核 | molecular_lri.cpp 和实际外部 LibRI | O/V 排列、共轭、q、2/Nk、A†/Bᵀ |
| 改波函数规范 | apply_wavefunction_gauge | 同网格速度协变因子、简并态、重启兼容 |
| 改矩阵布局 | transform_k_2dlocal、ArrayDesc 初始化 | 发送分块、元素累加、ScaLAPACK 1-based 起点 |
| 改振幅分布 | distributed_amplitudes.cpp | spectrum.cpp 的重复分区公式、I/O 格式 |
| 换求解器 | elpa_solver、run_bse | 返回布局、能量排序、full 正能量选择与辛归一化 |
| 改光谱公式 | spectrum.cpp | KS gap、Ry/Ha、Y 项负号、triplet、单位 |
| 增加展宽 | broaden_oscillator_spectrum | 归一化、宽度定义、输入校验、输出头 |
| 调 OpenMP | CMake、外部 LRI 模板、ELPA handle | 编译 flags、MPI FUNNELED、嵌套数学库线程 |
| 支持重启换 MPI 数 | 振幅 I/O 和 restart 分支 | 旧 rank 分区元数据、跨文件重分布 |
| 新上游程序 | fhi_aims_adapter 类似适配层 | reader 文件协议、波函数/RI/速度规范一致 |

具体例子：新增 Gaussian 展宽时，先定义一个解析参数，保持旧 Lorentz 为默认；在 broaden_oscillator_spectrum 选择归一化核；明确宽度是 σ 还是半高半宽；添加固定单态峰值和积分近似测试；通过后在小算例使用 spectrum 重启比较。无需修改核或重新跑 GW。

具体例子：想减小装配通信峰值，可把 transform_k_2dlocal 的 k1_batch_size 参数化，但全部 rank 必须使用一致值。用 test_molecular_lri_comm 的跨两批数据检查批边界，再检查每目标元素多来源累加，而不是仅比较总元素数。

<a id="section-13"></a>

## 13. 测试如何证明修改正确

| 测试源码 / CTest | 主要证明什么 |
|---|---|
| test_parameter / LibBSE_parameter | 空格/等号语法、重复项、路径、选项拒绝 |
| test_velocity_gauge / LibBSE_velocity_gauge | 复速度、TDA/full 的 Y 共轭与负号、零 gap、强度、Lorentz 峰 |
| test_profiler / LibBSE_profiler | 层次、次数、异常析构、表头 |
| test_progress / LibBSE_progress_mpi | 两 rank 只有 root 打 DONE |
| test_spectrum_mpi / LibBSE_spectrum_mpi | rank 1 的非零贡献到 root，局部振幅读写 |
| test_bse_files / LibBSE_bse_files | energy_qp 单位/核窗口、aims 链接、缺速度拒绝、局部 Wc |
| test_aims_mommat_to_velocity.py / LibBSE_aims_mommat_converter | 非对称网格轴、上三角、单位、binary v1 |
| test_molecular_lri_comm / LibBSE_molecular_lri_comm_mpi | 多 rank、多批 k-block 装配和 2/Nk |
| test_elpa_solver / LibBSE_elpa_solver_serial、LibBSE_elpa_solver_mpi | TDA/full 残差、辛规范、重分布、矩阵对称检查 |
| LibBSE_reject_command_line | 命令行参数按预期返回失败 |

CTest 中 spectrum、molecular_lri_comm、ELPA 设置 OMP_NUM_THREADS=2；其实际效果仍受第 9.1 节编译选项影响。旧 tests/README 有两处值得注意：write_wc 的说明被放到 Python 测试段；每态强度的 1/Nk 说法与当前实现不一致。本指南按源码归类。

~~~bash
ctest --test-dir build-dev -R 'LibBSE_(parameter|velocity_gauge)' --output-on-failure
ctest --test-dir build-dev -R 'LibBSE_(elpa_solver|molecular_lri_comm|spectrum)' --output-on-failure
~~~

算法改动需要针对其物理不变量验证：矩阵项看复数共轭和单位；求解器看残差与规范；并行看不同 rank 数下同一数学结果。不要只验证程序“正常结束”。

<a id="section-14"></a>

## 14. 性能、内存和排错

一个 D×D complex<double> 矩阵全局约 16D² 字节（通常双精度复数为 16 字节），局部按 lld*n_loc；增大 nocc 或 nvirt 会平方放大矩阵存储。full 还持有多个 2D×2D 实/复工作数组，峰值不只是 TDA 的两倍。spectrum 振幅约为 nstates*local_pairs 个复数每分量，X/Y 分开。

LibRI 的 Csk_ao_mo 中间量近似按局部 k、原子辅助基数、原子 AO 数、选择 band 数的乘积增长；线程局部 map 和 GEMM 临时量还会增加峰值。外部 LibRI 的 RI_MEMDBG 日志记录预分配估计及 /proc/self/status 的 RSS/HWM，读 libri_rank_N.log 可定位。

| 现象 | 优先查什么 |
|---|---|
| 修改后行为不变 | -I 是否指向另一 LibRI 副本；执行是否用旧 build |
| 缺 Wc，即使只求 rpa | 通用路径无条件读取 Wc |
| 零 KS gap | 光谱用 KS，不是 QP gap；带窗口与占据顺序 |
| B 不对称 | 复数共轭、q 的符号、R 最近胞和 O/V 排列 |
| A 不 Hermitian | 单位、RI/MO 约定、对角与核装配 |
| Cholesky 失败 | M 正定性、W 定义、A/B 检查、符号和输入一致性 |
| MPI 挂住 | 不一致分支、某 rank 提前异常、只有 root 调 collective |
| 多线程无加速 | core 是否有 -fopenmp；ELPA/BLAS 线程；任务规模 |
| restart 元素数量错误 | 原 MPI 数、D、nstates、文件完整性 |
| 光谱整体差 2/Nk 等倍数 | Ha/Ry、强度约定、重复 Nk 归一化 |
| 原子/辅助基输出对不上 | AO 全局/局部映射，CSC padding，辅助基排列 |

可设 LIBBSE_DIAG_B_SYMMETRY=1 打开分量诊断：检查 H_A/W_A 的 Hermiticity、H_B/W_B 的 symmetry；screened B 还按 k1,k2 分块汇总误差并列最大 12 块。V/W 傅里叶互易性诊断只支持单 rank，会比较转置互易与 Hermiticity；多 rank 明确跳过，不应将“未打印异常”当作该项通过。

<a id="section-15"></a>

## 15. 覆盖范围与阅读证据

本指南逐函数覆盖本项目 driver/src/tests/tools 的命名函数，并单列公开结构和内联访问器；匿名 lambda 在所属函数的签名/调用语境中解释。逐文件清单覆盖分析时 Git 管理的 551 项，其中第三方 485 项。构建产物按目录说明，不把每个 .o、CMake 生成文件误当维护源码。

第三方完整通用 API 远大于 LibBSE 使用范围：本指南对实际接入的接口、模板收缩链和并行实现给出详细说明，对其余随库测试/示例/底层模板提供逐文件定位及用途。没有把未进入当前调用链的 GreenX 模块、cereal 容器适配器都称为运行时调用。

源文件位置和函数签名来自本地源码；没有借用网上另一个版本的接口说明。索引中的“调用线索”是源码词法提取，包含构造、容器操作和局部 lambda，不是经过编译器重载解析的完整运行时调用图。修改模板、宏开关或外部库后，应重新核对实际构建路径。
