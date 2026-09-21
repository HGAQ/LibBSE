# LibBSE 外部接口与实现导航

返回[主指南](code_guide_zh.md)和[逐函数参考](code_functions_zh.md)。本页以本机实际外部源码与 build_external_libri 的构建记录为依据；“直接使用”“间接链接”“仅随库提供”分开说明。所有相邻路径链接以本 docs 目录为起点，若单独复制 LibBSE 仓库，需同时提供相邻源码才能打开。

## 1. 依赖总表

| 依赖 | 怎么接入 | LibBSE 实际用途 | 配置/代码证据 |
|---|---|---|---|
| LibRPA | add_subdirectory + rpa_lib | 全局环境、Dataset、文件读取、V(q)→R、矩阵包装 | [顶层 CMake](../CMakeLists.txt)、[公开 reader 头](../../LibRPA/include/librpa_file_reader.hpp) |
| LibRI | 头文件模板 | AO/MO RI 收缩、原子/k 分工、tensor、最近胞 | [LR.h](../../LibRI/LibRI/include/RI/physics/LR.h) |
| LibComm | 经 LibRI/LibRPA 头文件 | 通用 map/tensor 通信设施；具体路径可能关闭通信 | [实际 LibComm 目录](../../LibRPA/thirdparty/LibComm/include/Comm) |
| cereal | 头文件模板 | LibComm/LibRI 复杂容器序列化支持 | [实际 cereal 目录](../../LibRPA/thirdparty/cereal-1.3.0/include/cereal) |
| ELPA | 外部 libelpa_openmp.so | Hermitian TDA、本征向量、full Cholesky/skew | [elpa_solver.cpp](../src/bse/elpa_solver.cpp) |
| ScaLAPACK/BLACS | 经 rpa_lib 链接 | descriptor、进程网格、矩阵重分布/转置/GEMM | [scalapack_connector.h](../../LibRPA/src/math/scalapack_connector.h) |
| BLAS/LAPACK | 当前 OpenBLAS | 局部 GEMM、KS 矩阵 LU/逆、数值库基础 | [lapack_connector.h](../../LibRPA/src/math/lapack_connector.h) |
| MPI | 编译器 wrapper/传递链接 | 所有进程通信及 BLACS | [main.cpp](../driver/main.cpp) |
| OpenMP | libgomp、ELPA 设置、源码 pragma | 本地线程及模板收缩 | 注意主指南第 9.1 节的编译选项缺失 |
| GreenX MiniMax/GXCommon | 相邻 LibRPA 子构建静态库 | LibRPA 的 minimax 网格能力；当前 BSE 主流程没有直接网格调用 | [LibRPA CMake](../../LibRPA/CMakeLists.txt) |
| NumPy | Python import | 动量数组布局、单位和类型转换 | [转换器](../tools/aims_mommat_to_velocity.py) |
| h5py/HDF5 | Python import / h5py 底层 | 读取 mommat.h5；C++ BSE 不直接打开此 HDF5 | 同上 |
| C++/Fortran/系统运行库 | 编译器、数学库链接 | filesystem、复杂数、流、内存、Fortran ABI、quadmath 等 | 实际 link.txt 含 gfortran/quadmath/m/dl/pthread 等 |

ABACUS/FHI-aims 是外部**数据生产程序**，没有被链接为 LibBSE 内部库。ELSI CSC 是调试工具读取的文件格式，本项目没有因此直接链接 ELSI。HDF5 出现在 CMake 缓存中也不等于当前 LibBSE 链接命令包含 HDF5。

LibRPA 的 CMake 还有 CUDA/HIP/LibRI_GPU/LibDDLA/NCCL/RCCL 等可选路径；所检查构建全部 GPU 开关为 OFF，不能宣称当前 LibBSE 已支持这些后端。GreenX 随库的 PAW、解析延拓、局域基等模块也不等于当前主程序调用了它们。

## 2. LibRPA 的生命周期与公开接口

### 2.1 使用顺序

~~~text
MPI_Init_thread
LibRPA_API::initialize
  ├─ librpa::set_output_level(LIBRPA_VERBOSE_INFO)
  └─ librpa::init_global(LIBRPA_SWITCH_OFF)
       └─ librpa_init_global → librpa_init_global_common
            ├─ init_global_mpi
            ├─ init_global_io
            └─ elpa_init(ELPA_API_VERSION)（启用 ELPA）
read_dataset / run_bse
dataset.reset
LibRPA_API::finalize
  └─ librpa::finalize_global → librpa_finalize_global
       ├─ barrier、ELPA uninit
       ├─ finalize_instance_manager
       ├─ finalize_global_io
       └─ finalize_global_mpi
MPI_Finalize
~~~

实现：[src/api/global.cpp](../../LibRPA/src/api/global.cpp)。这里的 LibRPA 初始化不会替代主程序 MPI_Init_thread。最终清理包含 barrier，单 rank 提前进入 finalize 而其它 rank 仍在计算也可能造成等待。

### 2.2 read_dataset 的契约

LibBSE ReaderOptions 包含 input_dir/output_dir/input_format/read_ri/read_band_data。包装层将 aims 数据换为 reader view 后构造 librpa::FileReaderOptions，阈值来自 PARAM.constants，调用：

~~~cpp
std::shared_ptr<librpa_int::Dataset>
librpa::read_dataset_from_files(MPI_Comm comm,
                                const librpa::FileReaderOptions &opts);
~~~

[file_reader.cpp](../../LibRPA/src/api/file_reader.cpp) 的调用顺序：

| 函数/阶段 | 读什么、填什么 | 开发时如何使用 |
|---|---|---|
| normalize_input_dir | 非空目录加尾斜线 | 只处理路径文本 |
| require_input_file | 检查必要普通文件 | 缺少文件尽早报错 |
| driver::h.init | 用传入 comm 建 handler/Dataset | 不另起不相干通信域 |
| read_scf_meanfield | band_out 的维度、占据、KS 能量 | 内部 read_scf_occ_eigenvalues，再 h.set_scf_dimension、h.set_wg_ekb_efermi |
| read_stru | stru_out 的原子/晶格/周期 | 后续 R/q 和最近胞依赖它 |
| read_bz_sampling | 有 bz_sampling_out 时读采样 | 否则 read_bz_sampling_from_stru |
| read_basis_wfc_aux | 原子 AO/辅助基数量 | 验证辅助基原子分块长度等于原子数 |
| generate_atom_pair_from_nat | 生成 Coulomb/RI 读取原子任务 | read_ri 分支使用 |
| read_Vq_row | coulomb_unshrinked_cut 前缀 | 填 pds->vq_cut，应用 Coulomb threshold |
| read_Cs | Cs_data 等 RI 系数 | 填 cs_data，应用 cs_threshold |
| read_eigenvector | 粗 KS 波函数 | use_kpara_scf_eigvec 被设 OFF，粗波函数读取有复制 |
| read_velocity | canonical velocity_matrix | **无条件读取**，包括 IPA/spectrum |
| read_band_kpath_info | band_kpath_info | read_band_data=true 才要求 |
| read_band_meanfield_data | 独立 band 能量/波函数 | 形成 mf_band、kfrac_band_list |
| 无 band 分支 | mf_band=mf，kfrac_band_list=pbc.kfrac_list | 为后续提供统一 BSE 数据入口 |

run_reader 是给异常加“正在读哪个数据”的包装，不是跨 MPI rank 同步异常机制。

Reader 公开头虽然在 include/，仍包含 ../src/api/dataset.h、core/coulmat.h 和数学 connector 头。这是公开 C++ 入口暴露内部数据类型的现实结构，不是可以只复制单个头文件使用的完全独立 ABI。顶层检查 LIBRPA_INCLUDE_DIR 的父目录有 CMakeLists.txt，正是要求完整源码 checkout。

### 2.3 Dataset 里实际用到的字段/方法

| 对象 | LibBSE 使用的成员 | 用途 |
|---|---|---|
| Dataset | comm_h.comm | 全流程 collective 的通信域 |
| Dataset | blacs_h | 构造 A/B/full 的 ArrayDesc |
| Dataset | atoms.size()/coords | 原子数、坐标，LR 分配和最近胞 |
| Dataset | basis_wfc.nb_total/get_atom_nbs/get_i_atom | AO 总数、原子偏移、算符元素所属原子 |
| Dataset | basis_aux.nb_total/get_atom_nbs | 辅助基维度、库仑变换 |
| Dataset | mf / mf_band | 粗/BSE 网格 mean field |
| MeanField | get_n_kpoints/get_n_states/get_n_bands/get_n_aos | 维度校验和选择窗口 |
| MeanField | find_wfc(0,0,ik) | 获取 k 的波函数指针，缺失返回 nullptr |
| MeanField | get_eigenvals()[0] | 光谱的 KS gap |
| MeanField | get_weight().at(0) | KS 核态窗口偏移 |
| Dataset | pbc.kfrac_list/Rlist/latvec_array/period_array | Fourier 网格、周期与最近胞 |
| Dataset | kfrac_band_list | BSE k 顺序与输出 |
| Dataset | velocity_matrix[0][ik][direction] | 上游 MO 速度 |
| Dataset | vq_cut / symmetry_context | FT_Vq 输入 |
| Dataset | cs_data.data_libri / clear | 原始 RI tensor，转换后释放 |

这里硬编码自旋索引 0，速度还明确要求外层 spin 大小为 1；不能仅增加 bse_spin_types 就得到自旋极化或 SOC 支持。singlet/triplet 是核系数和选择规则的通道，不等同于一般多自旋 Dataset 计算。

## 3. LibRPA Fourier 和矩阵包装

### 3.1 FT_Vq

调用位置：[librpa_api.cpp](../src/interface/librpa_api.cpp) build_bare_coulomb。外部实现：[coulmat.cpp](../../LibRPA/src/core/coulmat.cpp#L403)。

参数依次为 comm_h、basis_aux、symmetry_context、vq_cut、pbc、return_ordered_atom_pair=true、use_symmetry_context=false。返回按原子 I/J/R 组织的矩阵指针。

内部遍历 R、原子对、提供的 q 及 map_irk_ks q-star；相位是负号、除 BvK cell 数。未直接提供的 inverse q 使用共轭。所走分支最后取实部，必要时补有序逆原子对。LibBSE 再用二维 RI::Tensor<Complex> 接收。调用后 vq_cut.clear()，是单次消费接口。

### 3.2 普通矩阵

| LibBSE 包装 | 外部调用 | 数学含义/输入契约 |
|---|---|---|
| inverse | LapackConnector::zgetrf、zgetri | 方阵 LU 后求逆，不是伪逆；奇异抛异常 |
| conjugate | librpa_int::conj | 每元素共轭 |
| transpose | librpa_int::transpose(m,false) | 普通转置，不共轭 |
| scale_accumulate | librpa_int::scale_accumulate | target += factor*source |
| ComplexMatrix operator* | LibRPA 矩阵乘法实现 | AO/MO 基变换，底层 BLAS；不是分布式 pgemm |
| ComplexMatrix::create/operator() | 数据分配/索引 | 普通矩阵 nr/nc/c 与 ArrayDesc 缓冲不是同一布局契约 |

### 3.3 分布式矩阵

所有接口都需要属于相容 BLACS 上下文的 descriptor，整数起点为 Fortran 1-based：

| 包装 | connector | 用在何处 |
|---|---|---|
| redistribute(double/Complex) | pgemr2d_f | 从 D×D 临时块复制到 full M 的四角 |
| distributed_transpose | ptranu_f | Bᵀ 检查、B 的 k-block 诊断 |
| distributed_conjugate_transpose | ptranc_f | A† 检查 |
| multiply(double/Complex) | pgemm_f | UJ、UJUᵀ、Uᵀz、Q 变换 |

ScaLAPACK connector 按数值类型路由到相应 p[d/z]gemm、p[d/z]gemr2d、pztranu/pztranc 一类底层例程；改重载时应核对 connector 的实参类型与 descriptor，而非直接手写 Fortran ABI。

ArrayDesc 源码入口在 [LibRPA mpi 目录](../../LibRPA/src/mpi)，BLACS 网格实现为 [base_blacs.cpp](../../LibRPA/src/mpi/base_blacs.cpp#L154)。set_square_grid 从 floor(sqrt(nprocs)) 向下找整除因子，默认 nprows≥npcols；素数进程数得到 nprocs×1。矩阵块大小另由 LibBSE run_bse 选择，不能把“方形进程网格”理解成每进程存方形连续矩阵。

## 4. LibRI 每个接入接口的用法

实际类型：RI::LR<int,int,3,Complex>。实现主要是头文件模板，修改头后需重新编译包含它的 LibBSE 翻译单元。

| 接口 | 传入/返回 | 调用要求与作用 |
|---|---|---|
| LR::init | kpoint vector、nocc、nvirt | 记录分数 k 与窗口；先于并行/收缩 |
| LR::set_parallel | comm、nat、nk、period | 原子/k 四维分工，产生 list_I/list_J/k1/k2 |
| LR::set_Cs | Cs、threshold、set_IJ、all_atoms | 安装 Cs_；实际 set_tensors_map2 的 flag_comm=false |
| LR::set_Vs | V、threshold、set_I、set_J | 安装 Vs_，flag_comm=false |
| LR::set_Ws | W、threshold、set_I、set_J | 安装 Ws_，flag_comm=false |
| LR::cal_Csk_ao_mo | 保存名 Cs_、ofstream | 用 map_psi/k_indices/list_IJ 预变换一个 AO 指标，存 Csk_ao_mo |
| LR::cal_cvc_mo_k_hartree_onthefly | O/V 列表、Vs_、is_A | Hartree q=0 收缩，返回 k1→k2→tensor |
| LR::cal_cvc_mo_k_onthefly | O/V 列表、Ws_、is_A | q_list/q2kpair 指定 screened 收缩 |
| LR::free_Cs/free_Vs/free_Ws | 可选存储名后缀 | 释放数据池对应 map，更新标志 |
| RI::Tensor | shape、operator()/ptr/data | 连续张量，构造时按 shape 分配；注意 shared 数据所有权 |
| Global_Func::convert<Complex> | 实数 tensor | 建复 tensor，保留形状 |
| Cell_Nearest::init | 原子坐标、晶格、BvK period | 初始化最近胞搜索 |
| Cell_Nearest::cell_nearest_direction | I,J,R,distance 引用 | 返回选定等价 R，distance 是附加输出 |
| Array_Operator 的减法/% | 三维数组 | 构造 q=(k2−k1) mod period |

set_Cs/Vs/Ws 的 list 参数名不能作为“内部自动按这个列表通信”的证据：当前外部 LR.h 中原 comm_map2_first 调用被注释，flag_comm 显式 false。LibBSE 必须在进入它之前就让本 rank 拥有所需输入。

### 4.1 数据安装到底做什么

[LRI-set.hpp](../../LibRI/LibRI/include/RI/ri/LRI-set.hpp) set_tensors_map2：

1. 合并默认参数与用户参数；默认 flag_period=true。
2. RI_Tools::cal_period 将等价晶胞按周期归类。
3. flag_comm 若为真才进入 parallel->comm_tensors_map2；当前 LR 安装路径为假。
4. 根据 threshold_filter 和 label 对 tensor 筛选，存入 data_pool。

所以既要检查 LibBSE 的最近胞重映射，也要看安装时周期处理，两者不是同一层。阈值滤掉的数据在后续求和中不会自动找回。

### 4.2 核计算向下追到 GEMM

入口文件：[LRI_k-cal_cvc_mo.hpp](../../LibRI/LibRI/include/RI/ri/LRI_k-cal_cvc_mo.hpp)。

~~~text
LR::cal_Csk_ao_mo
  → LRI_k::cal_Csk_ao_mo
      → R→k Fourier、LRI_Cal_Aux::add_Ds
      → Blas_Interface::gemm

LR::cal_cvc_mo_k_onthefly
  → LRI_k::cal_cvc_mo_k_onthefly
      → Divide_Atoms::traversal_atom_period
      → Global_Func::find、R→q Fourier
      → Cs_ao_mo_to_Cs_mo
          → switch_mo_type
          → Global_Func::get_conj
          → Blas_Interface::gemm（两个端点贡献）
      → Tensor_Multiply::x1x2y1_ax1x2_ay1（C×W）
      → Blas_Interface::gemm（再乘另一个 C 并排成 j,b,i,a）
      → add_Ds_omp_try_map / add_Ds_omp_wait_map（线程结果合并）
      → destroy_lock_result、malloc_trim

LR::cal_cvc_mo_k_hartree_onthefly
  → 先形成 V(q=0)
  → Cs_ao_mo_to_Cs_mo（按 is_A 切共轭）
  → Tensor_Multiply::x1x2y1_ax1x2_ay1
  → Tensor_Multiply::x1x2y0y1_ax1x2_y0y1a
  → 带锁合并局部结果
~~~

| 内部函数 | 怎么理解/怎么用 |
|---|---|
| switch_mo_type | O→起点 0、长度 nocc；V→起点 nocc、长度 nvirt；未知类型报错 |
| Cs_ao_mo_to_Cs_mo | 给两个 k、辅助原子、两个 O/V 范围和左/右共轭标志，构造完整 MO 对 RI 系数 |
| x1x2y1_ax1x2_ay1 | 把共同辅助指标 a 求和，保留两 MO 指标与另一辅助指标；名字就是指标映射 |
| x1x2y0y1_ax1x2_y0y1a | 再消去共同辅助指标，得到两组 MO 对的四维张量 |
| add_Ds | 若目标尚空则初始化，否则把源（可乘系数）相加 |
| init_lock_result | 为结果 map 外层键创建 OpenMP 锁 |
| add_Ds_omp_try_map | 尝试拿锁合并一部分线程局部结果，避免频繁阻塞 |
| add_Ds_omp_wait_map | 阶段结束时等待并清完剩余局部贡献 |
| destroy_lock_result | 在不再使用时销毁锁，不能和其它线程写入并发 |
| malloc_trim(0) | glibc 尝试归还空闲堆页面，不改变数学结果，也不保证 RSS 必然下降 |

cal_Csk_ao_mo 中 read_proc_status_kb/safe_mul 是局部 lambda：前者读 Linux /proc/self/status 做内存诊断，后者避免预估字节乘法溢出。它们不是独立物理接口。

### 4.3 分配链

[Distribute_Equally.hpp](../../LibRI/LibRI/include/RI/distribute/Distribute_Equally.hpp)：

~~~text
distribute_atom_and_k_pair
  → Split_Processes::split_all
      → 递归按 task_sizes 分配子通信域
  → MPI_Wrapper::mpi_get_rank（避免重复子组）
  → Divide_Atoms::divide_atoms（取得各维局部列表）
~~~

LRI 的分配任务空间、BLACS 的矩阵分布、光谱的连续 pair 分区三者不同。transform_k_2dlocal 和 redistribute_amplitudes 就是两座转换桥，改任意一种布局都要检查桥的两端。

## 5. LibComm 和 cereal：不要把 include 当已执行

LibRI 的 comm 目录和通用并行接口会引用 LibComm 的 Comm_Keys/Comm_Trans/Comm_Assemble 及 cereal。复杂 map 通信通常先根据 key 选择接收方，再将容器序列化成字节经 MPI 发送，接收后反序列化和组装。

当前 LibBSE 自己的两种矩阵交换直接使用 MPI_Alltoallv，发送 BlockHead/int/complex 数组，不调用 cereal。实际 LR 的 set_Cs/Vs/Ws 又关闭了 flag_comm。因此不能把 LibComm 的所有 map 交换函数写成当前主路径每次都执行。

若开发新数据分发，先确认需要的是：

- 固定布局的数值缓冲：优先沿已有 MPI 数组协议设计，显式统计 count/displacement。
- 可变长多层容器：可参考 [LibRI comm](../../LibRI/LibRI/include/RI/comm) 与 [LibComm include](../../LibRPA/thirdparty/LibComm/include/Comm)，明确 keys、owner、重复项相加/覆盖规则，再使用对应模板。
- 仅周期归类或阈值筛选：这不需要自动引入一次 MPI map 分发。

cereal 的 binary/json/xml 等 archive 及各容器适配是通用库能力；本仓库保存它们不表示生产 BSE 会写这些格式。当前振幅写 C++ 文本流，速度 binary v1 由 Python struct 写，Wc 由专用文本 reader 读。

## 6. ELPA 每个直接调用

| ELPA 函数 | 谁调用 | 关键契约 |
|---|---|---|
| elpa_init | LibRPA 全局初始化 | 必须成功，采用所用库 ELPA_API_VERSION |
| elpa_allocate | make_elpa_handle | 返回 handle，检查 status |
| elpa_set | make_elpa_handle | na/nev/local_nrows/local_ncols/nblk/mpi_comm_parent/process_row/process_col/solver/omp_threads |
| elpa_setup | make_elpa_handle | 所有尺寸与通信设置完成后执行 |
| elpa_eigenvectors | solve_tda_elpa | 复 Hermitian A，本征向量按原 descriptor 分布 |
| elpa_cholesky | solve_full_elpa | 实正定 M，输出上三角 U |
| elpa_skew_eigenvectors | solve_full_elpa | 实反对称 UJUᵀ，向量实部/虚部存两个平面 |
| elpa_deallocate | 两求解器结束 | 释放 handle，检查 status |
| elpa_uninit | LibRPA finalize | 结束全局 ELPA 环境 |

顶层仅定义 HAVE_SKEWSYMMETRIC 不会替外部 ELPA 编译出缺少的能力。外部安装必须真正提供 skew API；否则可能编译声明存在而链接缺符号，或接口不兼容。更换版本后先跑 test_elpa_solver 的复 B 残差和规范测试。

MPI_Comm_c2f 将 C communicator 转为 ELPA 要求的 Fortran 句柄，这个转换不是简单强制类型转换。solver 固定选 ELPA_SOLVER_2STAGE，当前没有输入项选择 1-stage、GPU kernel 或 autotuning。

## 7. MPI 与 OpenMP 完整直接调用类别

| API | 数据/用途 | 必须注意 |
|---|---|---|
| MPI_Init_thread / MPI_Finalize | 主程序生命周期 | 请求 FUNNELED |
| MPI_Comm_rank / MPI_Comm_size | 分区和 root 判断 | 对正确通信域调用 |
| MPI_Comm_c2f | ELPA handle | 转 Fortran 句柄 |
| MPI_Bcast | 规范参考、相位相关数据、错误、态数/能量 | 所有 rank 次序和 count 一致 |
| MPI_Allreduce | owner MIN、相位 SUM、矩阵范数 SUM、全局布尔检查 | MPI_IN_PLACE 只在支持位置用 |
| MPI_Reduce | 偶极、k 权重 | 非 root 接收缓冲不可当有效结果 |
| MPI_Alltoall | 交换每个目的 rank 的元素/块数量 | counts 用 int |
| MPI_Alltoallv | 块头/矩阵值、振幅索引/值、速度 pair/值 | displacement 单位是 datatype 元素，不是字节 |
| MPI_Gather / MPI_Gatherv | 筛选后的大跃迁贡献 | root 预分配，总 count 校验 |
| MPI_Barrier | aims reader view 完成 | 不在 done 中 |
| MPI_Type_create_struct / MPI_Type_commit | BlockHead 类型 | 位移来自 offsetof，避免 padding 假设 |
| omp_get_max_threads | 日志、ELPA omp_threads | 只报告运行库最大值 |
| omp_get_num_procs / omp_get_dynamic | 日志 | 不设置 CPU 绑定 |
| OpenMP parallel/for/barrier/master | ELPA 前后局部操作、外部 LibRI 收缩 | 需编译时启用；MPI 仍留主线程 |

具体每处行号见逐函数参考。当前源码没有为 MPI 通信提供通用分块到 64-bit count 的协议，大矩阵消息存在 int 上限；已有检查不能代替所有尺度下的溢出审计。不要把 max RSS 只按 D²/M 估算。

## 8. 其它运行与构建依赖

GreenX 在当前 LibRPA 配置中始终提供 minimax 能力（可选外部目标）；rpa_lib 私有链接 LibGXMiniMax，最终 LibBSE 链接记录出现其静态库及 GXCommon。但本 BSE 流程读取已经生成的 ifreq_0，不直接生成频率网格。需要改变频率采样时通常先修改上游 LibRPA 输入与 Wc 输出，再改 LibBSE reader 选择规则。

NumPy 用于 reshape/transpose/triu_indices/共轭/类型转换；h5py.File 读 HDF5 三个数据集。转换器还用 Python 标准库 argparse、pathlib、struct、tempfile、os、sys。RI debug 工具只使用 Python 标准库 argparse/math/re/struct/collections/dataclasses/pathlib。

examples/run.sh 使用 Bash、mpirun、文件复制/符号链接、test 检查；FHI-aims 例子还设置 ulimit -s unlimited。NPROCS 默认 4，OMP_NUM_THREADS 默认 1。Python 解释器是顶层测试依赖，NumPy/h5py 是否安装需要单独满足；find_package(Python3) 成功不代表二者可 import。

## 9. 外部升级时的实际核对步骤

1. 读实际 CMakeCache/flags/link，确认修改的是被使用的源码/库。
2. 先对比公开头签名，再对比存储约定和返回所有权，尤其 ComplexMatrix 与 RI::Tensor。
3. 检查 LR 的 set_parallel、flag_comm、flag_period 是否变化；它会影响本 rank 所需数据。
4. 检查 is_A 的共轭/排列、Fourier 正负号和返回 CVC shape，不能只保证编译通过。
5. 对 ELPA 检查 skew 布局和正谱排序，做 full 残差/辛规范。
6. 对 MPI/OpenMP 检查编译选项、主线程通信约束及 1/多 rank 一致性。
7. 记录外部源码版本；LibBSE 自身 git commit 不能唯一确定外部模板或共享库实现。
