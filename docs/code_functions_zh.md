# LibBSE 逐函数参考

返回[主指南](code_guide_zh.md)。按实际文件列出命名函数定义，包括私有/匿名 namespace 助手、模板、operator() 重载及测试。签名保留引用/const 与 Python 注解，默认值以头文件为准。

调用线索由函数体词法提取，包含容器操作、构造及局部 lambda，不解析模板实例、重载或动态分派；相同短名可能是不同函数。MPI/OpenMP 位置单列。公开调用须看对应头文件，匿名助手通过所属公共流程使用。

共识别 218 个命名函数定义/重载，覆盖 38 个本项目 C++/Python 源码与头文件。声明不重复计数；=delete 方法不是可调用函数。

源码链接标注 #L 行号；不支持行锚点的本地阅读器可按标注行号打开。

## 文件目录

- [driver/main.cpp](#file-driver-main-cpp)
- [src/bse/bse_calculation.cpp](#file-src-bse-bse-calculation-cpp)
- [src/bse/bse_calculation.h](#file-src-bse-bse-calculation-h)
- [src/bse/bse_types.h](#file-src-bse-bse-types-h)
- [src/bse/distributed_amplitudes.cpp](#file-src-bse-distributed-amplitudes-cpp)
- [src/bse/distributed_amplitudes.h](#file-src-bse-distributed-amplitudes-h)
- [src/bse/elpa_solver.cpp](#file-src-bse-elpa-solver-cpp)
- [src/bse/elpa_solver.h](#file-src-bse-elpa-solver-h)
- [src/bse/matrix_checks.cpp](#file-src-bse-matrix-checks-cpp)
- [src/bse/matrix_checks.h](#file-src-bse-matrix-checks-h)
- [src/bse/molecular_lri.cpp](#file-src-bse-molecular-lri-cpp)
- [src/bse/molecular_lri.h](#file-src-bse-molecular-lri-h)
- [src/bse/molecular_lri_comm.h](#file-src-bse-molecular-lri-comm-h)
- [src/bse/spectrum.cpp](#file-src-bse-spectrum-cpp)
- [src/bse/spectrum.h](#file-src-bse-spectrum-h)
- [src/interface/librpa_api.cpp](#file-src-interface-librpa-api-cpp)
- [src/interface/librpa_api.h](#file-src-interface-librpa-api-h)
- [src/io/bse_files.cpp](#file-src-io-bse-files-cpp)
- [src/io/bse_files.h](#file-src-io-bse-files-h)
- [src/io/fhi_aims_adapter.cpp](#file-src-io-fhi-aims-adapter-cpp)
- [src/io/fhi_aims_adapter.h](#file-src-io-fhi-aims-adapter-h)
- [src/parameter/parameter.cpp](#file-src-parameter-parameter-cpp)
- [src/parameter/parameter.h](#file-src-parameter-parameter-h)
- [src/utils/profiler.cpp](#file-src-utils-profiler-cpp)
- [src/utils/profiler.h](#file-src-utils-profiler-h)
- [src/utils/progress.cpp](#file-src-utils-progress-cpp)
- [src/utils/progress.h](#file-src-utils-progress-h)
- [tests/test_aims_mommat_to_velocity.py](#file-tests-test-aims-mommat-to-velocity-py)
- [tests/test_bse_files.cpp](#file-tests-test-bse-files-cpp)
- [tests/test_elpa_solver.cpp](#file-tests-test-elpa-solver-cpp)
- [tests/test_molecular_lri_comm.cpp](#file-tests-test-molecular-lri-comm-cpp)
- [tests/test_parameter.cpp](#file-tests-test-parameter-cpp)
- [tests/test_profiler.cpp](#file-tests-test-profiler-cpp)
- [tests/test_progress.cpp](#file-tests-test-progress-cpp)
- [tests/test_spectrum_mpi.cpp](#file-tests-test-spectrum-mpi-cpp)
- [tests/test_velocity_gauge.cpp](#file-tests-test-velocity-gauge-cpp)
- [tools/aims_mommat_to_velocity.py](#file-tools-aims-mommat-to-velocity-py)
- [tools/compare_ri_coeff_debug.py](#file-tools-compare-ri-coeff-debug-py)

<a id="file-driver-main-cpp"></a>
## driver/main.cpp

[打开源码](../driver/main.cpp)。程序唯一生产入口，负责 MPI/LibRPA 生命周期、参数和 run_bse。所有进程执行，root 只限制打印。

### mpi_thread_level_name（19—29 行）

[实现位置](../driver/main.cpp#L19)

~~~cpp
const char *mpi_thread_level_name(int level)
~~~

把 MPI 线程等级整数转成日志名称；未知值返回 unknown。无数据变换。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。

### print_parallel_configuration（31—43 行）

[实现位置](../driver/main.cpp#L31)

~~~cpp
void print_parallel_configuration(int mpi_size, int mpi_thread_level)
~~~

接收进程数及 MPI 实际提供等级，打印 OpenMP 最大线程、可用处理器与 dynamic 状态。调用者限制 root 输出；不设置线程。

调用线索：mpi_thread_level_name、omp_get_dynamic、omp_get_max_threads、omp_get_num_procs。

显式并行语句（被调函数还可能继续通信）：

- [第 39 行](../driver/main.cpp#L39)：<< omp_get_max_threads() << '\n'
- [第 40 行](../driver/main.cpp#L40)：<< "  OpenMP available processors: " << omp_get_num_procs() << '\n'
- [第 42 行](../driver/main.cpp#L42)：<< (omp_get_dynamic() ? "enabled" : "disabled") << '\n';

### main（47—131 行）

[实现位置](../driver/main.cpp#L47)

~~~cpp
int main(int argc, char **argv)
~~~

初始化 MPI FUNNELED，读 ./libbse.in，初始化 LibRPA、读 Dataset、run_bse，清理后返回 0/1；拒绝额外 CLI 参数。

调用线索：LibRPA_API::finalize、LibRPA_API::initialize、LibRPA_API::read_dataset、MPI_Comm_rank、MPI_Comm_size、MPI_Finalize、MPI_Init_thread、dataset->atoms.size、dataset->cs_data.n_keys、dataset->mf.get_n_kpoints、dataset->mf_band.get_n_kpoints、dataset->mf_band.get_n_states、dataset->vq_cut.size、dataset.reset、error.what、libbse::PARAM.inp.ipa_only、libbse::PARAM.inp.spectrum_only、libbse::PARAM.print、libbse::PARAM.read、libbse::done、libbse::global::profiler.display、libbse::global::profiler.start、libbse::global::profiler.stop、libbse::resolve_input_format、libbse::run_bse、print_parallel_configuration、std::invalid_argument、std::runtime_error、timer。

显式并行语句（被调函数还可能继续通信）：

- [第 50 行](../driver/main.cpp#L50)：MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
- [第 53 行](../driver/main.cpp#L53)：MPI_Comm_rank(MPI_COMM_WORLD, &rank);
- [第 54 行](../driver/main.cpp#L54)：MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
- [第 129 行](../driver/main.cpp#L129)：MPI_Finalize();

源码错误消息片段：

- "MPI does not provide MPI_THREAD_FUNNELED"
- "LibBSE takes no command-line parameters; configure ./libbse.in"


<a id="file-src-bse-bse-calculation-cpp"></a>
## src/bse/bse_calculation.cpp

[打开源码](../src/bse/bse_calculation.cpp)。BSE 工作流、分支、通道组合、文件名与诊断。匿名 namespace 助手只在本翻译单元使用。

### write_energies（33—40 行）

[实现位置](../src/bse/bse_calculation.cpp#L33)

~~~cpp
void write_energies(const fs::path &file, const std::vector<double> &energies)
~~~

将能量向量原样写文本；本项目传入 Ry，科学计数 8 位小数。调用者保证只由 root 写。

调用线索：file.string、output、std::runtime_error、std::setprecision。

源码错误消息片段：

- "cannot write "

### read_energies（42—52 行）

[实现位置](../src/bse/bse_calculation.cpp#L42)

~~~cpp
std::vector<double> read_energies(const fs::path &file, int nstates)
~~~

从指定文件读取 nstates 个能量，短文件报错；不转换单位，不检查额外尾部数据。重启由 root 调。

调用线索：file.string、input、std::runtime_error、std::vector。

源码错误消息片段：

- "cannot read "
- "truncated excitation-energy file "

### add_qp_diagonal（54—78 行）

[实现位置](../src/bse/bse_calculation.cpp#L54)

~~~cpp
void add_qp_diagonal(std::vector<Complex> &matrix, const librpa_int::ArrayDesc &descriptor, const QuasiparticleBands &qp, const InputParameters &options)
~~~

按 p=k*nocc*nvirt+i*nvirt+a 找本 rank 的对角位置，赋 QP 空带减占据带能量。是赋值，必须在相互作用累加前使用。

调用线索：descriptor.indx_g2l_c、descriptor.indx_g2l_r、descriptor.lld。

### energy_file（80—86 行）

[实现位置](../src/bse/bse_calculation.cpp#L80)

~~~cpp
fs::path energy_file(const InputParameters &options, const std::string &spin_type, bool full)
~~~

根据 output_dir、spin_type、full 布尔值构造激发能路径，不执行 I/O。

调用线索：fs::path、std::string。

### amplitude_file（88—96 行）

[实现位置](../src/bse/bse_calculation.cpp#L88)

~~~cpp
fs::path amplitude_file(const InputParameters &options, const std::string &spin_type, const std::string &component, int rank)
~~~

按输出目录、通道、分量前缀和 rank 构造振幅路径；TDA 分量为空，full 用 full_X_/full_Y_。

调用线索：fs::path、std::to_string。

### assemble_channel_matrix（98—122 行）

[实现位置](../src/bse/bse_calculation.cpp#L98)

~~~cpp
void assemble_channel_matrix( std::vector<Complex> &matrix, const librpa_int::ArrayDesc &descriptor, const QuasiparticleBands &qp, const InputParameters &options, const InteractionCoefficients &coefficients, const std::vector<Complex> &hartree, const std::vector<Complex> &screened)
~~~

重置本地 A，写 QP 对角，按 InteractionCoefficients 累加公共 H/W 核。需要使用的核尺寸必须匹配；不组 B。

调用线索：add_qp_diagonal、descriptor.lld、descriptor.n_loc、hartree.size、matrix.assign、matrix.size、screened.size、std::runtime_error。

源码错误消息片段：

- "missing Hartree matrix for BSE channel"
- "missing screened matrix for BSE channel"

### solve_ipa_tda（130—169 行）

[实现位置](../src/bse/bse_calculation.cpp#L130)

~~~cpp
IpaSolution solve_ipa_tda(const QuasiparticleBands &qp, const InputParameters &options, int nstates, MPI_Comm comm)
~~~

按 QP 跃迁能差排序，返回最低 nstates 能量及分布式单位基振幅；不分配 D×D 矩阵，不调用 ELPA。

调用线索：Complex、indices.begin、indices.end、make_distributed_amplitudes、result.amplitudes、result.energies.resize、std::iota、std::sort、std::vector。

### report_matrix_check（179—188 行）

[实现位置](../src/bse/bse_calculation.cpp#L179)

~~~cpp
void report_matrix_check(const MatrixCheckResult &result, const std::string &matrix_name, double threshold, int rank)
~~~

root 打印矩阵 PASS/WARNING；不会因为 passed=false 抛异常或停止求解。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。

### b_symmetry_diagnostics_enabled（190—194 行）

[实现位置](../src/bse/bse_calculation.cpp#L190)

~~~cpp
bool b_symmetry_diagnostics_enabled()
~~~

读取 LIBBSE_DIAG_B_SYMMETRY；非空且首字符不是 0 就开启。各 rank 应看到一致环境。

调用线索：std::getenv。

### fourier_interaction_block（203—235 行）

[实现位置](../src/bse/bse_calculation.cpp#L203)

~~~cpp
FourierInteractionBlock fourier_interaction_block( const TensorMap<Complex> &tensors, int iat, int jat, const std::array<double, 3> &q)
~~~

为指定原子对求 Σ_R tensor(R)*exp(+i2πqR)，返回行主序诊断块；检查各 R 的形状一致。

调用线索：result.values.assign、result.values.empty、std::polar、std::runtime_error、tensor、tensor.shape.size、tensors.end、tensors.find。

源码错误消息片段：

- "inconsistent interaction tensor dimensions in diagnostic"

### diagnose_interaction_reciprocity（237—320 行）

[实现位置](../src/bse/bse_calculation.cpp#L237)

~~~cpp
void diagnose_interaction_reciprocity( const TensorMap<Complex> &tensors, const librpa_int::Dataset &dataset, const char *name, int rank)
~~~

诊断开关启用且单 rank 时，在固定十个 q 上比较转置互易与 Hermiticity 的范数比；多 rank 跳过。

调用线索：MPI_Comm_size、b_symmetry_diagnostics_enabled、dataset.atoms.size、forward.values.empty、fourier_interaction_block、iat、jat、ratio、reverse_minus.values.empty、reverse_same.values.empty、std::conj、std::norm、std::runtime_error、std::sqrt。

显式并行语句（被调函数还可能继续通信）：

- [第 243 行](../src/bse/bse_calculation.cpp#L243)：MPI_Comm_size(dataset.comm_h.comm, &mpi_size);

源码错误消息片段：

- "reciprocal interaction tensor dimensions differ"

### diagnose_b_component（322—334 行）

[实现位置](../src/bse/bse_calculation.cpp#L322)

~~~cpp
void diagnose_b_component(const std::vector<Complex> &matrix, const librpa_int::ArrayDesc &descriptor, const char *name, int rank)
~~~

开关启用且矩阵非空时调 check_symmetric，检查一个 B 核分量；需要所有 rank 同步参与。

调用线索：b_symmetry_diagnostics_enabled、check_symmetric、matrix.empty。

### diagnose_a_component（336—348 行）

[实现位置](../src/bse/bse_calculation.cpp#L336)

~~~cpp
void diagnose_a_component(const std::vector<Complex> &matrix, const librpa_int::ArrayDesc &descriptor, const char *name, int rank)
~~~

开关启用且矩阵非空时调 check_hermitian，检查一个 A 核分量；需要所有 rank 同步参与。

调用线索：b_symmetry_diagnostics_enabled、check_hermitian、matrix.empty。

### diagnose_b_k_blocks（350—459 行）

[实现位置](../src/bse/bse_calculation.cpp#L350)

~~~cpp
void diagnose_b_k_blocks(const std::vector<Complex> &matrix, const librpa_int::ArrayDesc &descriptor, int pair_dimension, const librpa_int::Dataset &dataset, int rank)
~~~

把 B−Bᵀ 的平方差按 k1/k2 归类并 Allreduce，root 区分同 k/异 k 误差，列最差 12 块。用于排查 screened B 的 q/相位问题。

调用线索：LibRPA_API::distributed_transpose、MPI_Allreduce、b_symmetry_diagnostics_enabled、dataset.kfrac_band_list.at、dataset.mf_band.get_n_kpoints、descriptor.comm、descriptor.indx_l2g_c、descriptor.indx_l2g_r、descriptor.lld、descriptor.m、descriptor.m_loc、descriptor.n、descriptor.n_loc、difference_squared.data、difference_squared.size、matrix.data、matrix.empty、matrix.size、order.begin、order.end、order.size、ratio、std::floor、std::iota、std::min、std::norm、std::sort、std::sqrt、std::vector、sum_squared.data、sum_squared.size、transposed、transposed.data、wrap。

显式并行语句（被调函数还可能继续通信）：

- [第 385 行](../src/bse/bse_calculation.cpp#L385)：MPI_Allreduce(MPI_IN_PLACE, difference_squared.data(),
- [第 388 行](../src/bse/bse_calculation.cpp#L388)：MPI_Allreduce(MPI_IN_PLACE, sum_squared.data(),

### run_bse（463—995 行）

[实现位置](../src/bse/bse_calculation.cpp#L463)

~~~cpp
void run_bse(const InputParameters &options, const std::shared_ptr<librpa_int::Dataset> &dataset)
~~~

总编排入口：QP、规范、分支、核收缩、矩阵求解、振幅输出与光谱。所有 Dataset 通信域成员调用；会消费/清空 Dataset 的 vq_cut 与 cs_data。完整步骤见主指南第 6 节。

调用线索：LibRPA_API::build_bare_coulomb、MPI_Comm_rank、amplitude_file、apply_wavefunction_gauge、assemble_channel_matrix、channel.energies.front、check_hermitian、check_symmetric、convert_lri_coefficients、dataset->pbc.Rlist.size、descriptor、descriptor.init、descriptor.lld、descriptor.n_loc、diagnose_a_component、diagnose_b_component、diagnose_b_k_blocks、diagnose_interaction_reciprocity、done、energy_file、fs::create_directories、full_descriptor、full_descriptor.init、full_results.push_back、hartree_a.assign、hartree_b.assign、interaction_coefficients、matrix_b、molecular、molecular.add_hartree_a、molecular.add_hartree_b、molecular.add_screened_a、molecular.add_screened_b、molecular.initialize、molecular.local_i_atoms、molecular.local_j_atoms、molecular.release_interactions、options.ipa_only、options.requires_hartree、options.requires_screened、options.solve_full、options.solve_tda、options.spectrum_only、prepare_fine_velocity_mo、read_distributed_amplitudes、read_energies、read_qp_bands、read_screened_interaction、redistribute_amplitudes、remap_to_nearest_bvk_cell、report_matrix_check、run_timer、screened_a.assign、screened_b.assign、solve_full_elpa、solve_ipa_tda、solve_tda_elpa、std::invalid_argument、std::move、std::runtime_error、std::setprecision、tda_results.push_back、timer、write_distributed_amplitudes、write_energies、write_velocity_gauge_outputs。

显式并行语句（被调函数还可能继续通信）：

- [第 468 行](../src/bse/bse_calculation.cpp#L468)：MPI_Comm_rank(dataset->comm_h.comm, &rank);

源码错误消息片段：

- "number of requested states exceeds the BSE dimension"
- "failed to initialize BSE BLACS descriptor"
- "failed to initialize full-BSE BLACS descriptor"


<a id="file-src-bse-bse-calculation-h"></a>
## src/bse/bse_calculation.h

[打开源码](../src/bse/bse_calculation.h)。公共声明头，实现见同名 .cpp（若有）；通过此头使用接口，不 include .cpp。

本头没有独立函数体，以下给出声明/结构定义以查参数默认值、类型与访问权限；行为见实现文件及主指南。

~~~cpp
#pragma once

#include "bse_types.h"

#include <memory>

namespace librpa_int
{
class Dataset;
}

namespace libbse
{

void run_bse(const InputParameters &options,
             const std::shared_ptr<librpa_int::Dataset> &dataset);

} // namespace libbse
~~~


<a id="file-src-bse-bse-types-h"></a>
## src/bse/bse_types.h

[打开源码](../src/bse/bse_types.h)。公共类型与振幅访问器，不负责通信；布局和单位见主指南第 5 节。

### operator()（52—56 行）

[实现位置](../src/bse/bse_types.h#L52)

~~~cpp
Complex &operator()(int state, int local_pair)
~~~

访问本 rank 振幅 values[state*local_pairs+local_pair]；两个重载分别返回可写和只读引用。不做范围检查。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。

### operator()（58—62 行）

[实现位置](../src/bse/bse_types.h#L58)

~~~cpp
const Complex &operator()(int state, int local_pair) const
~~~

访问本 rank 振幅 values[state*local_pairs+local_pair]；两个重载分别返回可写和只读引用。不做范围检查。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。


<a id="file-src-bse-distributed-amplitudes-cpp"></a>
## src/bse/distributed_amplitudes.cpp

[打开源码](../src/bse/distributed_amplitudes.cpp)。二维块循环→连续 pair 分布，以及各 rank 文本振幅 I/O。offset 为 0-based。

### pair_partition（24—30 行）

[实现位置](../src/bse/distributed_amplitudes.cpp#L24)

~~~cpp
PairPartition pair_partition(int dimension, int mpi_size, int rank)
~~~

用商和余数把 D 个 pair 尽量平均分成连续区间，返回指定 rank 的 first/count。振幅与速度文件中的同名实现须保持一致。

调用线索：std::min。

### pair_owner（32—41 行）

[实现位置](../src/bse/distributed_amplitudes.cpp#L32)

~~~cpp
int pair_owner(int pair, int dimension, int mpi_size)
~~~

给定全局 pair，反求连续区间 owner；余数优先分给前面的 rank。调用前确保 pair 在 [0,D)。

调用线索：std::logic_error。

源码错误消息片段：

- "invalid owner for distributed amplitude pair"

### validate_shape（43—53 行）

[实现位置](../src/bse/distributed_amplitudes.cpp#L43)

~~~cpp
void validate_shape(const DistributedAmplitudes &amplitudes)
~~~

检查 DistributedAmplitudes 的正维度、正态数、本地范围和 values 大小；用于振幅输出前保护。

调用线索：amplitudes.dimension、std::invalid_argument。

源码错误消息片段：

- "invalid distributed excitation-amplitude shape"

### make_distributed_amplitudes（57—77 行）

[实现位置](../src/bse/distributed_amplitudes.cpp#L57)

~~~cpp
DistributedAmplitudes make_distributed_amplitudes( MPI_Comm comm, int dimension, int nstates)
~~~

给定 comm、D、nstates，查询 rank 数，分配本地 pair 区间并清零；返回的振幅按 state-major 存储。

调用线索：MPI_Comm_rank、MPI_Comm_size、pair_partition、result.values.assign、std::invalid_argument。

显式并行语句（被调函数还可能继续通信）：

- [第 65 行](../src/bse/distributed_amplitudes.cpp#L65)：MPI_Comm_rank(comm, &rank);
- [第 66 行](../src/bse/distributed_amplitudes.cpp#L66)：MPI_Comm_size(comm, &mpi_size);

源码错误消息片段：

- "distributed excitation-amplitude dimensions must be positive"

### redistribute_amplitudes（79—199 行）

[实现位置](../src/bse/distributed_amplitudes.cpp#L79)

~~~cpp
DistributedAmplitudes redistribute_amplitudes( MPI_Comm comm, const std::vector<Complex> &source, const librpa_int::ArrayDesc &source_descriptor, int row_offset, int column_offset, int dimension, int nstates)
~~~

把 ArrayDesc 所描述矩阵的指定子块转为连续 pair 分布；offset 为 0-based。所有 rank 调；检查大小、MPI count、接收数量及唯一 ownership。

调用线索：MPI_Alltoall、MPI_Alltoallv、MPI_Comm_size、assigned、begin、end、local_index、make_distributed_amplitudes、max、pair_owner、pair_partition、push_back、receive_counts.begin、receive_counts.data、receive_counts.end、receive_indices.data、receive_offsets.data、receive_values.data、result.values.size、row_offset、send_counts.begin、send_counts.data、send_counts.end、send_indices.begin、send_indices.data、send_offsets.data、send_values.begin、send_values.data、size、source_descriptor.indx_l2g_c、source_descriptor.indx_l2g_r、source_descriptor.lld、source_descriptor.m_loc、source_descriptor.n_loc、std::accumulate、std::copy、std::invalid_argument、std::overflow_error、std::runtime_error、std::vector。

显式并行语句（被调函数还可能继续通信）：

- [第 96 行](../src/bse/distributed_amplitudes.cpp#L96)：MPI_Comm_size(comm, &mpi_size);
- [第 141 行](../src/bse/distributed_amplitudes.cpp#L141)：MPI_Alltoall(send_counts.data(), 1, MPI_INT,
- [第 178 行](../src/bse/distributed_amplitudes.cpp#L178)：MPI_Alltoallv(send_indices.data(), send_counts.data(), send_offsets.data(),
- [第 181 行](../src/bse/distributed_amplitudes.cpp#L181)：MPI_Alltoallv(send_values.data(), send_counts.data(), send_offsets.data(),

源码错误消息片段：

- "invalid block-cyclic eigenvector block for redistribution"
- "distributed amplitude message exceeds the MPI count limit"
- "distributed amplitude exchange exceeds the MPI count limit"
- "incomplete distributed eigenvector redistribution"
- "invalid distributed eigenvector ownership"

### write_distributed_amplitudes（201—222 行）

[实现位置](../src/bse/distributed_amplitudes.cpp#L201)

~~~cpp
void write_distributed_amplitudes( const std::filesystem::path &file, const DistributedAmplitudes &amplitudes)
~~~

写单 rank 文件，每态一行、每列本地 pair；绝对值≤1e-10 置零。调用方必须提供互不冲突的 rank 文件名。

调用线索：amplitudes、file.string、output、std::abs、std::runtime_error、std::setprecision、validate_shape。

源码错误消息片段：

- "cannot write "

### read_distributed_amplitudes（224—242 行）

[实现位置](../src/bse/distributed_amplitudes.cpp#L224)

~~~cpp
DistributedAmplitudes read_distributed_amplitudes( const std::filesystem::path &file, MPI_Comm comm, int dimension, int nstates)
~~~

先按当前 MPI 分区分配，再顺序读复数；多/少元素均报错。文件不含元数据，因此还需调用方保证原 MPI 数与窗口/规范一致。

调用线索：file.string、input、make_distributed_amplitudes、std::runtime_error。

源码错误消息片段：

- "cannot read "
- "truncated or MPI-incompatible excitation-amplitude file "
- "oversized or MPI-incompatible excitation-amplitude file "


<a id="file-src-bse-distributed-amplitudes-h"></a>
## src/bse/distributed_amplitudes.h

[打开源码](../src/bse/distributed_amplitudes.h)。公共声明头，实现见同名 .cpp（若有）；通过此头使用接口，不 include .cpp。

本头没有独立函数体，以下给出声明/结构定义以查参数默认值、类型与访问权限；行为见实现文件及主指南。

~~~cpp
#pragma once

#include "bse_types.h"

#include <librpa_file_reader.hpp>

#include <mpi.h>

#include <filesystem>
#include <vector>

namespace libbse
{

DistributedAmplitudes make_distributed_amplitudes(
    MPI_Comm comm, int dimension, int nstates);

// Redistribute a block of a two-dimensional block-cyclic matrix into the
// pair-block layout used by spectrum analysis and amplitude files. Offsets are
// zero based. No rank receives the complete eigenvector matrix.
DistributedAmplitudes redistribute_amplitudes(
    MPI_Comm comm,
    const std::vector<Complex> &source,
    const librpa_int::ArrayDesc &source_descriptor,
    int row_offset, int column_offset,
    int dimension, int nstates);

void write_distributed_amplitudes(
    const std::filesystem::path &file,
    const DistributedAmplitudes &amplitudes);

DistributedAmplitudes read_distributed_amplitudes(
    const std::filesystem::path &file,
    MPI_Comm comm, int dimension, int nstates);

} // namespace libbse
~~~


<a id="file-src-bse-elpa-solver-cpp"></a>
## src/bse/elpa_solver.cpp

[打开源码](../src/bse/elpa_solver.cpp)。TDA/full 求解器。输入为 descriptor 所描述的本地列主序缓冲，所有 rank 调。公式见主指南 7.5。

### check_elpa（16—21 行）

[实现位置](../src/bse/elpa_solver.cpp#L16)

~~~cpp
void check_elpa(int status, const char *operation)
~~~

把非 ELPA_OK 状态转成包含操作名和错误码的异常；不做 MPI 异常同步。

调用线索：std::runtime_error、std::string、std::to_string。

### make_elpa_handle（23—50 行）

[实现位置](../src/bse/elpa_solver.cpp#L23)

~~~cpp
elpa_t make_elpa_handle(const librpa_int::ArrayDesc &descriptor, int nev)
~~~

按 ArrayDesc 设置 na、nev、本地行列、nblk、父通信域、进程坐标；选两阶段 ELPA，线程数为 omp_get_max_threads，再 setup。

调用线索：MPI_Comm_c2f、check_elpa、descriptor.comm、descriptor.m、descriptor.m_loc、descriptor.mb、descriptor.mypcol、descriptor.myprow、descriptor.n_loc、elpa_allocate、elpa_set、elpa_setup、omp_get_max_threads。

显式并行语句（被调函数还可能继续通信）：

- [第 38 行](../src/bse/elpa_solver.cpp#L38)：elpa_set(handle, "mpi_comm_parent", MPI_Comm_c2f(descriptor.comm()), &status);
- [第 46 行](../src/bse/elpa_solver.cpp#L46)：elpa_set(handle, "omp_threads", omp_get_max_threads(), &status);

### fill_pair_block（52—69 行）

[实现位置](../src/bse/elpa_solver.cpp#L52)

~~~cpp
template <typename Function> void fill_pair_block(const std::vector<Complex> &matrix_a, const std::vector<Complex> &matrix_b, const librpa_int::ArrayDesc &pair_descriptor, std::vector<double> &temporary, Function &&function)
~~~

模板局部循环，对 A/B 每对本地元素应用传入 lambda，写 double 临时块；full 的四种实/虚组合使用它。OpenMP static 按列分工。

调用线索：function、pair_descriptor.lld、pair_descriptor.m_loc、pair_descriptor.n_loc。

显式并行语句（被调函数还可能继续通信）：

- [第 59 行](../src/bse/elpa_solver.cpp#L59)：#pragma omp parallel for schedule(static)

### copy_pair_block（71—81 行）

[实现位置](../src/bse/elpa_solver.cpp#L71)

~~~cpp
void copy_pair_block(const std::vector<double> &source, const librpa_int::ArrayDesc &pair_descriptor, std::vector<double> &target, const librpa_int::ArrayDesc &full_descriptor, int target_row, int target_column)
~~~

通过 LibRPA_API::redistribute 将 D×D 实临时块放进 2D×2D 矩阵；目标 row/column 使用 ScaLAPACK 1-based 坐标。

调用线索：LibRPA_API::redistribute、pair_descriptor.m、pair_descriptor.n、source.data、target.data。

### solve_tda_elpa（85—110 行）

[实现位置](../src/bse/elpa_solver.cpp#L85)

~~~cpp
EigenSolution solve_tda_elpa(std::vector<Complex> &matrix, librpa_int::ArrayDesc &descriptor, int nstates)
~~~

解 Hermitian A，负 nstates 解释为 D；返回前 nstates 能量和 descriptor 对应的本地向量。输入 A 是可覆盖工作区；所有 rank 调。

调用线索：check_elpa、descriptor.lld、descriptor.m、descriptor.n、descriptor.n_loc、elpa_deallocate、elpa_eigenvectors、make_elpa_handle、matrix.data、result.energies_ry.data、result.energies_ry.resize、result.vectors_local.data、result.vectors_local.resize、std::invalid_argument。

源码错误消息片段：

- "TDA matrix must be square"
- "invalid number of requested TDA states"

### solve_full_elpa（112—280 行）

[实现位置](../src/bse/elpa_solver.cpp#L112)

~~~cpp
EigenSolution solve_full_elpa(std::vector<Complex> &matrix_a, std::vector<Complex> &matrix_b, const librpa_int::ArrayDesc &pair_descriptor, librpa_int::ArrayDesc &full_descriptor, int nstates)
~~~

把 A/B 化为实 M，经 Cholesky、skew 求解、Q 变换恢复 X/Y；返回正能量子集及 full 本地向量。清空 A/B；要求 M 正定、A†=A、Bᵀ=B。见主指南公式。

调用线索：Complex、LibRPA_API::multiply、a.imag、a.real、all_energies.begin、all_energies.data、b.imag、b.real、check_elpa、copy_pair_block、elpa_cholesky、elpa_deallocate、elpa_skew_eigenvectors、fill_pair_block、full_descriptor.indx_l2g_c、full_descriptor.indx_l2g_r、full_descriptor.lld、full_descriptor.m、full_descriptor.m_loc、full_descriptor.n、full_descriptor.n_loc、lz、lz.data、lz_imag、lz_imag.clear、lz_imag.data、lz_imag.shrink_to_fit、lz_real、lz_real.clear、lz_real.data、lz_real.shrink_to_fit、make_elpa_handle、matrix_a.clear、matrix_a.shrink_to_fit、matrix_a.size、matrix_b.clear、matrix_b.shrink_to_fit、matrix_b.size、matrix_j、matrix_j.clear、matrix_j.data、matrix_j.shrink_to_fit、matrix_m、matrix_m.clear、matrix_m.data、matrix_m.shrink_to_fit、pair_descriptor.lld、pair_descriptor.m、pair_descriptor.n、pair_descriptor.n_loc、q、q.data、result.energies_ry.assign、result.vectors_local.data、result.vectors_local.resize、skew_vectors、skew_vectors.clear、skew_vectors.data、skew_vectors.shrink_to_fit、std::abs、std::invalid_argument、std::sqrt、std::vector、temporary、temporary.clear、temporary.shrink_to_fit、uj、uj.clear、uj.data、uj.shrink_to_fit。

显式并行语句（被调函数还可能继续通信）：

- [第 164 行](../src/bse/elpa_solver.cpp#L164)：#pragma omp parallel for schedule(static)
- [第 205 行](../src/bse/elpa_solver.cpp#L205)：#pragma omp parallel for schedule(static)
- [第 237 行](../src/bse/elpa_solver.cpp#L237)：#pragma omp parallel for schedule(static)
- [第 248 行](../src/bse/elpa_solver.cpp#L248)：#pragma omp parallel for schedule(static)

源码错误消息片段：

- "full-BSE A/B matrices must be square"
- "invalid number of requested full-BSE states"
- "invalid full-BSE matrix descriptor"
- "invalid distributed A/B matrix storage"


<a id="file-src-bse-elpa-solver-h"></a>
## src/bse/elpa_solver.h

[打开源码](../src/bse/elpa_solver.h)。公共声明头，实现见同名 .cpp（若有）；通过此头使用接口，不 include .cpp。

本头没有独立函数体，以下给出声明/结构定义以查参数默认值、类型与访问权限；行为见实现文件及主指南。

~~~cpp
#pragma once

#include "bse_types.h"

#include <librpa_file_reader.hpp>

#include <vector>

namespace libbse
{

EigenSolution solve_tda_elpa(std::vector<Complex> &matrix,
                             librpa_int::ArrayDesc &descriptor,
                             int nstates);

EigenSolution solve_full_elpa(std::vector<Complex> &matrix_a,
                              std::vector<Complex> &matrix_b,
                              const librpa_int::ArrayDesc &pair_descriptor,
                              librpa_int::ArrayDesc &full_descriptor,
                              int nstates);

} // namespace libbse
~~~


<a id="file-src-bse-matrix-checks-cpp"></a>
## src/bse/matrix_checks.cpp

[打开源码](../src/bse/matrix_checks.cpp)。分布式复矩阵的普通/共轭转置比较，检查失败不会自动阻止求解。

### check_matrix（18—96 行）

[实现位置](../src/bse/matrix_checks.cpp#L18)

~~~cpp
MatrixCheckResult check_matrix(const std::vector<Complex> &matrix, const librpa_int::ArrayDesc &descriptor, double threshold, bool conjugate)
~~~

内部统一实现：选择普通/共轭分布式转置，Allreduce 范数与逐元素通过标志。检查方阵、存储和非负阈值，不修改原矩阵。

调用线索：LibRPA_API::distributed_conjugate_transpose、LibRPA_API::distributed_transpose、MPI_Allreduce、MPI_Comm_rank、descriptor.comm、descriptor.lld、descriptor.m、descriptor.m_loc、descriptor.n、descriptor.n_loc、infinity、matrix.data、matrix.size、max、std::abs、std::invalid_argument、std::norm、std::sqrt、transposed、transposed.data。

显式并行语句（被调函数还可能继续通信）：

- [第 32 行](../src/bse/matrix_checks.cpp#L32)：MPI_Comm_rank(descriptor.comm(), &rank);
- [第 64 行](../src/bse/matrix_checks.cpp#L64)：MPI_Allreduce(MPI_IN_PLACE, norms, 2, MPI_DOUBLE, MPI_SUM,
- [第 68 行](../src/bse/matrix_checks.cpp#L68)：MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LAND,

源码错误消息片段：

- "matrix symmetry check requires a square matrix"
- "invalid distributed matrix storage for symmetry check"
- "matrix symmetry threshold must be nonnegative"

### check_hermitian（100—105 行）

[实现位置](../src/bse/matrix_checks.cpp#L100)

~~~cpp
MatrixCheckResult check_hermitian( const std::vector<Complex> &matrix, const librpa_int::ArrayDesc &descriptor, double threshold)
~~~

对 A 检查 M−M†；返回 MatrixCheckResult。所有 descriptor 通信域成员参与，不是 root-only。

调用线索：check_matrix。

### check_symmetric（107—112 行）

[实现位置](../src/bse/matrix_checks.cpp#L107)

~~~cpp
MatrixCheckResult check_symmetric( const std::vector<Complex> &matrix, const librpa_int::ArrayDesc &descriptor, double threshold)
~~~

对 B 检查 M−Mᵀ，不做共轭。返回通过标志和 Frobenius 比值；所有 rank 参与。

调用线索：check_matrix。


<a id="file-src-bse-matrix-checks-h"></a>
## src/bse/matrix_checks.h

[打开源码](../src/bse/matrix_checks.h)。公共声明头，实现见同名 .cpp（若有）；通过此头使用接口，不 include .cpp。

本头没有独立函数体，以下给出声明/结构定义以查参数默认值、类型与访问权限；行为见实现文件及主指南。

~~~cpp
#pragma once

#include "bse_types.h"

#include <librpa_file_reader.hpp>

#include <vector>

namespace libbse
{

struct MatrixCheckResult
{
    bool passed = false;
    double difference_norm = 0.0;
    double sum_norm = 0.0;
    double relative_error = 0.0;
};

MatrixCheckResult check_hermitian(
    const std::vector<Complex> &matrix,
    const librpa_int::ArrayDesc &descriptor, double threshold);

MatrixCheckResult check_symmetric(
    const std::vector<Complex> &matrix,
    const librpa_int::ArrayDesc &descriptor, double threshold);

} // namespace libbse
~~~


<a id="file-src-bse-molecular-lri-cpp"></a>
## src/bse/molecular_lri.cpp

[打开源码](../src/bse/molecular_lri.cpp)。适配外部 RI::LR、规范与 q 映射，装配 LibRI k-block；会消费输入交互数据。

### block_head_mpi_type（30—48 行）

[实现位置](../src/bse/molecular_lri.cpp#L30)

~~~cpp
MPI_Datatype block_head_mpi_type()
~~~

首次构造并 commit BlockHead 对应的 MPI struct 类型，按 offsetof 描述四个 int 字段；静态缓存供块交换复用。

调用线索：MPI_Type_commit、MPI_Type_create_struct、offsetof。

显式并行语句（被调函数还可能继续通信）：

- [第 43 行](../src/bse/molecular_lri.cpp#L43)：MPI_Type_create_struct(4, block_lengths, displacements, types,
- [第 45 行](../src/bse/molecular_lri.cpp#L45)：MPI_Type_commit(&datatype);

### checked_total（50—59 行）

[实现位置](../src/bse/molecular_lri.cpp#L50)

~~~cpp
int checked_total(const std::vector<int> &counts, const char *description)
~~~

以 long long 汇总 int counts，超过 MPI int 上限抛 overflow_error；用于装配缓冲大小。

调用线索：counts.begin、counts.end、max、std::accumulate、std::overflow_error、std::string。

### apply_wavefunction_gauge（63—148 行）

[实现位置](../src/bse/molecular_lri.cpp#L63)

~~~cpp
std::vector<Complex> apply_wavefunction_gauge( librpa_int::Dataset &dataset, const InputParameters &options, const QuasiparticleBands &qp)
~~~

native 返回全 1；first_k 按首 k 同带 AO 系数重叠改选中 KS 行，广播参考并汇总相位表。会原地修改 mf_band；返回的相位供同网格速度协变使用。

调用线索：Complex、MPI_Allreduce、MPI_Bcast、MPI_Comm_rank、MPI_Comm_size、band_phase、dataset.mf_band.find_wfc、dataset.mf_band.get_n_aos、dataset.mf_band.get_n_kpoints、local_owners.data、owner_phases、owner_phases.data、owners.begin、owners.data、owners.end、phases.data、phases.size、reference.data、reference.size、std::abs、std::conj、std::find、std::runtime_error、std::vector。

显式并行语句（被调函数还可能继续通信）：

- [第 80 行](../src/bse/molecular_lri.cpp#L80)：MPI_Comm_rank(dataset.comm_h.comm, &rank);
- [第 81 行](../src/bse/molecular_lri.cpp#L81)：MPI_Comm_size(dataset.comm_h.comm, &mpi_size);
- [第 88 行](../src/bse/molecular_lri.cpp#L88)：MPI_Allreduce(local_owners.data(), owners.data(), nk, MPI_INT, MPI_MIN,
- [第 109 行](../src/bse/molecular_lri.cpp#L109)：MPI_Bcast(reference.data(), static_cast<int>(reference.size()),
- [第 144 行](../src/bse/molecular_lri.cpp#L144)：MPI_Allreduce(owner_phases.data(), phases.data(),

源码错误消息片段：

- "a fine-grid wavefunction is absent on every MPI rank"
- "reference wavefunction dimensions do not cover BSE bands"
- "fine-grid wavefunction dimensions do not cover BSE bands"
- "cannot align a BSE wavefunction with the k=0 phase reference"

### MolecularLri::MolecularLri（150—173 行）

[实现位置](../src/bse/molecular_lri.cpp#L150)

~~~cpp
MolecularLri::MolecularLri(librpa_int::Dataset &dataset, const InputParameters &options, const QuasiparticleBands &qp) : dataset_(dataset), options_(options), qp_(qp)
~~~

保存 Dataset/options/qp 引用，建立每 rank 日志，初始化 LR 带窗口与原子/k 并行，然后建 q 映射。引用对象必须比此对象活得久。

调用线索：MPI_Comm_rank、build_exact_q_map、dataset_.atoms.size、dataset_.kfrac_band_list.at、dataset_.mf_band.get_n_kpoints、fs::path、log_.open、lr_.init、lr_.set_parallel、std::move、std::runtime_error、std::to_string、std::vector。

显式并行语句（被调函数还可能继续通信）：

- [第 156 行](../src/bse/molecular_lri.cpp#L156)：MPI_Comm_rank(dataset_.comm_h.comm, &rank);

源码错误消息片段：

- "cannot create LibRI log file"

### MolecularLri::initialize（175—196 行）

[实现位置](../src/bse/molecular_lri.cpp#L175)

~~~cpp
void MolecularLri::initialize(TensorMap<Complex> &Cs_in, TensorMap<Complex> &Vs_in, TensorMap<Complex> &Ws_in)
~~~

将 Cs/V/W 安装进 LR 并清空输入 map，准备局部波函数、计算 Csk_ao_mo，释放原始 Cs。调用后不要再使用传入 map 的旧数据。

调用线索：Cs_in.clear、Vs_in.clear、Ws_in.clear、all_atoms.insert、build_wavefunctions、dataset_.atoms.size、lr_.cal_Csk_ao_mo、lr_.free_Cs、lr_.list_I.begin、lr_.list_I.end、lr_.list_IJ.begin、lr_.list_IJ.end、lr_.list_J.begin、lr_.list_J.end、lr_.set_Cs、lr_.set_Vs、lr_.set_Ws、set_i、set_ij、set_j。

### MolecularLri::build_exact_q_map（198—212 行）

[实现位置](../src/bse/molecular_lri.cpp#L198)

~~~cpp
void MolecularLri::build_exact_q_map()
~~~

对本地 k1/k2 取 (k2−k1) mod 1，将 pair 放入 q2kpair，再形成 q_list。使用浮点数组作键，没有额外容差聚类。

调用线索：emplace_back、lr_.q_list.push_back。

### MolecularLri::build_wavefunctions（214—243 行）

[实现位置](../src/bse/molecular_lri.cpp#L214)

~~~cpp
void MolecularLri::build_wavefunctions()
~~~

用每原子 AO 数累计偏移，把 mf_band 的选中带（含 ncore 偏移）复制到 map_psi[k][atom]。只为 LR k_indices 操作，缺波函数或形状不够报错。

调用线索：atom_sizes.size、dataset_.basis_wfc.get_atom_nbs、dataset_.mf_band.find_wfc、offsets、offsets.back、std::move、std::runtime_error、std::to_string、tensor。

源码错误消息片段：

- "fine-grid wavefunction is missing at k-point "
- "fine-grid wavefunction dimensions do not cover BSE bands"

### transform_k_2dlocal（245—475 行）

[实现位置](../src/bse/molecular_lri.cpp#L245)

~~~cpp
void transform_k_2dlocal( std::vector<Complex> &matrix, const KMatrixMap &blocks, const librpa_int::ArrayDesc &descriptor, int nk, int pair_dimension, double coefficient)
~~~

按 k1 每 64 点批量，将 LibRI P×P k-block 切到 BLACS 块边界，通过 Alltoall/Alltoallv 发送，只在目标 rank 累加。统一乘 coefficient*2/Nk。

调用线索：MPI_Alltoall、MPI_Alltoallv、MPI_Comm_rank、MPI_Comm_size、block_head_mpi_type、checked_total、descriptor.comm、descriptor.g2p_c、descriptor.g2p_r、descriptor.get_pnum、descriptor.indx_g2l_c、descriptor.indx_g2l_r、descriptor.lld、descriptor.mb、descriptor.n_loc、descriptor.nb、matrix.size、max、owner、receive_head_counts.data、receive_head_offsets.data、receive_heads.data、receive_value_counts.data、receive_value_offsets.data、receive_values.data、send_head_counts.begin、send_head_counts.data、send_head_counts.end、send_head_offsets.data、send_heads.data、send_value_counts.begin、send_value_counts.data、send_value_counts.end、send_value_offsets.data、send_values.data、std::fill、std::invalid_argument、std::min、std::overflow_error、std::runtime_error、std::vector、tensor.shape.get_shape_all。

显式并行语句（被调函数还可能继续通信）：

- [第 257 行](../src/bse/molecular_lri.cpp#L257)：MPI_Comm_rank(descriptor.comm(), &rank);
- [第 258 行](../src/bse/molecular_lri.cpp#L258)：MPI_Comm_size(descriptor.comm(), &mpi_size);
- [第 335 行](../src/bse/molecular_lri.cpp#L335)：MPI_Alltoall(send_head_counts.data(), 1, MPI_INT,
- [第 338 行](../src/bse/molecular_lri.cpp#L338)：MPI_Alltoall(send_value_counts.data(), 1, MPI_INT,
- [第 442 行](../src/bse/molecular_lri.cpp#L442)：MPI_Alltoallv(send_heads.data(), send_head_counts.data(),
- [第 447 行](../src/bse/molecular_lri.cpp#L447)：MPI_Alltoallv(send_values.data(), send_value_counts.data(),

源码错误消息片段：

- "invalid local BSE matrix storage"
- "LibRI returned a BSE block with unexpected dimensions"
- "BSE matrix block message exceeds the MPI count limit"
- "inconsistent BSE matrix send-buffer packing"
- "received a BSE matrix block on the wrong MPI rank"
- "inconsistent BSE matrix block communication"

### MolecularLri::add_hartree_a（477—485 行）

[实现位置](../src/bse/molecular_lri.cpp#L477)

~~~cpp
void MolecularLri::add_hartree_a(std::vector<Complex> &matrix, const librpa_int::ArrayDesc &descriptor, double coefficient)
~~~

调用 LR Hartree on-the-fly，O,V,O,V 与 is_A=true；收缩后调用 transform_k_2dlocal 累加到输入矩阵。调用方传零初始化/可累加缓冲。

调用线索：lr_.cal_cvc_mo_k_hartree_onthefly、transform_k_2dlocal。

### MolecularLri::add_hartree_b（487—495 行）

[实现位置](../src/bse/molecular_lri.cpp#L487)

~~~cpp
void MolecularLri::add_hartree_b(std::vector<Complex> &matrix, const librpa_int::ArrayDesc &descriptor, double coefficient)
~~~

调用 LR Hartree on-the-fly，O,V,O,V 与 is_A=false；区别在第二 MO 对共轭方式，随后分布式累加。

调用线索：lr_.cal_cvc_mo_k_hartree_onthefly、transform_k_2dlocal。

### MolecularLri::add_screened_a（497—505 行）

[实现位置](../src/bse/molecular_lri.cpp#L497)

~~~cpp
void MolecularLri::add_screened_a(std::vector<Complex> &matrix, const librpa_int::ArrayDesc &descriptor, double coefficient)
~~~

调用 LR screened on-the-fly，O,O,V,V 与 is_A=true，用 Ws_；这里只加正的收缩结果，通道负号在后续组合。

调用线索：lr_.cal_cvc_mo_k_onthefly、transform_k_2dlocal。

### MolecularLri::add_screened_b（507—515 行）

[实现位置](../src/bse/molecular_lri.cpp#L507)

~~~cpp
void MolecularLri::add_screened_b(std::vector<Complex> &matrix, const librpa_int::ArrayDesc &descriptor, double coefficient)
~~~

调用 LR screened on-the-fly，V,O,O,V 与 is_A=false，用 Ws_；带顺序及共轭不同于 A。

调用线索：lr_.cal_cvc_mo_k_onthefly、transform_k_2dlocal。

### MolecularLri::release_interactions（517—521 行）

[实现位置](../src/bse/molecular_lri.cpp#L517)

~~~cpp
void MolecularLri::release_interactions()
~~~

释放 LR 数据池里的 Vs_/Ws_；不等于立刻释放 MolecularLri 所有波函数和 MO 中间量。

调用线索：lr_.free_Vs、lr_.free_Ws。


<a id="file-src-bse-molecular-lri-h"></a>
## src/bse/molecular_lri.h

[打开源码](../src/bse/molecular_lri.h)。MolecularLri 声明及局部原子访问器；类保存 Dataset/options/qp 引用，生命周期须保证。

### local_i_atoms（34—34 行）

[实现位置](../src/bse/molecular_lri.h#L34)

~~~cpp
const std::vector<int> &local_i_atoms() const
~~~

返回 LR 本 rank list_I 的 const 引用，供 Wc 读取确定第一原子范围；不要保存到超出 MolecularLri 生命周期。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。

### local_j_atoms（35—35 行）

[实现位置](../src/bse/molecular_lri.h#L35)

~~~cpp
const std::vector<int> &local_j_atoms() const
~~~

返回 LR 本 rank list_J 的 const 引用，供 Wc 第二原子范围。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。


<a id="file-src-bse-molecular-lri-comm-h"></a>
## src/bse/molecular_lri_comm.h

[打开源码](../src/bse/molecular_lri_comm.h)。公共声明头，实现见同名 .cpp（若有）；通过此头使用接口，不 include .cpp。

本头没有独立函数体，以下给出声明/结构定义以查参数默认值、类型与访问权限；行为见实现文件及主指南。

~~~cpp
#pragma once

#include "bse_types.h"

#include <librpa_file_reader.hpp>

#include <RI/global/Tensor.h>

#include <map>
#include <vector>

namespace libbse
{

using KMatrixMap = std::map<int, std::map<int, RI::Tensor<Complex>>>;

// Each LibRI k block is sent only to the ranks that
// own its destination blocks; no process materializes the dense global matrix.
void transform_k_2dlocal(
    std::vector<Complex> &matrix,
    const KMatrixMap &blocks,
    const librpa_int::ArrayDesc &descriptor,
    int nk, int pair_dimension, double coefficient);

} // namespace libbse
~~~


<a id="file-src-bse-spectrum-cpp"></a>
## src/bse/spectrum.cpp

[打开源码](../src/bse/spectrum.cpp)。速度准备、KS-gap 偶极、强度、展宽和分析输出。区分 serial 全局输入与 MPI 局部输入。

### phase（28—34 行）

[实现位置](../src/bse/spectrum.cpp#L28)

~~~cpp
Complex phase(const librpa_int::Vector3_Order<double> &k, const librpa_int::Vector3_Order<int> &r, double sign)
~~~

计算 exp(i*sign*2π*k·R)，sign=-1 用于 k→R，+1 用于 R→k；纯标量计算。

调用线索：Complex、std::exp。

### pair_partition（42—48 行）

[实现位置](../src/bse/spectrum.cpp#L42)

~~~cpp
PairPartition pair_partition(int dimension, int mpi_size, int rank)
~~~

用商和余数把 D 个 pair 尽量平均分成连续区间，返回指定 rank 的 first/count。振幅与速度文件中的同名实现须保持一致。

调用线索：std::min。

### pair_owner（50—59 行）

[实现位置](../src/bse/spectrum.cpp#L50)

~~~cpp
int pair_owner(int pair, int dimension, int mpi_size)
~~~

给定全局 pair，反求连续区间 owner；余数优先分给前面的 rank。调用前确保 pair 在 [0,D)。

调用线索：std::logic_error。

源码错误消息片段：

- "invalid owner for velocity-matrix pair"

### velocity_index（61—64 行）

[实现位置](../src/bse/spectrum.cpp#L61)

~~~cpp
std::size_t velocity_index(int direction, int local_pair, int local_pairs)
~~~

返回 direction*local_pairs+local_pair；速度为 direction-major，不能使用振幅的 state-major 索引直接替代。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。

### mean_squared（66—70 行）

[实现位置](../src/bse/spectrum.cpp#L66)

~~~cpp
double mean_squared(const std::array<Complex, 3> &dipole)
~~~

返回三个复偶极模平方的算术平均，不是先平均复偶极再平方。

调用线索：std::norm。

### write_dipoles（72—96 行）

[实现位置](../src/bse/spectrum.cpp#L72)

~~~cpp
void write_dipoles(const fs::path &file, const std::vector<double> &energies, const std::vector<std::array<Complex, 3>> &dipoles, const std::vector<double> &mean_squared_dipoles)
~~~

写 root 偶极表，能量从 Ry 转 eV，包含复偶极、方向模平方与平均；调用者负责 root 限制。

调用线索：energies.size、file.string、output、std::norm、std::runtime_error、std::setprecision、std::setw。

源码错误消息片段：

- "cannot write "

### write_oscillator_strengths（98—114 行）

[实现位置](../src/bse/spectrum.cpp#L98)

~~~cpp
void write_oscillator_strengths( const fs::path &file, const std::vector<OscillatorStrength> &strengths)
~~~

写每态 energy_eV、fx/fy/fz/fiso，标注 ABACUS Ry 约定，不再进行归一化。

调用线索：file.string、output、std::runtime_error、std::setprecision、strengths.size。

源码错误消息片段：

- "cannot write "

### write_spectrum（116—131 行）

[实现位置](../src/bse/spectrum.cpp#L116)

~~~cpp
void write_spectrum(const fs::path &file, const std::vector<SpectrumPoint> &spectrum, double broadening_ev)
~~~

写展宽密度及半宽注释，输入 SpectrumPoint 已有 eV 能量；自身不执行展宽。

调用线索：file.string、output、std::runtime_error、std::setprecision。

源码错误消息片段：

- "cannot write "

### validate_velocity_mo（133—149 行）

[实现位置](../src/bse/spectrum.cpp#L133)

~~~cpp
void validate_velocity_mo(const InputParameters &options, const FineVelocityMo &velocity_mo)
~~~

检查窗口带数、本地范围、3*local_pairs 大小和 gaps_ha 大小；零 KS gap 报错。

调用线索：std::abs、std::invalid_argument、std::runtime_error、velocity_mo.first_pair、velocity_mo.gaps_ha.size。

源码错误消息片段：

- "invalid fine-grid velocity_mo shape"
- "zero KS gap in velocity-gauge spectrum"

### validate_velocity_inputs（151—165 行）

[实现位置](../src/bse/spectrum.cpp#L151)

~~~cpp
void validate_velocity_inputs( int nstates, const InputParameters &options, const FineVelocityMo &velocity_mo, const std::vector<Complex> &amplitudes_x, const std::vector<Complex> *amplitudes_y)
~~~

串行接口形状校验：振幅必须覆盖 nstates*D，速度必须完整覆盖所有 pair，Y 若存在必须同大小。

调用线索：amplitudes_x.size、amplitudes_y->size、nstates、std::invalid_argument、validate_velocity_mo。

源码错误消息片段：

- "invalid excitation amplitudes for velocity gauge"

### validate_distributed_velocity_inputs（167—203 行）

[实现位置](../src/bse/spectrum.cpp#L167)

~~~cpp
void validate_distributed_velocity_inputs( MPI_Comm comm, const InputParameters &options, const FineVelocityMo &velocity_mo, const DistributedAmplitudes &amplitudes_x, const DistributedAmplitudes *amplitudes_y)
~~~

根据 comm 的 rank 数重新计算预期连续区间，检查 X/Y 和速度的 ownership 与态数；只允许协议一致的局部块。

调用线索：MPI_Comm_rank、MPI_Comm_size、amplitudes.values.size、shape_matches、std::invalid_argument、std::min、validate_velocity_mo。

显式并行语句（被调函数还可能继续通信）：

- [第 175 行](../src/bse/spectrum.cpp#L175)：MPI_Comm_rank(comm, &rank);
- [第 176 行](../src/bse/spectrum.cpp#L176)：MPI_Comm_size(comm, &mpi_size);

源码错误消息片段：

- "invalid distributed excitation amplitudes for velocity gauge"
- "velocity_mo and excitation-amplitude ownership do not match"

### transition_dipole_slice（205—237 行）

[实现位置](../src/bse/spectrum.cpp#L205)

~~~cpp
std::array<Complex, 3> transition_dipole_slice( int state, int first_pair, int pair_stride, const InputParameters &options, const FineVelocityMo &velocity_mo, const std::vector<Complex> &amplitudes_x, const std::vector<Complex> *amplitudes_y)
~~~

对完整振幅从 first_pair 起以 pair_stride 遍历，求 i√2 Σ(vX−conj(v)Y)/gap。生产 MPI 路径用另一个局部函数；此函数服务串行接口。

调用线索：prefactor、std::abs、std::conj、std::invalid_argument、std::runtime_error、std::sqrt、velocity_index。

源码错误消息片段：

- "serial velocity_mo does not cover all excitation pairs"
- "zero KS gap in velocity-gauge spectrum"

### distributed_transition_dipole（239—266 行）

[实现位置](../src/bse/spectrum.cpp#L239)

~~~cpp
std::array<Complex, 3> distributed_transition_dipole( int state, const InputParameters &options, const FineVelocityMo &velocity_mo, const DistributedAmplitudes &amplitudes_x, const DistributedAmplitudes *amplitudes_y)
~~~

求某态本地连续 pair 的偶极贡献，不通信；X/Y 用本地访问器，返回三个复数，供上层 Reduce。

调用线索：amplitudes_x、prefactor、std::conj、std::sqrt、velocity_index。

### broadcast_vector（268—277 行）

[实现位置](../src/bse/spectrum.cpp#L268)

~~~cpp
template <typename T> void broadcast_vector(std::vector<T> &values, MPI_Datatype datatype, int count, MPI_Comm comm)
~~~

模板助手：检查 count，resize 各 rank vector，再以 rank 0 为源 MPI_Bcast。datatype 必须与 T 匹配。

调用线索：MPI_Bcast、count、std::overflow_error、values.data、values.max_size、values.resize。

显式并行语句（被调函数还可能继续通信）：

- [第 276 行](../src/bse/spectrum.cpp#L276)：MPI_Bcast(values.data(), count, datatype, 0, comm);

源码错误消息片段：

- "invalid MPI broadcast vector size"

### calculate_oscillator_strengths（281—313 行）

[实现位置](../src/bse/spectrum.cpp#L281)

~~~cpp
std::vector<OscillatorStrength> calculate_oscillator_strengths( const std::vector<double> &energies_ry, const std::vector<std::array<Complex, 3>> &dipoles, int kpoint_count)
~~~

输入 Ry 能量与复偶极，返回每态 2*Ω_Ry*|d|² 及方向平均；拒绝负/非有限激发能。kpoint_count 只做正数检查，内部不除 Nk。

调用线索：dipoles.size、energies_ry.size、result、std::invalid_argument、std::isfinite、std::norm。

源码错误消息片段：

- "invalid excitation energies, dipoles, or k-point count"
- "oscillator strength requires non-negative excitation energies"

### broaden_oscillator_spectrum（315—358 行）

[实现位置](../src/bse/spectrum.cpp#L315)

~~~cpp
std::vector<SpectrumPoint> broaden_oscillator_spectrum( const InputParameters &options, const std::vector<OscillatorStrength> &strengths)
~~~

按输入 eV 网格叠加归一化 Lorentz 核；负最大值自动确定上界；返回方向密度。限制跨度/步长不超过 1e7，γ和步长必须正。

调用线索：result、std::floor、std::invalid_argument、std::max、std::overflow_error。

源码错误消息片段：

- "invalid optical-spectrum grid"
- "optical-spectrum grid is too large"

### velocity_gauge_transition_dipole（360—373 行）

[实现位置](../src/bse/spectrum.cpp#L360)

~~~cpp
std::array<Complex, 3> velocity_gauge_transition_dipole( int state, const InputParameters &options, const FineVelocityMo &velocity_mo, const std::vector<Complex> &amplitudes_x, const std::vector<Complex> *amplitudes_y)
~~~

串行单态 API；校验完整全局数组，调 transition_dipole_slice。不要把 MPI 局部振幅传到此接口。

调用线索：amplitudes_x.size、state、std::invalid_argument、transition_dipole_slice、validate_velocity_inputs。

源码错误消息片段：

- "invalid excitation state for velocity gauge"

### velocity_gauge_transition_dipoles_mpi（375—397 行）

[实现位置](../src/bse/spectrum.cpp#L375)

~~~cpp
std::vector<std::array<Complex, 3>> velocity_gauge_transition_dipoles_mpi( MPI_Comm comm, const InputParameters &options, const FineVelocityMo &velocity_mo, const DistributedAmplitudes &amplitudes_x, const DistributedAmplitudes *amplitudes_y)
~~~

校验局部布局，对每态做本地 contraction，以 MPI_Reduce 将 3*nstates 复偶极送 root。非 root 返回空 vector。

调用线索：MPI_Comm_rank、MPI_Reduce、distributed_transition_dipole、local.data、result.data、result.resize、std::vector、validate_distributed_velocity_inputs。

显式并行语句（被调函数还可能继续通信）：

- [第 384 行](../src/bse/spectrum.cpp#L384)：MPI_Comm_rank(comm, &rank);
- [第 394 行](../src/bse/spectrum.cpp#L394)：MPI_Reduce(local.data(), rank == 0 ? result.data() : nullptr,

### prepare_fine_velocity_mo（399—708 行）

[实现位置](../src/bse/spectrum.cpp#L399)

~~~cpp
FineVelocityMo prepare_fine_velocity_mo( const InputParameters &options, const QuasiparticleBands &qp, const std::shared_ptr<librpa_int::Dataset> &dataset, const std::vector<Complex> &band_gauge_phases)
~~~

生成与振幅相同 pair ownership 的紧凑速度和 KS gap。同网格直接选元素并改规范；异网格逆 MO→AO、k→R→细 k、AO→MO，再 Alltoallv。会分配复制的粗 AO 中间矩阵。

调用线索：LibRPA_API::conjugate、LibRPA_API::inverse、LibRPA_API::scale_accumulate、LibRPA_API::transpose、MPI_Allreduce、MPI_Alltoall、MPI_Alltoallv、MPI_Comm_rank、MPI_Comm_size、band_gauge_phases.size、create、dataset->atoms.size、dataset->basis_wfc.get_i_atom、dataset->mf.find_wfc、dataset->mf.get_n_aos、dataset->mf.get_n_kpoints、dataset->mf.get_n_states、dataset->mf_band.find_wfc、dataset->mf_band.get_eigenvals、dataset->mf_band.get_n_kpoints、dataset->mf_band.get_n_states、dataset->pbc.Rlist.size、dataset->velocity_matrix.size、eigenvalues、iat、jat、local_sources.data、matrix、max、nearest.cell_nearest_direction、nearest.init、pair_owner、pair_partition、phase、r、receive_pair_counts.begin、receive_pair_counts.data、receive_pair_counts.end、receive_pair_offsets.data、receive_pairs.data、receive_value_counts.data、receive_value_offsets.data、receive_values.data、result.gaps_ha.resize、result.values.assign、send_pair_counts.begin、send_pair_counts.data、send_pair_counts.end、send_pair_offsets.data、send_pairs.data、send_value_counts.data、send_value_offsets.data、send_values.data、size、sources.data、std::abs、std::accumulate、std::conj、std::invalid_argument、std::overflow_error、std::runtime_error、std::vector、velocity_ao、velocity_ao_k.clear、velocity_ao_r、velocity_index、velocity_mo。

显式并行语句（被调函数还可能继续通信）：

- [第 406 行](../src/bse/spectrum.cpp#L406)：MPI_Comm_rank(dataset->comm_h.comm, &rank);
- [第 407 行](../src/bse/spectrum.cpp#L407)：MPI_Comm_size(dataset->comm_h.comm, &mpi_size);
- [第 571 行](../src/bse/spectrum.cpp#L571)：MPI_Allreduce(local_sources.data(), sources.data(), fine_nk, MPI_INT,
- [第 587 行](../src/bse/spectrum.cpp#L587)：MPI_Alltoall(send_pair_counts.data(), 1, MPI_INT,
- [第 669 行](../src/bse/spectrum.cpp#L669)：MPI_Alltoallv(send_pairs.data(), send_pair_counts.data(),
- [第 685 行](../src/bse/spectrum.cpp#L685)：MPI_Alltoallv(send_values.data(), send_value_counts.data(),

源码错误消息片段：

- "coarse-grid velocity matrix was not loaded"
- "invalid fine-grid band-gauge phase table"
- "fine-grid eigenvalues do not contain the selected BSE bands"
- "velocity_mo does not contain the selected BSE bands"
- "velocity interpolation requires a complete square coarse KS basis"
- "coarse k/R grids are incompatible for velocity interpolation"
- "coarse-grid KS wavefunction is missing or incomplete"
- "fine-grid KS wavefunction is absent on every MPI rank"
- "incomplete distributed velocity_mo ownership"
- "distributed velocity_mo exchange exceeds the MPI count limit"
- "fine-grid KS wavefunction is incomplete"
- "inconsistent distributed velocity_mo packing"
- "invalid distributed velocity_mo pair ownership"

### write_velocity_gauge_outputs（710—950 行）

[实现位置](../src/bse/spectrum.cpp#L710)

~~~cpp
void write_velocity_gauge_outputs( const InputParameters &options, const librpa_int::Dataset &dataset, const FineVelocityMo &velocity, const std::vector<double> &energies_ry, const DistributedAmplitudes &amplitudes_x, const DistributedAmplitudes *amplitudes_y, const std::string &spin_type, const std::string &solution_type)
~~~

集体光谱入口：广播态数和能量、Reduce 偶极/k 权重、Gatherv 大 X 贡献，root 施加 triplet 零光学规则并写五类表。仅 root 的输入 energies 需有效。

调用线索：MPI_Allreduce、MPI_Bcast、MPI_Comm_rank、MPI_Comm_size、MPI_Gather、MPI_Gatherv、MPI_Reduce、amplitudes_x、analysis、broadcast_vector、broaden_oscillator_spectrum、calculate_oscillator_strengths、contribution_counts.data、contribution_counts.resize、contribution_indices.data、contribution_indices.resize、contribution_indices.size、contribution_offsets.data、contribution_offsets.resize、contribution_values.data、contribution_values.resize、contributions.begin、contributions.end、emplace_back、energies_ry.size、kweight、local_contribution_indices.data、local_contribution_indices.push_back、local_contribution_indices.size、local_contribution_values.data、local_contribution_values.push_back、local_weight1.data、local_weight2.data、max、mean_squared、oscillator_strengths.begin、oscillator_strengths.end、output_dir、std::abs、std::accumulate、std::conj、std::fill、std::invalid_argument、std::norm、std::overflow_error、std::runtime_error、std::setprecision、std::setw、std::sort、std::to_string、std::vector、velocity_gauge_transition_dipoles_mpi、velocity_index、weight1.data、weight2.begin、weight2.data、weight2.end、write_dipoles、write_oscillator_strengths、write_spectrum。

显式并行语句（被调函数还可能继续通信）：

- [第 722 行](../src/bse/spectrum.cpp#L722)：MPI_Comm_rank(dataset.comm_h.comm, &rank);
- [第 723 行](../src/bse/spectrum.cpp#L723)：MPI_Comm_size(dataset.comm_h.comm, &mpi_size);
- [第 725 行](../src/bse/spectrum.cpp#L725)：MPI_Bcast(&nstates, 1, MPI_INT, 0, dataset.comm_h.comm);
- [第 776 行](../src/bse/spectrum.cpp#L776)：MPI_Reduce(local_weight1.data(), rank == 0 ? weight1.data() : nullptr,
- [第 778 行](../src/bse/spectrum.cpp#L778)：MPI_Reduce(local_weight2.data(), rank == 0 ? weight2.data() : nullptr,
- [第 787 行](../src/bse/spectrum.cpp#L787)：MPI_Allreduce(&local_count_valid, &all_counts_valid, 1, MPI_INT,
- [第 796 行](../src/bse/spectrum.cpp#L796)：MPI_Gather(&local_contribution_count, 1, MPI_INT,
- [第 823 行](../src/bse/spectrum.cpp#L823)：MPI_Bcast(&total_count_valid, 1, MPI_INT, 0, dataset.comm_h.comm);
- [第 827 行](../src/bse/spectrum.cpp#L827)：MPI_Gatherv(local_contribution_indices.data(), local_contribution_count,
- [第 833 行](../src/bse/spectrum.cpp#L833)：MPI_Gatherv(local_contribution_values.data(), local_contribution_count,

源码错误消息片段：

- "excitation energies and distributed amplitudes do not match"
- "zero KS gap in velocity-gauge spectrum"
- "local transition analysis exceeds the MPI count limit"
- "transition analysis exceeds the MPI count limit"
- "cannot write transition analysis"
- "invalid distributed transition-analysis index"


<a id="file-src-bse-spectrum-h"></a>
## src/bse/spectrum.h

[打开源码](../src/bse/spectrum.h)。公共声明头，实现见同名 .cpp（若有）；通过此头使用接口，不 include .cpp。

本头没有独立函数体，以下给出声明/结构定义以查参数默认值、类型与访问权限；行为见实现文件及主指南。

~~~cpp
#pragma once

#include "bse_types.h"

#include <mpi.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace librpa_int
{
class Dataset;
}

namespace libbse
{

struct FineVelocityMo
{
    int nk = 0;
    int nbands = 0;
    int first_pair = 0;
    int local_pairs = 0;
    // Direction-major velocity elements for the locally owned (k,i,a) pairs.
    // Only <i|v|a>, which enters the velocity-gauge spectrum, is retained.
    std::vector<Complex> values;
    std::vector<double> gaps_ha;
};

struct OscillatorStrength
{
    double energy_ev = 0.0;
    // Cartesian ABACUS-Ry oscillator strengths and orientational average.
    std::array<double, 3> directional{};
    double isotropic = 0.0;
};

struct SpectrumPoint
{
    double energy_ev = 0.0;
    // Lorentz-broadened oscillator-strength density in eV^-1.
    std::array<double, 3> directional{};
    double isotropic = 0.0;
};

std::vector<OscillatorStrength> calculate_oscillator_strengths(
    const std::vector<double> &energies_ry,
    const std::vector<std::array<Complex, 3>> &dipoles,
    int kpoint_count);

std::vector<SpectrumPoint> broaden_oscillator_spectrum(
    const InputParameters &options,
    const std::vector<OscillatorStrength> &strengths);

// Return the velocity matrix elements needed by the locally owned BSE pairs.
// When the SCF and BSE grids coincide this selects the supplied velocity_mo
// directly; otherwise it interpolates the operator through its localized
// AO/R form and sends each pair only to its excitation-amplitude owner.
FineVelocityMo prepare_fine_velocity_mo(
    const InputParameters &options, const QuasiparticleBands &qp,
    const std::shared_ptr<librpa_int::Dataset> &dataset,
    const std::vector<Complex> &band_gauge_phases);

std::array<Complex, 3> velocity_gauge_transition_dipole(
    int state, const InputParameters &options, const FineVelocityMo &velocity_mo,
    const std::vector<Complex> &amplitudes_x,
    const std::vector<Complex> *amplitudes_y);

// Contract each rank's local electron-hole-pair block and reduce the
// transition dipoles to rank 0. Excitation amplitudes remain distributed.
std::vector<std::array<Complex, 3>> velocity_gauge_transition_dipoles_mpi(
    MPI_Comm comm, const InputParameters &options,
    const FineVelocityMo &velocity_mo,
    const DistributedAmplitudes &amplitudes_x,
    const DistributedAmplitudes *amplitudes_y);

void write_velocity_gauge_outputs(
    const InputParameters &options,
    const librpa_int::Dataset &dataset,
    const FineVelocityMo &velocity_mo,
    const std::vector<double> &energies_ry,
    const DistributedAmplitudes &amplitudes_x,
    const DistributedAmplitudes *amplitudes_y,
    const std::string &spin_type,
    const std::string &solution_type);

} // namespace libbse
~~~


<a id="file-src-interface-librpa-api-cpp"></a>
## src/interface/librpa_api.cpp

[打开源码](../src/interface/librpa_api.cpp)。LibRPA/LAPACK/ScaLAPACK 薄包装。带 descriptor 操作通常需集体参与，ScaLAPACK 坐标为 1-based。

### initialize（10—14 行）

[实现位置](../src/interface/librpa_api.cpp#L10)

~~~cpp
void initialize()
~~~

LibRPA_API 全局初始化：设 INFO 输出，调用 librpa::init_global(LIBRPA_SWITCH_OFF)。必须在 MPI 初始化后。

调用线索：librpa::init_global、librpa::set_output_level。

### finalize（16—19 行）

[实现位置](../src/interface/librpa_api.cpp#L16)

~~~cpp
void finalize()
~~~

调用 librpa::finalize_global，包含对应外部全局资源清理；Dataset 应先销毁，MPI 应最后结束。

调用线索：librpa::finalize_global。

### read_dataset（21—44 行）

[实现位置](../src/interface/librpa_api.cpp#L21)

~~~cpp
std::shared_ptr<librpa_int::Dataset> read_dataset( MPI_Comm comm, const ReaderOptions &options)
~~~

根据 ReaderOptions 可先建 aims 视图，再调用 librpa::read_dataset_from_files。速度读取无条件；read_ri/read_band_data 控制其余分支。

调用线索：libbse::prepare_fhi_aims_reader_view、librpa::read_dataset_from_files、string。

### build_bare_coulomb（46—73 行）

[实现位置](../src/interface/librpa_api.cpp#L46)

~~~cpp
libbse::TensorMap<libbse::Complex> build_bare_coulomb( librpa_int::Dataset &dataset)
~~~

调 LibRPA FT_Vq 将 vq_cut 变实空间，重装为 TensorMap<Complex>，清空 Dataset vq_cut。调用后原 q 数据不再可用。

调用线索：dataset.vq_cut.clear、librpa_int::FT_Vq、std::move、tensor。

### inverse（75—90 行）

[实现位置](../src/interface/librpa_api.cpp#L75)

~~~cpp
librpa_int::ComplexMatrix inverse(const librpa_int::ComplexMatrix &matrix)
~~~

检查 ComplexMatrix 方阵，复制后 LAPACK zgetrf LU + zgetri 求逆，奇异或失败报错；原矩阵不修改。用于完整粗 KS 基逆变换。

调用线索：librpa_int::LapackConnector::zgetrf、librpa_int::LapackConnector::zgetri、pivots.data、result、std::invalid_argument、std::runtime_error、std::vector、work.data、work.size。

源码错误消息片段：

- "cannot invert a nonsquare matrix"
- "KS wavefunction matrix is singular"
- "failed to invert KS wavefunction matrix"

### conjugate（92—95 行）

[实现位置](../src/interface/librpa_api.cpp#L92)

~~~cpp
librpa_int::ComplexMatrix conjugate(const librpa_int::ComplexMatrix &matrix)
~~~

返回逐元素共轭矩阵，调用 librpa_int::conj；不转置。

调用线索：librpa_int::conj。

### transpose（97—100 行）

[实现位置](../src/interface/librpa_api.cpp#L97)

~~~cpp
librpa_int::ComplexMatrix transpose(const librpa_int::ComplexMatrix &matrix)
~~~

返回普通转置，调用 librpa_int::transpose(matrix,false)；不共轭。

调用线索：librpa_int::transpose。

### scale_accumulate（102—107 行）

[实现位置](../src/interface/librpa_api.cpp#L102)

~~~cpp
void scale_accumulate(libbse::Complex factor, const librpa_int::ComplexMatrix &source, librpa_int::ComplexMatrix &target)
~~~

target += factor*source，委托 LibRPA 实现；source/target 形状需匹配。

调用线索：librpa_int::scale_accumulate。

### redistribute（109—119 行）

[实现位置](../src/interface/librpa_api.cpp#L109)

~~~cpp
void redistribute(int rows, int columns, const libbse::Complex *source, int source_row, int source_column, const librpa_int::ArrayDesc &source_descriptor, libbse::Complex *target, int target_row, int target_column, const librpa_int::ArrayDesc &target_descriptor)
~~~

double/Complex 两个重载封装 ScaLAPACK pgemr2d_f，子块起点为 1-based。source descriptor 的 BLACS context 用于跨分布复制，全部相关 rank 调。

调用线索：librpa_int::ScalapackConnector::pgemr2d_f、source_descriptor.ictxt。

### redistribute（121—131 行）

[实现位置](../src/interface/librpa_api.cpp#L121)

~~~cpp
void redistribute(int rows, int columns, const double *source, int source_row, int source_column, const librpa_int::ArrayDesc &source_descriptor, double *target, int target_row, int target_column, const librpa_int::ArrayDesc &target_descriptor)
~~~

double/Complex 两个重载封装 ScaLAPACK pgemr2d_f，子块起点为 1-based。source descriptor 的 BLACS context 用于跨分布复制，全部相关 rank 调。

调用线索：librpa_int::ScalapackConnector::pgemr2d_f、source_descriptor.ictxt。

### distributed_transpose（133—140 行）

[实现位置](../src/interface/librpa_api.cpp#L133)

~~~cpp
void distributed_transpose( int rows, int columns, const libbse::Complex *source, const librpa_int::ArrayDesc &descriptor, libbse::Complex *target)
~~~

封装复数 ptranu_f：target=sourceᵀ，系数 alpha=1,beta=0；沿相同 descriptor 分布，集体调用。

调用线索：libbse::Complex、librpa_int::ScalapackConnector::ptranu_f。

### distributed_conjugate_transpose（142—149 行）

[实现位置](../src/interface/librpa_api.cpp#L142)

~~~cpp
void distributed_conjugate_transpose( int rows, int columns, const libbse::Complex *source, const librpa_int::ArrayDesc &descriptor, libbse::Complex *target)
~~~

封装 ptranc_f：target=source†；与 B 的普通转置区别是元素共轭。

调用线索：libbse::Complex、librpa_int::ScalapackConnector::ptranc_f。

### multiply（151—162 行）

[实现位置](../src/interface/librpa_api.cpp#L151)

~~~cpp
void multiply(char trans_a, char trans_b, int m, int n, int k, double alpha, const double *a, const librpa_int::ArrayDesc &a_descriptor, const double *b, const librpa_int::ArrayDesc &b_descriptor, double beta, double *c, const librpa_int::ArrayDesc &c_descriptor)
~~~

double/Complex 重载封装分布式 pgemm_f：C=alpha*op(A)*op(B)+beta*C。m,n,k 为全局操作维度，各输入从 1,1 开始，descriptor 决定局部存储。

调用线索：librpa_int::ScalapackConnector::pgemm_f。

### multiply（164—175 行）

[实现位置](../src/interface/librpa_api.cpp#L164)

~~~cpp
void multiply(char trans_a, char trans_b, int m, int n, int k, libbse::Complex alpha, const libbse::Complex *a, const librpa_int::ArrayDesc &a_descriptor, const libbse::Complex *b, const librpa_int::ArrayDesc &b_descriptor, libbse::Complex beta, libbse::Complex *c, const librpa_int::ArrayDesc &c_descriptor)
~~~

double/Complex 重载封装分布式 pgemm_f：C=alpha*op(A)*op(B)+beta*C。m,n,k 为全局操作维度，各输入从 1,1 开始，descriptor 决定局部存储。

调用线索：librpa_int::ScalapackConnector::pgemm_f。


<a id="file-src-interface-librpa-api-h"></a>
## src/interface/librpa_api.h

[打开源码](../src/interface/librpa_api.h)。公共声明头，实现见同名 .cpp（若有）；通过此头使用接口，不 include .cpp。

本头没有独立函数体，以下给出声明/结构定义以查参数默认值、类型与访问权限；行为见实现文件及主指南。

~~~cpp
#pragma once

#include "bse/bse_types.h"

#include <librpa_file_reader.hpp>

#include <memory>
#include <string>

namespace LibRPA_API
{

void initialize();
void finalize();

struct ReaderOptions
{
    std::string input_dir;
    std::string output_dir;
    std::string input_format = "librpa";
    bool read_ri = true;
    bool read_band_data = true;
};

std::shared_ptr<librpa_int::Dataset> read_dataset(
    MPI_Comm comm, const ReaderOptions &options);

libbse::TensorMap<libbse::Complex> build_bare_coulomb(
    librpa_int::Dataset &dataset);

librpa_int::ComplexMatrix inverse(const librpa_int::ComplexMatrix &matrix);
librpa_int::ComplexMatrix conjugate(const librpa_int::ComplexMatrix &matrix);
librpa_int::ComplexMatrix transpose(const librpa_int::ComplexMatrix &matrix);
void scale_accumulate(libbse::Complex factor,
                      const librpa_int::ComplexMatrix &source,
                      librpa_int::ComplexMatrix &target);

void redistribute(int rows, int columns,
                  const libbse::Complex *source, int source_row, int source_column,
                  const librpa_int::ArrayDesc &source_descriptor,
                  libbse::Complex *target, int target_row, int target_column,
                  const librpa_int::ArrayDesc &target_descriptor);
void redistribute(int rows, int columns,
                  const double *source, int source_row, int source_column,
                  const librpa_int::ArrayDesc &source_descriptor,
                  double *target, int target_row, int target_column,
                  const librpa_int::ArrayDesc &target_descriptor);

void distributed_transpose(
    int rows, int columns, const libbse::Complex *source,
    const librpa_int::ArrayDesc &descriptor, libbse::Complex *target);
void distributed_conjugate_transpose(
    int rows, int columns, const libbse::Complex *source,
    const librpa_int::ArrayDesc &descriptor, libbse::Complex *target);

void multiply(char trans_a, char trans_b, int m, int n, int k,
              double alpha,
              const double *a, const librpa_int::ArrayDesc &a_descriptor,
              const double *b, const librpa_int::ArrayDesc &b_descriptor,
              double beta,
              double *c, const librpa_int::ArrayDesc &c_descriptor);
void multiply(char trans_a, char trans_b, int m, int n, int k,
              libbse::Complex alpha,
              const libbse::Complex *a, const librpa_int::ArrayDesc &a_descriptor,
              const libbse::Complex *b, const librpa_int::ArrayDesc &b_descriptor,
              libbse::Complex beta,
              libbse::Complex *c, const librpa_int::ArrayDesc &c_descriptor);

} // namespace LibRPA_API
~~~


<a id="file-src-io-bse-files-cpp"></a>
## src/io/bse_files.cpp

[打开源码](../src/io/bse_files.cpp)。QP/Wc 协议、单位、带窗口与周期胞。公共声明在 bse_files.h，其余为文件内助手。

### move_tensor（24—37 行）

[实现位置](../src/io/bse_files.cpp#L24)

~~~cpp
void move_tensor(TensorMap<Complex> &map, int iat, int jat, const Cell &old_cell, const Cell &new_cell)
~~~

把指定 (I,J,old_R) 移到 new_R；找不到旧键则无操作，目标已存在则报错。移动不复制，不累加冲突。

调用线索：blocks.count、blocks.emplace、blocks.end、blocks.erase、blocks.find、map.end、map.find、std::move、std::runtime_error。

源码错误消息片段：

- "nearest-cell remap produced a duplicate tensor key"

### periodic_coordinate_distance（48—52 行）

[实现位置](../src/io/bse_files.cpp#L48)

~~~cpp
double periodic_coordinate_distance(double left, double right)
~~~

返回 abs((left-right)-round(left-right))，使相差整数的分数坐标等价。

调用线索：std::abs、std::round。

### match_kpoint（54—74 行）

[实现位置](../src/io/bse_files.cpp#L54)

~~~cpp
int match_kpoint(const KPoint &k, const std::vector<librpa_int::Vector3_Order<double>> &grid, double tolerance, const std::string &source)
~~~

在 BSE grid 中按逐坐标周期距离匹配，返回唯一 0-based k 索引；缺失或多个匹配报错。

调用线索：grid.size、periodic_coordinate_distance、std::runtime_error。

源码错误消息片段：

- "ambiguous periodic k-point in "
- "QP k-point is absent from the BSE grid: "

### wavefunction_core_offset（76—96 行）

[实现位置](../src/io/bse_files.cpp#L76)

~~~cpp
int wavefunction_core_offset(const librpa_int::Dataset &dataset, int ik, int nocc, int record_offset)
~~~

从 mf_band 占据权重找最高占据带，减 nocc 得 KS 选择窗口起点；合成数据无占据时退回 QP record_offset。

调用线索：at、dataset.mf_band.get_n_bands、dataset.mf_band.get_n_kpoints、dataset.mf_band.get_weight、std::runtime_error、weights。

源码错误消息片段：

- "mean-field data contain fewer occupied bands than nocc"

### store_requested_bands（98—141 行）

[实现位置](../src/io/bse_files.cpp#L98)

~~~cpp
void store_requested_bands(QuasiparticleBands &result, int ik, const QpRecord &record, const InputParameters &options, const librpa_int::Dataset &dataset)
~~~

检查 QP 占据后空态的连续窗口、带数及共同核偏移，把选择能量存到 k*nbands+b；occupation_scale 区分文件权重约定。

调用线索：occupied_count、record.energies_ry.size、record.occupations.begin、record.occupations.end、record.occupations.size、std::any_of、std::distance、std::find_if、std::runtime_error、wavefunction_core_offset。

源码错误消息片段：

- "inconsistent QP record columns in "
- "not enough occupied/virtual QP bands in "
- "QP occupations are not an occupied-then-virtual window in "
- "inconsistent wavefunction core-band offset in QP data"

### assign_qp_records（143—170 行）

[实现位置](../src/io/bse_files.cpp#L143)

~~~cpp
QuasiparticleBands assign_qp_records( const std::vector<QpRecord> &records, const InputParameters &options, const librpa_int::Dataset &dataset, double tolerance)
~~~

把任意文件顺序记录匹配到 Dataset k 顺序，要求每个 k 恰好一次；返回完整 QuasiparticleBands，能量在子 reader 已统一 Ry。

调用线索：assigned.begin、assigned.end、dataset.mf_band.get_n_kpoints、match_kpoint、result.energies_ry.resize、std::distance、std::find、std::runtime_error、std::to_string、std::vector、store_requested_bands。

源码错误消息片段：

- "duplicate QP data for BSE k-point "
- "missing QP data for BSE k-point "

### data_file（172—177 行）

[实现位置](../src/io/bse_files.cpp#L172)

~~~cpp
fs::path data_file(const InputParameters &options, const char *default_name)
~~~

qp_data 为空用 input_dir；若路径是目录则拼默认文件名，否则直接使用该文件路径。

调用线索：fs::is_directory、options.qp_data.empty、path。

### read_fine_qp_bands（179—213 行）

[实现位置](../src/io/bse_files.cpp#L179)

~~~cpp
QuasiparticleBands read_fine_qp_bands(const InputParameters &options, const librpa_int::Dataset &dataset)
~~~

读 GW_band_spin_1.dat 风格逐行 QP 数据，energy_eV→Ry，occupation_scale=Nk，再以 1e-6 容差统一 k 顺序。

调用线索：assign_qp_records、data_file、dataset.mf_band.get_n_kpoints、file.string、input、line.empty、parser、record.energies_ry.push_back、record.occupations.push_back、records.push_back、std::getline、std::move、std::runtime_error、std::to_string。

源码错误消息片段：

- "cannot open GW band file: "
- "invalid k-point header in "

### read_coarse_qp_bands（215—272 行）

[实现位置](../src/io/bse_files.cpp#L215)

~~~cpp
QuasiparticleBands read_coarse_qp_bands(const InputParameters &options, const librpa_int::Dataset &dataset)
~~~

读 energy_qp 的 K_point 块，采用 QP_Ha 列乘 2，使用 5.1e-5 周期 k 容差。文件格式名称不代表只能匹配粗网格，最终仍看 Dataset BSE grid。

调用线索：assign_qp_records、data_file、file.string、header、input、line.find、line.find_first_not_of、record.energies_ry.push_back、record.occupations.push_back、records.push_back、state_line、std::getline、std::move、std::runtime_error、std::to_string。

源码错误消息片段：

- "cannot open coarse-grid quasiparticle file: "
- "invalid k-point header in "
- "empty QP block in "

### calculate_gaps（274—293 行）

[实现位置](../src/io/bse_files.cpp#L274)

~~~cpp
void calculate_gaps(QuasiparticleBands &result, const InputParameters &options)
~~~

在选择窗口内，逐 k 找占据最大值和空态最小值；直接隙=min 同 k 差，间接隙=全局 CBM−VBM。

调用线索：max、std::max、std::min。

### read_qp_bands（297—325 行）

[实现位置](../src/io/bse_files.cpp#L297)

~~~cpp
QuasiparticleBands read_qp_bands(const InputParameters &options, const librpa_int::Dataset &dataset)
~~~

分派 auto/energy_qp/fine_band，读完计算两种隙。auto 优先 energy_qp；明确不支持 aims 自身 QP 文件。

调用线索：calculate_gaps、fs::is_regular_file、options.qp_data.empty、path、path.filename、read_coarse_qp_bands、read_fine_qp_bands、std::invalid_argument、std::runtime_error。

源码错误消息片段：

- "cannot find LibRPA energy_qp; FHI-aims quasiparticle files are intentionally unsupported"
- "unsupported quasiparticle format: "

### convert_lri_coefficients（327—336 行）

[实现位置](../src/io/bse_files.cpp#L327)

~~~cpp
TensorMap<Complex> convert_lri_coefficients(librpa_int::Dataset &dataset)
~~~

把 Dataset cs_data.data_libri 的实 tensor 转 Complex，并转原子键为 int；清空原 cs_data 以释放内存。

调用线索：RI::Global_Func::convert、dataset.cs_data.clear。

### read_screened_interaction（338—413 行）

[实现位置](../src/io/bse_files.cpp#L338)

~~~cpp
TensorMap<Complex> read_screened_interaction( const InputParameters &options, const TensorMap<Complex> &bare_coulomb, std::size_t cell_count, const std::vector<int> &local_i_atoms, const std::vector<int> &local_j_atoms)
~~~

按本 rank list_I×list_J×R 读取 ifreq_0 复坐标 Wc，验证 R/尺寸/条目，加对应 V 返回 W。所有要求的局部文件必须存在。

调用线索：Complex、bare_coulomb.at、bare_iter->second.shape.size、dims、end、file.string、find、fs::path、input、line.empty、line.find、line.front、line.substr、name.str、options.screened_dir.empty、parent_path、rs、std::getline、std::move、std::runtime_error、tensor。

源码错误消息片段：

- "cannot open screened interaction: "
- "missing R vector in "
- "invalid MatrixMarket dimensions in "
- "Wc R vector is absent from bare Coulomb map"
- "Wc and bare Coulomb dimensions differ in "
- "invalid MatrixMarket entry in "

### remap_to_nearest_bvk_cell（415—439 行）

[实现位置](../src/io/bse_files.cpp#L415)

~~~cpp
void remap_to_nearest_bvk_cell(TensorMap<Complex> &tensors, const librpa_int::Dataset &dataset)
~~~

用 RI::Cell_Nearest 对每原子对和 canonical R 选择最近等价胞，必要时 move_tensor；原地修改 map 的键。

调用线索：dataset.atoms.size、move_tensor、nearest.cell_nearest_direction、nearest.init。


<a id="file-src-io-bse-files-h"></a>
## src/io/bse_files.h

[打开源码](../src/io/bse_files.h)。公共声明头，实现见同名 .cpp（若有）；通过此头使用接口，不 include .cpp。

本头没有独立函数体，以下给出声明/结构定义以查参数默认值、类型与访问权限；行为见实现文件及主指南。

~~~cpp
#pragma once

#include "bse/bse_types.h"

namespace librpa_int
{
class Dataset;
}

namespace libbse
{

QuasiparticleBands read_qp_bands(const InputParameters &options,
                                 const librpa_int::Dataset &dataset);

TensorMap<Complex> read_screened_interaction(
    const InputParameters &options,
    const TensorMap<Complex> &bare_coulomb,
    std::size_t cell_count,
    const std::vector<int> &local_i_atoms,
    const std::vector<int> &local_j_atoms);

TensorMap<Complex> convert_lri_coefficients(librpa_int::Dataset &dataset);

void remap_to_nearest_bvk_cell(TensorMap<Complex> &tensors,
                               const librpa_int::Dataset &dataset);

} // namespace libbse
~~~


<a id="file-src-io-fhi-aims-adapter-cpp"></a>
## src/io/fhi_aims_adapter.cpp

[打开源码](../src/io/fhi_aims_adapter.cpp)。识别 aims、建立只读源数据链接视图；root 建视图后同步错误并 Barrier。

### basis_identifies_aims（14—23 行）

[实现位置](../src/io/fhi_aims_adapter.cpp#L14)

~~~cpp
bool basis_identifies_aims(const fs::path &input_dir)
~~~

检查 basis_out 开头 atoms nao naux producer，producer 精确为 aims 返回 true；文件缺失或读失败返回 false。

调用线索：input。

### create_source_link（25—36 行）

[实现位置](../src/io/fhi_aims_adapter.cpp#L25)

~~~cpp
void create_source_link(const fs::path &source, const fs::path &target)
~~~

创建指向绝对 source 的符号链接；目标若已是相同 canonical source 的链接可复用，否则已有目标报错。

调用线索：fs::absolute、fs::create_symlink、fs::exists、fs::is_symlink、fs::weakly_canonical、std::runtime_error、target.string。

源码错误消息片段：

- "FHI-aims adapter target already exists: "

### validate_adapter_source（38—56 行）

[实现位置](../src/io/fhi_aims_adapter.cpp#L38)

~~~cpp
void validate_adapter_source(const fs::path &view, const fs::path &source)
~~~

读取/创建视图 .source_directory 标记；已有标记必须等于当前源 canonical 路径，防止混用来源。

调用线索：fs::exists、fs::path、fs::weakly_canonical、input、output、std::getline、std::runtime_error、string。

源码错误消息片段：

- "the existing FHI-aims reader view belongs to another input directory: "
- "cannot create FHI-aims adapter marker"

### collective_error（58—68 行）

[实现位置](../src/io/fhi_aims_adapter.cpp#L58)

~~~cpp
std::string collective_error(MPI_Comm comm, const std::string &root_error)
~~~

root 将错误字符串长度与内容广播，所有 rank 得到同样错误；自身不抛异常，由上层统一处理。

调用线索：MPI_Bcast、MPI_Comm_rank、error、error.data、root_error.size。

显式并行语句（被调函数还可能继续通信）：

- [第 61 行](../src/io/fhi_aims_adapter.cpp#L61)：MPI_Comm_rank(comm, &rank);
- [第 63 行](../src/io/fhi_aims_adapter.cpp#L63)：MPI_Bcast(&length, 1, MPI_INT, 0, comm);
- [第 66 行](../src/io/fhi_aims_adapter.cpp#L66)：if (length != 0) MPI_Bcast(error.data(), length, MPI_CHAR, 0, comm);

### install_velocity（70—82 行）

[实现位置](../src/io/fhi_aims_adapter.cpp#L70)

~~~cpp
void install_velocity(const fs::path &source, const fs::path &view)
~~~

要求源 canonical velocity_matrix 存在，更新视图链接；缺失时提示先运行 Python 转换器，不现场伪造速度。

调用线索：create_source_link、fs::exists、fs::is_regular_file、fs::is_symlink、fs::remove、std::runtime_error。

源码错误消息片段：

- "FHI-aims input has no canonical velocity_matrix; run tools/aims_mommat_to_velocity.py AIMS_EXPORT_DIR first"

### resolve_input_format（86—104 行）

[实现位置](../src/io/fhi_aims_adapter.cpp#L86)

~~~cpp
void resolve_input_format(InputParameters &options)
~~~

auto 按 aims 标记解析格式；显式 aims 也验证标记；auto wavefunction_gauge 对 aims 取 native，其它取 first_k。原地改 options。

调用线索：basis_identifies_aims、std::runtime_error。

源码错误消息片段：

- "input_format fhi_aims requires an aims producer tag in basis_out"

### prepare_fhi_aims_reader_view（106—154 行）

[实现位置](../src/io/fhi_aims_adapter.cpp#L106)

~~~cpp
fs::path prepare_fhi_aims_reader_view(MPI_Comm comm, const InputParameters &options)
~~~

root 建读取视图/marker/原文件链接和 cut Coulomb 别名，安装速度；同步错误并 Barrier 后返回视图路径。

调用线索：MPI_Barrier、MPI_Comm_rank、collective_error、create_source_link、entry.is_regular_file、entry.path、error.empty、error.what、filename、fs::create_directories、fs::directory_iterator、fs::exists、fs::is_symlink、fs::path、install_velocity、length、name.rfind、name.substr、source、std::runtime_error、string、validate_adapter_source。

显式并行语句（被调函数还可能继续通信）：

- [第 113 行](../src/io/fhi_aims_adapter.cpp#L113)：MPI_Comm_rank(comm, &rank);
- [第 152 行](../src/io/fhi_aims_adapter.cpp#L152)：MPI_Barrier(comm);


<a id="file-src-io-fhi-aims-adapter-h"></a>
## src/io/fhi_aims_adapter.h

[打开源码](../src/io/fhi_aims_adapter.h)。公共声明头，实现见同名 .cpp（若有）；通过此头使用接口，不 include .cpp。

本头没有独立函数体，以下给出声明/结构定义以查参数默认值、类型与访问权限；行为见实现文件及主指南。

~~~cpp
#pragma once

#include "parameter/parameter.h"

#include <mpi.h>

#include <filesystem>

namespace libbse
{

/** Detect the producer recorded in basis_out and resolve automatic options. */
void resolve_input_format(InputParameters &options);

/**
 * Build a non-destructive input view for LibRPA's public file reader.
 *
 * The view aliases the FHI-aims files, maps coulomb_cut_* to the historical
 * LibRPA Coulomb name, and aliases a canonical velocity_matrix created by
 * tools/aims_mommat_to_velocity.py. Velocity is mandatory.
 */
std::filesystem::path prepare_fhi_aims_reader_view(
    MPI_Comm comm, const InputParameters &options);

} // namespace libbse
~~~


<a id="file-src-parameter-parameter-cpp"></a>
## src/parameter/parameter.cpp

[打开源码](../src/parameter/parameter.cpp)。字符串→参数→验证→绝对路径。全局 PARAM 供主程序使用，测试可构造独立 Parameter。

### trim（19—26 行）

[实现位置](../src/parameter/parameter.cpp#L19)

~~~cpp
std::string trim(std::string value)
~~~

去字符串前后空白，返回新字符串；内部解析助手。

调用线索：base、std::find_if_not、std::isspace、std::string、value.begin、value.end、value.rbegin、value.rend。

### lower（28—33 行）

[实现位置](../src/parameter/parameter.cpp#L28)

~~~cpp
std::string lower(std::string value)
~~~

将字符转小写，使用 unsigned char 避免 ctype 未定义行为；内部助手。

调用线索：std::tolower、std::transform、value.begin、value.end。

### parse_integer（35—50 行）

[实现位置](../src/parameter/parameter.cpp#L35)

~~~cpp
int parse_integer(const std::string &key, const std::string &value)
~~~

stoi 并检查剩余非空白字符；拒绝尾随垃圾，抛包含关键字的 invalid_argument。

调用线索：empty、std::invalid_argument、std::stoi、trim、value.substr。

源码错误消息片段：

- "invalid integer for "
- "invalid integer for "

### parse_double（52—69 行）

[实现位置](../src/parameter/parameter.cpp#L52)

~~~cpp
double parse_double(const std::string &key, const std::string &value)
~~~

stod 并检查尾随字符；有限性与正值约束在 validate_and_resolve 阶段检查。

调用线索：empty、std::invalid_argument、std::stod、trim、value.substr。

源码错误消息片段：

- "invalid real value for "
- "invalid real value for "

### parse_boolean（71—81 行）

[实现位置](../src/parameter/parameter.cpp#L71)

~~~cpp
bool parse_boolean(const std::string &key, const std::string &value)
~~~

接受 1/true/t/.true./yes 与 0/false/f/.false./no，不分大小写，其他值报错。

调用线索：lower、std::invalid_argument、trim。

源码错误消息片段：

- "invalid boolean for "

### parse_string_list（83—91 行）

[实现位置](../src/parameter/parameter.cpp#L83)

~~~cpp
std::vector<std::string> parse_string_list(std::string value)
~~~

逗号变空格后分词转小写；通道去重/合法性由后续验证完成。

调用线索：input、lower、result.push_back、std::replace、value.begin、value.end。

### parse_assignments（93—121 行）

[实现位置](../src/parameter/parameter.cpp#L93)

~~~cpp
std::map<std::string, std::string> parse_assignments(const std::string &contents)
~~~

按行去 #/! 注释，跳可选标题，读等号或空白赋值，后一次覆盖同键，返回字符串 map。

调用线索：input、key.empty、line.empty、line.erase、line.find、line.find_first_of、line.substr、lower、std::getline、std::invalid_argument、std::min、std::to_string、trim、value.empty、value.front、value.substr。

源码错误消息片段：

- "invalid libbse.in assignment at line "

### InputParameters::solve_tda（127—130 行）

[实现位置](../src/parameter/parameter.cpp#L127)

~~~cpp
bool InputParameters::solve_tda() const noexcept
~~~

bse_tda 为 tda 或 both 时返回 true；不做 I/O。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。

### InputParameters::solve_full（132—135 行）

[实现位置](../src/parameter/parameter.cpp#L132)

~~~cpp
bool InputParameters::solve_full() const noexcept
~~~

bse_tda 为 full 或 both 时返回 true。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。

### InputParameters::spectrum_only（137—140 行）

[实现位置](../src/parameter/parameter.cpp#L137)

~~~cpp
bool InputParameters::spectrum_only() const noexcept
~~~

bse_solver 是否为 spectrum；用来跳核数据读取和求解。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。

### InputParameters::has_spin_type（142—146 行）

[实现位置](../src/parameter/parameter.cpp#L142)

~~~cpp
bool InputParameters::has_spin_type(const std::string &spin_type) const noexcept
~~~

线性搜索 bse_spin_types 中是否包含给定字符串，调用方传规范化名称。

调用线索：bse_spin_types.begin、bse_spin_types.end、std::find。

### InputParameters::requires_hartree（148—151 行）

[实现位置](../src/parameter/parameter.cpp#L148)

~~~cpp
bool InputParameters::requires_hartree() const noexcept
~~~

包含 singlet 或 rpa 时 true，决定 H 核收缩；不是整个 reader 的条件。

调用线索：has_spin_type。

### InputParameters::requires_screened（153—156 行）

[实现位置](../src/parameter/parameter.cpp#L153)

~~~cpp
bool InputParameters::requires_screened() const noexcept
~~~

包含 singlet 或 triplet 时 true，决定 W 核收缩。

调用线索：has_spin_type。

### InputParameters::ipa_only（158—161 行）

[实现位置](../src/parameter/parameter.cpp#L158)

~~~cpp
bool InputParameters::ipa_only() const noexcept
~~~

恰好一个通道且为 ipa 才 true；控制免求解器快捷路径。

调用线索：bse_spin_types.front、bse_spin_types.size。

### interaction_coefficients（163—170 行）

[实现位置](../src/parameter/parameter.cpp#L163)

~~~cpp
InteractionCoefficients interaction_coefficients(const std::string &spin_type)
~~~

将 singlet/triplet/rpa/ipa 转为 (2,-1)/(0,-1)/(2,0)/(0,0)，未知通道抛异常。

调用线索：std::invalid_argument。

源码错误消息片段：

- "unsupported BSE spin type: "

### Parameter::read（172—180 行）

[实现位置](../src/parameter/parameter.cpp#L172)

~~~cpp
void Parameter::read(const fs::path &filename)
~~~

读整个输入文件，以其绝对父目录为 base 调 parse；默认文件名 libbse.in。

调用线索：absolute.parent_path、contents.str、filename.string、fs::absolute、input、input.rdbuf、parse、std::runtime_error。

源码错误消息片段：

- "cannot open LibBSE input file: "

### Parameter::parse（182—242 行）

[实现位置](../src/parameter/parameter.cpp#L182)

~~~cpp
void Parameter::parse(const std::string &contents, const fs::path &base_directory)
~~~

重置 inp 默认值，逐键取值、转类型，剩余未知键报错，再 validate_and_resolve；可在测试直接传字符串。

调用线索：assignments.begin、assignments.empty、assignments.end、assignments.erase、assignments.find、lower、parse_assignments、parse_boolean、parse_double、parse_integer、parse_string_list、std::invalid_argument、take、validate_and_resolve、value.empty。

源码错误消息片段：

- "unknown LibBSE input parameter: "

### Parameter::validate_and_resolve（244—336 行）

[实现位置](../src/parameter/parameter.cpp#L244)

~~~cpp
void Parameter::validate_and_resolve(const fs::path &base_directory)
~~~

统一枚举、检查模式/范围/不支持项，按输入文件 base 将路径转绝对规范路径；不检查所有计算数据已存在。

调用线索：fs::absolute、fs::path、inp.bse_spin_types.empty、inp.input_dir.empty、inp.ipa_only、inp.output_dir.empty、inp.qp_data.empty、inp.requires_hartree、inp.screened_dir.empty、input_path、input_path.is_relative、interaction_coefficients、lexically_normal、lower、output_path、output_path.is_relative、parent_path、qp_path、qp_path.is_relative、screened_path.is_relative、std::invalid_argument、std::isfinite、string、trim、unique_spin_types.insert。

源码错误消息片段：

- "input_dir is required in libbse.in"
- "output_dir must not be empty"
- "input_format must be auto, librpa, or fhi_aims"
- "qp_format must be auto, energy_qp, or fine_band"
- "invalid nocc, nvirt, or bse_nstates"
- "bse_solver must be elpa or spectrum"
- "bse_tda must be tda, full, or both"
- "bse_spin_types must not be empty"
- "duplicate BSE spin type: "
- "IPA requires bse_tda tda"
- "this LibBSE path currently requires bse_continue 0"
- "singlet and RPA channels require bse_ri_hartree 1 in LibBSE"
- "bse_use_fine_kgrid must be 0 or 1"
- "this LibBSE path currently requires bse_q_approx_mode 0"
- "out_bse_ab is not implemented in LibBSE"
- "LibBSE supports only abs_gauge velocity"
- "wavefunction_gauge must be auto, native, or first_k"
- "invalid optical-spectrum energy grid or broadening"

### Parameter::print（338—369 行）

[实现位置](../src/parameter/parameter.cpp#L338)

~~~cpp
void Parameter::print(std::ostream &output) const
~~~

打印解析后关键参数；只负责流输出，调用者决定 root 限制。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。


<a id="file-src-parameter-parameter-h"></a>
## src/parameter/parameter.h

[打开源码](../src/parameter/parameter.h)。公共声明头，实现见同名 .cpp（若有）；通过此头使用接口，不 include .cpp。

本头没有独立函数体，以下给出声明/结构定义以查参数默认值、类型与访问权限；行为见实现文件及主指南。

~~~cpp
#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace libbse
{

struct Constants
{
    static constexpr double pi = 3.141592653589793238462643383279502884;
    static constexpr double ry_to_ev = 13.605693122994;
    static constexpr double ha_to_ry = 2.0;
    static constexpr double cs_threshold = 1.0e-12;
    static constexpr double coulomb_threshold = 1.0e-12;
    static constexpr double kpoint_tolerance = 1.0e-10;
    static constexpr double band_file_kpoint_tolerance = 1.0e-6;
    static constexpr double energy_qp_kpoint_tolerance = 5.1e-5;
    static constexpr double occupation_tolerance = 0.1;
    static constexpr double matrix_symmetry_threshold = 1.0e-6;
    static constexpr double output_zero_tolerance = 1.0e-10;
    static constexpr double zero_gap_tolerance = 1.0e-14;
    static constexpr const char *input_filename = "libbse.in";
};

struct InputParameters
{
    // Names exposed in libbse.in
    std::string input_dir;
    std::string output_dir = "libbse.d";
    std::string input_format = "auto";
    std::string qp_data;
    std::string qp_format = "auto";
    std::string screened_dir;
    int bse_nstates = -1;
    int nocc = 4;
    int nvirt = 4;
    std::string bse_solver = "elpa";
    std::vector<std::string> bse_spin_types{"singlet", "triplet"};
    int bse_continue = 0;
    std::string bse_tda = "both";
    bool bse_ri_hartree = true;
    int bse_use_fine_kgrid = 1;
    int bse_q_approx_mode = 0;
    bool out_bse_ab = false;
    std::string abs_gauge = "velocity";
    std::string wavefunction_gauge = "auto";
    double spectrum_broadening_ev = 0.10;
    double spectrum_energy_step_ev = 0.01;
    double spectrum_energy_min_ev = 0.0;
    // A negative maximum selects max(excitation energy) + 5*broadening.
    double spectrum_energy_max_ev = -1.0;

    bool solve_tda() const noexcept;
    bool solve_full() const noexcept;
    bool spectrum_only() const noexcept;
    bool has_spin_type(const std::string &spin_type) const noexcept;
    bool requires_hartree() const noexcept;
    bool requires_screened() const noexcept;
    bool ipa_only() const noexcept;
};

struct InteractionCoefficients
{
    double hartree = 0.0;
    double screened = 0.0;
};

InteractionCoefficients interaction_coefficients(const std::string &spin_type);

class Parameter
{
  public:
    InputParameters inp;
    const Constants constants{};

    void read(const std::filesystem::path &filename = Constants::input_filename);
    void parse(const std::string &contents,
               const std::filesystem::path &base_directory = ".");
    void print(std::ostream &output) const;

  private:
    void validate_and_resolve(const std::filesystem::path &base_directory);
};

extern Parameter PARAM;

} // namespace libbse
~~~


<a id="file-src-utils-profiler-cpp"></a>
## src/utils/profiler.cpp

[打开源码](../src/utils/profiler.cpp)。本进程主线程层次计时，非 MPI 汇总、非线程安全。

### Profiler::find_child（12—19 行）

[实现位置](../src/utils/profiler.cpp#L12)

~~~cpp
std::size_t Profiler::find_child(std::size_t parent, const std::string &name) const noexcept
~~~

按父 timer 索引和名字找节点，找不到返回 timers_.size()，供 start 创建新节点。

调用线索：timers_.size。

### Profiler::find_timer（21—27 行）

[实现位置](../src/utils/profiler.cpp#L21)

~~~cpp
const Profiler::Timer *Profiler::find_timer(const std::string &name) const noexcept
~~~

按名字返回第一个匹配 timer 指针，未找到 nullptr；同名不同父节点查询有歧义，应使用有区分度的名称。

调用线索：std::find_if、timers_.begin、timers_.end。

### Profiler::start（29—51 行）

[实现位置](../src/utils/profiler.cpp#L29)

~~~cpp
void Profiler::start(const std::string &name, const std::string &note)
~~~

在当前栈顶父节点下找/建 timer，累计调用次数，记录 CPU/wall 起点并入栈；主控制线程使用。

调用线索：WallClock::now、active_.back、active_.empty、active_.push_back、find_child、std::clock、stop、timers_.push_back、timers_.size。

### Profiler::stop（53—69 行）

[实现位置](../src/utils/profiler.cpp#L53)

~~~cpp
void Profiler::stop(const std::string &name) noexcept
~~~

只停止与栈顶同名的 timer，累加本次 CPU/wall 并出栈；空栈/名字不符不操作，noexcept。

调用线索：WallClock::now、active_.back、active_.empty、active_.pop_back、count、std::chrono::duration、std::clock。

### Profiler::terminate（71—74 行）

[实现位置](../src/utils/profiler.cpp#L71)

~~~cpp
void Profiler::terminate() noexcept
~~~

按栈顺序停止所有仍运行 timer，不删除记录。

调用线索：active_.back、active_.empty、stop。

### Profiler::reset（76—81 行）

[实现位置](../src/utils/profiler.cpp#L76)

~~~cpp
void Profiler::reset() noexcept
~~~

先 terminate，再清空 timer 数据与活动栈。

调用线索：active_.clear、terminate、timers_.clear。

### Profiler::get_num_timers（83—86 行）

[实现位置](../src/utils/profiler.cpp#L83)

~~~cpp
std::size_t Profiler::get_num_timers() const noexcept
~~~

返回层次节点数量，不是总调用次数。

调用线索：timers_.size。

### Profiler::get_call_count（88—92 行）

[实现位置](../src/utils/profiler.cpp#L88)

~~~cpp
std::size_t Profiler::get_call_count(const std::string &name) const noexcept
~~~

按名字返回第一个 timer 的调用次数，找不到返回 0。

调用线索：find_timer。

### Profiler::get_cpu_time_last（94—98 行）

[实现位置](../src/utils/profiler.cpp#L94)

~~~cpp
double Profiler::get_cpu_time_last(const std::string &name) const noexcept
~~~

返回对应 timer 最近一次 CPU 秒数，未找到返回 -1。

调用线索：find_timer。

### Profiler::get_wall_time_last（100—104 行）

[实现位置](../src/utils/profiler.cpp#L100)

~~~cpp
double Profiler::get_wall_time_last(const std::string &name) const noexcept
~~~

返回最近一次 steady-clock wall 秒数，未找到返回 -1。

调用线索：find_timer。

### Profiler::append_timer（106—119 行）

[实现位置](../src/utils/profiler.cpp#L106)

~~~cpp
void Profiler::append_timer(std::ostream &output, std::size_t index, int level) const
~~~

递归按层次写累计次数/CPU/wall，label 优先 note，否则 name；内部表格格式函数。

调用线索：append_timer、indent、std::setprecision、std::setw、timer.note.empty、timers_.size。

### Profiler::get_profile_string（121—133 行）

[实现位置](../src/utils/profiler.cpp#L121)

~~~cpp
std::string Profiler::get_profile_string() const
~~~

将根 timer 与子树格式化为完整计时表字符串。

调用线索：append_timer、output.str、std::setw、std::string、timers_.size。

### Profiler::display（135—138 行）

[实现位置](../src/utils/profiler.cpp#L135)

~~~cpp
void Profiler::display(std::ostream &output) const
~~~

把 get_profile_string 结果写到给定 ostream。

调用线索：get_profile_string。

### ScopedTimer::ScopedTimer（140—144 行）

[实现位置](../src/utils/profiler.cpp#L140)

~~~cpp
ScopedTimer::ScopedTimer(Profiler &profiler, std::string name, std::string note) : profiler_(&profiler), name_(std::move(name))
~~~

保存 profiler 指针和 name，构造即 start；禁止复制/赋值以避免重复 stop。

调用线索：profiler_->start。

### ScopedTimer::~ScopedTimer（146—149 行）

[实现位置](../src/utils/profiler.cpp#L146)

~~~cpp
ScopedTimer::~ScopedTimer()
~~~

析构 stop，异常展开也执行；要求被引用 profiler 仍存在。

调用线索：profiler_->stop。


<a id="file-src-utils-profiler-h"></a>
## src/utils/profiler.h

[打开源码](../src/utils/profiler.h)。公共声明头，实现见同名 .cpp（若有）；通过此头使用接口，不 include .cpp。

本头没有独立函数体，以下给出声明/结构定义以查参数默认值、类型与访问权限；行为见实现文件及主指南。

~~~cpp
#pragma once

#include <chrono>
#include <cstddef>
#include <ctime>
#include <iosfwd>
#include <string>
#include <vector>

namespace libbse
{

//! Hierarchical CPU and wall-clock profiler for the LibBSE control thread.
class Profiler
{
  public:
    void start(const std::string &name, const std::string &note = {});
    void stop(const std::string &name) noexcept;
    void terminate() noexcept;
    void reset() noexcept;

    std::size_t get_num_timers() const noexcept;
    std::size_t get_call_count(const std::string &name) const noexcept;
    double get_cpu_time_last(const std::string &name) const noexcept;
    double get_wall_time_last(const std::string &name) const noexcept;
    std::string get_profile_string() const;
    void display(std::ostream &output) const;

  private:
    using WallClock = std::chrono::steady_clock;
    static constexpr std::size_t no_parent = static_cast<std::size_t>(-1);

    struct Timer
    {
        std::string name;
        std::string note;
        std::size_t parent = no_parent;
        std::size_t calls = 0;
        std::clock_t cpu_start = 0;
        WallClock::time_point wall_start{};
        double cpu_time = 0.0;
        double wall_time = 0.0;
        double cpu_time_last = 0.0;
        double wall_time_last = 0.0;
        bool running = false;
    };

    std::size_t find_child(std::size_t parent, const std::string &name) const noexcept;
    const Timer *find_timer(const std::string &name) const noexcept;
    void append_timer(std::ostream &output, std::size_t index, int level) const;

    std::vector<Timer> timers_;
    std::vector<std::size_t> active_;
};

//! Exception-safe start/stop pair for a profiler entry.
class ScopedTimer
{
  public:
    ScopedTimer(Profiler &profiler, std::string name, std::string note = {});
    ~ScopedTimer();

    ScopedTimer(const ScopedTimer &) = delete;
    ScopedTimer &operator=(const ScopedTimer &) = delete;

  private:
    Profiler *profiler_;
    std::string name_;
};

namespace global
{
extern Profiler profiler;
}

} // namespace libbse
~~~


<a id="file-src-utils-progress-cpp"></a>
## src/utils/progress.cpp

[打开源码](../src/utils/progress.cpp)。只有 root 打进度，done 没有 Barrier。

### done（16—31 行）

[实现位置](../src/utils/progress.cpp#L16)

~~~cpp
void done(const std::string &description, MPI_Comm comm)
~~~

查询通信域 rank，只有 root 输出从程序静态时钟起点开始的 elapsed 秒；恢复流格式，不含 Barrier。

调用线索：MPI_Comm_rank、count、std::chrono::duration、std::chrono::steady_clock::now、std::cout.flags、std::cout.precision、std::setprecision、std::setw。

显式并行语句（被调函数还可能继续通信）：

- [第 19 行](../src/utils/progress.cpp#L19)：MPI_Comm_rank(comm, &rank);


<a id="file-src-utils-progress-h"></a>
## src/utils/progress.h

[打开源码](../src/utils/progress.h)。公共声明头，实现见同名 .cpp（若有）；通过此头使用接口，不 include .cpp。

本头没有独立函数体，以下给出声明/结构定义以查参数默认值、类型与访问权限；行为见实现文件及主指南。

~~~cpp
#pragma once

#include <mpi.h>

#include <string>

namespace libbse
{

//! Print an cumulative completion marker on communicator rank 0.
void done(const std::string &description, MPI_Comm comm = MPI_COMM_WORLD);

} // namespace libbse
~~~


<a id="file-tests-test-aims-mommat-to-velocity-py"></a>
## tests/test_aims_mommat_to_velocity.py

[打开源码](../tests/test_aims_mommat_to_velocity.py)。独立测试文件，助手不是生产 API。生成非对称 k 网格 HDF5 fixture，检查转换轴序、三角、单位和二进制头。

### read_block（21—30 行）

[实现位置](../tests/test_aims_mommat_to_velocity.py#L21)

~~~python
def read_block(path: Path, ik: int, nk: int, nbands: int) -> np.ndarray:
~~~

Python 测试按 binary v1 偏移读取某 k 的速度块，用 NumPy 解释 complex128 后与预期比较。

调用线索：list、np.fromfile、np.fromfile(stream, dtype='<c16', count=3 * nbands * nbands).reshape、path.open、range、stream.read、stream.seek、struct.unpack。

### main（33—68 行）

[实现位置](../tests/test_aims_mommat_to_velocity.py#L33)

~~~python
def main() -> None:
~~~

生成非对称 k 网格 HDF5 fixture，检查转换轴序、三角、单位和二进制头。测试失败以异常或非零退出报告。

调用线索：(directory / 'band_out').write_text、CONVERTER.convert、Path、complex、h5py.File、int、np.allclose、np.array、np.conjugate、np.ndindex、np.prod、np.zeros、range、read_block、tempfile.TemporaryDirectory。


<a id="file-tests-test-bse-files-cpp"></a>
## tests/test_bse_files.cpp

[打开源码](../tests/test_bse_files.cpp)。独立测试文件，助手不是生产 API。构造临时 QP/aims/Wc，验证窗口、视图与局部 W=V+Wc。

### write_wc（21—29 行）

[实现位置](../tests/test_bse_files.cpp#L21)

~~~cpp
void write_wc(const fs::path &file, int iat, int jat, double value)
~~~

测试辅助函数，写单原子对最小 MatrixMarket Wc，验证局部 reader 只访问所需文件且加裸 V。

调用线索：output、std::runtime_error。

源码错误消息片段：

- "cannot create Wc test file"

### test_coarse_qp_reader（31—71 行）

[实现位置](../tests/test_bse_files.cpp#L31)

~~~cpp
void test_coarse_qp_reader(const fs::path &input_dir)
~~~

构造 energy_qp 与 Dataset 验证 Ha→Ry、核态跳过、选择窗口及直接/间接隙。

调用线索：dataset、expected.size、input_dir.string、libbse::read_qp_bands、librpa_int::MeanField、output、output.close、qp.energies_ry.size、std::abs、std::runtime_error。

源码错误消息片段：

- "cannot create energy_qp test file"
- "coarse-grid quasiparticle dimensions are incorrect"
- "coarse-grid quasiparticle energy is incorrect"
- "coarse-grid quasiparticle gap is incorrect"

### test_aims_reader_view（73—101 行）

[实现位置](../tests/test_bse_files.cpp#L73)

~~~cpp
void test_aims_reader_view(const fs::path &input_dir, const fs::path &output_dir)
~~~

检查 aims 标记识别、原文件/库仑别名/已转换速度链接和源目录标记。

调用线索：band、basis、coulomb、fs::file_size、fs::is_symlink、input_dir.string、libbse::prepare_fhi_aims_reader_view、libbse::resolve_input_format、output_dir.string、std::runtime_error、velocity。

源码错误消息片段：

- "FHI-aims input auto-detection failed"
- "FHI-aims reader compatibility view is incomplete"

### test_aims_reader_rejects_missing_velocity（103—128 行）

[实现位置](../tests/test_bse_files.cpp#L103)

~~~cpp
void test_aims_reader_rejects_missing_velocity( const fs::path &input_dir, const fs::path &output_dir)
~~~

刻意不提供 canonical velocity_matrix，检查 adapter 同步拒绝并提示转换。

调用线索：band、basis、fs::create_directories、input_dir.string、libbse::prepare_fhi_aims_reader_view、output_dir.string、std::runtime_error。

源码错误消息片段：

- "FHI-aims input without velocity data was not rejected"

### main（132—186 行）

[实现位置](../tests/test_bse_files.cpp#L132)

~~~cpp
int main(int argc, char **argv)
~~~

构造临时 QP/aims/Wc，验证窗口、视图与局部 W=V+Wc。测试失败以异常或非零退出报告。

调用线索：MPI_Finalize、MPI_Init、at、error.what、fs::create_directories、fs::remove_all、fs::temp_directory_path、getpid、input_dir.string、libbse::Complex、libbse::read_screened_interaction、screened.at、screened.size、size、std::abs、std::move、std::runtime_error、std::to_string、tensor、test_aims_reader_rejects_missing_velocity、test_aims_reader_view、test_coarse_qp_reader、write_wc。

显式并行语句（被调函数还可能继续通信）：

- [第 134 行](../tests/test_bse_files.cpp#L134)：MPI_Init(&argc, &argv);
- [第 184 行](../tests/test_bse_files.cpp#L184)：MPI_Finalize();

源码错误消息片段：

- "local-pair screened interaction is incorrect"


<a id="file-tests-test-elpa-solver-cpp"></a>
## tests/test_elpa_solver.cpp

[打开源码](../tests/test_elpa_solver.cpp)。独立测试文件，助手不是生产 API。初始化 MPI/LibRPA/ELPA/BLACS，检查固定参考、矩阵性质、TDA/full 残差与规范，最后清理。

### require（20—23 行）

[实现位置](../tests/test_elpa_solver.cpp#L20)

~~~cpp
void require(bool condition, const std::string &message)
~~~

测试助手：条件不成立就抛包含说明的异常，让 CTest 获得非零退出；不属于生产 API。

调用线索：std::runtime_error。

### require_close（25—31 行）

[实现位置](../tests/test_elpa_solver.cpp#L25)

~~~cpp
void require_close(double actual, double expected, double tolerance, const std::string &message)
~~~

测试 double/Complex 两个重载：比较数值误差并给出上下文信息。

调用线索：std::abs、std::runtime_error、std::to_string。

### require_close（33—38 行）

[实现位置](../tests/test_elpa_solver.cpp#L33)

~~~cpp
void require_close(Complex actual, Complex expected, double tolerance, const std::string &message)
~~~

测试 double/Complex 两个重载：比较数值误差并给出上下文信息。

调用线索：std::abs、std::runtime_error。

### make_descriptor（40—47 行）

[实现位置](../tests/test_elpa_solver.cpp#L40)

~~~cpp
librpa_int::ArrayDesc make_descriptor(const librpa_int::BlacsCtxtHandler &blacs, int dimension)
~~~

测试用 BLACS descriptor 工厂，按给定维度构造块循环矩阵，失败抛异常。

调用线索：descriptor、descriptor.init、std::runtime_error。

源码错误消息片段：

- "failed to initialize test matrix descriptor"

### localize（49—69 行）

[实现位置](../tests/test_elpa_solver.cpp#L49)

~~~cpp
std::vector<Complex> localize(const std::vector<Complex> &global, const librpa_int::ArrayDesc &descriptor)
~~~

测试助手：从小 dense 参考提取当前 rank 的块循环元素，用于给求解器喂确定输入。

调用线索：descriptor.indx_l2g_c、descriptor.indx_l2g_r、descriptor.lld、descriptor.m、descriptor.m_loc、descriptor.n、descriptor.n_loc、global.size、require、std::vector。

### globalize（71—94 行）

[实现位置](../tests/test_elpa_solver.cpp#L71)

~~~cpp
std::vector<Complex> globalize(const std::vector<Complex> &local, const librpa_int::ArrayDesc &descriptor)
~~~

测试助手：MPI 汇总小局部矩阵得到完整参考供残差检查；生产流程不使用这种全局化。

调用线索：MPI_Allreduce、descriptor.comm、descriptor.indx_l2g_c、descriptor.indx_l2g_r、descriptor.lld、descriptor.m、descriptor.m_loc、descriptor.n、descriptor.n_loc、global.data、global.size、local.size、require、std::vector。

显式并行语句（被调函数还可能继续通信）：

- [第 91 行](../tests/test_elpa_solver.cpp#L91)：MPI_Allreduce(MPI_IN_PLACE, global.data(), static_cast<int>(global.size()),

### deterministic_hermitian（96—115 行）

[实现位置](../tests/test_elpa_solver.cpp#L96)

~~~cpp
std::vector<Complex> deterministic_hermitian(int dimension)
~~~

构造确定性复 Hermitian 测试 A，避免随机种子影响回归。

调用线索：Complex、std::conj、std::vector。

### deterministic_symmetric（117—131 行）

[实现位置](../tests/test_elpa_solver.cpp#L117)

~~~cpp
std::vector<Complex> deterministic_symmetric(int dimension)
~~~

构造确定性复对称测试 B；允许非实元素，能暴露错误共轭。

调用线索：std::vector、value。

### full_bse_matrix（133—154 行）

[实现位置](../tests/test_elpa_solver.cpp#L133)

~~~cpp
std::vector<Complex> full_bse_matrix(const std::vector<Complex> &a, const std::vector<Complex> &b, int dimension)
~~~

测试中显式组 [[A,B],[-conj(B),-conj(A)]]，作为 full 残差参考。

调用线索：std::conj、std::vector。

### test_distributed_matrix_checks（156—196 行）

[实现位置](../tests/test_elpa_solver.cpp#L156)

~~~cpp
void test_distributed_matrix_checks( const librpa_int::BlacsCtxtHandler &blacs)
~~~

对正确及故意破坏的分布式 A/B 验证 Hermitian/symmetric 检查，涵盖跨 rank 元素。

调用线索：Complex、deterministic_hermitian、deterministic_symmetric、libbse::check_hermitian、libbse::check_symmetric、localize、make_descriptor、require。

### test_skew_solver_reference（198—220 行）

[实现位置](../tests/test_elpa_solver.cpp#L198)

~~~cpp
void test_skew_solver_reference(const librpa_int::BlacsCtxtHandler &blacs)
~~~

使用固定 2×2 A/B 参考，检查 full 正激发能数值。

调用线索：libbse::solve_full_elpa、localize、make_descriptor、require、require_close、solution.energies_ry.size。

### test_tda_solver_residual（222—259 行）

[实现位置](../tests/test_elpa_solver.cpp#L222)

~~~cpp
void test_tda_solver_residual(const librpa_int::BlacsCtxtHandler &blacs)
~~~

检查 ELPA TDA 的 A*v−Ω*v，再检查向连续 pair 振幅的重分布。

调用线索：deterministic_hermitian、distributed、globalize、libbse::redistribute_amplitudes、libbse::solve_tda_elpa、localize、make_descriptor、require_close。

### test_full_solver_residual_and_metric（261—341 行）

[实现位置](../tests/test_elpa_solver.cpp#L261)

~~~cpp
void test_full_solver_residual_and_metric( const librpa_int::BlacsCtxtHandler &blacs)
~~~

检查 full Hamiltonian 残差与 X†X−Y†Y=I，以及 X/Y 两部分重分布。

调用线索：Complex、deterministic_hermitian、deterministic_symmetric、distributed_x、distributed_y、full_bse_matrix、globalize、libbse::redistribute_amplitudes、libbse::solve_full_elpa、localize、make_descriptor、require_close、std::conj。

### main（345—381 行）

[实现位置](../tests/test_elpa_solver.cpp#L345)

~~~cpp
int main(int argc, char **argv)
~~~

初始化 MPI/LibRPA/ELPA/BLACS，检查固定参考、矩阵性质、TDA/full 残差与规范，最后清理。测试失败以异常或非零退出报告。

调用线索：LibRPA_API::finalize、LibRPA_API::initialize、MPI_Allreduce、MPI_Comm_rank、MPI_Finalize、MPI_Init_thread、blacs、blacs.init、blacs.set_square_grid、error.what、std::runtime_error、test_distributed_matrix_checks、test_full_solver_residual_and_metric、test_skew_solver_reference、test_tda_solver_residual。

显式并行语句（被调函数还可能继续通信）：

- [第 348 行](../tests/test_elpa_solver.cpp#L348)：MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
- [第 350 行](../tests/test_elpa_solver.cpp#L350)：MPI_Comm_rank(MPI_COMM_WORLD, &rank);
- [第 378 行](../tests/test_elpa_solver.cpp#L378)：MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
- [第 379 行](../tests/test_elpa_solver.cpp#L379)：MPI_Finalize();

源码错误消息片段：

- "MPI does not provide MPI_THREAD_FUNNELED"


<a id="file-tests-test-molecular-lri-comm-cpp"></a>
## tests/test_molecular_lri_comm.cpp

[打开源码](../tests/test_molecular_lri_comm.cpp)。独立测试文件，助手不是生产 API。跨两个 k1 批次逐元素检查分布式装配和 2/Nk。

### block_value（15—19 行）

[实现位置](../tests/test_molecular_lri_comm.cpp#L15)

~~~cpp
libbse::Complex block_value(int k1, int k2, int i, int j)
~~~

为通信测试按 k1/k2/行/列产生可预测复数值，使每个装配元素可逐项校验。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。

### main（23—114 行）

[实现位置](../tests/test_molecular_lri_comm.cpp#L23)

~~~cpp
int main(int argc, char **argv)
~~~

跨两个 k1 批次逐元素检查分布式装配和 2/Nk。测试失败以异常或非零退出报告。

调用线索：LibRPA_API::finalize、LibRPA_API::initialize、MPI_Allreduce、MPI_Comm_rank、MPI_Comm_size、MPI_Finalize、MPI_Init_thread、blacs、blacs.init、blacs.set_square_grid、block_value、descriptor、descriptor.indx_l2g_c、descriptor.indx_l2g_r、descriptor.init、descriptor.lld、descriptor.m_loc、descriptor.n_loc、error.what、libbse::transform_k_2dlocal、std::abs、std::move、std::runtime_error、std::vector、tensor。

显式并行语句（被调函数还可能继续通信）：

- [第 26 行](../tests/test_molecular_lri_comm.cpp#L26)：MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
- [第 29 行](../tests/test_molecular_lri_comm.cpp#L29)：MPI_Comm_rank(MPI_COMM_WORLD, &rank);
- [第 30 行](../tests/test_molecular_lri_comm.cpp#L30)：MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
- [第 110 行](../tests/test_molecular_lri_comm.cpp#L110)：MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX,
- [第 112 行](../tests/test_molecular_lri_comm.cpp#L112)：MPI_Finalize();

源码错误消息片段：

- "test requires two FUNNELED-capable MPI ranks"
- "failed to initialize communication-test descriptor"
- "transform_k_2dlocal differs from the dense reference"


<a id="file-tests-test-parameter-cpp"></a>
## tests/test_parameter.cpp

[打开源码](../tests/test_parameter.cpp)。独立测试文件，助手不是生产 API。运行参数语法、路径、选项与拒绝测试。

### require（10—13 行）

[实现位置](../tests/test_parameter.cpp#L10)

~~~cpp
void require(bool condition, const std::string &message)
~~~

测试助手：条件不成立就抛包含说明的异常，让 CTest 获得非零退出；不属于生产 API。

调用线索：std::runtime_error。

### test_whitespace_separated_parameters（15—71 行）

[实现位置](../tests/test_parameter.cpp#L15)

~~~cpp
void test_whitespace_separated_parameters()
~~~

覆盖空格语法、主要选项、多通道列表、相对路径和 full 分支。

调用线索：parameter.inp.solve_full、parameter.inp.solve_tda、parameter.inp.spectrum_only、parameter.parse、require、std::vector。

### test_librpa_style_assignments_and_last_value_wins（73—96 行）

[实现位置](../tests/test_parameter.cpp#L73)

~~~cpp
void test_librpa_style_assignments_and_last_value_wins()
~~~

覆盖等号语法、注释、布尔值、重复键最后生效与默认 output_dir。

调用线索：parameter.inp.solve_full、parameter.inp.solve_tda、parameter.parse、require、std::vector。

### test_coarse_kgrid_mode（98—106 行）

[实现位置](../tests/test_parameter.cpp#L98)

~~~cpp
void test_coarse_kgrid_mode()
~~~

验证 bse_use_fine_kgrid=0 可被解析并保留为粗网格模式。

调用线索：parameter.parse、require。

### test_invalid_or_unsupported_parameters_are_rejected（108—156 行）

[实现位置](../tests/test_parameter.cpp#L108)

~~~cpp
void test_invalid_or_unsupported_parameters_are_rejected()
~~~

验证 length gauge、未知/重复通道、非法纯 IPA、未知字段和畸形数字等拒绝路径。

调用线索：parameter.parse、rejected、require。

### main（160—175 行）

[实现位置](../tests/test_parameter.cpp#L160)

~~~cpp
int main()
~~~

运行参数语法、路径、选项与拒绝测试。测试失败以异常或非零退出报告。

调用线索：error.what、test_coarse_kgrid_mode、test_invalid_or_unsupported_parameters_are_rejected、test_librpa_style_assignments_and_last_value_wins、test_whitespace_separated_parameters。


<a id="file-tests-test-profiler-cpp"></a>
## tests/test_profiler.cpp

[打开源码](../tests/test_profiler.cpp)。独立测试文件，助手不是生产 API。验证层次、次数、last 时间、异常清理和格式。

### require（11—14 行）

[实现位置](../tests/test_profiler.cpp#L11)

~~~cpp
void require(bool condition, const char *message)
~~~

测试助手：条件不成立就抛包含说明的异常，让 CTest 获得非零退出；不属于生产 API。

调用线索：std::runtime_error。

### exercise_exception_path（16—21 行）

[实现位置](../tests/test_profiler.cpp#L16)

~~~cpp
void exercise_exception_path(libbse::Profiler &profiler)
~~~

在 ScopedTimer 活动期间抛并捕获异常，检查析构确实停止计时。

调用线索：std::runtime_error、timer。

源码错误消息片段：

- "expected test exception"

### main（25—82 行）

[实现位置](../tests/test_profiler.cpp#L25)

~~~cpp
int main()
~~~

验证层次、次数、last 时间、异常清理和格式。测试失败以异常或非零退出报告。

调用线索：error.what、exercise_exception_path、profiler.get_call_count、profiler.get_cpu_time_last、profiler.get_num_timers、profiler.get_profile_string、profiler.get_wall_time_last、profiler.start、profiler.stop、report.find、require、timer。


<a id="file-tests-test-progress-cpp"></a>
## tests/test_progress.cpp

[打开源码](../tests/test_progress.cpp)。独立测试文件，助手不是生产 API。验证 DONE 格式和只有 root 打印。

### main（10—46 行）

[实现位置](../tests/test_progress.cpp#L10)

~~~cpp
int main(int argc, char **argv)
~~~

验证 DONE 格式和只有 root 打印。测试失败以异常或非零退出报告。

调用线索：MPI_Allreduce、MPI_Comm_rank、MPI_Finalize、MPI_Init、capture.rdbuf、capture.str、error.what、libbse::done、output.empty、output.find、std::cout.rdbuf、std::runtime_error。

显式并行语句（被调函数还可能继续通信）：

- [第 12 行](../tests/test_progress.cpp#L12)：MPI_Init(&argc, &argv);
- [第 14 行](../tests/test_progress.cpp#L14)：MPI_Comm_rank(MPI_COMM_WORLD, &rank);
- [第 42 行](../tests/test_progress.cpp#L42)：MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX,
- [第 44 行](../tests/test_progress.cpp#L44)：MPI_Finalize();

源码错误消息片段：

- "rank 0 DONE output has the wrong format"
- "non-root rank printed a DONE marker"


<a id="file-tests-test-spectrum-mpi-cpp"></a>
## tests/test_spectrum_mpi.cpp

[打开源码](../tests/test_spectrum_mpi.cpp)。独立测试文件，助手不是生产 API。rank 1 非零贡献归约到 root，检查局部振幅读写。

### velocity_index（18—21 行）

[实现位置](../tests/test_spectrum_mpi.cpp#L18)

~~~cpp
std::size_t velocity_index(int direction, int local_pairs)
~~~

返回 direction*local_pairs+local_pair；速度为 direction-major，不能使用振幅的 state-major 索引直接替代。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。

### main（25—108 行）

[实现位置](../tests/test_spectrum_mpi.cpp#L25)

~~~cpp
int main(int argc, char **argv)
~~~

rank 1 非零贡献归约到 root，检查局部振幅读写。测试失败以异常或非零退出报告。

调用线索：MPI_Allreduce、MPI_Barrier、MPI_Bcast、MPI_Comm_rank、MPI_Comm_size、MPI_Finalize、MPI_Init_thread、amplitudes、dipoles.size、error.what、expected、getpid、libbse::make_distributed_amplitudes、libbse::read_distributed_amplitudes、libbse::velocity_gauge_transition_dipoles_mpi、libbse::write_distributed_amplitudes、std::abs、std::filesystem::create_directories、std::filesystem::remove、std::filesystem::temp_directory_path、std::runtime_error、std::sqrt、std::to_string、velocity.values.assign、velocity_index。

显式并行语句（被调函数还可能继续通信）：

- [第 28 行](../tests/test_spectrum_mpi.cpp#L28)：MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
- [第 31 行](../tests/test_spectrum_mpi.cpp#L31)：MPI_Comm_rank(MPI_COMM_WORLD, &rank);
- [第 32 行](../tests/test_spectrum_mpi.cpp#L32)：MPI_Comm_size(MPI_COMM_WORLD, &size);
- [第 75 行](../tests/test_spectrum_mpi.cpp#L75)：MPI_Bcast(&directory_id, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
- [第 80 行](../tests/test_spectrum_mpi.cpp#L80)：MPI_Barrier(MPI_COMM_WORLD);
- [第 95 行](../tests/test_spectrum_mpi.cpp#L95)：MPI_Barrier(MPI_COMM_WORLD);
- [第 105 行](../tests/test_spectrum_mpi.cpp#L105)：MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
- [第 106 行](../tests/test_spectrum_mpi.cpp#L106)：MPI_Finalize();

源码错误消息片段：

- "test requires two FUNNELED-capable MPI ranks"
- "MPI velocity-gauge reduction is incorrect"
- "distributed excitation-amplitude I/O is inconsistent"


<a id="file-tests-test-velocity-gauge-cpp"></a>
## tests/test_velocity_gauge.cpp

[打开源码](../tests/test_velocity_gauge.cpp)。独立测试文件，助手不是生产 API。单 pair 检查复速度、TDA/full、零 gap、强度及 Lorentz 峰。

### close（11—15 行）

[实现位置](../tests/test_velocity_gauge.cpp#L11)

~~~cpp
bool close(const libbse::Complex &actual, const libbse::Complex &expected, double tolerance = 1.0e-13)
~~~

速度规范测试的复数近似比较，返回布尔值。

调用线索：std::abs。

### velocity_index（17—20 行）

[实现位置](../tests/test_velocity_gauge.cpp#L17)

~~~cpp
std::size_t velocity_index(int direction)
~~~

返回 direction*local_pairs+local_pair；速度为 direction-major，不能使用振幅的 state-major 索引直接替代。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。

### main（24—130 行）

[实现位置](../tests/test_velocity_gauge.cpp#L24)

~~~cpp
int main()
~~~

单 pair 检查复速度、TDA/full、零 gap、强度及 Lorentz 峰。测试失败以异常或非零退出报告。

调用线索：close、error.what、libbse::broaden_oscillator_spectrum、libbse::calculate_oscillator_strengths、libbse::velocity_gauge_transition_dipole、spectrum.size、std::abs、std::conj、std::polar、std::runtime_error、std::sqrt、strengths.size、velocity.values.assign、velocity_index。

源码错误消息片段：

- "TDA velocity-gauge contraction is incorrect"
- "full-BSE velocity-gauge Y contraction is incorrect"
- "velocity-gauge conjugation is incorrect"
- "oscillator-strength prefactor is incorrect"
- "full-BSE band-gauge covariance is broken"
- "Lorentz-broadened spectrum is incorrect"
- "zero KS gap was not rejected"


<a id="file-tools-aims-mommat-to-velocity-py"></a>
## tools/aims_mommat_to_velocity.py

[打开源码](../tools/aims_mommat_to_velocity.py)。独立转换器，依赖 NumPy/h5py；CLI 或 import 后 convert(Path,Path) 使用。

### _exact_integer（29—33 行）

[实现位置](../tools/aims_mommat_to_velocity.py#L29)

~~~python
def _exact_integer(value: float, description: str) -> int:
~~~

把 HDF5 数值验证为整数（1e-8 误差），返回 Python int；拒绝不合法值。

调用线索：ValueError、abs、float、np.isfinite、round。

### _read_band_dimensions（36—46 行）

[实现位置](../tools/aims_mommat_to_velocity.py#L36)

~~~python
def _read_band_dimensions(input_dir: Path) -> tuple[int, int, int, int]:
~~~

读 band_out 前四个正整数 nk,nspin,nbands,nao，要求 nspin=1，供动量转换验证。

调用线索：(input_dir / 'band_out').read_text、(input_dir / 'band_out').read_text(encoding='utf-8').split、ValueError、any、int、len、tuple。

### _read_momentum（49—95 行）

[实现位置](../tools/aims_mommat_to_velocity.py#L49)

~~~python
def _read_momentum( input_dir: Path, dimensions: tuple[int, int, int, int] ) -> tuple[np.ndarray, int, int]:
~~~

从 mommat.h5 读窗口、k_points、Momentummatrix；检查形状，反转三网格轴，按 0/1-based k 索引重新排序；返回 packed 数据和窗口边界。

调用线索：ValueError、_exact_integer、enumerate、h5py.File、int、k_points.transpose、k_points[..., 0].reshape、momentum.reshape、momentum.transpose、np.allclose、np.arange、np.array_equal、np.asarray、np.asarray(handle['Energy_window']).reshape、np.empty、np.prod、np.rint、np.rint(k_points[..., 0]).astype、np.rint(k_points[..., 0]).astype(np.int64).reshape、np.sort。

### _velocity_blocks（98—128 行）

[实现位置](../tools/aims_mommat_to_velocity.py#L98)

~~~python
def _velocity_blocks( packed_momentum: np.ndarray, dimensions: tuple[int, int, int, int], state_min: int, state_max: int, ) -> np.ndarray:
~~~

将 packed gradient 乘 -i 得速度，恢复上三角和共轭下三角、实对角，窗口外填零，再转 eV·Å；返回小端 complex128 块。

调用线索：np.conjugate、np.triu_indices、np.zeros、range。

### convert（131—171 行）

[实现位置](../tools/aims_mommat_to_velocity.py#L131)

~~~python
def convert(input_dir: Path, output_file: Path) -> None:
~~~

转换完整 aims 导出目录，输出 canonical binary v1；同目录临时文件+fsync+原子替换，拒绝符号链接目标。成功无返回值。

调用线索：ValueError、_read_band_dimensions、_read_momentum、_velocity_blocks、input_dir.resolve、np.ascontiguousarray、np.ascontiguousarray(block, dtype='<c16').tobytes、os.fsync、os.replace、os.unlink、output.fileno、output.flush、output.write、output_file.absolute、output_file.is_symlink、output_file.parent.mkdir、range、struct.pack、tempfile.NamedTemporaryFile。

### main（174—189 行）

[实现位置](../tools/aims_mommat_to_velocity.py#L174)

~~~python
def main(argv: list[str] | None = None) -> int:
~~~

解析输入目录与可选输出路径，默认写 input_dir/velocity_matrix；调用 convert，成功 0，常见读写/数据错误返回 1。

调用线索：argparse.ArgumentParser、convert、parser.add_argument、parser.parse_args、print。


<a id="file-tools-compare-ri-coeff-debug-py"></a>
## tools/compare_ri_coeff_debug.py

[打开源码](../tools/compare_ri_coeff_debug.py)。只比较实部的 RI 系数诊断工具，函数主要接收 Entry 字典和 CscMeta。基组元数据需匹配真实体系。

### CheckResult.percent（91—92 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L91)

~~~python
def percent(self) -> float:
~~~

只读属性：100*correct/checked；没有检查样本时为 0。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。

### parse_args（95—180 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L95)

~~~python
def parse_args() -> argparse.Namespace:
~~~

读取 RI 调试工具 CLI；包含输入目录、CSC/text、每原子基数、容差、过滤上限和重排诊断开关。

调用线索：argparse.ArgumentParser、parser.add_argument、parser.parse_args。

### read_max_n_basis_sp（183—194 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L183)

~~~python
def read_max_n_basis_sp(case_dir: Path) -> int | None:
~~~

从 case_dir/check.txt 中匹配 max_n_basis_sp 调试行，找不到返回 None；main 在无显式参数时依赖此值。

调用线索：check_path.exists、check_path.open、int、match.group、pattern.search、re.compile。

### is_close（197—198 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L197)

~~~python
def is_close(a: float, b: float, atol: float, rtol: float) -> bool:
~~~

比较 abs(a-b)≤atol+rtol*abs(b)，b 是参考值，尺度不对称。

调用线索：abs。

### count_skip（201—213 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L201)

~~~python
def count_skip(stats: LoadStats, reason: str) -> None:
~~~

按原因给 LoadStats 对应过滤计数加一；不修改 Entry。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。

### parse_int_list（216—223 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L216)

~~~python
def parse_int_list(text: str, label: str) -> list[int]:
~~~

解析逗号分隔正整数，用于每原子 AO/辅助基数；空或非正值直接 SystemExit。

调用线索：SystemExit、any、int、part.strip、text.split。

### offsets（226—230 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L226)

~~~python
def offsets(counts: list[int]) -> list[int]:
~~~

对每原子计数形成前缀偏移，输出首项 0，长度与输入相同。

调用线索：result.append。

### atom_from_aux（233—237 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L233)

~~~python
def atom_from_aux(row: int, aux_offsets: list[int], aux_per_atom: list[int]) -> int | None:
~~~

由 1-based 全局辅助基行号找 0-based 原子，范围外返回 None。

调用线索：enumerate、zip。

### parse_real_token（240—249 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L240)

~~~python
def parse_real_token(token: str, max_abs: float) -> tuple[float | None, str | None]:
~~~

读取浮点文本，支持其实现中的 Fortran 指数处理，过滤非法/非有限/超过 max_abs 的值；返回 value 与跳过原因。

调用线索：abs、float、math.isfinite、token.replace、token.replace('D', 'E').replace。

### read_mine_text（252—279 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L252)

~~~python
def read_mine_text(mine_dir: Path, pattern: str, max_abs: float) -> tuple[list[Entry], list[Path], LoadStats]:
~~~

按 glob 读本地 RI 调试文本，提取 k/aux/basis1/basis2/value 和来源位置；按规则过滤，返回记录、文件列表和统计。

调用线索：Entry、LoadStats、any、count_skip、entries.append、enumerate、file.open、int、len、line.split、mine_dir.glob、parse_real_token、range、sorted、str、tuple。

### parse_csc_k（282—286 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L282)

~~~python
def parse_csc_k(path: Path) -> int:
~~~

从 lvl_tricoeff_tmp_k_N.csc 文件名解析 k，格式不匹配报错。

调用线索：CSC_K_RE.match、ValueError、int、match.group。

### parse_text_k（289—293 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L289)

~~~python
def parse_text_k(path: Path) -> int:
~~~

从 lvl_tricoeff_tmp_k_N[_row_M].txt 名解析 k，支持文本分片。

调用线索：TEXT_K_ROW_RE.match、ValueError、int、match.group。

### read_one_elsi_csc（296—334 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L296)

~~~python
def read_one_elsi_csc(path: Path) -> tuple[int, int, list[tuple[int, int, float]]]:
~~~

按 ELSI 二进制头、列指针、行索引和值读取单 CSC 文件，返回矩阵阶数、nnz、坐标值；比较流程只使用实部。

调用线索：ValueError、int、len、path.read_bytes、range、records.append、struct.unpack_from。

### read_reference_csc（337—391 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L337)

~~~python
def read_reference_csc( case_dir: Path, pattern: str, max_n_basis_sp: int, basis_per_atom: list[int], aux_per_atom: list[int], max_abs: float, ) -> tuple[list[Entry], list[Path], LoadStats, CscMeta]:
~~~

批读参考 CSC，根据每原子基数把列打包中的局部 basis1 转全局，过滤 padding 和非法值，返回 Entries/文件/统计/CscMeta。

调用线索：CscMeta、Entry、LoadStats、abs、atom_from_aux、case_dir.glob、count_skip、entries.append、math.isfinite、offsets、parse_csc_k、read_one_elsi_csc、sorted、str。

### read_reference_text（394—461 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L394)

~~~python
def read_reference_text( case_dir: Path, pattern: str, max_n_basis_sp: int, basis_per_atom: list[int], aux_per_atom: list[int], max_abs: float, ) -> tuple[list[Entry], list[Path], LoadStats, CscMeta]:
~~~

读取参考文本或分片，统一为四元键和 CscMeta，保留能完整表示的 basis2 信息。

调用线索：Counter、CscMeta、Entry、LoadStats、atom_from_aux、case_dir.glob、count_skip、dict、entries.append、enumerate、file.open、int、len、line.split、offsets、parse_real_token、parse_text_k、sorted、str、sum。

### unique_map（464—488 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L464)

~~~python
def unique_map(entries: list[Entry], stats: LoadStats, atol: float, rtol: float) -> dict[tuple[int, int, int, int], Entry]:
~~~

按四元键去重，相近重复与冲突重复分别统计；冲突处理按函数实现，不把重复当新的独立样本。

调用线索：conflicted.values、is_close、kept.get、len、sum。

### csc_col（491—493 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L491)

~~~python
def csc_col(key: tuple[int, int, int, int], max_n_basis_sp: int) -> int:
~~~

按 col=(basis2−1)*max_n_basis_sp+basis1 得 1-based CSC 列号；用于约定的局部 key。

函数体没有识别到显式调用，通过直接索引/算术/返回完成。

### csc_col_from_global_key（496—504 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L496)

~~~python
def csc_col_from_global_key(key: tuple[int, int, int, int], meta: CscMeta) -> int | None:
~~~

利用原子元数据将全局 basis1 还原为对应局部列打包，无法表示返回 None。

调用线索：atom_from_aux。

### key_is_representable（507—517 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L507)

~~~python
def key_is_representable(key: tuple[int, int, int, int], meta: CscMeta) -> bool:
~~~

检查四元键能否落在参考 CSC/text 表示范围；用于区分隐式零与根本不能表示的项。

调用线索：csc_col_from_global_key、meta.n_by_k.get、sum。

### hist（520—522 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L520)

~~~python
def hist(keys: list[tuple[int, int, int, int]], index: int) -> list[tuple[int, int]]:
~~~

对四元 key 的指定位置做 Counter 并按频数输出，供 k/AO/辅助基误差直方图。

调用线索：Counter、counts.items、sorted。

### print_stats（525—536 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L525)

~~~python
def print_stats(label: str, stats: LoadStats) -> None:
~~~

打印加载、过滤、同值/冲突重复计数，不执行比较。

调用线索：print。

### print_key_examples（539—546 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L539)

~~~python
def print_key_examples(title: str, keys: list[tuple[int, int, int, int]], entries: dict[tuple[int, int, int, int], Entry], max_report: int) -> None:
~~~

按 max_report 打印特定键集、数值及源文件行号，辅助定位差异。

调用线索：print。

### atom_aux_range（549—552 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L549)

~~~python
def atom_aux_range(atom: int, meta: CscMeta) -> range:
~~~

返回指定 0-based 原子对应的 1-based 全局辅助基 range。

调用线索：range。

### atom_basis_range（555—558 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L555)

~~~python
def atom_basis_range(atom: int, meta: CscMeta) -> range:
~~~

返回指定原子的 1-based 全局 AO range。

调用线索：range。

### atom_from_basis（561—565 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L561)

~~~python
def atom_from_basis(basis: int, basis_offsets: list[int], basis_per_atom: list[int]) -> int | None:
~~~

由 1-based 全局 AO 编号反查 0-based 原子，范围外 None。

调用线索：enumerate、zip。

### build_aux_permutation（568—666 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L568)

~~~python
def build_aux_permutation( ref_map: dict[tuple[int, int, int, int], Entry], mine_map: dict[tuple[int, int, int, int], Entry], meta: CscMeta, k_filter: int, min_abs: float, atol: float, rtol: float, ) -> None:
~~~

在同 k/AO 对和同原子内按实数值寻找辅助基匹配，按命中数/误差贪心一一配对；剩余按序补齐并标 residual。实际返回 AuxPermutationResult，源码 ->None 注解与返回值不一致。

调用线索：AuxPermutationResult、Counter、abs、atom_aux_range、atom_from_aux、candidates.append、candidates.sort、defaultdict、is_close、len、list、mine_by_column.get、mine_by_column[k_point, basis1, basis2].append、mine_map.values、pair_hits.items、pairs.sort、range、ref_map.values、residual_rows[atom].add、set、used_mine.add、used_ref.add、zip。

### infer_aux_permutation（669—727 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L669)

~~~python
def infer_aux_permutation( ref_map: dict[tuple[int, int, int, int], Entry], mine_map: dict[tuple[int, int, int, int], Entry], meta: CscMeta, k_filter: int, min_abs: float, atol: float, rtol: float, ) -> None:
~~~

调用 build_aux_permutation，打印 strong/weak/residual 证据、映射和自检；不改输入文件。

调用线索：', '.join、','.join、Counter、atom_aux_range、build_aux_permutation、len、list、print、range、result.inferred[atom].get、result.inferred[atom].values、sorted、str、target_counts.items、validate_aux_permutation。

### validate_aux_permutation（730—771 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L730)

~~~python
def validate_aux_permutation( ref_map: dict[tuple[int, int, int, int], Entry], mine_map: dict[tuple[int, int, int, int], Entry], inferred: dict[int, dict[int, int]], meta: CscMeta, k_filter: int, atol: float, rtol: float, ) -> None:
~~~

应用候选辅助基映射，再比较参考记录，打印正确/错误/缺失计数，防止仅看命中数下结论。

调用线索：atom_from_aux、inferred.get、inferred.get(atom, {}).get、is_close、mine_map.get、print、ref_map.values。

### check_reference_against_mine（774—820 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L774)

~~~python
def check_reference_against_mine( ref_map: dict[tuple[int, int, int, int], Entry], mine_map: dict[tuple[int, int, int, int], Entry], aux_map: dict[int, dict[int, int]], meta: CscMeta, k_filter: int, atol: float, rtol: float, target_atom: int | None = None, pair_map: dict[tuple[int, int], tuple[int, int]] | None = None, aux_pair_map: dict[tuple[int, int, int], tuple[int, int]] | None = None, ) -> CheckResult:
~~~

应用辅助基映射及可选目标原子 AO 对映射，返回 checked/correct/wrong/missing 的 CheckResult。

调用线索：CheckResult、atom_from_aux、aux_map.get、aux_map.get(aux_atom, {}).get、aux_pair_map.get、is_close、mine_map.get、pair_map.get、ref_map.values。

### print_check_result（823—827 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L823)

~~~python
def print_check_result(label: str, result: CheckResult) -> None:
~~~

输出一次映射自检结果和百分比。

调用线索：print。

### collect_wrong_after_mapping（830—862 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L830)

~~~python
def collect_wrong_after_mapping( ref_map: dict[tuple[int, int, int, int], Entry], mine_map: dict[tuple[int, int, int, int], Entry], aux_map: dict[int, dict[int, int]], meta: CscMeta, k_filter: int, atol: float, rtol: float, target_atom: int, pair_map: dict[tuple[int, int], tuple[int, int]], ) -> list[tuple[float, Entry, Entry | None, tuple[int, int]]]:
~~~

收集给定映射后的错误/缺失项及误差；缺失以无穷误差表示，供后续排序诊断。

调用线索：abs、atom_from_aux、aux_map.get、aux_map.get(aux_atom, {}).get、is_close、mine_map.get、pair_map.get、ref_map.values、wrong.append。

### infer_o_basis_pair_relocation（865—1047 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L865)

~~~python
def infer_o_basis_pair_relocation( ref_map: dict[tuple[int, int, int, int], Entry], mine_map: dict[tuple[int, int, int, int], Entry], meta: CscMeta, k_filter: int, aux_min_abs: float, pair_min_abs: float, min_hits: int, target_atom_1based: int, max_report: int, atol: float, rtol: float, ) -> None:
~~~

对目标原子的大错误样本进一步推断 AO 对位置，比较映射前后并列剩余差异；默认原子 2，不是通用自动修复。

调用线索：' '.join、Counter、SystemExit、abs、atom_aux_range、atom_basis_range、atom_from_aux、atom_from_basis、aux_hist.most_common、aux_map.get、aux_map.get(aux_atom, {}).get、aux_pair_hits.items、aux_pair_hits.most_common、aux_pair_map.get、build_aux_permutation、candidates.append、candidates.sort、check_reference_against_mine、choices.sort、collect_wrong_after_mapping、defaultdict、is_close、len、math.isinf、min、mine_by_k_aux.get、mine_by_k_aux[k_point, mine_aux].append、mine_map.get、mine_map.values、pair_hist.most_common、pair_hits.items、pair_hits.most_common、pair_map.get、print、print_check_result、ref_map.values、set、sorted。

### main（1050—1243 行）

[实现位置](../tools/compare_ri_coeff_debug.py#L1050)

~~~python
def main() -> int:
~~~

组织读取和比较；普通比较有 wrong/mine-only/ref-only 返回 1。推断模式完成返回 0，不意味着全部数据正确或已修复。

调用线索：' '.join、', '.join、Entry、SystemExit、abs、args.case_dir.resolve、args.mine_dir.resolve、case_dir.glob、case_dir.is_dir、correct.append、csc_meta.n_by_k.values、csc_meta.nnz_by_k.values、hist、infer_aux_permutation、infer_o_basis_pair_relocation、is_close、key_is_representable、len、max、min、mine_dir.is_dir、parse_args、parse_int_list、print、print_key_examples、print_stats、read_max_n_basis_sp、read_mine_text、read_reference_csc、read_reference_text、ref_map.get、set、sorted、sum、unique_map、wrong.append。

