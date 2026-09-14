# FHI-aims → LibRPA → LibBSE 数据路径

[English](fhi_aims_io.md)

该接口保留 LibRPA/LibRI 中的多体计算，仅添加读取 FHI-aims 文件所需的
数据生产程序专用转换；它不是一套独立实现的 BSE 方法。

## RI 约定与屏蔽相互作用

FHI-aims 导出局域辅助基组展开：

```text
phi_i*(k+q,r) phi_j(k,r)
  = sum_mu Ctilde_ij^mu(k+q,k) P_mu^{q*}(r),

C_mn^mu(k+q,k)
  = sum_ij c_i,m*(k+q) c_j,n(k) Ctilde_ij^mu(k+q,k).
```

LibRPA 在该辅助基组中构造独立粒子响应：

```text
chi0_mu,nu(q,iw) = sum_(s,m,n,k) w_k
  C_mn,s^mu(k+q,k) C_nm,s^nu(k,k+q)
  / (epsilon_m,s(k+q) - epsilon_n,s(k) - iw),
```

并使用对称化介电矩阵：

```text
epsilon(q,0) = I - V^(1/2)(q) chi0(q,0) V^(1/2)(q),
W(q,0)       = V^(1/2)(q) epsilon^(-1)(q,0) V^(1/2)(q).
```

独立运行的 LibRPA 输出相关部分
`Wc(q,iw) = W(q,iw) - V(q)`。在严格静态极限下，由 Fourier 变换的线性性
可得：

```text
W_mu,nu(R,0) = V_mu,nu(R) + Wc_mu,nu(R,0),
W_mu,nu(R,0) = (1/Nq) sum_q exp(-i q.R) W_mu,nu(q,0).
```

因此，LibBSE 读取实空间 `Wc_Mu_*` 块，加上 `FT_Vq` 生成的实空间裸相互
作用，将两者映射到相同的最近 Born--von Karman 晶胞，然后把得到的静态极限
`W(R,iw_0)` 传给 LibRI。BSE 直接项由两个 transition density 与静态 W
收缩得到；交换/Hartree 项使用对应的裸 V。统一的 `1/Nk` Fourier 归一化和
Hartree 到 Rydberg 的单位转换，只在 LibRI k 空间块写入分布式 BSE 矩阵时
施加一次。

这条路径避免导出稠密的四指标屏蔽电子-空穴张量，程序只保留
`W(R,iw_0)` 的局域辅助基组块。

在跃迁基 `(i k -> a k)` 中，LibRI kernel 实现的两个辅助基组收缩对应于：

```text
<i k, a k | V | j k', b k'>
  = sum_(mu,nu) C_ia^mu(k,k)* V_mu,nu(q=0) C_jb^nu(k',k'),

<i k, j k' | W | a k, b k'>
  = sum_(mu,nu) C_ij^mu(k,k')* W_mu,nu(k'-k) C_ab^nu(k,k').
```

第一个收缩是排斥的交换/Hartree 项，其自旋系数取决于计算通道；第二个收缩
以吸引的负号进入共振 BSE 块。在实空间表示中，LibRI 对 R 的相位求和恢复
动量转移。因此 FHI-aims 只需导出局域 vertex，LibRPA 只需导出 W 的辅助
基组块；IO 层不会写出或重建稠密的 `(i,a,j,b,k,k')` 对象。

LibRPA 的对称性设置必须与 FHI-aims 导出一致。示例 Si 输入使用
`periodic_gw_optimize_kgrid_symmetry inverse`；此时程序根据
`bz_sampling_out` 通过共轭恢复 q/-q，LibRPA 的点群对称性开关保持关闭。
如果 FHI-aims 使用 `all`，则 LibRPA 必须同时设置
`use_symmetry_gw = t` 和 `use_symmetry_exx = t`，分别把相关与交换路径恢复到
完整 Born--von Karman 网格；`none` 导出也关闭这两个开关。不能仅根据 SCF
k 点数判断。
`i_state_low`、`i_state_high` 应限制在 BSE 能带窗口内（边界从零开始、左闭
右开），因为更远的 QP 根既不需要也更不稳定。LibBSE 在对角化前会报告 A 的
Hermitian 和 B 的对称性诊断；偶数 k 网格在当前有限网格存储约定下可以合理地
使 B 呈现非 Hermitian/非对称性质。

当前 LibRPA MatrixMarket writer 对 Minimax 虚频网格采样，而该网格不包含
严格的零频。因此 `ifreq_0` 是最低正频率节点，LibBSE 把
`W(R,iw_0)` 用作静态极限近似。实际频率记录在每个 MatrixMarket 文件头中。
使用 `option_dielect_func = 0` 时，LibRPA 的 `nfreq` 必须与 FHI-aims 的
`frequency_points` 相等；需要检查该近似时应同步增加二者并做收敛测试。接口
不会把该节点静默标记成严格零频值。

## 能带规范

对于能带相位变换
`c_i,n(k) -> exp(i theta_n(k)) c_i,n(k)`，MO RI vertex 会获得相反的
bra/ket 相位。这些相位与电子-空穴基的变换在 BSE kernel 中相消。因此
FHI-aims 数据以其 native gauge 使用。

把每条能带分别对齐到它与 k=0 处同一能带的 overlap，在简并点尤其不安全：
简并子空间中允许的规范变换是酉旋转，而不是单个标量相位。

`wavefunction_gauge auto` 对 FHI-aims 选择 `native`，对其他 LibRPA 数据集
保留历史 `first_k` 约定。诊断时也可以显式指定该选项。

## FHI-aims 文件与兼容视图

在已扩展的 FHI-aims 接口中使用 `output librpa bse`。该预设生成规则网格
KS 本征矢、折叠后的 RI 系数、Coulomb 矩阵和压缩存储的
`mommat_ks_kpt_*.dat`；LibRPA reader 会自动识别所支持的 legacy binary 或
v1 容器。标准 FHI-aims 命令
`compute_momentummatrix Emin Emax 0` 还可以为规则网格的全部 k 点生成
`mommat.h5`。示例中的 `0` 是当前接口提供的全 k 点扩展；对于未修改的团簇
计算，应按 FHI-aims 文档使用 k 点 1。

`tools/aims_mommat_to_velocity.py` 读取 `mommat.h5` 中单位为 bohr^-1 的三个
笛卡尔梯度矩阵分量，并应用 `v = p = -i nabla`。脚本将 h5py 看到的 Fortran
HDF5 网格轴序 `(kz,ky,kx)` 显式反转为 `(kx,ky,kz)`，由压缩的能带上三角
重建 Hermitian 矩阵，再按公共 reader 所需的 eV*angstrom 单位写出 LibRPA v1
`velocity_matrix`。运行 LibBSE 前必须先执行该脚本；兼容视图只链接已经存在的
标准 `velocity_matrix`，不再包含另一套转换实现。缺少速度数据会直接报错。
速度矩阵是光学/BSE 输入，不是 GW 输入。

FHI-aims 将截断矩阵命名为 `coulomb_cut_*`，而发布版 LibRPA reader 使用旧
前缀 `coulomb_unshrinked_cut_*`。LibBSE 在
`output_dir/fhi_aims_reader_view` 中建立旧名称的符号链接，不会重命名或修改
源导出数据。

## 准粒子能量

在这条流程中，FHI-aims 只负责产生 DFT/RI/动量数据，不是 LibBSE 使用的 QP
数据源。独立 LibRPA 读取 FHI-aims 导出，执行 G0W0，同时写出 `energy_qp`
和静态屏蔽相互作用块。LibBSE 使用 `qp_format energy_qp` 读取该文件。已经
删除直接解析 FHI-aims `GW_band*.out` 的路径，因此 LibRPA 与 LibBSE 使用
完全相同的能带顺序、单位约定和 GW 结果。

全电子 KS 文件可能包含芯态。`energy_qp` 中的占据数确定占据窗口；LibBSE
选择最后 `nocc` 个占据态和最前 `nvirt` 个空态，并对 KS 本征矢和速度矩阵
应用相同的能带偏移。

## 三阶段运行流程

1. 使用 3x3x3 `k_grid`、`output librpa bse` 和动量导出运行 FHI-aims。该
   步骤只提供 DFT、RI、Coulomb、波函数和速度数据。
2. 把 `mommat.h5` 转成 LibBSE 的标准 `velocity_matrix`，然后以
   `output_energy_qp = t`、`output_wc_rf = t` 和
   `ifreq_output_wc_end = 1` 独立运行 LibRPA。使用
   `option_dielect_func = 0` 与 `replace_w_head = t` 修正 FHI-aims 介电矩阵头；
   令 `nfreq` 与 FHI-aims 频率网格一致，并把 QP 状态范围限制在 BSE
   能带窗口。示例的 `inverse` 导出关闭 LibRPA 点群对称性；如果 FHI-aims
   使用 `all`，则同时开启 `use_symmetry_gw` 和 `use_symmetry_exx`。对于示例
   中的大辅助基组，使用
   `sqrt_coulomb_threshold = 1e-8` 去除数值零 Coulomb 通道，并使用
   `option_qpe_solver = 2` 选择与 KS 态连续相连的微扰根，避免跳到远处卫星峰。
3. 使用 `input_format fhi_aims`、`qp_format energy_qp` 运行 LibBSE；将
   `qp_data` 指向 LibRPA 的 `energy_qp`，将 `screened_dir` 指向 LibRPA 的
   `Wc_Mu_*` 输出目录。

可移植的输入模板位于 `examples/fhi_aims_si_333`。
