# W 全对称恢复回归测试

测试使用真实 Si 3×3×3、694 个辅助基函数的 aims 导出数据。`all` 分支只读取 4 个代表点 W，通过 Δ 恢复 27 个 q 点；`none` 分支读取未约化 W。两次 BSE 使用同一程序、KS、QP 和 4v3c 能带窗口，不重新运行 DFT/GW。

[执行脚本](../backend/run_w_all_symmetry.py) 注册两项测试：

- `LibBSE_regression_w_all_symmetry_mpi1`：单进程。
- `LibBSE_regression_w_all_symmetry_mpi2`：双进程，覆盖 MPI 原子对分配与结果收集。

每项先运行 `none` 和 `all`，再检查：

1. 全部 324 个激子能量，绝对误差不超过 `2e-6 eV`。
2. 激子振幅的归一化、正交性，以及去掉整体复相位／简并子空间旋转后的匹配，使用已有回归比较器的 `2e-4` 容差。
3. 三个方向各自的 Im ε，频率区间 0–12 eV、步长 0.0025 eV、Lorentz 半宽 0.10 eV；每个方向的相对 L2 误差不超过 `2e-5`。不只比较方向平均或未归一化吸收强度。
4. 新运行的 `none` 与冻结的未约化 BSE 参考比较，避免两个读取分支同时发生相同错误却通过测试。

介电函数由 `momentum_strength_singlet_tda.dat` 重建，采用 singlet 自旋简并约定：

\[
f_{S\alpha}=\frac{4|P_{S\alpha}|^2}{N_k\Omega_S^{\rm Ha}},\qquad
\operatorname{Im}\varepsilon_\alpha(E)=\frac{2\pi^2}{V_{\rm bohr^3}}
(E_{\rm Ha\to eV})^2\sum_S\frac{f_{S\alpha}}{\Omega_S^{\rm eV}}
\frac{\gamma/\pi}{(E-\Omega_S^{\rm eV})^2+\gamma^2}.
\]

## 数据与运行

真实输入约数 GB，作为外部 fixture，不随 Git 发布。fixture 根目录需要：

```text
aims_export/                  # 同一套 KS、RI、Coulomb、velocity_matrix、geometry.in
aims_none/                    # 未约化 periodic_gw_w_q_*_rank_*.dat 和 energy_qp
aims_all/                     # 代表点 W、aims_symmetry_all.dat、aims_delta_q_*.dat
bse_none/libbse.d/             # 冻结的未约化 BSE 能量、所有 rank 振幅、momentum_strength
```

在仓库根目录，设置 `W_FIXTURES` 为上述数据目录后运行：

```bash
cmake -S . -B build -DLIBBSE_W_ALL_SYMMETRY_FIXTURES="$W_FIXTURES"
cmake --build build -j 4
ctest --test-dir build -L w_all_symmetry --output-on-failure
```

没有配置外部目录时不注册这两项测试，因此普通 GitHub CI 不会自动运行真实 Si 333 回归。配置后缺失／不兼容的文件会失败，不跳过。小矩阵的 Δ 旋转、反幺正共轭、坏文件拒绝，以及 W 读取串行／MPI 测试仍在默认快速测试中运行。

每次运行在 `build/w_all_symmetry_runs` 下创建独立目录，保留输入、输出、执行日志和可执行文件 SHA256；`inputs.sha256` 记录数据来源，`comparison.txt` 记录误差和通过状态，`*_dielectric.dat` 保存对照曲线。不会更新冻结参考，也不生成 JSON 计算记录。

这些测试验证固定 Si 数据下的读取与恢复，不代表磁性、SOC 或任意材料对称性已验证。
