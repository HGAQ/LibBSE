# Real Si: 222 grid, 2v2c

该数据集用于固定输入的软件回归，不是 Si 激发能／光谱的收敛基准。CI 直接读取已生成的电子结构数据，不下载或运行 FHI-aims。

## 输入来源

- 两原子 Si 原胞、PBE、scalar atomic ZORA、2×2×2 k 网格。
- 轨道与积分设置来自 FHI-aims `species_defaults/defaults_2020/light/14_Si_default`。
- 为缩小回归数据，辅助基设置为 `default_prodbas_acc 1e-3`、`default_max_l_prodbas 3`；导出和原生 GW 阶段使用相同设置。这不再是未经修改的 light 辅助基。实际为 50 个轨道基函数、218 个辅助基函数。
- 两阶段的实际输入：[export](producer/export/control.in)、[GW](producer/gw/control.in)、[geometry](producer/gw/geometry.in)。
- QPE 来自这次原生 AIMS GW，未施加额外 scissor。
- BSE 默认选 2 条价带和 2 条导带：8×2×2=32 个电子–空穴跃迁。额外有 1v1c 算例。
- W 与 χ₀ 使用 AIMS 导出的最低虚频切片，保持原频率约 0.00266388359 Ha，**不是严格零频 W／χ₀**。原计算使用 16 个频率点。

压缩包包含 C、完整与截断 V、KS 能量／波函数、速度矩阵、QP 能量、W 和 χ₀；没有 FHI-aims 源码或可执行程序。W／χ₀ 的文本条目用 NumPy float64 数组无损保存，运行时恢复文本，并逐值检查恢复前后相同。输入不裁剪到 2v2c：head/wing 的求和仍需要其他 KS 态。

输入归档约 **18.9 MiB**，参考输出约 **18.4 MiB**，合计约 **37.2 MiB**。原始 light 辅助基有 428 维；更激进的 116 维试算虽压到 5.7 MiB，但产生负激发能，因此未作为本数据集发布。

## 实际运行的八项测试

| 名称 | 输入与检查 |
|---|---|
| `native_w` | 原生 AIMS W＋QPE → 完整 BSE 构造、32 个激发能／振幅、光谱 |
| `chi0_headwing` | 原生 χ₀＋head/wing → W → BSE；比较 W、激发能、振幅、光谱 |
| `dynamic` | χ₀ 路径，plasma energy=15 eV、运行中计算 binding energy |
| `native_w_1v1c` | 同一份数据，缩小为 8 个电子–空穴跃迁，检查带窗口选择 |
| `native_w_unbatched` | 关闭 RI 分批优化，与 `native_w` 冻结参考比较 |
| `native_w_mpi2` | 2 MPI 进程，与 `native_w` 单进程参考比较 |
| `chi0_headwing_mpi2` | 2 MPI 进程，检查 head/wing、W 与 BSE 的进程数一致性 |
| `boundary` | 在原始 222 网格开启 boundary，与关闭时参考比较 |

χ₀／dynamic 测试还将诊断矩阵重建为 W，再读取 W 求 BSE，比较两条路径的激发能和振幅。动态重放用 NumPy 独立实现有效介电矩阵公式。

`native_w` 与 `chi0_headwing` 各有自己的参考，不假定 AIMS 原生 W 与 LibBSE 重建 W 完全相等。boundary 测试验证原粗网格上的不变性，不证明细网格插值收敛。

## 比较标准和边界

- 激发能绝对差 ≤2×10⁻⁶ eV。
- 振幅按整体相位／简并子空间对齐，误差 ≤2×10⁻⁴，同时检查归一化和正交性。
- W 按 q／频率匹配，相对 Frobenius 误差 ≤2×10⁻⁸。
- 光谱能量网格相同，谱值整体相对误差 ≤2×10⁻⁵。

参考源码和工具链记录见 [reference_version.toml](reference_version.toml)。参考由固定源码版本生成，代表初始回归基线，不是独立高精度物理解。CI 不允许自动更新参考文件。此节目前不覆盖真实 full-BSE、triplet、ABACUS 和 coarse→fine 插值；这些不能由八项测试的通过推断。

## 运行与维护

算例参数在 [cases.toml](cases.toml)，执行入口为 [run_real_si.py](../../backend/run_real_si.py)。从仓库根目录运行：

```bash
ctest --test-dir build -L real_si --output-on-failure
```

归档校验值在 `archives.txt`。输入内部包含逐文件校验及生产程序哈希。更新参考时应先独立审核输入、源码和数值变化，再显式重新生成归档并更新校验值；不能因测试失败直接接受新结果。
