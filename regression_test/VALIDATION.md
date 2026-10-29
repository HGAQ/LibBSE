# 2026-10-29 本地验证

本轮采用 [模板算例](../../../3_LibBSE_templatetest)，将原 `tests` 目录迁到 `regression_test`，保留已有测试。`src` 文件逐个 SHA-256 核对，与本轮开始时一致；已有工作区修改未撤销。

## 实际执行结果

| 检查 | 结果 |
|---|---|
| Release 编译 | 通过 |
| 快速 CTest（含比较器自检） | 24/24 通过 |
| 固定输入集成回归 | 10/10 通过 |
| χ₀ / dynamic 生成 W 后再读取 W 求 BSE | 5/5 通过 |
| 真实算例通过 CTest 入口运行 | ABACUS Wc 算例通过 |
| ABACUS 静态 χ₀，参考 1 MPI 进程、候选 2 MPI 进程 | 通过 |

十组当前基线对照中，激发能在文本输出精度内一致。MPI 进程数对照的最大激发能差为 0 eV，最大振幅子空间误差约 5.37×10⁻⁸，W 最大相对误差约 5.38×10⁻¹⁴。振幅比较处理整体复相位与简并态旋转，同时检查归一化、正交性。

初次 MPI 启动配置遇到 Slurm slots／绑定问题。最终使用每个 MPI 进程分配 `OMP_NUM_THREADS` 个核心、绑定核心的配置；完成任务为 4969365_8。未修改数值算法解决启动问题。

## 基线与历史结果须分开理解

冻结参考程序和候选程序使用相同物理源码，本轮首先建立可复现的回归基线。通过不代表已经独立证明物理实现正确，也不代表与所有历史结果一致。

历史 AIMS 模板通过本次比较阈值。三个历史 ABACUS 模板有差异：激发能最大差约 1.7–2.1×10⁻⁶ eV，振幅子空间误差约 0.00269–0.0669，超过 2×10⁻⁴ 阈值。扩大简并态分组不能完全消除差异，因此尚不能归因为简并态任意旋转；本轮没有放宽阈值或修改物理源码消除差异。

## 覆盖边界

覆盖 AIMS / ABACUS 输入路径、给定 W 与 QPE、给定 χ₀ 与 QPE、动态屏蔽、粗网格计算和 W 的 3³→4³ 插值。当前动态生产接口由静态计算确定 binding energy；直接给定 binding energy 的公式检查由已有单元测试覆盖。尚未增加真实 χ₀ 粗到细网格插值算例。集成算例为 singlet TDA；full-BSE 的求解器性质由已有测试检查。

GitHub 远程工作流尚未部署；测试目录、可选 CTest 集成入口、JUnit 报告已准备好。外部大文件与冻结参考程序需要在运行环境中另行提供。

## 详细记录

- [使用方法与容差](README.md)
- [逐项集成结果](../../../regression_work_20261029/INTEGRATION_RESULTS.md)
- [集成 JUnit](../../../regression_work_20261029/integration.xml)
- [快速 CTest JUnit](../../../regression_work_20261029/final_ctest.xml)
- [MPI 进程数对照](../../../regression_work_20261029/mpi_bound/abacus_chi0_coarse/comparison.json)
- [历史 ABACUS 进一步检查](../../../regression_work_20261029/historical_abacus_diagnostic.json)
