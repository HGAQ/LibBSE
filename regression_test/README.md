# LibBSE regression tests

本目录替代原 `tests`，保留原有小型测试，并增加固定输入的真实算例回归。现有功能清单见 [UNIT_TESTS.md](UNIT_TESTS.md)。

## 覆盖什么

算例清单见 [cases.json](cases.json)，对应 [已完成的模板算例](../../../3_LibBSE_templatetest/README.md)。

| 用户关注的功能 | 算例与检查 |
|---|---|
| 给定 W、QPE 求 BSE | AIMS 原生 full W、AIMS/ABACUS 的 LibRPA Wc；比较全部 singlet TDA eigenvalue、amplitude |
| 给定 χ₀、QPE 求 W 和 BSE | AIMS χ₀、ABACUS/LibRPA χ₀；按 q 和频率比较完整 W，再比较 eigenvalue、amplitude |
| dynamic | AIMS/ABACUS 的 plasma energy 15 eV，以及 AIMS 20 eV；比较动态结果，并用 NumPy 独立构造有效 W 后重放 BSE |
| χ₀ 的 coarse-grid 计算 | AIMS、ABACUS 222；读粗网格 QPE，关闭 fine-grid 模式 |
| W 的 coarse-grid 计算与插值 | 222 同网格；AIMS、ABACUS 333→444 |
| AIMS/ABACUS 与版本差异 | 各自产物分别比较冻结程序和候选程序；历史模板差异另列，不要求两个独立电子结构计算数值相等 |

这里不重新运行 DFT/GW，也不改历史文件。输入使用已完成算例；运行前后记录并复核输入文件 SHA256。尚未增加真实 χ₀ 的粗细网格插值案例；不要把 W 的 333→444 案例称为 χ₀ 插值验证。

当前端到端算例只覆盖 singlet TDA；full BSE 的求解器残差和度量归一化由现有小型 CTest 覆盖，不能据此声称完成了真实材料 full-BSE 回归。

## 如何运行

先配置、编译项目，再运行默认的小型测试：

```sh
cmake --build build -j4
ctest --test-dir build --output-on-failure --output-junit ctest.xml
```

运行真实数据测试需要指定两个外部位置：完成的模板数据，以及冻结的参考可执行文件。候选程序由当前构建提供。

```sh
cmake -S . -B build \
  -DLIBBSE_REGRESSION_FIXTURES=../../3_LibBSE_templatetest \
  -DLIBBSE_REGRESSION_REFERENCE_EXE=../../regression_work_20261029/LibBSE.baseline
cmake --build build -j4
ctest --test-dir build -L integration --output-on-failure
```

上面的路径以源码根目录为运行目录，指向本地已完成模板和本次冻结程序；其他机器需提供各自的固定数据与参考程序。也可单独运行：

```sh
python3 regression_test/run_case.py \
  --fixtures ../fixtures \
  --reference-exe ../reference/LibBSE \
  --candidate-exe build/LibBSE \
  --work build/regression_runs \
  --case aims_chi0_coarse
```

可分别指定 `--reference-ranks` 和 `--candidate-ranks`。新结果不覆盖旧结果，重复运行自动创建带时间标识的目录。需要计算节点、可用的 MPI/ELPA 和 Python NumPy；真实数据未随源码分发，未配置时这些测试不注册。不能把“未注册”报告为“已通过”。

## eigenvalue 与 amplitude 怎么比

[compare_outputs.py](compare_outputs.py) 读取 `Excitation_Energy_singlet.dat`（Ry，转为 eV），按 rank 拼接每个状态的 `Excitation_Amplitude_singlet_*.dat`，允许两次计算使用不同 MPI 数量。

- 能量最大绝对误差上限：2×10⁻⁶ eV。
- 非简并态：先对齐整体复相位，再比较 amplitude 的 L² 差异。
- 简并组：按参考能量、组内跨度不超过 10⁻⁵ eV 分组，比较 QR 正交化后的子空间主角，不要求每条本征矢相同。
- amplitude／子空间误差上限：2×10⁻⁴，同时检查归一化。阈值考虑现有 amplitude 文本输出的 8 位精度及输出零截断，不代表能量收敛精度。
- 输入、gauge、带窗口和 transition 顺序必须相同；不同生产者的 amplitude 不能直接套这个比较器。

[test_compare_outputs.py](test_compare_outputs.py) 专门检查比较器能接受任意相位／简并态旋转，并能拒绝能量变化、跨不同能量的混合、错误归一化和 W 变化。

## W 与 dynamic 怎么比

χ₀ 案例启用 `out_screening_matrices`。诊断文件包含 q、频率、√Vcut、ε⁻¹。比较器重建：

```text
W = sqrt(Vcut) × epsilon_inverse × sqrt(Vcut)
```

按 q 和频率匹配，而不是按文件编号匹配，要求覆盖相同；相对 Frobenius 容差为 2×10⁻⁸。只读 W 的案例使用同一份给定 W，检查的是 BSE 消费路径，不声称重新生成了 W。

χ₀ 案例还将诊断矩阵转换为可读取的 W 文件（严格零频使用 full-W binary；AIMS 非零最低虚频使用保留原频率的文本 W），再直接给 BSE 求解，要求 eigenvalue 和 amplitude 与原 χ₀ 路径一致。

动态模型为：

```text
epsilon_inverse = X diag(d) X†
f(d) = 1 − P(1−d)/(P + Eb sqrt(1−d))
W_effective = sqrt(Vcut) X diag(f(d)) X† sqrt(Vcut)
```

P 对应 `bse_plasma_energy_ev`。**当前生产接口不支持直接输入 Eb**：Eb 从静态最低激发能和直接 QP 带隙计算。重放测试读取该 Eb，用 NumPy 独立对角化和计算有效 W，再通过直接读取有效 W 的路径求 BSE。已有 `test_chi0_screening.cpp` 还直接给定 P、Eb，检查公式、Eb=0 静态极限和酉变换协变性。

重放测试验证实现一致性，不证明动态模型的物理精度。

## 参考版本与历史结果

冻结参考程序与候选程序的回归通过，只说明固定输入下保持一致。首次建立测试时，冻结程序取自迁移测试目录前的当前工作区构建；两边物理源码相同，属于建立可复现基线，不能冒称独立求解器正确性证明。

历史模板另生成 `historical_template` 数值报告。它们可能与当前程序存在真实版本差异，包括 AIMS head/wing 行列处理；不能自动把旧结果设成金标准，也不能静默覆盖差异。不同 AIMS/ABACUS 基组、带窗口、W 生成方式的差异不属于同输入软件回归。

## GitHub 接入范围

默认 CTest 可在依赖就绪的 CI 环境运行；真实算例需要另外准备固定数据和参考程序。本次完成本地编译与回归框架，不会自动上传生产数据、安装集群 runner 或开启远程工作流。日后 CI 应固定 LibRPA/LibRI/ELPA 版本，保留 JUnit、comparison.json 和输入哈希，且不得把缺少 fixture 当作 PASS。

GitHub 自动运行的配置和首次启用步骤见 [GITHUB_ACTIONS.md](GITHUB_ACTIONS.md)。工作流已写入仓库，尚未部署 runner 或进行远程验证。
