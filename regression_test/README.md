# LibBSE regression tests

测试分为人工小算例和真实材料算例。测试通过表示对应固定输入与检查项通过，不等于所有 BSE 功能或物理近似都已验证。

| 目录 | 用途 |
|---|---|
| [general_testcase](general_testcase) | 24 项快速 CTest 的 C++／Python 小算例：求解器、屏蔽公式、文件转换、光谱和 MPI |
| [physical_testcases](physical_testcases) | 真实 Si 输入、冻结参考输出和 TOML 算例配置 |
| [backend](backend) | 执行、结果比较、报告汇总、依赖安装脚本 |
| [doc](doc) | 测试说明、GitHub 配置和验证记录 |

配置采用 TOML；数值数据采用压缩归档／NumPy；报告采用文本和 JUnit XML。数值比较会处理本征态整体复相位和简并态子空间，检查归一化、正交性与非有限值，不直接逐元素比较未对齐的振幅。

## 运行

在仓库根目录编译后：

```bash
python3 -m pip install -r regression_test/backend/requirements.txt
ctest --test-dir build -LE 'integration|real_si' --output-on-failure
ctest --test-dir build -L real_si --output-on-failure
```

第二条执行快速测试，第三条执行仓库自带的真实 Si 222、2v2c 测试。运行需要 MPI、ELPA 和配套编译依赖；Python 至少为 3.11。真实 Si 测试默认注册，可用 `-DLIBBSE_TEST_REAL_SI=OFF` 关闭。

原先十组外部模板回归仍保留，配置见 [templates.toml](physical_testcases/templates.toml)。通过 CMake 的 `LIBBSE_REGRESSION_FIXTURES`、`LIBBSE_REGRESSION_REFERENCE_EXE` 注册，标签为 `integration`。该流程使用外部冻结程序；自带 Si 流程使用归档的参考数值，CI 不会自动重建参考答案。

## 文档

- [快速测试清单](doc/UNIT_TESTS.md)
- [真实 Si 测试与数据来源](physical_testcases/si222/README.md)
- [GitHub Actions](doc/GITHUB_ACTIONS.md)
- [Si 222 验证记录](doc/SI222_VALIDATION.md)
- [初次本地回归验证记录](doc/VALIDATION.md)
