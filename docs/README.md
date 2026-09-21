# LibBSE 文档

第一次接手代码，先读 **[代码上手与修改指南（中文）](code_guide_zh.md)**。

| 文档 | 查什么 |
|---|---|
| [代码上手与修改指南](code_guide_zh.md) | 架构、构建、全部输入参数、数学公式、数据布局、MPI/OpenMP、修改入口和验证方法 |
| [逐函数参考](code_functions_zh.md) | 本项目 218 个命名函数定义/重载：签名、作用、调用线索、源码行号、显式并行语句 |
| [逐文件清单](code_files_zh.md) | 基准版本 551 个受版本控制文件，含第三方源码/测试/资源及定义入口 |
| [外部接口与实现导航](code_external_zh.md) | 实际 LibRPA/LibRI 接口、向下调用链、ELPA/数学库/MPI、LibComm/cereal 和 Python 依赖 |
| [FHI-aims 数据接入（中文）](fhi_aims_io_zh.md) | aims 导出、动量转换和读取视图 |
| [FHI-aims data interface（English）](fhi_aims_io.md) | 英文数据接入说明 |

开发指南基于 2026-09-14 本地源码及相邻依赖实现。当前构建使用的外部库路径与本仓库 thirdparty 副本不同，修改前先读指南第 2 节；OpenMP 编译配置的实际发现见第 9.1 节。

本次文档工作没有修改生产代码，也没有运行材料计算。文档已检查文件链接、行号范围、章节锚点和本项目函数/文件覆盖。调用索引为词法线索，不替代编译器对模板、重载和宏条件的解析。
