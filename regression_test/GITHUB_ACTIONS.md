# GitHub-hosted 自动编译与测试

[工作流](../.github/workflows/regression.yml) 使用 GitHub 提供的 `ubuntu-24.04` 临时虚拟机。无需在集群注册 runner，无需 `LIBBSE_CI_ENV`、SSH 密钥或集群文件访问权限。

## 启用步骤

1. 将更新的 workflow、[ELPA 构建脚本](build_ci_elpa.sh) 和配套文档一起提交、推送。
2. 仓库 Settings → Actions → General 中允许使用本工作流的 GitHub 官方 actions；通常默认已允许。
3. 打开 Actions → LibBSE regression，查看 `Comparator self-tests` 与 `Build and fast tests`。两者成功才说明本轮 GitHub 快速检查通过。
4. 可在分支规则中要求 `Build and fast tests` 通过后才能合并。

触发事件为 push、pull_request 和 workflow_dispatch。本地 commit 不会触发远程运行；一次 push 多个 commit 时测试最新提交。每次运行结束上传日志和 JUnit，失败时也尝试上传。没有使用 `pull_request_target`。

## 自动执行的工作

- 安装 Ubuntu 的 GCC/GFortran、OpenMPI、OpenBLAS、ScaLAPACK 和构建工具。
- 下载兼容的 LibRPA：`HGAQ/LibRPA_LibBSE`，固定 commit `7551e28df26175167eb2415e5dbbaf5bec7d1ce6`；不跟随 master 漂移。
- 下载 ELPA 2024.05.001，核对 SHA256，从源码启用 OpenMP 和 skew-symmetric 接口。LibBSE 的 full-BSE 路径需要这个接口。
- 使用仓库内的 LibRI、LibComm、cereal，Release 编译 LibBSE 和测试程序。
- 执行 24 项快速 CTest，包括 MPI、ELPA、χ₀、文件读取、转换器和比较器自检。

为控制内存，编译并行度为 2，CTest 顺序执行。MPI 允许 oversubscription，使四进程／线程测试可在标准 runner 的 CPU 数量内完成。Ubuntu 工具链与本地集群不同，第一次云端构建仍需核实兼容性；不能把之前的集群通过结果当作云端实测。

## 尚未包含的完整算例

十组真实算例没有自动上传，也不在该工作流中运行。完整回归需要另行准备可下载、允许公开的数据包与固定参考结果，并核实标准 runner 的内存和磁盘容量。缺少它们时，快速检查通过不代表真实算例回归通过。

现有 [ci.sh](ci.sh) 保留为本地完整回归入口，GitHub workflow 不再调用它。历史 [VALIDATION.md](VALIDATION.md) 记录的是集群实测。

## 配置验证状态

已核实固定 LibRPA commit 在公开仓库可访问，下载并核对 ELPA 发布包 SHA256，检查 shell 语法和比较器自检。新工作流尚未在 GitHub Ubuntu 上完成编译验证。推送后应以 Actions 的实际日志为准。

参考：[标准 runner](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)、[工作流触发](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)。
