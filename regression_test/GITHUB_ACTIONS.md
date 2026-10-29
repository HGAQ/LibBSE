# GitHub 自动回归

[工作流](../.github/workflows/regression.yml) 已配置 `push`、`pull_request` 和手动触发。普通本地 commit 不会触发 GitHub；push 后测试推送的最新提交。一次 push 包含多个 commit 时，不会逐个测试中间提交。要逐个测试，应逐次 push。本工作流不取消前一次运行。

## 两个检查

- `Comparator self-tests`：GitHub Ubuntu 机器运行 Python 比较器自检；每次 push 和 PR 都执行。这不包含 C++ 编译和 BSE 求解。
- `Build and full regression`：专用 runner 编译候选程序，运行快速测试与十组真实算例，比较 eigenvalue、amplitude、W；每次仓库 push 或手动触发执行。PR 事件本身不执行此项，但仓库内分支的 push 会执行。外部 fork 的代码不能直接在持有本地计算数据的 runner 上运行。

完整测试没有在线 runner 时会排队，不会退化为只跑比较器并称为完整通过。

## 首次启用

1. 将本轮 `regression_test`、CMake 修改和 `.github/workflows/regression.yml` 一起提交并推送到 GitHub。仅上传 workflow 不够。
2. 在仓库 Settings → Actions → Runners → New self-hosted runner 注册专用 Linux x64 机器，增加标签 `libbse`。按 GitHub 页面给出的命令安装并启动 runner。不要将计算任务直接放在集群登录节点；若使用 Slurm，应先获得计算资源，再在该分配内启动 runner，或另外实现提交和等待 Slurm 的桥接脚本。当前脚本直接在 runner 所在机器运行。
3. 机器准备至少 8 个可用 CPU、建议 48 GB 内存，并安装已经验证的 GCC、OpenMPI、CMake/CTest、OpenBLAS、ScaLAPACK、ELPA、Python ≥3.11 和 NumPy。准备固定版本的本地扩展 LibRPA、LibRI、LibComm、cereal；不能直接假定任意上游 LibRPA 可替代。
4. 在 runner 工作区之外创建环境文件，加载编译器／MPI 环境并设置下表变量。在仓库 Settings → Secrets and variables → Actions → Variables 中添加 `LIBBSE_CI_ENV`，其值为该环境文件在 runner 上的实际路径。
5. 推送一个提交，在 Actions 查看两个检查。完整测试成功后下载 `regression-…` artifact，其中包含 JUnit 和比较结果。

| 环境文件变量 | 含义 |
|---|---|
| `LIBRPA_SOURCE` | 固定的 LibRPA 源码目录，包含本地文件读取接口 |
| `LIBRI_HEADERS` | LibRI 的 include 目录 |
| `ELPA_PREFIX` | ELPA 安装前缀 |
| `REGRESSION_FIXTURES` | 固定的模板数据目录，结构对应 [cases.json](cases.json) |
| `REGRESSION_REFERENCE` | 已审核的冻结 LibBSE 可执行文件 |
| `REGRESSION_REFERENCE_SHA256` | 上述程序的 SHA256；脚本在编译前核对 |

变量指向 runner 上的实际路径，不能使用 GitHub Ubuntu 机器访问不到的本地文件。模板输入中的路径也必须在 runner 上有效。首次冻结基线的物理验证范围见 [VALIDATION.md](VALIDATION.md)。基线不能随每次候选编译自动更新，否则真实回归会被掩盖。

[ci.sh](ci.sh) 在独立 `build-ci` 目录编译；测试失败返回非零，GitHub 显示失败。没有改动生产源码。依赖和 fixture 应由管理员固定保存，更新时另行记录版本、输入哈希与审核理由。

## 合并门槛

启用并观察完整检查后，可在仓库分支规则中要求 `Build and full regression` 通过才允许合并。当前外部 fork PR 没有完整检查：维护者需审核代码，将其放到仓库内的测试分支触发 push 测试。不要改用 `pull_request_target` 自动执行未审核代码。

## 本轮验证范围

已检查 shell 语法并运行比较器自检（6/6）。底层本地测试此前已通过；新 GitHub 工作流尚未在远程执行，也尚未注册 runner。首次远程运行仍需验证全新检出、依赖路径、MPI 资源与 artifact 上传。

参考：[触发事件](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)、[添加 runner](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)、[runner 安全边界](https://docs.github.com/en/actions/concepts/security/compromised-runners)。
