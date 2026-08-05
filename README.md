# NsysScope

基于内置的 `sglang-nsys-static-analysis` skill，提供交互式的 SGLang Nsight Systems
分析能力。

## 便携式可执行文件

轻量发布版是一个独立的 Linux 自解压文件：

```bash
chmod +x nsysscope-linux-x86_64.run
./nsysscope-linux-x86_64.run
```

它包含预构建的前端页面、Analyzer 源码和一份经过校验的兜底 Skill。不包含
Node.js、npm 依赖包、任务数据、Provider 凭证或 NVIDIA Nsight Systems。首次启动
会在用户缓存目录下创建一个可复用的 Python 环境。仅做"导入结果"用途只需要
Python 3；要分析新的 `.nsys-rep`，还需要 `nsys` 和一个已登录的 Codex 或 Comate
Provider。

构建便携版：

```bash
python3 scripts/build_portable.py
```

## 一条命令启动

```bash
cd /home/users/zhongyuanming/NsysScope
./nsysscope start
```

该命令会自动：

- 检查 Python、Nsight Systems、至少一个 Agent Provider 以及分析 skill 是否就位；
- 需要时创建或更新那个小型 Python 环境；
- 把预构建的前端页面和 Analyzer 作为同一个 FastAPI 进程启动；
- 自动挑选一个可用的本机端口；
- 按 `Ctrl-C` 时清理临时状态。

正常本地使用不需要 Node.js 和 npm。它们只在你要重新构建已提交的前端产物，或
部署托管版 demo 时才用得上。

打开命令打印出的地址，通常是：

```text
http://127.0.0.1:3000
```

如果是通过 SSH 连接的服务器，保持 `./nsysscope start` 运行，并在本地电脑上执行
打印出来的单端口 SSH 转发命令。远端 Analyzer 端口和本地浏览器端口是特意分开的，
默认本地端口是 `远端端口 + 10000`，避免和本地 3000 端口冲突。需要时可以用
`NSYSSCOPE_SSH_LOCAL_PORT` 覆盖。浏览器已经不再需要填写 Analyzer URL 或
API token。

本机页面还会根据 hostname 自动识别本地 Analyzer，忽略过期的远程 API 配置，并
自动重试瞬时性的启动失败。前端首页不会被缓存，所以新启动的服务不会误用旧的
托管模式连接配置。

单独运行环境检查：

```bash
./nsysscope doctor
```

缺少 `nsys` 或 Agent Provider 不再会阻止启动，页面仍然可以用来导入已有的六表
分析包。

## 可替换的分析 Skill

NsysScope 内置了一份精简版 Skill，同时也支持跟踪一个外部维护的 Skill 目录，而
不需要把它复制进应用里：

```bash
./nsysscope skill status
./nsysscope skill use /path/to/sglang-nsys-static-analysis
./nsysscope skill sync
./nsysscope skill validate
./nsysscope skill reset
```

执行 `skill use` 之后，对那个外部目录的修改会在下次启动时生效。每次新分析都会
把选用的 Skill 路径、来源和 SHA256 记录到 `metadata/skill.json`。解析优先级依次
是：显式设置的 `NSYSSCOPE_SKILL_DIR`、配置的外部路径、用户 Codex 安装的副本、
内置的兜底版本。

## Agent Provider

NsysScope 支持两种可互换的分析 Provider：

- **Codex CLI**，通过非交互式的 `codex exec`；
- **Comate Zulu CLI**，通过非交互式的 `zulu run`。

两个 Provider 接收相同的 prompt 和任务材料，激活同一个
`sglang-nsys-static-analysis` skill，并且必须产出相同的六表分析包，通过相同的
确定性校验，页面才会接受结果。

对每一个新的组合式/异构模型，Skill 会先建立一份针对当前模型的架构分类体系。每
个算子和功能阶段都带有其结构位置、具体 unit ID 和架构变体。聚合方式是
`(位置, unit, 变体, 功能模块)`，所以像 `KDA,KDA,KDA,MLA` 这样的混合模式不会被
折叠成一个笼统的 Attention 平均值。页面会把完整周期和每一个结构单元都展示为
独立视图。细粒度的源码归因保留在 `module` 字段里；功能视图默认每个变体呈现
5–8 个架构阶段，避免投影、归一化、门控、缓存操作和路由细节把对比层面淹没。

每个新生成的 CSV 都以一行总计结尾。算子/分类/阶段的总计代表累计 GPU 耗时，在
重叠场景下可能超过墙钟时间。算子总览表还会在紧凑算子名之前保留原始的
`module` 列。

Provider 是否就位会显示在"新建分析"对话框里。需要登录时执行：

```bash
./nsysscope login codex
./nsysscope login comate internal
```

登录后重启 `./nsysscope start`，再在任务表单里选择 **Codex CLI** 或
**Comate Zulu**。启动脚本会自动发现随已安装的 Comate 插件一起分发的 Zulu
可执行文件；可以用 `NSYSSCOPE_COMATE_BIN` 覆盖该路径。默认 Comate 平台是
`internal`；公网 SaaS 场景使用 `./nsysscope login comate saas`。内部版 Comate
命令会绕开 shell 的 HTTP 代理，因为内部服务地址是可以直连的。启动脚本还会给
Zulu 传入所需的 `PLATFORM` 参数，让 `status` 和 `run` 能复用 `login` 写入的
token。`NSYSSCOPE_COMATE_PLATFORM`、`NSYSSCOPE_COMATE_MODEL` 和
`NSYSSCOPE_COMATE_TIMEOUT_SECONDS` 分别控制平台、可选模型和超时时间。

任务表单里还有一个"Agent 基座模型"选择框：

- Codex 的候选项来自当前 CLI 的本地模型缓存，"自动"选项会保留
  `~/.codex/config.toml` 里配置的模型；
- Comate 的候选项通过 `zulu model list --ids` 从已登录账号拉取；
- 选中的值只会随这次任务一起保存，并只在这次运行时传给对应 Provider（即
  `codex --model ... exec` 或 `zulu run --model ...`）。

这个任务级别的选择不会改写任一 Provider 的全局模型配置。

## 创建一次分析

点击"新建分析"，然后提供：

- `.nsys-rep` 或已导出的 `.sqlite`；
- 模型的 `config.json`；
- 真实的部署 YAML 或启动脚本；
- 对应的 SGLang/模型源码根目录；
- 模型名称、推理阶段和硬件；
- 一个空的或尚未创建的结果目录；
- 可选的设计文档；
- 分析范围和硬性验收标准。

范围约束是强制性的。如果请求了某个具体的 layer 子类型或分支，Agent 必须精确
选中那个单元，或者带着证据失败退出；不能悄悄改用一个更宽泛的架构周期替代。举
例来说，请求分析单独一个 GLM5.2 非共享 Indexer 层时，不能改成分析四层的
全量/共享 Indexer 周期。

路径解析发生在运行 NsysScope 的这台机器上。默认情况下，工具只允许访问本仓库
所在目录之下的材料。便携版 `.run` 文件则使用启动时所在的目录作为范围，而不是
它私有的解压目录。需要时可以覆盖这个范围：

```bash
NSYSSCOPE_ALLOWED_ROOTS=/reports:/model-source ./nsysscope start
```

结果目录是可移植的存储单元。一次成功的运行会写出：

```text
result-package/
├── analysis.json
├── nsysscope-package.json
├── csv/                 # 六张规范化的 CSV 表
├── xlsx/                # 每张 CSV 对应一份 XLSX 工作簿
├── trace/               # 导出或复制的 SQLite trace
├── logs/job.log         # 有大小上限的 Agent/任务日志
└── metadata/            # 请求、prompt、manifest 和校验证据
```

默认情况下，`./nsysscope start` 把任务索引、锁文件和运行日志放在一个临时运行
目录里，工具停止时会一起删除。结果包是唯一会持久保留的任务数据。

持久化任务历史是可选功能，需要显式开启：

```bash
NSYSSCOPE_PERSIST_STATE=true \
NSYSSCOPE_DATA_DIR=/path/to/small-state-dir \
./nsysscope start
```

即使开启了持久化模式，这个目录也只存放轻量的任务元数据；CSV、XLSX、SQLite 和
任务日志仍然留在各自的结果包里。

Agent 日志默认有大小上限：

- Comate 使用 Zulu 的 `task-json` 模式，只保留最终任务结果，不会保留累积的会话
  快照；
- Agent 静默期间每 30 秒记录一次轻量心跳；
- 单条日志记录上限 16 KiB，单个任务日志总大小上限 2 MiB；
- 页面按字节偏移增量读取日志，不会加载整个文件。

这些限制可以通过 `NSYSSCOPE_AGENT_HEARTBEAT_SECONDS`、
`NSYSSCOPE_JOB_LOG_LINE_MAX_BYTES` 和 `NSYSSCOPE_JOB_LOG_MAX_BYTES` 调整。

## 导入已有结果

使用"导入结果目录（无需 Agent）"，填入包目录或 ZIP 文件路径。如果目录里已经有
`analysis.json`，NsysScope 会直接打开；如果只有规范化的六张 CSV，NsysScope 会
自动构建 `analysis.json`。新版的 `csv/` 子目录布局和旧版的平铺六表目录都支持。
"导入 JSON"只用于纯浏览器端的单次预览。

ZIP 导入会要求指定一个新的结果目录，解压前会先校验路径，并且只会把规范化的
CSV、XLSX、分析数据、可选的 trace、metadata 和有大小上限的日志写入那个指定
目录。

六张 CSV 文件本身就足够使用。Manifest、语义映射、校验报告和统计 sidecar 能
提升可追溯性，但对于快速导入来说是可选的。缺少这些 sidecar 时，分类信息会从
核心计算表和辅助算子表里恢复，剩余的行会被视为通信类别，导入的包也会被标记
为"缺少外部校验证据"。

## 可选的 Codex 插件

经过校验的插件源码位于 `plugins/nsysscope`。它把同一个分析 Skill 提供给
Codex，并附带一个启动适配器；独立可执行文件仍然是主要的使用入口，同时也支持
Comate 以及无需 Agent 的导入场景。
