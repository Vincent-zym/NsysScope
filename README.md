# NsysScope

面向 SGLang 推理框架的交互式 GPU 性能分析工具。核心分析逻辑封装在
`bundled/skills/sglang-nsys-static-analysis` 这个 skill 里，NsysScope 本身是
它的可视化前端与任务编排层——两者是解耦的：skill 可以脱离本工具、交给任意支持
skill 机制的 Agent CLI（Codex、Comate Zulu）独立运行，产出的结果包能被本工具
直接导入查看；也可以在本工具里一键新建分析，由工具驱动 Agent 跑完整流程。

## 这个工具做什么

把一份 Nsight Systems 的 `.nsys-rep`（或已导出的 `.sqlite`）转换成一套结构化
的性能分析包：

- 按重复结构单元（layer/block）拆解耗时,正确处理 KDA/MLA/NSA/MoE 等异构层
  混合架构——不会被错误折叠成一个没有物理意义的平均值
- 把 CUDA kernel 映射回具体的 Python 调用栈与源码位置
- 区分核心计算 / 通信 / 辅助算子，计算关键 GEMM 的 MFU/MBU
- 生成一份人读的分析报告（结论 + 优化建议 + 详细数据），而不是只给一堆表格
- 提供交互式前端查看结果、发布分享链接

## 快速开始

**便携版可执行文件**（推荐，无需 Node.js/npm）：

```bash
chmod +x nsysscope-linux-x86_64.run
./nsysscope-linux-x86_64.run
```

首次启动会在用户缓存目录下创建一个可复用的 Python 环境。仅导入已有结果只需
Python 3.10+；分析新的 `.nsys-rep` 还需要 `nsys` 和一个已登录的 Codex 或 Comate
Provider。

`.run` 文件是构建产物，未提交到仓库（见 `.gitignore` 里的 `release/*.run`）。
从源码重新构建：

```bash
python3 scripts/build_portable.py
```

生成的文件位于 `release/nsysscope-linux-x86_64.run`，只包含运行时后端、已构建
的前端页面和内置 skill，不含 Node.js、npm 依赖或任务数据。

**固定下载链接（自动更新到最新构建）**：

```text
# 完整工具
https://github.com/Vincent-zym/NsysScope/releases/latest/download/nsysscope-linux-x86_64.run

# 只要分析 skill（单独使用，不需要本工具）
https://github.com/Vincent-zym/NsysScope/releases/latest/download/sglang-nsys-static-analysis.zip
```

`.github/workflows/release-run.yml` 在每次 push 到 `main` 时跑 backend 测试、
skill evals、`skill_manager validate`，全部通过后才构建这两个产物、把 `latest`
tag 强制指向本次提交并替换 GitHub Release 里的对应文件——所以这两个链接本身
永远不用改，之后每次拿到的都是当时 `main` 上最新的、通过全部校验的构建。
CI 任一步失败都不会更新链接，上一份能用的构建会继续留在原地。

skill 的 zip 解开后是 `sglang-nsys-static-analysis/`，直接放进 Agent CLI 的
skills 目录即可使用；本地重新打包：

```bash
python3 scripts/build_skill_zip.py
```

这个 zip 是可复现构建——条目排序固定、时间戳归零，同一个 commit 无论在本地还是
CI 打包，产出的字节完全一致，方便核对拿到的是不是预期的那一版。

**从源码启动**：

```bash
cd /path/to/NsysScope
./nsysscope start
```

该命令会自动完成环境检查、启动 FastAPI 服务并选择可用端口，浏览器打开打印出
的地址（通常是 `http://127.0.0.1:3000`）即可使用。单独执行环境检查：

```bash
./nsysscope doctor
```

远程服务器场景下，保持 `./nsysscope start` 运行，在本地执行打印出的 SSH 转发
命令即可通过浏览器访问。

## 环境依赖与首次配置

`./nsysscope start` 会自己准备 Python 依赖，所以正常情况下不需要手工装包。它需要
机器上先有这几样东西：

- `python3` **3.10 及以上**。低于 3.10 时 `backend/models.py` 里的 `str | None`
  注解会让 pydantic 在 import 阶段就报错，所以启动脚本直接拒绝。
- `curl`、`flock`、`sha256sum`、`awk`、`tar`、`mktemp`、`find`（`./nsysscope doctor`
  会检查这些；常见 Linux 发行版自带）。
- 只有分析新的 `.nsys-rep` 才需要 `nsys` 和一个已登录的 Codex / Comate
  Provider；纯导入已有结果包不需要。

Python 依赖写在 `backend/requirements.txt`，只声明**最低版本**而不钉死：

```
fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.9.0
tomli>=2.0.1; python_version < "3.11"
```

这些包全部从**官方 PyPI**（`https://pypi.org/simple`）安装。启动脚本会用
`PIP_CONFIG_FILE=/dev/null` 屏蔽机器上已有的 `pip.conf`，避免落到内部镜像源上；
下载前先探测通路：先试直连，直连不通再依次试 `NSYSSCOPE_DOWNLOAD_PROXY`、当前
shell 的 `https_proxy`/`HTTPS_PROXY`，第一个能连通的就用它。代理只作用于下载
子进程，不会导出成全局环境变量（Comate/zulu 在 `HTTP_PROXY` 已设置时会失败）。
内网环境可以通过 `NSYSSCOPE_PIP_INDEX_URL` 指定内部 PyPI 源。

若网络环境特殊，可以手工指定：

```bash
export NSYSSCOPE_DOWNLOAD_PROXY=http://your-proxy:port
./nsysscope start
```

首次安装被中断（例如网络中断）不会留下半成品：下次 `start` 会重新检测虚拟环境
是否能正常 import 所需依赖，检测失败自动重装,不需要手工删除重来。

其余几个容易踩的点：

- `./nsysscope start` 固定监听 `127.0.0.1` 并把 `NSYSSCOPE_API_TOKEN` 清空，
  按本机自用设计，**没有鉴权**。要对外暴露只能改走 `backend/run.sh`（它认
  `NSYSSCOPE_HOST`），那时必须同时设置 `NSYSSCOPE_API_TOKEN`。
- 用 Comate Provider 时按需设置 `NSYSSCOPE_COMATE_USERNAME` /
  `NSYSSCOPE_COMATE_MODEL`；不设置则用 CLI 自身的默认值。
- `~/.codex/skills/sglang-nsys-static-analysis` 若存在，会**优先于**仓库内置
  skill 生效。任务日志开头会打印实际使用的 skill 来源，被外部副本遮蔽时会给出
  警告——排查"改了 skill 没生效"先看这一行。

## 核心能力

- **双 Agent Provider**：支持 Codex CLI（`codex exec`）和 Comate Zulu CLI
  （`zulu run`），两者产出相同的结果包并通过统一的确定性校验。
- **架构感知分析**：为每个新模型建立专属的架构分类体系，按
  `(结构位置, unit, 变体, 功能模块)` 聚合，正确处理 KDA/MLA 等异构层混合模式，
  不会被错误折叠成单一平均值。
- **skill 与工具解耦**：skill 单独运行即可产出规范布局的完整结果包（`csv/`
  `xlsx/` `metadata/` `trace/` `analysis.json` `final_report.md`
  `nsysscope-package.json`），工具与 skill 共用同一份打包脚本
  `finalize_package.py`，两条路径产出的目录结构完全一致。
- **前端契约自愈**：打包前会自动校验 `analysis.json` 是否满足前端渲染要求；
  不满足时先尝试用现有表重新生成一次再校验，仍不满足才报错并指出具体缺什么
  ——不会让一次跑到最后的完整分析因为一个可派生的文件而失败。
- **可替换 Skill**：同时支持跟踪外部维护的 skill 目录，外部更新在下次启动时
  自动生效。
- **日志与状态管理**：Agent 日志按大小和心跳频率限制，避免无限增长；心跳线程
  在产物齐全且会话长时间静默时会自动判定任务完成，不需要等到 CLI 自身的超时；
  任务索引默认使用临时目录，停止服务后自动清理，结果包是唯一持久化数据。

## 创建一次分析

点击「新建分析」，提供：

- `.nsys-rep` 或已导出的 `.sqlite`
- 可选的 torch profiler trace（`.trace.json` / `.trace.json.gz`，采集时 `activities` 需含 GPU）：
  提供后先从中解析每个 kernel 的 Python 调用栈与源码位置供 Agent 查表，省去逐个 kernel
  检索源码；解析失败只记日志，不影响主分析
- 模型 `config.json`、真实部署脚本、对应的模型源码根目录
- 模型名称、推理阶段、硬件信息
- 一个空的或未创建的结果目录
- 分析范围与硬性验收标准（约束是强制性的，Agent 必须精确匹配或带证据失败，
  不能静默扩大分析范围）

成功运行后，结果目录是一套规范包：

```text
result-package/
├── analysis.json          # 前端读取的稳定契约
├── final_report.md        # 人读的分析报告
├── nsysscope-package.json # 包清单：前缀、目录结构、表清单
├── csv/                    # 六或七张规范化 CSV 表
├── xlsx/                   # 对应的 XLSX 工作簿
├── trace/                   # 导出或复制的 SQLite trace
├── logs/job.log             # 有大小上限的任务日志（仅走工具产生）
├── dispatch_sites/          # 仅在提供 torch profiler trace 时生成的调用栈查表
└── metadata/                 # 架构分类、manifest、语义映射、统计 sidecar、
                               # 校验报告，以及请求/prompt 等任务自身的痕迹
```

这套布局由 skill 自身的 `scripts/finalize_package.py` 生成，工具调用的是同一
个脚本——单独运行 skill 得到的目录与走本工具得到的完全一致；工具只多出
`logs/job.log` 和 `metadata/` 里 prompt/request/skill 这些任务自身的痕迹。

## 单独使用 skill（不经过本工具）

`bundled/skills/sglang-nsys-static-analysis` 是一个完整独立的 skill，按其
`SKILL.md` 的步骤交给任意支持 skill 机制的 Agent CLI 即可运行，不依赖本工具、
不需要启动任何服务。产出的结果包同样规范、同样能被前端直接渲染——最后一步
`finalize_package.py` 会自动校验 `analysis.json` 是否满足前端契约，不满足会
先尝试用现有表重建一次再校验。

产出的结果包可以直接用本工具的「导入结果目录」打开查看，或者把 `analysis.json`
单独拖进浏览器展示——两者拿到的展示效果与工具全流程跑出来的一致。

## 导入已有结果

使用「导入结果目录（无需 Agent）」指定包目录或 ZIP：已有 `analysis.json` 则直接
打开，只有 CSV 表则自动构建。前六张 CSV 为必需契约；第 7 张 forward 链路表是可选的
补充视图（缺失时会用包内 trace 补齐，补不出则跳过并记录日志，不影响导入成功），
manifest/语义映射/校验报告/统计 sidecar 为可选的可追溯性增强。

## 常用环境变量

| 变量 | 用途 |
| --- | --- |
| `NSYSSCOPE_ALLOWED_ROOTS` | 限定可访问的材料路径范围 |
| `NSYSSCOPE_PERSIST_STATE` / `NSYSSCOPE_DATA_DIR` | 开启持久化任务历史 |
| `NSYSSCOPE_COMATE_PLATFORM` / `NSYSSCOPE_COMATE_MODEL` | Comate 平台与模型配置 |
| `NSYSSCOPE_AGENT_HEARTBEAT_SECONDS` | Agent 心跳间隔 |
| `NSYSSCOPE_AGENT_STALL_TIMEOUT_SECONDS` | 无产出、无输出、无 CPU、无会话更新多久判为停滞（默认 1800） |
| `NSYSSCOPE_COMATE_STORE_DIR` | Comate 会话目录，用于判断 Agent 是否真的还在跑（默认 `~/.comate-engine/store`） |
| `NSYSSCOPE_JOB_LOG_MAX_BYTES` | 单任务日志总大小上限 |
| `NSYSSCOPE_TEST_PACKAGE` | 指向一份七表结果包，供 `pytest backend/test_service.py` 中依赖真实数据的测试使用；未设置时这些测试会显式 skip |
| `NSYSSCOPE_DOWNLOAD_PROXY` | 指定安装依赖时使用的代理，跳过自动探测 |
| `NSYSSCOPE_PIP_INDEX_URL` | 覆盖默认的 `https://pypi.org/simple` |
| `NSYSSCOPE_PROXY_PROBE_URL` | 覆盖代理连通性探测的目标 URL |
| `NSYSSCOPE_API_TOKEN` | 开启接口鉴权；仅在改用 `backend/run.sh` 对外暴露时需要（`./nsysscope start` 会将其清空） |

## 发布到 popo

页面顶部的「发布到 popo」按钮可将当前分析结果一键生成可分享链接。由于出网
代理对单次请求体大小有限制，发布逻辑会自动拆分为两次上传（先发布页面骨架，
再追加分析数据），无需手动干预。

## 开发

前端源码位于 `app/`，改动后需要重新构建：

```bash
npm install          # 仅开发环境需要
npm run build:local   # 单次构建
npm run watch:local   # 监听文件变化自动重新构建
```

日常使用（跑分析、看结果）不依赖 `node_modules`，只有修改前端界面才需要。

运行测试：

```bash
pytest backend/test_service.py -q                              # 后端单测
pytest bundled/skills/sglang-nsys-static-analysis/evals -q      # skill 契约测试
python3 scripts/skill_manager.py validate bundled/skills/sglang-nsys-static-analysis  # skill 完整性校验
```

## 可选的 Codex 插件

经过校验的插件源码位于 `plugins/nsysscope`，向 Codex 提供同一份分析 skill：
`plugins/nsysscope/skills/` 下每一项都是指向 `bundled/skills/` 的符号链接，
不是副本，所以插件用户与 Web 用户拿到的 skill 完全一致
（`test_plugin_serves_the_bundled_skills` 会在两者分叉时失败）。
