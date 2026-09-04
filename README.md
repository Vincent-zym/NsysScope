# NsysScope

面向 SGLang 的交互式 Nsight Systems 性能分析工具，基于内置的
`sglang-nsys-static-analysis` skill 实现自动化的算子归因与可视化。

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
的前端页面和兜底 skill，不含 Node.js、npm 依赖或任务数据。

**固定下载链接（自动更新到最新构建）**：

```text
https://github.com/Vincent-zym/NsysScope/releases/latest/download/nsysscope-linux-x86_64.run
```

`.github/workflows/release-run.yml` 在每次 push 到 `main` 时跑测试、跑
evals、构建 `.run`，然后把 `latest` tag 强制指向本次提交并更新 GitHub
Release 里的这份文件——所以这个链接本身永远不用改，指给同事一次即可，之后
每次拿到的都是当时 `main` 上最新的构建。CI 任一步失败（测试、evals、skill
校验）都不会更新这个链接，上一份能用的构建会继续留在原地。

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
- `curl` 和 `flock`（`flock` 用来保证同一目录只跑一个实例，常见发行版自带）。
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
shell 的 `https_proxy`/`HTTPS_PROXY`、`http://agent.baidu.com:8891`、
`http://10.162.37.16:8128`，第一个能连通的就用它。代理只作用于下载子进程，不会
导出成全局环境变量（Comate/zulu 在 `HTTP_PROXY` 已设置时会失败）。

若网络环境特殊，可以手工指定：

```bash
export NSYSSCOPE_DOWNLOAD_PROXY=http://your-proxy:port
./nsysscope start
```

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
  （`zulu run`），两者产出相同的七表分析包并通过统一的确定性校验。
- **架构感知分析**：为每个新模型建立专属的架构分类体系，按
  `(结构位置, unit, 变体, 功能模块)` 聚合，正确处理 KDA/MLA 等异构层混合模式，
  不会被错误折叠成单一平均值。
- **可替换 Skill**：内置精简版分析 skill，同时支持跟踪外部维护的 skill 目录，
  外部更新在下次启动时自动生效。
- **日志与状态管理**：Agent 日志按大小和心跳频率限制，避免无限增长；任务索引
  默认使用临时目录，停止服务后自动清理，结果包是唯一持久化数据。

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

成功运行后，结果目录包含：

```text
result-package/
├── analysis.json
├── final_report.md      # 人读的分析报告
├── nsysscope-package.json
├── csv/                 # 六或七张规范化 CSV 表
├── xlsx/                # 对应的 XLSX 工作簿
├── trace/               # 导出或复制的 SQLite trace
├── logs/job.log         # 有大小上限的任务日志
├── dispatch_sites/      # 仅在提供 torch profiler trace 时生成的调用栈查表
└── metadata/            # 请求、prompt、manifest 与校验证据
```

这套布局由 Skill 的 `scripts/finalize_package.py` 生成，后端调用的是同一个脚本，
所以单独运行 Skill 得到的目录与走工具得到的一致；工具只多出 `logs/job.log` 和
`metadata/` 里 prompt/request/skill 这些任务自身的痕迹。

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
| `NSYSSCOPE_TEST_PACKAGE` | 指向一份七表结果包，供 `pytest backend/test_service.py` 中 5 个导入/端到端测试使用；未设置时这些测试会显式 skip |
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

## 可选的 Codex 插件

经过校验的插件源码位于 `plugins/nsysscope`，向 Codex 提供同一份分析 skill：
`plugins/nsysscope/skills/` 下每一项都是指向 `bundled/skills/` 的符号链接，
不是副本，所以插件用户与 Web 用户拿到的 skill 完全一致
（`test_plugin_serves_the_bundled_skills` 会在两者分叉时失败）。
