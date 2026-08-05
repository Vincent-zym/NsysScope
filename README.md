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
Python 3；分析新的 `.nsys-rep` 还需要 `nsys` 和一个已登录的 Codex 或 Comate
Provider。

`.run` 文件是构建产物，未提交到仓库（见 `.gitignore` 里的 `release/*.run`）。
从源码重新构建：

```bash
python3 scripts/build_portable.py
```

生成的文件位于 `release/nsysscope-linux-x86_64.run`，只包含运行时后端、已构建
的前端页面和兜底 skill，不含 Node.js、npm 依赖或任务数据。

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

## 核心能力

- **双 Agent Provider**：支持 Codex CLI（`codex exec`）和 Comate Zulu CLI
  （`zulu run`），两者产出相同的六表分析包并通过统一的确定性校验。
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
- 模型 `config.json`、真实部署脚本、对应的模型源码根目录
- 模型名称、推理阶段、硬件信息
- 一个空的或未创建的结果目录
- 分析范围与硬性验收标准（约束是强制性的，Agent 必须精确匹配或带证据失败，
  不能静默扩大分析范围）

成功运行后，结果目录包含：

```text
result-package/
├── analysis.json
├── nsysscope-package.json
├── csv/                 # 六张规范化 CSV 表
├── xlsx/                # 对应的 XLSX 工作簿
├── trace/               # 导出或复制的 SQLite trace
├── logs/job.log         # 有大小上限的任务日志
└── metadata/            # 请求、prompt、manifest 与校验证据
```

## 导入已有结果

使用「导入结果目录（无需 Agent）」指定包目录或 ZIP：已有 `analysis.json` 则直接
打开，只有六表 CSV 则自动构建。六张 CSV 是最小可用契约，manifest/语义映射/
校验报告/统计 sidecar 为可选的可追溯性增强。

## 常用环境变量

| 变量 | 用途 |
| --- | --- |
| `NSYSSCOPE_ALLOWED_ROOTS` | 限定可访问的材料路径范围 |
| `NSYSSCOPE_PERSIST_STATE` / `NSYSSCOPE_DATA_DIR` | 开启持久化任务历史 |
| `NSYSSCOPE_COMATE_PLATFORM` / `NSYSSCOPE_COMATE_MODEL` | Comate 平台与模型配置 |
| `NSYSSCOPE_AGENT_HEARTBEAT_SECONDS` | Agent 心跳间隔 |
| `NSYSSCOPE_JOB_LOG_MAX_BYTES` | 单任务日志总大小上限 |

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

经过校验的插件源码位于 `plugins/nsysscope`，向 Codex 提供同一份分析 skill。
