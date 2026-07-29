# NsysScope

Interactive SGLang Nsight Systems analysis powered by the installed
`sglang-nsys-static-analysis` skill.

## One-command start

```bash
cd /home/users/zhongyuanming/NsysScope
./nsysscope start
```

The command automatically:

- verifies Python, Nsight Systems, at least one Agent Provider and the analysis
  skill;
- creates or updates the small Python environment when needed;
- starts the prebuilt dashboard and Analyzer as one FastAPI process;
- chooses an available loopback port;
- removes temporary state when `Ctrl-C` is pressed.

Node.js and npm are not required for normal local use. They are only development
dependencies used to rebuild the checked-in browser bundle or deploy the hosted
demo.

Open the page printed by the command, normally:

```text
http://127.0.0.1:3000
```

For an SSH-connected server, keep `./nsysscope start` running and execute the
printed one-port SSH forwarding command on the local computer. The browser no
longer needs an Analyzer URL or API token.

Run the prerequisite check by itself with:

```bash
./nsysscope doctor
```

## Agent providers

NsysScope supports two interchangeable analysis providers:

- **Codex CLI** through non-interactive `codex exec`;
- **Comate Zulu CLI** through non-interactive `zulu run`.

Both providers receive the same prompt and task materials, activate the same
`sglang-nsys-static-analysis` skill, and must produce the same six-table package
and pass the same deterministic validation before the dashboard accepts the
result.

Provider readiness is shown in the **新建分析** dialog. Log in when needed:

```bash
./nsysscope login codex
./nsysscope login comate internal
```

Restart `./nsysscope start` after login, then select **Codex CLI** or
**Comate Zulu** in the task form. The launcher discovers the Zulu executable
shipped with the installed Comate extension; `NSYSSCOPE_COMATE_BIN` can override
that path. The default Comate platform is `internal`; use
`./nsysscope login comate saas` for public SaaS. Internal Comate commands bypass
the shell HTTP proxy because the internal endpoint is directly reachable. The
launcher also supplies Zulu's required `PLATFORM` selector so that `status` and
`run` reuse the token written by `login`.
`NSYSSCOPE_COMATE_PLATFORM`, `NSYSSCOPE_COMATE_MODEL` and
`NSYSSCOPE_COMATE_TIMEOUT_SECONDS` control the platform, optional model and
timeout.

The task form also has an **Agent 基座模型** selector:

- Codex choices come from the current CLI model cache, while the automatic
  option preserves the model configured in `~/.codex/config.toml`;
- Comate choices are fetched from the logged-in account with
  `zulu model list --ids`;
- the selected value is stored with the job and forwarded only to that run as
  `codex --model ... exec` or `zulu run --model ...`.

This task-level selection does not rewrite either Provider's global model
configuration.

## Create an analysis

Click **新建分析**, then provide:

- `.nsys-rep` or exported `.sqlite`;
- model `config.json`;
- the real deployment YAML or launch script;
- the corresponding SGLang/model source root;
- model name, inference stage and hardware;
- an empty or not-yet-created result directory;
- optional design notes;
- analysis scope and hard acceptance criteria.

Scope criteria are binding. If a specific layer subtype or branch is requested,
the Agent must select that exact unit or fail with evidence; it may not silently
replace it with a wider architectural period. For example, requesting a single
GLM5.2 non-shared Indexer layer must not select the four-layer full/shared
Indexer cycle.

The paths are resolved on the machine running NsysScope. By default, the tool
allows materials under the directory containing this repository. Override the
scope when needed:

```bash
NSYSSCOPE_ALLOWED_ROOTS=/reports:/model-source ./nsysscope start
```

The result directory is the portable unit of storage. A successful run writes:

```text
result-package/
├── analysis.json
├── nsysscope-package.json
├── csv/                 # exactly six normalized CSV tables
├── xlsx/                # one XLSX workbook for each CSV
├── trace/               # exported or copied SQLite trace
├── logs/job.log         # bounded Agent/job log
└── metadata/            # request, prompt, manifest and validation evidence
```

By default, `./nsysscope start` keeps its job index, lock and operational log
in a temporary runtime directory and deletes that directory when the tool
stops. The result packages are the only persistent task data.

Persistent task history is optional. Enable it explicitly when needed:

```bash
NSYSSCOPE_PERSIST_STATE=true \
NSYSSCOPE_DATA_DIR=/path/to/small-state-dir \
./nsysscope start
```

Even in persistent mode, this directory contains only lightweight task
metadata; CSV, XLSX, SQLite and task logs remain in their result packages.

Agent logs are bounded by default:

- Comate uses Zulu `task-json`, so only the final task result is consumed instead
  of cumulative conversation snapshots;
- a lightweight heartbeat is recorded every 30 seconds while an Agent is
  silent;
- individual log records are capped at 16 KiB and each job log is capped at
  2 MiB;
- the dashboard reads logs incrementally by byte offset rather than loading the
  whole file.

The limits can be adjusted with `NSYSSCOPE_AGENT_HEARTBEAT_SECONDS`,
`NSYSSCOPE_JOB_LOG_LINE_MAX_BYTES`, and `NSYSSCOPE_JOB_LOG_MAX_BYTES`.

## Existing results

Use **导入结果目录（无需 Agent）** and enter the package directory. If it
already contains `analysis.json`, NsysScope opens it directly; if it only has
the normalized six CSV files, NsysScope builds `analysis.json` automatically.
Both the new `csv/` layout and legacy flat six-table directories are supported.
Use **导入 JSON** only for a standalone browser-side preview.
