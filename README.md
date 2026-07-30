# NsysScope

Interactive SGLang Nsight Systems analysis powered by the installed
`sglang-nsys-static-analysis` skill.

## Portable executable

The lightweight release is a single self-extracting Linux file:

```bash
chmod +x nsysscope-linux-x86_64.run
./nsysscope-linux-x86_64.run
```

It contains the prebuilt dashboard, Analyzer source and a validated fallback
Skill. It does not contain Node.js, npm packages, task data, Provider
credentials or NVIDIA Nsight Systems. On first start it creates a reusable
Python environment under the user's cache directory. Import-only use needs
Python 3; new `.nsys-rep` analysis additionally needs `nsys` and a logged-in
Codex or Comate Provider.

Build the portable file with:

```bash
python3 scripts/build_portable.py
```

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

Missing `nsys` or Agent Providers no longer blocks startup. The dashboard
remains available for importing an existing six-table package.

## Replaceable analysis Skill

NsysScope includes a small built-in Skill and can track an externally maintained
Skill directory without copying it into the application:

```bash
./nsysscope skill status
./nsysscope skill use /path/to/sglang-nsys-static-analysis
./nsysscope skill sync
./nsysscope skill validate
./nsysscope skill reset
```

After `skill use`, edits to that external directory take effect on the next
start. Every new analysis records the selected Skill path, source and SHA256 in
`metadata/skill.json`. Resolution order is an explicit
`NSYSSCOPE_SKILL_DIR`, the configured external path, the user's Codex-installed
copy, then the bundled fallback.

## Agent providers

NsysScope supports two interchangeable analysis providers:

- **Codex CLI** through non-interactive `codex exec`;
- **Comate Zulu CLI** through non-interactive `zulu run`.

Both providers receive the same prompt and task materials, activate the same
`sglang-nsys-static-analysis` skill, and must produce the same six-table package
and pass the same deterministic validation before the dashboard accepts the
result.

For every new composite or heterogeneous model, the Skill first creates a
current-model architecture taxonomy. Each operator and functional stage carries
its structural position, concrete unit ID and architecture variant. Aggregation
uses `(position, unit, variant, functional module)`, so mixed patterns such as
`KDA,KDA,KDA,MLA` cannot collapse into a generic Attention average. The
dashboard exposes the complete cycle and every structural unit as separate
views. Fine-grained source attribution stays in `module`; the functional view
defaults to 5–8 architecture stages per variant so projections, norms, gates,
cache operations and dispatch details do not overwhelm the comparison layer.

Every newly generated CSV ends with a total row. Operator/category/stage totals
represent accumulated GPU work and may exceed wall time under overlap. The
operator overview also preserves the origin `module` column immediately before
the compact operator name.

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
allows materials under the directory containing this repository. The portable
`.run` file instead uses the directory from which it was launched, never its
private extraction directory. Override the scope when needed:

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

Use **导入结果目录（无需 Agent）** and enter the package directory or ZIP. If it
already contains `analysis.json`, NsysScope opens it directly; if it only has
the normalized six CSV files, NsysScope builds `analysis.json` automatically.
Both the new `csv/` layout and legacy flat six-table directories are supported.
Use **导入 JSON** only for a standalone browser-side preview.

ZIP import asks for a new result directory, validates paths before extraction,
and writes the normalized CSV, XLSX, analysis data, optional trace, metadata and
bounded log only into that selected directory.

The six CSV files are sufficient. Manifests, semantic maps, validation reports
and statistics sidecars improve provenance but are optional for quick import.
When sidecars are absent, categories are recovered from the core and auxiliary
tables, remaining rows are treated as communication, and the imported package
is marked as lacking external validation evidence.

## Optional Codex plugin

The validated plugin source lives at `plugins/nsysscope`. It contributes the
same analysis Skill to Codex and includes a launcher adapter, while the
standalone executable remains the primary UI and also supports Comate and
Agent-free imports.
