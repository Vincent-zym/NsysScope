# NsysScope

Interactive SGLang Nsight Systems analysis powered by the installed
`sglang-nsys-static-analysis` skill.

## One-command start

```bash
cd /home/users/zhongyuanming/NsysScope
./nsysscope start
```

The command automatically:

- verifies Python, Node.js, Nsight Systems, at least one Agent Provider and the
  analysis skill;
- creates or updates the local Python and Node.js dependencies when needed;
- builds the dashboard;
- chooses available internal ports;
- generates an internal API token;
- starts the Analyzer and dashboard together;
- stops both processes when `Ctrl-C` is pressed.

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
- optional design notes and task-specific requirements.

The paths are resolved on the machine running NsysScope. By default, the tool
allows materials under the directory containing this repository. Override the
scope when needed:

```bash
NSYSSCOPE_ALLOWED_ROOTS=/reports:/model-source ./nsysscope start
```

Generated jobs and logs are stored under `.data/` by default. Override this with
`NSYSSCOPE_DATA_DIR`.

Agent logs are bounded by default:

- Comate uses Zulu `task-json`, so only the final task result is consumed instead
  of cumulative conversation snapshots;
- a lightweight heartbeat is recorded every 30 seconds while an Agent is
  silent;
- individual log records are capped at 16 KiB and each job log is capped at
  8 MiB;
- the dashboard reads logs incrementally by byte offset rather than loading the
  whole file.

The limits can be adjusted with `NSYSSCOPE_AGENT_HEARTBEAT_SECONDS`,
`NSYSSCOPE_JOB_LOG_LINE_MAX_BYTES`, and `NSYSSCOPE_JOB_LOG_MAX_BYTES`.

## Existing results

Use **已有六表分析包** to convert a completed normalized package without
running an Agent again, or use **导入 JSON** to open an existing
`analysis.json`.
