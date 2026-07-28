# NsysScope

Interactive SGLang Nsight Systems analysis powered by the installed
`sglang-nsys-static-analysis` Codex skill.

## One-command start

```bash
cd /home/users/zhongyuanming/NsysScope
./nsysscope start
```

The command automatically:

- verifies Python, Node.js, Nsight Systems, Codex login and the analysis skill;
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

## Existing results

Use **已有六表分析包** to convert a completed normalized package without
running Codex again, or use **导入 JSON** to open an existing
`analysis.json`.
