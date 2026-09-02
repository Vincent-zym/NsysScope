# NsysScope architecture

NsysScope separates report analysis from visualization.

## Data plane

1. The analyzer runs where NVIDIA Nsight Systems, model source and deployment
   materials are available.
2. It writes the exported SQLite, bounded logs and analysis evidence directly
   into the user-selected result directory.
3. The Skill's `scripts/finalize_package.py` lays the result directory out: the
   six or seven tables in `csv/`, one workbook per table in `xlsx/`, every
   sidecar in `metadata/`, the trace in `trace/`, and `nsysscope-package.json` at
   the root. The service calls that same script, so a Skill run done by hand and
   a job produce the same directory.
4. `bundled/skills/sglang-nsys-static-analysis/scripts/build_analysis_json.py`
   converts that package into the versioned frontend contract. It lives in the
   Skill too, so the only things the service adds to a package are its own job
   traces: `logs/job.log` and the prompt/request/skill records in `metadata/`.
5. The dashboard loads `analysis.json` and remains model-independent.

## Contract

`schemaVersion: "1.0"` contains:

- `metadata`: model, stage, hardware, report and repeating-unit evidence
- `summary`: wall-span, operator count, stable sample count, devices and MFU
- `stages`: architecture-level functional-stage aggregates
- `classifications`: core, communication and auxiliary totals
- `operators`: timeline, stable statistics, semantic mapping and source evidence
- `evidence`: boundary proof, uncertainty, taxonomy and MFU assumptions

## Analyzer service

`backend.app` is a local FastAPI service that:

- accepts trusted server-side material paths;
- restricts paths to `NSYSSCOPE_ALLOWED_ROOTS`;
- requires a new or empty user-selected result directory for Agent runs;
- authenticates job APIs with `X-NsysScope-Token`;
- persists jobs in SQLite and runs them with bounded concurrency;
- exports `.nsys-rep` with `nsys`;
- invokes the installed analysis skill through a selectable Agent Provider:
  non-interactive `codex exec` or Comate `zulu run`;
- exposes cursor-paginated logs, activity timestamps and the resulting
  `analysis.json`;
- can retry only the deterministic conversion/validation stage when a completed
  seven-table package survives an agent-side failure.

The one-command launcher uses an ephemeral jobs database by default and removes
it together with the operational log on normal shutdown. Persistent state is an
explicit opt-in and still stores only task metadata and absolute package paths.
Bulk artifacts live in the selected result directory, which can later be
re-imported without rerunning an Agent.

The package converter accepts both the legacy `accepted_unit_count` sidecar and
the current `accepted_full_template_sample_count` schema. It prefers package-local
manifest, semantic-map, statistics and validation files so copied packages do
not depend on stale absolute paths.

Codex execution is disabled by default. Enable it only on a controlled runner
with `NSYSSCOPE_CODEX_ENABLED=true`. The dashboard can also submit
`existing_package` jobs, which makes the deterministic conversion path easy to
test without invoking an agent.

Raw `.nsys-rep` parsing is intentionally not performed in the browser. Large
reports remain on the analyzer host; the browser submits their paths and only
receives normalized analysis data.

Import-only mode can start without `nsys` or an Agent Provider. The six
normalized CSV files are the minimum portable contract; optional sidecars add
provenance but are not required to build the dashboard payload.

## Local run

The supported local entry point is:

```bash
cd /path/to/NsysScope
./nsysscope start
```

The launcher checks prerequisites, prepares only the Python dependencies and
starts one FastAPI process on loopback. FastAPI serves both the checked-in
browser bundle and `/api/*`, so normal local use needs neither Node.js nor a
second frontend process. The local same-origin API does not require a token;
loopback binding remains the security boundary.

The browser bundle is rebuilt for development with `npm run build:local`.
Node.js dependencies are not installed or inspected by the normal launcher.

The lower-level `npm run analyzer` command remains available for development or
an externally managed deployment. For a remotely deployed dashboard, expose
the Analyzer API through an authenticated HTTPS reverse proxy and configure the
advanced connection settings. Do not expose the analyzer directly to the public
internet.

## Agent provider boundary

The Analyzer selects a provider from each job's `agent_provider` field:

- `codex` runs the installed skill with `codex exec`, adding each material
  directory as a read workspace;
- `comate` copies the runtime-relevant skill files into the job-local
  `.comate/skills/sglang-nsys-static-analysis` directory and activates it with
  `zulu run`.

Provider-specific event output is written to the same job log. Cancellation
terminates the active provider subprocess. After the provider exits, both paths
share `find_package`, conversion and validation, so a provider cannot bypass
the seven-table or `analysis.json` contracts.

On success, provider staging files are removed and the package is normalized as
`csv/`, `xlsx/`, `trace/`, `logs/`, `metadata/`, `analysis.json`, and
`nsysscope-package.json`.

The Skill is resolved in this order: explicit environment override, configured
external directory, Codex-installed copy, bundled fallback. The external
directory is referenced rather than copied into NsysScope, so updates take
effect at the next start. Each Agent run stores `metadata/skill.json` with the
exact path, source and content hash. Comate receives a temporary job-local copy;
Codex receives the selected path in both its read workspace and binding prompt.

`GET /api/providers/{provider}/models` exposes the model catalog used by the
task form. Codex models are read from its local account cache and Comate models
are queried from Zulu under the same platform, identity and proxy environment
used for analysis. A non-empty job `agent_model` is forwarded to the selected
Provider; an empty value preserves that Provider's configured default.

## Bounded Agent logs

The runner never stores Comate's cumulative `event-stream`. Zulu runs with
`task-json`, and its final output is reduced to a compact status record. Both
Providers emit periodic heartbeat lines while their subprocess remains alive.
The runner enforces per-record and per-job byte limits before writing, touching
the capped log on later activity so liveness remains observable without
unbounded growth.

`GET /api/jobs/{id}/logs` uses a byte-offset cursor and bounded reads. It does
not load the entire log into memory. The response can request a client reset if
the underlying log was externally truncated.

## Portable distribution

`scripts/build_portable.py` creates a small self-extracting Linux executable
containing only the runtime backend, checked-in browser bundle, converters and
bundled Skill. It excludes development sources, Node dependencies, virtual
environments, caches and task data. The extracted immutable runtime lives in
the user's data directory and its reusable Python environment lives in the
user's cache directory.
