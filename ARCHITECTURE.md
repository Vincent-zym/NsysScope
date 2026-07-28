# NsysScope architecture

NsysScope separates report analysis from visualization.

## Data plane

1. The analyzer runs where NVIDIA Nsight Systems, model source and deployment
   materials are available.
2. It exports `.nsys-rep` to SQLite and generates the normalized six-table
   package plus manifest.
3. `scripts/build_analysis_json.py` converts that package into the versioned
   frontend contract.
4. The dashboard loads `analysis.json` and remains model-independent.

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
- authenticates job APIs with `X-NsysScope-Token`;
- persists jobs in SQLite and runs them with bounded concurrency;
- exports `.nsys-rep` with `nsys`;
- invokes the installed analysis skill through a selectable Agent Provider:
  non-interactive `codex exec` or Comate `zulu run`;
- exposes cursor-paginated logs, activity timestamps and the resulting
  `analysis.json`;
- can retry only the deterministic conversion/validation stage when a completed
  six-table package survives an agent-side failure.

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

## Local run

The supported local entry point is:

```bash
cd /path/to/NsysScope
./nsysscope start
```

The launcher checks prerequisites, prepares dependencies, generates an
ephemeral internal token, starts the Analyzer on loopback, builds and starts the
dashboard, and shuts both down together. The browser uses the same-origin
`/analyzer-api/*` route; the server-side proxy injects the token, so normal local
use requires one port and no connection form.

The lower-level `npm run analyzer` command remains available for development or
an externally managed deployment. For a privately deployed dashboard, expose
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
the six-table or `analysis.json` contracts.
