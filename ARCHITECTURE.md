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
- invokes the installed analysis skill through non-interactive `codex exec`;
- streams logs and exposes the resulting `analysis.json`.

Codex execution is disabled by default. Enable it only on a controlled runner
with `NSYSSCOPE_CODEX_ENABLED=true`. The dashboard can also submit
`existing_package` jobs, which makes the deterministic conversion path easy to
test without invoking an agent.

Raw `.nsys-rep` parsing is intentionally not performed in the browser. Large
reports remain on the analyzer host; the browser submits their paths and only
receives normalized analysis data.

## Local run

```bash
cd /path/to/NsysScope
python3 -m pip install -r backend/requirements.txt

export NSYSSCOPE_DATA_DIR=/path/to/nsysscope-data
export NSYSSCOPE_ALLOWED_ROOTS=/path/to/reports:/path/to/model-source
export NSYSSCOPE_API_TOKEN="$(openssl rand -hex 24)"
export NSYSSCOPE_CODEX_ENABLED=true
export NSYSSCOPE_CORS_ORIGINS=http://localhost:3000

npm run analyzer
```

In another terminal, run `npm run dev` and open `http://localhost:3000`.
For the privately deployed dashboard, expose the Analyzer API through an
authenticated HTTPS reverse proxy and add that dashboard origin to
`NSYSSCOPE_CORS_ORIGINS`. Do not expose the analyzer directly to the public
internet.
