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

The production analyzer API should accept paths or uploaded artifacts, enqueue
an isolated job, invoke the existing SGLang analysis pipeline and expose job
status plus the generated contract. Raw `.nsys-rep` parsing is intentionally not
performed in the browser.
