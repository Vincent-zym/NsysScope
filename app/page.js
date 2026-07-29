"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const COLORS = [
  "#5b8cff", "#7b61ff", "#15b8a6", "#f59e0b", "#ff6b6b",
  "#35a7ff", "#a78bfa", "#22c55e", "#fb7185", "#38bdf8",
];
const CATEGORY = {
  core: { label: "核心计算", color: "#5b8cff" },
  communication: { label: "通信", color: "#f59e0b" },
  auxiliary: { label: "辅助算子", color: "#64748b" },
};

const fmt = (value, digits = 2) =>
  value == null ? "—" : new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);

const basename = (path) => path?.split("/").pop() || "—";

function Metric({ label, value, suffix, note, accent }) {
  return (
    <article className="metric">
      <span>{label}</span>
      <strong className={accent ? "accent" : ""}>{value}<small>{suffix}</small></strong>
      <p>{note}</p>
    </article>
  );
}

function StageBars({ stages, selected, onSelect, colors }) {
  const max = Math.max(...stages.map((x) => x.durationUs));
  return (
    <div className="stage-list">
      {stages.map((stage) => (
        <button
          key={stage.name}
          className={`stage-row ${selected === stage.name ? "active" : ""}`}
          onClick={() => onSelect(selected === stage.name ? null : stage.name)}
        >
          <div className="stage-label">
            <span>{stage.name}</span>
            <b>{fmt(stage.durationUs)} μs</b>
          </div>
          <div className="bar-track">
            <i style={{
              width: `${stage.durationUs / max * 100}%`,
              background: colors[stage.name],
            }} />
          </div>
          <em>{fmt(stage.durationPct)}%</em>
        </button>
      ))}
    </div>
  );
}

function Timeline({ operators, selectedStage, colors, onPick }) {
  const start = Math.min(...operators.map((op) => op.startNs));
  const end = Math.max(...operators.map((op) => op.endNs));
  const span = end - start;
  const streams = [...new Set(operators.map((op) => op.stream))].sort((a, b) => a - b);
  return (
    <div className="timeline">
      <div className="axis">
        <span>0</span><span>{fmt(span / 2000)} ms</span><span>{fmt(span / 1000)} ms</span>
      </div>
      {streams.map((stream) => (
        <div className="lane" key={stream}>
          <label>Stream {stream}</label>
          <div className="lane-track">
            {operators.filter((op) => op.stream === stream).map((op) => {
              const left = (op.startNs - start) / span * 100;
              const width = Math.max((op.endNs - op.startNs) / span * 100, 0.12);
              const dim = selectedStage && op.stage !== selectedStage;
              return <button
                key={op.index}
                title={`${op.name} · ${fmt(op.durationUs)} μs`}
                className={`kernel ${dim ? "dim" : ""}`}
                style={{ left: `${left}%`, width: `${width}%`, background: colors[op.stage] }}
                onClick={() => onPick(op)}
              />;
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

function Insight({ index, title, children, tone = "blue" }) {
  return (
    <article className={`insight ${tone}`}>
      <span>{String(index).padStart(2, "0")}</span>
      <div><strong>{title}</strong><p>{children}</p></div>
    </article>
  );
}

function JobDialog({ open, onClose, onLoaded }) {
  const [api, setApi] = useState("");
  const [token, setToken] = useState("");
  const [form, setForm] = useState({
    mode: "codex_skill", agent_provider: "codex", agent_model: "", model_name: "",
    stage: "prefill", hardware: "Nvidia B200",
    report_path: "", config_path: "", launch_path: "", source_path: "",
    design_path: "", existing_package_path: "", result_path: "",
    prefix: "analysis", notes: "",
  });
  const [job, setJob] = useState(null);
  const [logs, setLogs] = useState([]);
  const logCursor = useRef(0);
  const analysisLoaded = useRef(false);
  const [logHasMore, setLogHasMore] = useState(true);
  const [health, setHealth] = useState({
    state: "checking", message: "正在连接本地 Analyzer…", providers: null,
  });
  const [modelCatalog, setModelCatalog] = useState({
    state: "idle", default_model: "", models: [], message: "",
  });
  const [error, setError] = useState("");
  const terminal = ["succeeded", "failed", "cancelled"];

  useEffect(() => {
    if (!open) return;
    setApi(localStorage.getItem("nsysscope.api.v2") || "/analyzer-api");
    setToken(sessionStorage.getItem("nsysscope.token") || "");
  }, [open]);

  useEffect(() => {
    if (!open || !api) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setHealth({
        state: "checking", message: "正在连接本地 Analyzer…", providers: null,
      });
      try {
        const response = await fetch(`${api.replace(/\/$/, "")}/api/health`, {
          headers: { "X-NsysScope-Token": token },
          signal: controller.signal,
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
        const providers = payload.providers || {
          codex: {
            enabled: Boolean(payload.codex_enabled),
            ready: Boolean(payload.codex_enabled),
            message: payload.codex_enabled ? "Codex CLI 可用" : "Codex 未启用",
          },
        };
        const ready = Object.entries(providers)
          .filter(([, value]) => value.ready)
          .map(([name]) => name === "codex" ? "Codex" : "Comate");
        const firstReadyProvider = Object.entries(providers)
          .find(([, value]) => value.ready)?.[0];
        if (firstReadyProvider) {
          setForm((current) => providers[current.agent_provider]?.ready
            ? current
            : { ...current, agent_provider: firstReadyProvider, agent_model: "" });
        }
        setHealth({
          state: ready.length ? "ready" : "warning",
          message: ready.length
            ? `可用 Provider：${ready.join("、")}`
            : "Analyzer 已连接，但没有可用的 Agent Provider",
          providers,
        });
      } catch (cause) {
        if (cause.name !== "AbortError") {
          setHealth({ state: "offline", message: cause.message, providers: null });
        }
      }
    }, 250);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [api, open, token]);

  useEffect(() => {
    const ready = health.providers?.[form.agent_provider]?.ready;
    if (!open || !api || form.mode !== "codex_skill" || !ready) {
      setModelCatalog({
        state: "idle", default_model: "", models: [], message: "",
      });
      return;
    }
    const controller = new AbortController();
    setModelCatalog({
      state: "loading", default_model: "", models: [], message: "正在读取可用模型…",
    });
    fetch(`${api.replace(/\/$/, "")}/api/providers/${form.agent_provider}/models`, {
      headers: { "X-NsysScope-Token": token },
      signal: controller.signal,
    }).then(async (response) => {
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
      setModelCatalog({
        state: "ready",
        default_model: payload.default_model || "",
        models: payload.models || [],
        message: "",
      });
    }).catch((cause) => {
      if (cause.name !== "AbortError") {
        setModelCatalog({
          state: "error", default_model: "", models: [], message: cause.message,
        });
      }
    });
    return () => controller.abort();
  }, [api, form.agent_provider, form.mode, health.providers, open, token]);

  useEffect(() => {
    if (!job || (terminal.includes(job.status) && !logHasMore)) return;
    const timer = setTimeout(async () => {
      try {
        const headers = { "X-NsysScope-Token": token };
        const [jobResponse, logResponse] = await Promise.all([
          fetch(`${api}/api/jobs/${job.id}`, { headers }),
          fetch(`${api}/api/jobs/${job.id}/logs?after=${logCursor.current}&limit=200`, { headers }),
        ]);
        if (!jobResponse.ok) throw new Error(`任务状态请求失败 (${jobResponse.status})`);
        const next = await jobResponse.json();
        setJob(next);
        if (logResponse.ok) {
          const payload = await logResponse.json();
          logCursor.current = payload.next;
          setLogHasMore(payload.has_more);
          setLogs((current) => (
            payload.reset ? payload.lines : [...current, ...payload.lines]
          ).slice(-500));
        }
        if (next.status === "succeeded" && !analysisLoaded.current) {
          const result = await fetch(`${api}/api/jobs/${next.id}/analysis`, { headers });
          if (!result.ok) throw new Error("分析产物读取失败");
          const payload = await result.json();
          analysisLoaded.current = true;
          onLoaded(payload);
        }
      } catch (cause) {
        setError(cause.message);
      }
    }, 1600);
    return () => clearTimeout(timer);
  }, [api, job, logHasMore, onLoaded, token]);

  if (!open) return null;

  const set = (key) => (event) => setForm({ ...form, [key]: event.target.value });
  const setProvider = (event) => setForm({
    ...form, agent_provider: event.target.value, agent_model: "",
  });
  const provider = health.providers?.[form.agent_provider];
  const importingPackage = form.mode === "existing_package";
  async function submit(event) {
    event.preventDefault();
    setError("");
    setLogs([]);
    logCursor.current = 0;
    analysisLoaded.current = false;
    setLogHasMore(true);
    localStorage.setItem("nsysscope.api.v2", api.replace(/\/$/, ""));
    sessionStorage.setItem("nsysscope.token", token);
    try {
      const payload = { ...form };
      if (payload.mode === "existing_package") {
        payload.model_name ||= "Imported analysis";
        payload.hardware ||= "Unknown";
      }
      for (const key of Object.keys(payload)) {
        if (payload[key] === "") delete payload[key];
      }
      const response = await fetch(`${api.replace(/\/$/, "")}/api/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-NsysScope-Token": token },
        body: JSON.stringify(payload),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || `提交失败 (${response.status})`);
      setApi(api.replace(/\/$/, ""));
      setJob(body);
    } catch (cause) {
      setError(`${cause.message}。请确认本地工具由 ./nsysscope start 启动。`);
    }
  }

  async function retryConversion() {
    setError("");
    try {
      const response = await fetch(`${api}/api/jobs/${job.id}/retry-conversion`, {
        method: "POST",
        headers: { "X-NsysScope-Token": token },
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || `重试失败 (${response.status})`);
      analysisLoaded.current = false;
      setLogHasMore(true);
      setJob(body);
    } catch (cause) {
      setError(cause.message);
    }
  }

  async function cancelJob() {
    setError("");
    try {
      const response = await fetch(`${api}/api/jobs/${job.id}/cancel`, {
        method: "POST",
        headers: { "X-NsysScope-Token": token },
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || `取消失败 (${response.status})`);
      setJob(body);
    } catch (cause) {
      setError(cause.message);
    }
  }

  return <div className="dialog-backdrop" onMouseDown={onClose}>
    <section className="job-dialog" onMouseDown={(event) => event.stopPropagation()}>
      <button className="close" onClick={onClose}>×</button>
      <span className="drawer-kicker">ANALYZER SERVICE</span>
      <h2>创建性能分析任务</h2>
      <p className="dialog-note">路径由运行 Analyzer Service 的机器解析。大型 nsys 无需经过浏览器上传。</p>

      {!job ? <form onSubmit={submit}>
        <div className={`analyzer-connection ${health.state}`}>
          <i />
          <div><b>{health.state === "ready" ? "本地执行器就绪" : "本地执行器需要配置"}</b><span>{health.message}</span></div>
        </div>
        <details className="advanced-connection">
          <summary>高级连接设置</summary>
          <div className="form-grid">
            <label className="span-2">Analyzer API<input required value={api} onChange={(e) => setApi(e.target.value)} placeholder="/analyzer-api" /></label>
            <label className="span-2">API Token（直接连接远端时填写）<input type="password" value={token} onChange={(e) => setToken(e.target.value)} autoComplete="off" /></label>
          </div>
        </details>
        <div className="form-grid">
          <label className={importingPackage ? "span-2" : ""}>执行模式<select value={form.mode} onChange={set("mode")}><option value="codex_skill">Agent 静态分析</option><option value="existing_package">导入结果目录（无需 Agent）</option></select></label>
          {!importingPackage && <label>阶段<select value={form.stage} onChange={set("stage")}><option value="prefill">Prefill</option><option value="decode">Decode</option></select></label>}
          {form.mode === "codex_skill" && <>
            <label>Agent Provider<select value={form.agent_provider} onChange={setProvider}>
              <option value="codex" disabled={health.providers && !health.providers.codex?.ready}>Codex CLI{health.providers && !health.providers.codex?.ready ? "（未就绪）" : ""}</option>
              <option value="comate" disabled={health.providers && !health.providers.comate?.ready}>Comate Zulu{health.providers && !health.providers.comate?.ready ? "（未登录或未就绪）" : ""}</option>
            </select></label>
            <label>Agent 基座模型<select value={form.agent_model} onChange={set("agent_model")} disabled={modelCatalog.state === "loading"}>
              <option value="">{modelCatalog.state === "loading"
                ? "正在读取模型…"
                : `自动 / Provider 默认${modelCatalog.default_model ? `（${modelCatalog.default_model}）` : ""}`}</option>
              {modelCatalog.models.map((model) =>
                <option key={model.id} value={model.id}>{model.label}</option>
              )}
            </select></label>
            {provider && !provider.ready && <div className="provider-warning span-2">{provider.message}</div>}
            {modelCatalog.state === "error" && <div className="provider-warning span-2">模型列表读取失败，将使用 Provider 默认模型：{modelCatalog.message}</div>}
          </>}
          {!importingPackage && <>
            <label>模型名称<input required value={form.model_name} onChange={set("model_name")} placeholder="GLM5.2" /></label>
            <label>硬件<input required value={form.hardware} onChange={set("hardware")} /></label>
          </>}
          {form.mode === "existing_package" ? <>
            <label className="span-2">NsysScope 结果目录<input required value={form.existing_package_path} onChange={set("existing_package_path")} placeholder="/path/to/result-package" /></label>
            <p className="package-hint span-2">目录内有 analysis.json 时直接展示；只有六表 CSV 时会自动转换。支持新版 csv/ 子目录和旧版平铺目录。</p>
          </> : <>
            <label className="span-2">Nsight 报告<input required value={form.report_path} onChange={set("report_path")} placeholder="/path/to/report.nsys-rep" /></label>
            <label>Model config<input required value={form.config_path} onChange={set("config_path")} placeholder="/path/to/config.json" /></label>
            <label>部署 YAML / 脚本<input required value={form.launch_path} onChange={set("launch_path")} placeholder="/path/to/launch.yaml" /></label>
            <label className="span-2">模型源码根目录<input required value={form.source_path} onChange={set("source_path")} placeholder="/path/to/sglang/source" /></label>
            <label className="span-2">设计说明（可选）<input value={form.design_path} onChange={set("design_path")} placeholder="/path/to/design.md" /></label>
            <label className="span-2">结果保存目录<input required value={form.result_path} onChange={set("result_path")} placeholder="/path/to/result-package（必须为空或不存在）" /></label>
          </>}
          <label>输出前缀<input required value={form.prefix} onChange={set("prefix")} pattern="[a-zA-Z0-9_-]+" /></label>
          {!importingPackage && <label>补充要求<input value={form.notes} onChange={set("notes")} placeholder="目标层、batch、特殊分支…" /></label>}
        </div>
        {error && <div className="error">{error}</div>}
        <div className="dialog-actions"><button type="button" onClick={onClose}>取消</button><button className="primary" type="submit">提交分析</button></div>
      </form> : <div className="job-progress">
        <div className="job-status"><span>{job.status.toUpperCase()}</span><b>{job.message}</b><em>{job.progress}%</em></div>
        <div className="progress-track"><i style={{ width: `${job.progress}%` }} /></div>
        {job.status === "succeeded" && <div className="result-location"><span>结果目录</span><code>{job.output_dir}</code></div>}
        {job.idle_seconds > 180 && <div className="activity-warning">执行器超过 {Math.floor(job.idle_seconds / 60)} 分钟没有新输出，请检查日志；任务仍在运行，可随时取消。</div>}
        <div className="job-log">{logs.length ? logs.map((line, index) => <code key={`${index}-${line}`}>{line}</code>) : <code>等待执行器输出…</code>}</div>
        {(error || job.error) && <div className="error">{error || job.error}</div>}
        <div className="dialog-actions">
          <button onClick={() => { setJob(null); setLogs([]); logCursor.current = 0; analysisLoaded.current = false; setLogHasMore(true); setError(""); }}>新建任务</button>
          {!terminal.includes(job.status) && <button onClick={cancelJob}>取消任务</button>}
          {job.status === "failed" && <button onClick={retryConversion}>仅重试转换</button>}
          <button className="primary" onClick={onClose}>{job.status === "succeeded" ? "查看结果" : "后台运行"}</button>
        </div>
      </div>}
    </section>
  </div>;
}

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [selectedStage, setSelectedStage] = useState(null);
  const [selectedOp, setSelectedOp] = useState(null);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [error, setError] = useState("");
  const [jobOpen, setJobOpen] = useState(false);
  const fileRef = useRef(null);

  useEffect(() => {
    fetch("/demo-analysis.json").then((r) => r.json()).then((payload) => {
      setData(payload);
      setSelectedOp(payload.operators?.[0] || null);
    })
      .catch(() => setError("示例数据加载失败"));
  }, []);

  const colors = useMemo(() => data ? Object.fromEntries(
    data.stages.map((stage, index) => [stage.name, COLORS[index % COLORS.length]])
  ) : {}, [data]);

  const filtered = useMemo(() => {
    if (!data) return [];
    return data.operators.filter((op) => {
      const stageOk = !selectedStage || op.stage === selectedStage;
      const categoryOk = category === "all" || op.category === category;
      const q = query.trim().toLowerCase();
      const queryOk = !q || `${op.name} ${op.module} ${op.stage}`.toLowerCase().includes(q);
      return stageOk && categoryOk && queryOk;
    }).sort((a, b) => b.durationUs - a.durationUs);
  }, [data, selectedStage, category, query]);

  async function loadFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (parsed.schemaVersion !== "1.0" || !Array.isArray(parsed.operators)) {
        throw new Error("schema");
      }
      setData(parsed);
      setSelectedStage(null);
      setSelectedOp(parsed.operators?.[0] || null);
      setError("");
    } catch {
      setError("无法读取：请选择 NsysScope analysis.json（schemaVersion 1.0）");
    }
  }

  const loadAnalysis = useCallback((payload) => {
    setData(payload);
    setSelectedStage(null);
    setSelectedOp(payload.operators?.[0] || null);
    setError("");
  }, []);

  if (!data) return <main className="loading"><div className="pulse" />正在载入分析数据…</main>;

  const topStage = data.stages[0];
  const highVariance = [...data.operators]
    .filter((op) => op.durationUs > 0)
    .sort((a, b) => (b.diffUs / b.durationUs) - (a.diffUs / a.durationUs))[0];
  const compute = data.classifications.find((x) => x.name === "核心计算");
  const displayModel = String(data.metadata.model || "Unknown model")
    .split(" / ", 1)[0].split(" (", 1)[0].slice(0, 80);

  return (
    <main>
      <header className="topbar">
        <div className="brand"><i>NS</i><div><b>NsysScope</b><span>GPU INFERENCE PROFILER</span></div></div>
        <div className="report-chip"><span className="status-dot" />{displayModel}</div>
        <div className="top-actions">
          <button className="ghost-action" onClick={() => fileRef.current?.click()}>导入 JSON</button>
          <button className="import" onClick={() => setJobOpen(true)}>新建分析</button>
        </div>
        <input ref={fileRef} hidden type="file" accept=".json,application/json" onChange={loadFile} />
      </header>

      <section className="shell">
        <div className="title-row compact-title">
          <div>
            <p className="eyebrow">PERFORMANCE REPORT</p>
            <h1>{displayModel}</h1>
            <p>{basename(data.metadata.report)} </p>
          </div>
          <div className="run-meta">
            <span>STATUS <b>VALIDATED</b></span>
            <span>STAGE <b>{data.metadata.stage?.toUpperCase()}</b></span>
            <span>HARDWARE <b>{data.metadata.hardware}</b></span>
            <span>DEVICES <b>{data.summary.devices.length} × GPU</b></span>
          </div>
        </div>
        {error && <div className="error">{error}</div>}

        <section className="metrics">
          <Metric label="重复单元平均耗时" value={fmt(data.summary.totalDurationUs / 1000, 3)} suffix=" ms" note={`${data.summary.stableSamples} 个稳定样本`} accent />
          <Metric label="算子数量" value={data.summary.operatorCount} suffix="" note={`${data.stages.length} 个功能阶段`} />
          <Metric label="核心计算占比" value={fmt(compute?.durationPct)} suffix="%" note={`${compute?.count || 0} 个核心算子`} />
          <Metric label="最高实测 MFU" value={fmt(data.summary.maxMfu)} suffix="%" note="仅统计 shape 可证明的 GEMM" />
        </section>

        <section className="analysis-grid">
          <article className="panel stage-panel">
            <div className="panel-head"><div><span>BREAKDOWN</span><h2>功能模块耗时</h2></div><small>点击模块联动筛选</small></div>
            <StageBars stages={data.stages} selected={selectedStage} onSelect={setSelectedStage} colors={colors} />
          </article>

          <article className="panel insight-panel">
            <div className="panel-head"><div><span>FINDINGS</span><h2>性能结论</h2></div><small>基于当前重复单元</small></div>
            <div className="insights">
              <Insight index={1} title="首要耗时阶段">
                {topStage.name} 占 {fmt(topStage.durationPct)}%，平均 {fmt(topStage.durationUs)} μs。
              </Insight>
              <Insight index={2} title="计算密集度" tone="violet">
                核心计算累计 {fmt(compute?.durationPct)}%；百分比以 wall-span 为分母，重叠流不做归一化。
              </Insight>
              <Insight index={3} title="稳定性关注点" tone="amber">
                {highVariance.name} 的 max-min 为 {fmt(highVariance.diffUs)} μs，建议进一步按 rank 和输入形态拆分。
              </Insight>
            </div>
          </article>
        </section>

        <section className="panel timeline-panel">
          <div className="panel-head"><div><span>CUDA TIMELINE</span><h2>重复单元时间线</h2></div><small>按真实 stream 展示 · 点击 kernel 查看证据</small></div>
          <Timeline operators={data.operators} selectedStage={selectedStage} colors={colors} onPick={setSelectedOp} />
        </section>

        <section className="operator-workbench">
          <article className="panel table-panel">
            <div className="panel-head table-tools">
              <div><span>OPERATORS</span><h2>算子明细</h2></div>
              <div className="filters">
                <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="搜索算子、module…" />
                {["all", "core", "communication", "auxiliary"].map((key) => (
                  <button key={key} className={category === key ? "active" : ""} onClick={() => setCategory(key)}>
                    {key === "all" ? "全部" : CATEGORY[key].label}
                  </button>
                ))}
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>#</th><th>功能模块 / 算子</th><th>分类</th><th>耗时</th><th>占比</th><th>MFU</th><th>波动</th></tr></thead>
                <tbody>
                  {filtered.map((op) => (
                    <tr key={op.index} className={selectedOp?.index === op.index ? "selected" : ""} onClick={() => setSelectedOp(op)}>
                      <td className="mono muted">{op.index}</td>
                      <td><div className="op-title"><i style={{ background: colors[op.stage] }} /><div><b>{op.name}</b><span>{op.stage} · {op.module}</span></div></div></td>
                      <td><span className={`category ${op.category}`}>{CATEGORY[op.category]?.label}</span></td>
                      <td className="mono">{fmt(op.durationUs)} μs</td>
                      <td className="mono">{fmt(op.durationPct)}%</td>
                      <td className="mono">{op.mfu ? `${fmt(op.mfu)}%` : "—"}</td>
                      <td className="mono muted">{fmt(op.diffUs)} μs</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
          <aside className="panel evidence-panel">
            <div className="panel-head"><div><span>EVIDENCE</span><h2>算子证据</h2></div><small>{selectedOp ? `#${selectedOp.index}` : "选择算子"}</small></div>
            {selectedOp ? <div className="evidence-body">
              <div className="evidence-title"><i style={{ background: colors[selectedOp.stage] }} /><div><b>{selectedOp.name}</b><span>{selectedOp.stage} · {selectedOp.module}</span></div></div>
              <div className="evidence-stats">
                <div><span>AVG</span><b>{fmt(selectedOp.durationUs)} μs</b></div>
                <div><span>MIN / MAX</span><b>{fmt(selectedOp.minUs)} / {fmt(selectedOp.maxUs)}</b></div>
                <div><span>SHAPE / MFU</span><b>{selectedOp.shape || "N/A"} · {selectedOp.mfu ? `${fmt(selectedOp.mfu)}%` : "N/A"}</b></div>
              </div>
              <h3>功能说明</h3><p>{selectedOp.introduction || "暂无说明"}</p>
              <h3>Python 调用链</h3><code>{selectedOp.pythonFunction || "暂无调用链证据"}</code>
              <h3>映射依据</h3><p>{selectedOp.mappingReason || "暂无映射依据"}</p>
              <details><summary>完整 CUDA 符号</summary><code>{selectedOp.fullName}</code></details>
            </div> : <div className="evidence-empty">从左侧选择一个算子查看映射证据。</div>}
          </aside>
        </section>
      </section>
      <JobDialog open={jobOpen} onClose={() => setJobOpen(false)} onLoaded={loadAnalysis} />
    </main>
  );
}
