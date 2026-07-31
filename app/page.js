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

function compactKernelName(rawName) {
  const source = String(rawName || "").trim().replace(/^(?:void|int|float|double|bool)\s+/, "");
  let depth = 0;
  let cut = source.length;
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (char === "<") depth += 1;
    else if (char === ">" && depth) depth -= 1;
    else if (char === "(" && depth === 0) {
      cut = index;
      break;
    }
  }
  const symbol = source.slice(0, cut).trim();
  depth = 0;
  let leafStart = 0;
  for (let index = 0; index + 1 < symbol.length; index += 1) {
    if (symbol[index] === "<") depth += 1;
    else if (symbol[index] === ">" && depth) depth -= 1;
    else if (symbol.slice(index, index + 2) === "::" && depth === 0) {
      leafStart = index + 2;
      index += 1;
    }
  }
  const leaf = symbol.slice(leafStart).trim() || String(rawName || "").trim();
  const templateAt = leaf.indexOf("<");
  return templateAt > 0 && leaf.length > 96 ? `${leaf.slice(0, templateAt)}<…>` : leaf;
}

function operatorDisplayName(operator) {
  return operator.kernelName || compactKernelName(operator.fullName) || operator.name || "未知算子";
}

const normalizeStageKey = (value) => {
  const text = String(value || "");
  if (text.startsWith("__pattern_total__::")) return text.slice("__pattern_total__::".length);
  // Compatibility with analysis.json files generated before pattern-level
  // stage keys were introduced.
  return text.includes("::") ? text.slice(text.lastIndexOf("::") + 2) : text;
};
const stageKey = (stage) => normalizeStageKey(stage?.key || stage?.stageKey || stage?.name);
const operatorStageKey = (operator) => normalizeStageKey(operator?.stageKey || operator?.stage);
const unitKey = (unit) => String(unit?.id || unit?.unitId || unit?.position || unit?.unitPosition || "");
const operatorUnitKey = (operator) => String(
  operator?.unitId || operator?.unitPosition || "",
);

function calculateOverlapPct(operators) {
  const intervals = operators
    .map((op) => [Number(op.startNs), Number(op.endNs)])
    .filter(([start, end]) => Number.isFinite(start) && Number.isFinite(end) && end > start)
    .sort((a, b) => a[0] - b[0]);
  if (!intervals.length) return null;

  const accumulatedNs = intervals.reduce((sum, [start, end]) => sum + end - start, 0);
  let unionNs = 0;
  let [currentStart, currentEnd] = intervals[0];
  for (const [start, end] of intervals.slice(1)) {
    if (start <= currentEnd) {
      currentEnd = Math.max(currentEnd, end);
    } else {
      unionNs += currentEnd - currentStart;
      [currentStart, currentEnd] = [start, end];
    }
  }
  unionNs += currentEnd - currentStart;
  return accumulatedNs ? Math.max(0, (accumulatedNs - unionNs) / accumulatedNs * 100) : null;
}

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
  const max = Math.max(...stages.map((x) => x.durationUs), 1);
  return (
    <div className="stage-list">
      {stages.map((stage) => {
        const key = stageKey(stage);
        return (
        <button
          key={key}
          className={`stage-row ${selected === key ? "active" : ""}`}
          onClick={() => onSelect(key)}
        >
          <div className="stage-label">
            <span>{stage.name}</span>
            {stage.unitVariant && <small>{stage.unitPosition}. {stage.unitVariant}</small>}
            <b>{fmt(stage.durationUs)} μs</b>
          </div>
          <div className="bar-track">
            <i style={{
              width: `${stage.durationUs / max * 100}%`,
              background: colors[key],
            }} />
          </div>
          <em>{fmt(stage.durationPct)}%</em>
        </button>
        );
      })}
    </div>
  );
}

function Timeline({ operators, stages, selectedStage, selectedOp, colors, onPick, onStage }) {
  if (!operators.length) return <div className="timeline-empty">当前结构单元没有可展示的算子。</div>;
  const start = Math.min(...operators.map((op) => op.startNs));
  const end = Math.max(...operators.map((op) => op.endNs));
  const span = Math.max(end - start, 1);
  const streams = [...new Set(operators.map((op) => op.stream))].sort((a, b) => a - b);
  const stageByKey = Object.fromEntries(stages.map((stage) => [stageKey(stage), stage]));
  const orderedStageKeys = [...new Set(
    [...operators].sort((a, b) => a.startNs - b.startNs).map(operatorStageKey),
  )];
  const orderedStages = [
    ...orderedStageKeys.map((key) => stageByKey[key]).filter(Boolean),
    ...stages.filter((stage) => !orderedStageKeys.includes(stageKey(stage))),
  ];
  return (
    <div className="timeline">
      <div className="axis">
        <span>0</span><span>{fmt(span / 2_000_000)} ms</span><span>{fmt(span / 1_000_000)} ms</span>
      </div>
      {streams.map((stream) => (
        <div className="lane" key={stream}>
          <label>Stream {stream}</label>
          <div className="lane-track">
            {operators.filter((op) => op.stream === stream).map((op) => {
              const left = (op.startNs - start) / span * 100;
              const width = Math.max((op.endNs - op.startNs) / span * 100, 0.12);
              const operatorSelected = selectedOp?.index === op.index;
              const dim = selectedOp
                ? !operatorSelected
                : selectedStage && operatorStageKey(op) !== selectedStage;
              return <button
                key={op.index}
                title={`${op.unitVariant ? `${op.unitVariant} · ` : ""}${operatorDisplayName(op)} · ${fmt(op.durationUs)} μs`}
                className={`kernel ${dim ? "dim" : ""} ${operatorSelected ? "selected" : ""}`}
                style={{ left: `${left}%`, width: `${width}%`, background: colors[operatorStageKey(op)] }}
                onClick={() => onPick(op)}
              />;
            })}
          </div>
        </div>
      ))}
      <div className="timeline-stage-list">
        <div className="timeline-stage-heading">
          <b>功能模块</b>
          <span>选择全部、模块或时间线中的单个算子</span>
        </div>
        <div className="timeline-stage-buttons">
          <button
            className={`total-stage ${!selectedStage && !selectedOp ? "active" : ""}`}
            onClick={() => onStage(null)}
          >
            <i>全</i>
            <span><b>总模块</b><small>{operators.length} 个算子 · 全部时间区间</small></span>
          </button>
          {orderedStages.map((stage, index) => (
            <button
              key={stageKey(stage)}
              className={
                selectedStage === stageKey(stage) ||
                operatorStageKey(selectedOp) === stageKey(stage) ? "active" : ""
              }
              onClick={() => onStage(stageKey(stage))}
            >
              <i style={{ background: colors[stageKey(stage)] }}>{index + 1}</i>
              <span>
                <b>{stage.name}</b>
                <small>{stage.unitVariant ? `${stage.unitPosition}. ${stage.unitVariant} · ` : ""}{fmt(stage.durationUs)} us</small>
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function ClassificationDonut({ classifications }) {
  const baseRows = [
    { name: "核心计算", label: "计算", color: "#6a91ff" },
    { name: "通信", label: "通信", color: "#f4b860" },
    { name: "辅助算子", label: "辅助", color: "#66758c" },
  ].map((item) => ({
    ...item,
    ...(classifications.find((entry) => entry.name === item.name) || {
      count: 0, durationUs: 0,
    }),
  }));
  const totalDuration = baseRows.reduce((sum, row) => sum + row.durationUs, 0);
  const rows = baseRows.map((row) => ({
    ...row,
    share: totalDuration ? row.durationUs / totalDuration * 100 : 0,
  }));
  const adviceText = {
    核心计算: "优先检查高耗时、低 MFU 的核心算子，结合 Shape 评估矩阵尺寸、精度和布局是否匹配硬件。",
    通信: "检查通信与计算的重叠窗口、集合通信粒度和 Stream 排布，减少串行等待与同步空洞。",
    辅助算子: "关注量化、归一化、数据重排及缓存管理等碎片化算子，优先评估融合与批量化。",
  };
  const advice = [...rows].sort((a, b) => b.share - a.share);
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  let cursor = 0;
  return (
    <div className="classification-summary">
      <div className="classification-upper">
        <div className="donut">
          <svg viewBox="0 0 144 144" aria-label="计算、通信和辅助算子耗时占比">
            <circle className="donut-track" cx="72" cy="72" r={radius} />
            {rows.map((row) => {
              const offset = -cursor / 100 * circumference;
              cursor += row.share;
              return row.share > 0 && <circle
                key={row.name}
                className="donut-segment"
                cx="72"
                cy="72"
                r={radius}
                stroke={row.color}
                strokeDasharray={`${Math.max(row.share / 100 * circumference - 4, 1)} ${circumference}`}
                strokeDashoffset={offset}
              />;
            })}
          </svg>
          <div><b>{rows.reduce((sum, row) => sum + row.count, 0)}</b><span>算子总数</span></div>
        </div>
        <div className="classification-legend">
          {rows.map((row) => <div key={row.name}>
            <i style={{ background: row.color }} />
            <span><b>{row.label}</b><small>{row.count} 个算子 · {fmt(row.durationUs)} us</small></span>
            <strong>{fmt(row.share)}%</strong>
          </div>)}
        </div>
      </div>
      <div className="optimization-advice">
        <div className="advice-heading">
          <b>优化建议</b>
          <span>按当前耗时占比排序</span>
        </div>
        <div className="advice-list">
          {advice.map((row, index) => (
            <article key={row.name}>
              <i>{String(index + 1).padStart(2, "0")}</i>
              <div>
                <b>{row.label} · {fmt(row.share)}%</b>
                <p>{adviceText[row.name]}</p>
              </div>
            </article>
          ))}
        </div>
      </div>
    </div>
  );
}

function CallStack({ value }) {
  const frames = String(value || "").split(/\s*(?:->|→)\s*/).filter(Boolean);
  return (
    <ol className="call-stack">
      {frames.length
        ? frames.map((frame, index) => <li key={`${index}-${frame}`}><span>{index + 1}</span><code>{frame}</code></li>)
        : <li className="empty-stack">暂无调用栈证据</li>}
    </ol>
  );
}

function JobDialog({ open, onClose, onLoaded }) {
  const [api, setApi] = useState("");
  const [token, setToken] = useState("");
  const [form, setForm] = useState({
    mode: "codex_skill", agent_provider: "codex", agent_model: "", model_name: "GLM5.2",
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
  const [modelRefresh, setModelRefresh] = useState(0);
  const [error, setError] = useState("");
  const terminal = ["succeeded", "failed", "cancelled"];

  useEffect(() => {
    if (!open) return;
    const isLoopback = new Set(["localhost", "127.0.0.1", "::1", "[::1]"])
      .has(window.location.hostname);
    const isLocal = Boolean(window.__NSYSSCOPE_LOCAL__) || isLoopback;
    const defaultApi = isLocal ? "." : "/analyzer-api";
    // A local one-command launch must always talk to the Analyzer that served
    // this page. Reusing an old remote/manual URL makes Provider discovery
    // appear broken even though the local service is healthy.
    setApi(isLocal
      ? defaultApi
      : (localStorage.getItem("nsysscope.api.v2") ?? defaultApi));
    setToken(sessionStorage.getItem("nsysscope.token") || "");
  }, [open]);

  useEffect(() => {
    if (!open || !api) return;
    let stopped = false;
    let retryTimer;
    let controller;
    let failures = 0;

    async function requestHealth(base) {
      const response = await fetch(`${base.replace(/\/$/, "")}/api/health`, {
        headers: { "X-NsysScope-Token": token },
        signal: controller.signal,
        cache: "no-store",
      });
      const text = await response.text();
      let payload;
      try {
        payload = JSON.parse(text);
      } catch {
        payload = { detail: text.trim() };
      }
      if (!response.ok) {
        const error = new Error(payload.detail || `HTTP ${response.status}`);
        error.status = response.status;
        throw error;
      }
      return payload;
    }

    async function checkHealth() {
      controller = new AbortController();
      if (failures === 0) {
        setHealth({
          state: "checking", message: "正在连接本地 Analyzer…", providers: null,
        });
      }
      try {
        let payload;
        try {
          payload = await requestHealth(api);
        } catch (cause) {
          if (cause.status !== 404 || api === ".") throw cause;
          payload = await requestHealth(".");
          if (!stopped) {
            setApi(".");
          }
        }
        if (stopped) return;
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
        if (stopped || cause.name === "AbortError") return;
        failures += 1;
        const retryDelay = Math.min(5000, 500 * (2 ** Math.min(failures - 1, 4)));
        setHealth({
          state: "offline",
          message: `${cause.message || "连接失败"}；正在自动重试…`,
          providers: null,
        });
        retryTimer = setTimeout(checkHealth, retryDelay);
      }
    }

    retryTimer = setTimeout(checkHealth, 250);
    return () => {
      stopped = true;
      clearTimeout(retryTimer);
      controller?.abort();
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
      cache: "no-store",
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
  }, [api, form.agent_provider, form.mode, health.providers, modelRefresh, open, token]);

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
          <div>
            <b>{health.state === "ready"
              ? "本地执行器就绪"
              : health.state === "warning"
                ? "本地执行器已连接"
                : health.state === "checking"
                  ? "正在连接本地执行器"
                  : "本地执行器正在重连"}</b>
            <span>{health.message}</span>
          </div>
        </div>
        <details className="advanced-connection">
          <summary>高级连接设置</summary>
          <div className="form-grid">
            <label className="span-2">Analyzer API<input required value={api} onChange={(e) => setApi(e.target.value)} placeholder="/analyzer-api" /></label>
            <label className="span-2">API Token（直接连接远端时填写）<input type="password" value={token} onChange={(e) => setToken(e.target.value)} autoComplete="off" /></label>
          </div>
        </details>
        <div className="form-grid">
          <label>执行模式<select value={form.mode} onChange={set("mode")}><option value="codex_skill">Agent 静态分析</option><option value="existing_package">导入结果目录（无需 Agent）</option></select></label>
          <label>阶段<select value={form.stage} onChange={set("stage")}><option value="prefill">Prefill</option><option value="decode">Decode</option></select></label>
          {form.mode === "codex_skill" && <>
            <label>Agent Provider<select value={form.agent_provider} onChange={setProvider}>
              <option value="codex" disabled={health.providers && !health.providers.codex?.ready}>Codex CLI{health.providers && !health.providers.codex?.ready ? "（未就绪）" : ""}</option>
              <option value="comate" disabled={health.providers && !health.providers.comate?.ready}>Comate Zulu{health.providers && !health.providers.comate?.ready ? "（未登录或未就绪）" : ""}</option>
            </select></label>
            <label className="model-picker">
              <span className="field-title">
                Agent 基座模型
                {modelCatalog.state === "ready" && <em>{modelCatalog.models.length} 个可选</em>}
              </span>
              <span className="model-select-row">
                <select value={form.agent_model} onChange={set("agent_model")} disabled={modelCatalog.state === "loading"}>
                  <option value="">{modelCatalog.state === "loading"
                    ? "正在读取模型…"
                    : `自动 / Provider 默认${modelCatalog.default_model ? `（${modelCatalog.default_model}）` : ""}`}</option>
                  {modelCatalog.models.map((model) =>
                    <option key={model.id} value={model.id}>{model.label}</option>
                  )}
                </select>
                <button
                  type="button"
                  className="model-refresh"
                  onClick={() => setModelRefresh((value) => value + 1)}
                  disabled={modelCatalog.state === "loading" || !provider?.ready}
                >
                  刷新
                </button>
              </span>
              {modelCatalog.state === "ready" && modelCatalog.models.length > 0 &&
                <small>已从 {form.agent_provider === "codex" ? "Codex CLI" : "Comate Zulu"} 自动读取</small>}
            </label>
            {provider && !provider.ready && <div className="provider-warning span-2">{provider.message}</div>}
            {modelCatalog.state === "error" && <div className="provider-warning span-2">模型列表读取失败，将使用 Provider 默认模型：{modelCatalog.message}</div>}
            {modelCatalog.state === "ready" && modelCatalog.models.length === 0 &&
              <div className="provider-warning span-2">Provider 已就绪，但没有返回可选模型；可点击“刷新”重试。</div>}
          </>}
          <label>模型<select required value={form.model_name} onChange={set("model_name")}>
            <option value="GLM5.2">GLM5.2</option>
            <option value="DeepSeekV4">DeepSeekV4</option>
            <option value="KiMi3">KiMi3</option>
          </select></label>
          <label>硬件<select required value={form.hardware} onChange={set("hardware")}>
            <option value="Nvidia B200">Nvidia B200</option>
            <option value="Nvidia B300">Nvidia B300</option>
          </select></label>
          {form.mode === "existing_package" ? <>
            <label className="span-2">六表 / NsysScope 结果目录<input required value={form.existing_package_path} onChange={set("existing_package_path")} placeholder="/path/to/result-package" /></label>
            {form.existing_package_path.trim().toLowerCase().endsWith(".zip") &&
              <label className="span-2">ZIP 解压结果保存目录<input required value={form.result_path} onChange={set("result_path")} placeholder="/path/to/new-result-package（必须为空或不存在）" /></label>}
            <p className="package-hint span-2">目录内有 analysis.json 时直接展示；只有六张规范 CSV 也能自动转换，不要求额外 sidecar。支持 csv/ 子目录和旧版平铺目录。</p>
          </> : <>
            <label className="span-2">Nsight 报告<input required value={form.report_path} onChange={set("report_path")} placeholder="/path/to/report.nsys-rep" /></label>
            <label>Model config<input required value={form.config_path} onChange={set("config_path")} placeholder="/path/to/config.json" /></label>
            <label>部署 YAML / 脚本<input required value={form.launch_path} onChange={set("launch_path")} placeholder="/path/to/launch.yaml" /></label>
            <label className="span-2">模型源码根目录<input required value={form.source_path} onChange={set("source_path")} placeholder="/path/to/sglang/source" /></label>
            <label className="span-2">设计说明（可选）<input value={form.design_path} onChange={set("design_path")} placeholder="/path/to/design.md" /></label>
            <label className="span-2">结果保存目录<input required value={form.result_path} onChange={set("result_path")} placeholder="/path/to/result-package（必须为空或不存在）" /></label>
          </>}
          <label>输出前缀<input required value={form.prefix} onChange={set("prefix")} pattern="[a-zA-Z0-9_-]+" /></label>
          {!importingPackage && <label className="span-2">分析范围与硬性要求<textarea value={form.notes} onChange={set("notes")} placeholder="例如：只分析 GLM5.2 的单个非 shared Indexer 层，不要扩展为 4 层周期。" /><small>Agent 必须按这里限定重复单元和分支；无法满足时任务应失败，不能静默改用其他范围。</small></label>}
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
  const [selectedUnit, setSelectedUnit] = useState(null);
  const [selectedStage, setSelectedStage] = useState(null);
  const [selectedOp, setSelectedOp] = useState(null);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [error, setError] = useState("");
  const [jobOpen, setJobOpen] = useState(false);
  const fileRef = useRef(null);

  useEffect(() => {
    fetch("/demo-analysis.json?default=kimi3-fixed-v2-20260731", {
      cache: "no-store",
    }).then((r) => r.json()).then((payload) => {
      setData(payload);
      setSelectedOp(null);
    })
      .catch(() => setError("示例数据加载失败"));
  }, []);

  const colors = useMemo(() => data ? Object.fromEntries(
    data.stages.map((stage, index) => [stageKey(stage), COLORS[index % COLORS.length]])
  ) : {}, [data]);

  const visibleOperators = useMemo(() => {
    if (!data) return [];
    return selectedUnit
      ? data.operators.filter((op) => operatorUnitKey(op) === selectedUnit)
      : data.operators;
  }, [data, selectedUnit]);

  const visibleStages = useMemo(() => {
    if (!data) return [];
    // Functional stages are pattern-level aggregates.  Keep the layer/unit
    // selector for operator and timeline drill-down, but never split the
    // module-duration panel by layer.
    return data.stages;
  }, [data]);

  const visibleClassifications = useMemo(() => {
    if (!data || !selectedUnit) return data?.classifications || [];
    const names = {
      core: "核心计算",
      communication: "通信",
      auxiliary: "辅助算子",
    };
    return Object.entries(names).map(([categoryName, name]) => {
      const rows = visibleOperators.filter((op) => op.category === categoryName);
      return {
        name,
        count: rows.length,
        durationUs: rows.reduce((sum, op) => sum + (op.durationUs || 0), 0),
      };
    });
  }, [data, selectedUnit, visibleOperators]);

  const filtered = useMemo(() => {
    if (!data) return [];
    const operators = visibleOperators.filter((op) => {
      const stageOk = !selectedStage || operatorStageKey(op) === selectedStage;
      const categoryOk = category === "all" || op.category === category;
      const q = query.trim().toLowerCase();
      const queryOk = !q || `${operatorDisplayName(op)} ${op.name} ${op.module} ${op.stage} ${op.unitVariant || ""} ${op.unitId || ""}`.toLowerCase().includes(q);
      return stageOk && categoryOk && queryOk;
    });
    return operators.sort(category === "all"
      ? (a, b) => a.startNs - b.startNs || a.index - b.index
      : (a, b) => b.durationUs - a.durationUs || a.startNs - b.startNs);
  }, [data, visibleOperators, selectedStage, category, query]);

  async function loadFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (parsed.schemaVersion !== "1.0" || !Array.isArray(parsed.operators)) {
        throw new Error("schema");
      }
      setData(parsed);
      setSelectedUnit(null);
      setSelectedStage(null);
      setSelectedOp(null);
      setError("");
    } catch {
      setError("无法读取：请选择 NsysScope analysis.json（schemaVersion 1.0）");
    }
  }

  const loadAnalysis = useCallback((payload) => {
    setData(payload);
    setSelectedUnit(null);
    setSelectedStage(null);
    setSelectedOp(null);
    setError("");
  }, []);

  if (!data) return <main className="loading"><div className="pulse" />正在载入分析数据…</main>;

  const topStage = [...visibleStages].sort((a, b) => b.durationUs - a.durationUs)[0];
  const compute = visibleClassifications.find((x) => x.name === "核心计算");
  const classificationTotal = visibleClassifications.reduce((sum, row) => sum + row.durationUs, 0);
  const computePct = classificationTotal ? compute?.durationUs / classificationTotal * 100 : 0;
  const selectedUnitData = data.units?.find((unit) => unitKey(unit) === selectedUnit);
  const primaryDurationUs = selectedUnitData?.representativeWallSpanUs ||
    data.summary.primaryDurationUs ||
    data.summary.normalizedLayerDurationUs ||
    data.summary.totalDurationUs;
  const overlapPct = calculateOverlapPct(visibleOperators);
  const displayModel = String(data.metadata.model || "Unknown model")
    .split(" / ", 1)[0].split(" (", 1)[0].slice(0, 80);
  const highlightedStage = operatorStageKey(selectedOp) || selectedStage;

  const selectStage = (stageName) => {
    setSelectedOp(null);
    setSelectedStage((current) => (
      stageName && current !== stageName ? stageName : null
    ));
  };

  const selectOperator = (operator) => {
    setSelectedStage(null);
    setSelectedOp(operator);
  };

  const selectUnit = (value) => {
    setSelectedUnit((current) => current === value ? null : value);
    setSelectedStage(null);
    setSelectedOp(null);
  };

  const unitNote = selectedUnitData
    ? `位置 ${selectedUnitData.position} · Layer ${selectedUnitData.layerId ?? "—"} · 代表墙钟跨度`
    : data.summary.heterogeneous
      ? `${data.summary.unitLayerCount || data.units?.length || 1} 个异构单元/周期 · ${(data.summary.distinctUnitVariants || []).join(" / ")}`
      : `${data.summary.stableSamples} 个稳定样本`;

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

        {data.units?.length > 1 && <section className="unit-selector">
          <div><span>STRUCTURAL UNITS</span><b>结构周期</b><small>先选 KDA / MLA 等真实单元，再查看各自模块</small></div>
          <button className={!selectedUnit ? "active" : ""} onClick={() => selectUnit(null)}>
            <b>完整周期</b><small>{data.units.length} 个结构单元</small>
          </button>
          {data.units.map((unit) => (
            <button
              key={unitKey(unit)}
              className={selectedUnit === unitKey(unit) ? "active" : ""}
              onClick={() => selectUnit(unitKey(unit))}
            >
              <b>{unit.position}. {unit.variant || unit.id}</b>
              <small>Layer {unit.layerId ?? "—"} · {fmt(unit.representativeWallSpanUs)} μs</small>
            </button>
          ))}
        </section>}

        <section className="metrics">
          <Metric
            label={selectedUnitData ? `${selectedUnitData.variant || selectedUnitData.id} 单元耗时` : (data.summary.durationLabel || "单层耗时")}
            value={fmt(primaryDurationUs / 1000, 3)}
            suffix=" ms"
            note={unitNote}
            accent
          />
          <Metric label="算子数量" value={visibleOperators.length} suffix="" note={`${visibleStages.length} 个功能阶段`} />
          <Metric label="最高耗时功能模块" value={fmt(topStage?.durationUs)} suffix=" μs" note={topStage ? `${topStage.name} · ${topStage.unitVariant || "通用"}` : "—"} />
          <Metric label="核心计算占比" value={fmt(computePct)} suffix="%" note={`${compute?.count || 0} 个核心算子`} />
          <Metric label="Overlap 率" value={fmt(overlapPct)} suffix="%" note="基于算子时间区间并集" />
        </section>

        <section className="analysis-grid">
          <article className="panel stage-panel">
            <div className="panel-head"><div><span>BREAKDOWN</span><h2>功能模块耗时</h2></div><small>点击模块联动筛选</small></div>
            <StageBars stages={visibleStages} selected={highlightedStage} onSelect={selectStage} colors={colors} />
          </article>

          <article className="panel classification-panel">
            <div className="panel-head"><div><span>COMPOSITION</span><h2>计算 · 通信 · 辅助算子占比</h2></div><small>按三类累计耗时归一化</small></div>
            <ClassificationDonut classifications={visibleClassifications} />
          </article>
        </section>

        <section className="panel timeline-panel">
          <div className="panel-head"><div><span>CUDA TIMELINE</span><h2>多 Stream 时间线</h2></div><small>按真实 Stream 展示 · 点击算子查看证据</small></div>
          <Timeline
            operators={visibleOperators}
            stages={visibleStages}
            selectedStage={selectedStage}
            selectedOp={selectedOp}
            colors={colors}
            onPick={selectOperator}
            onStage={selectStage}
          />
        </section>

        <section className="operator-workbench">
          <article className="panel table-panel">
            <div className="panel-head table-tools">
              <div><span>OPERATORS</span><h2>算子明细</h2></div>
              <div className="filters">
                <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="搜索算子、功能模块…" />
                {["all", "core", "communication", "auxiliary"].map((key) => (
                  <button key={key} className={category === key ? "active" : ""} onClick={() => setCategory(key)}>
                    {key === "all" ? "全部算子" : CATEGORY[key].label}
                  </button>
                ))}
              </div>
            </div>
            <div className="table-wrap">
              <table className="operator-table">
                <colgroup>
                  <col className="col-index" />
                  <col className="col-operator" />
                  <col className="col-category" />
                  <col className="col-duration" />
                  <col className="col-share" />
                  <col className="col-shape" />
                  <col className="col-mfu" />
                </colgroup>
                <thead><tr><th className="center">#</th><th>算子名</th><th>分类</th><th className="numeric">耗时(us)</th><th className="numeric">占比(%)</th><th>Shape</th><th className="numeric">MFU(%)</th></tr></thead>
                <tbody>
                  {filtered.map((op) => (
                    <tr key={op.index} className={selectedOp?.index === op.index ? "selected" : ""} onClick={() => selectOperator(op)}>
                      <td className="mono muted center">{op.index}</td>
                      <td><div className="op-title"><i style={{ background: colors[operatorStageKey(op)] }} /><div><b title={operatorDisplayName(op)}>{operatorDisplayName(op)}</b><span>{op.unitVariant ? `${op.unitVariant} · ` : ""}{op.stage}</span></div></div></td>
                      <td><span className={`category ${op.category}`}>{CATEGORY[op.category]?.label}</span></td>
                      <td className="mono numeric">{fmt(op.durationUs)}</td>
                      <td className="mono numeric">{fmt(op.durationPct)}</td>
                      <td className="mono shape-cell" title={op.shape || ""}>{op.shape || "—"}</td>
                      <td className="mono numeric">{op.mfu != null ? fmt(op.mfu) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
          <aside className="panel evidence-panel">
            <div className="panel-head"><div><span>EVIDENCE</span><h2>算子证据</h2></div><small>{selectedOp ? `#${selectedOp.index}` : "选择算子"}</small></div>
            {selectedOp ? <div className="evidence-body">
              <div className="evidence-title">
                <i style={{ background: colors[operatorStageKey(selectedOp)] }}>{selectedOp.index}</i>
                <div><b>{operatorDisplayName(selectedOp)}</b><span>{selectedOp.unitVariant ? `${selectedOp.unitVariant} · ` : ""}{selectedOp.stage}</span></div>
                <em className={`category ${selectedOp.category}`}>{CATEGORY[selectedOp.category]?.label}</em>
              </div>
              <div className="evidence-summary">
                <div><span>平均耗时</span><b>{fmt(selectedOp.durationUs)} us</b></div>
                <div><span>最小 / 最大</span><b>{fmt(selectedOp.minUs)} / {fmt(selectedOp.maxUs)} us</b></div>
                <div><span>MFU</span><b>{selectedOp.mfu != null ? `${fmt(selectedOp.mfu)}%` : "N/A"}</b></div>
                <div><span>Shape</span><b>{selectedOp.shape || "N/A"}</b></div>
                <div><span>设备 / Stream</span><b>GPU {selectedOp.device} / Stream {selectedOp.stream}</b></div>
                <div><span>结构单元</span><b>{selectedOp.unitPosition || "—"}. {selectedOp.unitVariant || selectedOp.unitId || "N/A"}</b></div>
                <div><span>技术模块</span><b>{selectedOp.module || "N/A"}</b></div>
              </div>
              <section className="evidence-section">
                <h3>功能说明</h3><p>{selectedOp.introduction || "暂无说明"}</p>
              </section>
              <section className="evidence-section call-stack-section">
                <h3>Python 调用链</h3><CallStack value={selectedOp.pythonFunction} />
              </section>
              <details className="cuda-symbol"><summary>查看完整 CUDA 符号</summary><code>{selectedOp.fullName}</code></details>
            </div> : <div className="evidence-empty">从左侧选择一个算子查看详细信息。</div>}
          </aside>
        </section>
      </section>
      <JobDialog open={jobOpen} onClose={() => setJobOpen(false)} onLoaded={loadAnalysis} />
    </main>
  );
}
