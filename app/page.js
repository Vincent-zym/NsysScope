"use client";

import { useEffect, useMemo, useRef, useState } from "react";

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

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [selectedStage, setSelectedStage] = useState(null);
  const [selectedOp, setSelectedOp] = useState(null);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [error, setError] = useState("");
  const fileRef = useRef(null);

  useEffect(() => {
    fetch("/demo-analysis.json").then((r) => r.json()).then(setData)
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
      setSelectedOp(null);
      setError("");
    } catch {
      setError("无法读取：请选择 NsysScope analysis.json（schemaVersion 1.0）");
    }
  }

  if (!data) return <main className="loading"><div className="pulse" />正在载入分析数据…</main>;

  const topStage = data.stages[0];
  const highVariance = [...data.operators]
    .filter((op) => op.durationUs > 0)
    .sort((a, b) => (b.diffUs / b.durationUs) - (a.diffUs / a.durationUs))[0];
  const compute = data.classifications.find((x) => x.name === "核心计算");

  return (
    <main>
      <header className="topbar">
        <div className="brand"><i>NS</i><div><b>NsysScope</b><span>GPU INFERENCE PROFILER</span></div></div>
        <div className="report-chip"><span className="status-dot" />{data.metadata.model}</div>
        <button className="import" onClick={() => fileRef.current?.click()}>
          导入 analysis.json
        </button>
        <input ref={fileRef} hidden type="file" accept=".json,application/json" onChange={loadFile} />
      </header>

      <section className="shell">
        <div className="breadcrumb">ANALYSES <span>/</span> {basename(data.metadata.report)}</div>
        <div className="title-row">
          <div>
            <p className="eyebrow">REPEATING UNIT ANALYSIS</p>
            <h1>{data.metadata.model}</h1>
            <p>{data.metadata.repeatingUnit} · {data.metadata.stage?.toUpperCase()} · {data.metadata.hardware}</p>
          </div>
          <div className="run-meta">
            <span>STATUS <b>VALIDATED</b></span>
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

        <section className="panel table-panel">
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
              <thead><tr><th>#</th><th>功能模块 / 算子</th><th>分类</th><th>耗时</th><th>占比</th><th>Shape</th><th>MFU</th><th>波动</th></tr></thead>
              <tbody>
                {filtered.map((op) => (
                  <tr key={op.index} onClick={() => setSelectedOp(op)}>
                    <td className="mono muted">{op.index}</td>
                    <td><div className="op-title"><i style={{ background: colors[op.stage] }} /><div><b>{op.name}</b><span>{op.stage} · {op.module}</span></div></div></td>
                    <td><span className={`category ${op.category}`}>{CATEGORY[op.category]?.label}</span></td>
                    <td className="mono">{fmt(op.durationUs)} μs</td>
                    <td className="mono">{fmt(op.durationPct)}%</td>
                    <td className="mono muted">{op.shape || "—"}</td>
                    <td className="mono">{op.mfu ? `${fmt(op.mfu)}%` : "—"}</td>
                    <td className="mono muted">{fmt(op.diffUs)} μs</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="pipeline">
          <div><span>01</span><b>输入物料</b><p>.nsys-rep · config · YAML · model source</p></div>
          <i>→</i><div><span>02</span><b>静态分析</b><p>边界识别 · 模块映射 · 稳定统计 · MFU</p></div>
          <i>→</i><div><span>03</span><b>标准契约</b><p>analysis.json · 六张 CSV · manifest</p></div>
          <i>→</i><div><span>04</span><b>交互洞察</b><p>阶段 · 时间线 · 算子 · 证据 · 对比</p></div>
        </section>
      </section>

      {selectedOp && <div className="drawer-backdrop" onClick={() => setSelectedOp(null)}>
        <aside className="drawer" onClick={(e) => e.stopPropagation()}>
          <button className="close" onClick={() => setSelectedOp(null)}>×</button>
          <span className="drawer-kicker">OPERATOR #{selectedOp.index}</span>
          <h2>{selectedOp.name}</h2>
          <p className="drawer-module">{selectedOp.stage} · {selectedOp.module}</p>
          <div className="drawer-stats">
            <div><span>AVG</span><b>{fmt(selectedOp.durationUs)} μs</b></div>
            <div><span>MIN / MAX</span><b>{fmt(selectedOp.minUs)} / {fmt(selectedOp.maxUs)}</b></div>
            <div><span>MFU</span><b>{selectedOp.mfu ? `${fmt(selectedOp.mfu)}%` : "N/A"}</b></div>
          </div>
          <h3>功能说明</h3><p>{selectedOp.introduction}</p>
          <h3>Python 调用链</h3><code>{selectedOp.pythonFunction}</code>
          <h3>映射依据</h3><p>{selectedOp.mappingReason}</p>
          <h3>完整 CUDA 符号</h3><code>{selectedOp.fullName}</code>
        </aside>
      </div>}
    </main>
  );
}
