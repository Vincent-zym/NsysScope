# glm5_next Prefill 性能分析报告

- 结果包：`/home/users/zhongyuanming/record_NsysScope_analysis/glm5_next_prefill_analysis_0828_1`
- 硬件：Nvidia B200 × 8；采样 rank：device 0
- 阶段：prefill；trace：`sglang_glm5next_chunk32_p_base_0828.sqlite`
- 模型结构：45 层 = 11 × DSA-MoE + 34 × KDA-MoE（DSA : KDA = 1 : 3），未启用 MTP

# 1. 结论

1. **MoE Experts 计算是绝对热点**，占一个结构单元（4 层）的 47.2%（34.46 ms / 72.93 ms）。其中 4 个路由专家 GEMM 各 2.70 ms、MFU 仅 36.8%，是整个模型耗时上限的决定项。
2. **小算子占比过高**：辅助算子 227 个、31.03 ms，占单元 42.6%，与核心计算（47.1%）几乎相当。单是 `elementwise_kernel` 一族就有 26 次启动、合计 10.89 ms，占单元 14.9% —— 算子融合空间很大。
3. **DSA 层比 KDA 层贵 61%**：24.33 ms/层 vs 15.08 ms/层。差距全部来自 DSA 独有的两块：核心稀疏注意力 7.06 ms 与 Indexer 稀疏索引 3.97 ms。好在 DSA 只占 11/45 层。
4. **步间间隙 12.64 ms（1.56%）几乎全在步的头部**——锚点到第一层之间的调度/输入准备区间（该区间忙碌 2.80 ms，其余为空洞），层间与尾部几乎无空隙。

#### 潜在优化点

1. **辅助小算子融合（收益最大）**：优先 `elementwise_kernel`（26 次 / 10.89 ms）、`mhc_post_tilelang_kernel`（8 次 / 2.86 ms）、`per_token_group_quant_flat_kernel`（19 次 / 1.60 ms）。mHC 超连接整块 6.36 ms 里前置 GEMM 的 MFU 只有 11.4%，属于典型的访存/启动开销主导，适合与相邻 norm、量化算子合并。
2. **MoE Experts 提效**：4 个路由专家 GEMM MFU 36.8%、MBU 5.1%，两头都不饱和，说明受 tile 划分/专家负载不均影响，值得核对 EP=8 下的专家分布与 group GEMM 配置；共享专家的 gate_up 可考虑与路由专家合批。
3. **DSA Indexer 评估**：3.97 ms（5.4%）为 DSA 层独有，可评估 `index_topk=2048` 的取值与 Indexer query 上投影（1536→4096）的精度/规模是否可降。
4. **步头部气泡**：>50 µs 空洞均值 10.5 ms/步（最高 14.3 ms），集中在 batch 分配、KV 记账、position / attention metadata 之间的 host 等待，可考虑对该段做 CUDA graph 捕获或把 host 侧准备提前来压缩。

<h1 style="margin:0">2. 链路与算子耗时分析</h1>
<p style="margin:0"><b>结论</b>：耗时集中在 MoE Experts 计算（单元占比 47.2%），且辅助小算子占到 42.6%（227 个算子、31.03 ms），与核心计算量级相当——算子融合的空间明显大于单个 kernel 调优的空间。DSA 层的稀疏注意力 + Indexer（合计 11.03 ms）是层间差异的唯一来源。</p>
<p style="margin:0">&nbsp;</p>
<p style="margin:0">以下是具体分析过程：</p>
<p style="margin:0"><b>分析思路</b>：glm5_next 的注意力层按 DSA : KDA = 1 : 3 交替（<code>layer_types</code> 中每 4 层出现一次 <code>deepseek_sparse_attention</code>），因此以 <b>4 层为一个分析单元</b>（1 × DSA-MoE + 3 × KDA-MoE），单元耗时 72.93 ms。forward 链路层面则按 45 层整体统计。</p>
<h2 style="margin:0">2.1 forward 链路耗时</h2>
<p style="margin:0">当前配置：chunked-prefill-size = 32768、TP = 8、EP = 8、PP = 1、<code>--disable-overlap-schedule</code>；无 CUDA graph（prefill eager）。</p>
<p style="margin:0"><b>Token 链路耗时</b></p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">阶段</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">Forward 耗时</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">Target 耗时</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">Draft 耗时</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">Token间间隙</th>
</tr>
<tr>
<th style="border:1px solid #999;background:#d9e2f3;text-align:center;vertical-align:middle">耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">808.76</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">796.12</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—（未启用投机）</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">12.64</td>
</tr>
<tr>
<th style="border:1px solid #999;background:#d9e2f3;text-align:center;vertical-align:middle">耗时百分比</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">100%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">98.44%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.56%</td>
</tr>
</table>
<p style="margin:0"><b>Target 内部构成</b></p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">环节</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">调度与输入准备</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">DSA-MoE 层 × 11</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">KDA-MoE 层 × 34</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">其他</th>
</tr>
<tr>
<th style="border:1px solid #999;background:#d9e2f3;text-align:center;vertical-align:middle">耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.80</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">267.57</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">512.57</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">13.18</td>
</tr>
<tr>
<th style="border:1px solid #999;background:#d9e2f3;text-align:center;vertical-align:middle">占 forward step</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.35%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">33.08%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">63.38%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.63%</td>
</tr>
<tr>
<th style="border:1px solid #999;background:#d9e2f3;text-align:center;vertical-align:middle">单层耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">24.33</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">15.08</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
</tr>
</table>
<h2 style="margin:0">2.2 算子耗时分析</h2>
<p style="margin:0">以下口径为<b>一个 4 层结构单元</b>内、稳定样本（640 个）逐算子平均耗时之和，单元合计 72.93 ms，下表覆盖其中 72.13 ms（98.9%，余量为未归类的零散算子）。</p>
<p style="margin:0"><b>按功能模块划分（按执行顺序）</b></p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">功能模块</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">pattern</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">mHC 超连接混合与合并</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">DSA 输入与投影</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">DSA Indexer 稀疏索引</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">DSA 核心注意力</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">DSA 输出与通信</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">KDA 输入投影与状态预处理</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">KDA 核心状态更新</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">KDA 输出重建与通信</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">MoE 输入与路由</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">MoE Experts 计算</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">MoE 输出与通信</th>
</tr>
<tr>
<th style="border:1px solid #999;background:#d9e2f3;text-align:center;vertical-align:middle">耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">72.93</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">6.36</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.98</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.97</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">7.06</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.15</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4.56</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.19</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.42</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.06</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">34.46</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4.91</td>
</tr>
<tr>
<th style="border:1px solid #999;background:#d9e2f3;text-align:center;vertical-align:middle">耗时百分比</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">100%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">8.72%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.35%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">5.45%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">9.67%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.57%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">6.25%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4.38%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4.69%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.82%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">47.25%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">6.73%</td>
</tr>
</table>
<ul style="margin:0;padding-left:22px">
<li>DSA 相关模块只在单元内 1 层出现，KDA 相关模块在 3 层出现，MoE 与 mHC 每层都有——比较模块间成本时需按出现层数折算。</li>
<li>单层最贵的单个 kernel：DSA 层是 <code>fmhaSm100fKernel_...H512PagedKvDense</code>（7.06 ms），KDA 层是带门控的 delta-rule 状态更新（约 1.07 ms/层）。</li>
</ul>
<p style="margin:0">&nbsp;</p>
<p style="margin:0"><b>按算子类型划分</b></p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">算子类型</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">核心计算</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">通信</th>
<th style="border:1px solid #999;background:#b4c7e7;text-align:center;vertical-align:middle">小算子（辅助算子）</th>
</tr>
<tr>
<th style="border:1px solid #999;background:#d9e2f3;text-align:center;vertical-align:middle">算子数量</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">66</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">8</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">227</td>
</tr>
<tr>
<th style="border:1px solid #999;background:#d9e2f3;text-align:center;vertical-align:middle">耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">34.33</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">6.76</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">31.03</td>
</tr>
<tr>
<th style="border:1px solid #999;background:#d9e2f3;text-align:center;vertical-align:middle">耗时百分比</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">47.07%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">9.27%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">42.55%</td>
</tr>
</table>
<p style="margin:0">小算子 Top 5（单元内合计 / 启动次数）：<code>elementwise_kernel</code> 10.89 ms / 26 次、<code>act_and_mul_kernel</code> 3.19 ms / 4 次、<code>mhc_post_tilelang_kernel</code> 2.86 ms / 8 次、<code>CatArrayBatchedCopy_vectorized</code> 2.77 ms / 4 次、<code>mhc_pre_big_fuse_with_norm_tilelang_kernel</code> 1.89 ms / 8 次。</p>
<p style="margin:0">效率上界参考：本包内单算子 MFU 最高 78.65%、MBU 最高 78.18%；而占比最大的 4 个路由专家 GEMM 只有 MFU 36.8% / MBU 5.1%。</p>

# 3. 算子分析工具数据

popo 发布页面链接：待补（人工发布后填入）

# 4. 物料

nsys 文件：`/home/users/zhongyuanming/dev_dir/v15.8.1.0/test_dir/sglang_glm5next_chunk32_p_base_0828.sqlite`（包内副本 `trace/sglang_glm5next_chunk32_p_base_0828.sqlite`）
