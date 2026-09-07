<h1 style="margin:0">glm5_next prefill 典型shape Nsys TimeLine分析结果</h1>
<h1 style="margin:0">1. 输入配置</h1>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">模型</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">glm5_next</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">硬件</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">Nvidia B200；采样 rank：device 0</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">阶段</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">prefill</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">代码版本</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">引擎配置</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">tp_size=8, ep_size=8, pp_size=1</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">运行时 shape</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">chunked-prefill-size=32768、batch_size=1、未启用 MTP</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">nsys 文件</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sglang_glm5next_chunk32_p_base_0828.nsys-rep</code></td>
</tr>
</table>
<h1 style="margin:0">2. 分析结果</h1>
<p style="margin:0"><b>分析思路</b>：glm5_next 的注意力层按 DSA : KDA = 1 : 3 交替，以 <b>4 层为一个分析单元</b>（1 × DSA-MoE + 3 × KDA-MoE），单元耗时 72.93 ms；forward 链路层面按 45 层整体统计。</p>
<h2 style="margin:0">2.1 整体耗时统计</h2>
<p style="margin:0"><b>Token 链路耗时</b></p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">阶段</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">Forward step</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">Target 主模型</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">Draft 模型</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">Token间间隙</th>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">808.76</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">796.12</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—（未启用投机）</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">12.64</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">耗时百分比</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">100%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">98.44%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.56%</td>
</tr>
</table>
<h2 style="margin:0">2.2 Target耗时统计</h2>
<h3 style="margin:0">2.2.1 整体耗时统计</h3>
<p style="margin:0"><b>Target 内部构成</b></p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">环节</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">调度与输入准备</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">DSA-MoE 层 × 11</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">KDA-MoE 层 × 34</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">其他</th>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.80</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">267.57</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">512.57</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">13.18</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">占 forward step</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.35%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">33.08%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">63.38%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.63%</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">单层耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">24.33</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">15.08</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
</tr>
</table>
<h3 style="margin:0">2.2.2 按功能模块划分统计</h3>
<p style="margin:0">以下口径为<b>一个重复单元</b>内、稳定样本逐算子平均耗时之和，单元合计 72.93 ms，下表覆盖其中 72.13 ms（98.9%，余量为未归类的零散算子）。</p>
<p style="margin:0"><b>按功能模块划分（按执行顺序）</b></p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">功能模块</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">pattern</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">mHC 超连接混合与合并</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">DSA 输入与投影</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">DSA Indexer 稀疏索引</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">DSA 核心注意力</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">DSA 输出与通信</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">MoE Experts 计算</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">MoE 输入与路由</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">MoE 输出与通信</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">KDA 输入投影与状态预处理</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">KDA 核心状态更新</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">KDA 输出重建与通信</th>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">72.93</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">6.36</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.98</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.97</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">7.06</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.15</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">34.46</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.06</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4.91</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4.56</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.19</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.42</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">耗时百分比</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">100%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">8.72%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.35%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">5.45%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">9.67%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.57%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">47.25%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.82%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">6.73%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">6.25%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4.38%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4.69%</td>
</tr>
</table>
<h3 style="margin:0">2.2.3 按算子大类划分统计</h3>
<p style="margin:0">按算子类型划分</p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">算子类型</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">核心计算</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">通信</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">小算子（辅助算子）</th>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">算子数量</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">66</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">8</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">227</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">34.33</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">6.76</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">31.03</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">耗时百分比</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">47.07%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">9.27%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">42.55%</td>
</tr>
</table>
<h3 style="margin:0">2.2.4 按算子小类划分统计</h3>
<p style="margin:0">按算子合计耗时从高到低排列，Top 15；同一 kernel 跨多个模块出现时，耗时/次数为跨模块合计，所属模块列出全部（按各自贡献从高到低排序）。</p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">算子名称</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">所属模块</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">耗时(ms)</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">占单元耗时</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#b4c7e7">启动次数</th>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_fp8_fp4_gemm_1d1d_impl<…></code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MoE Experts 计算、DSA 输入与投影、DSA 输出与通信</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">17.60</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">24.13%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">19</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>elementwise_kernel<…></code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MoE Experts 计算、KDA 输入投影与状态预处理、DSA 输入与投影</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">10.89</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">14.93%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">26</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparseP1VarSeqQ8Kv128PersistentSwapsAbForGen</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">DSA 核心注意力</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">7.06</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">9.67%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>ncclDevKernel_AllReduce_Sum_bf16_RING_LL</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MoE 输出与通信、KDA 输出重建与通信、DSA 输出与通信</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">6.76</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">9.27%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">8</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>chunk_gated_delta_rule_fwd_kernel_h_blockdim64</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">KDA 核心状态更新</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.19</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4.38%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind)0, (bool)1, (bool)0, (bool)0, (bool)0></code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MoE Experts 计算</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.19</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4.37%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>mhc_post_tilelang_kernel</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">mHC 超连接混合与合并</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.86</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.93%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">8</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>CatArrayBatchedCopy_vectorized<…></code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MoE Experts 计算</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.77</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.80%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">KDA 输入投影与状态预处理、KDA 输出重建与通信、DSA Indexer 稀疏索引、DSA 输入与投影、DSA 输出与通信</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.36</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.23%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">15</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>mhc_pre_big_fuse_with_norm_tilelang_kernel</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">mHC 超连接混合与合并</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.89</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.59%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">8</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_tf32_hc_prenorm_gemm_impl<…></code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">mHC 超连接混合与合并</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.61</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.20%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">8</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>per_token_group_quant_flat_kernel<…></code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MoE Experts 计算、MoE 输入与路由、DSA 输入与投影、DSA 输出与通信</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.60</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.20%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">19</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>kpool_topk_transform_kernel<(int)512></code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">DSA Indexer 稀疏索引</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.46</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.00%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>vectorized_elementwise_kernel<…></code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MoE 输出与通信、DSA 输入与投影、MoE 输入与路由、KDA 输入投影与状态预处理</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.27</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.75%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">69</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_mqa_logits<…></code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">DSA Indexer 稀疏索引</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.97</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.33%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1</td>
</tr>
</table>
<h1 style="margin:0">3. 算子分析工具数据</h1>
<p style="margin:0">popo 发布页面链接：待补（人工发布后填入）</p>
<h1 style="margin:0">4. 输出物料</h1>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">工具版本</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sglang-nsys-static-analysis</code>，sha256 <code>e1d7e9fb9c1a614763b47fd03b2ccff72137cbe3731bbd757a320e9ff40cce8e</code></td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">工具启动指令</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">model=glm5_next、stage=prefill、hardware=Nvidia B200；采样 rank：device 0、nsys=<code>sglang_glm5next_chunk32_p_base_0828.sqlite</code>，无额外范围约束</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background:#d9e2f3">工具产物</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>analysis.json</code>（前端契约）、<code>final_report.md</code>（本报告）、<code>nsysscope-package.json</code>（包清单）、<code>csv/</code>（规范化表）、<code>xlsx/</code>（对应工作簿）、<code>trace/sglang_glm5next_chunk32_p_base_0828.sqlite</code>（导出的 SQLite trace，原始 nsys 文件：<code>/home/users/zhongyuanming/record_NsysScope_analysis/glm5_next_prefill_analysis_0828_1/trace/sglang_glm5next_chunk32_p_base_0828.sqlite</code>）</td>
</tr>
</table>