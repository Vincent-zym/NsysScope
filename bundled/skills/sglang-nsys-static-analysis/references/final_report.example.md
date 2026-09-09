<h1 style="margin:0">GLM5.2 prefill 典型shape Nsys TimeLine分析结果</h1>
<h1 style="margin:0">1. 输入配置</h1>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">模型</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">GLM5.2</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">硬件</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">Nvidia B200</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">阶段</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">prefill</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">代码版本</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">引擎配置</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">TP=8、DP=1（未启用 dp-attention）、CP=8、megamoe、EAGLE 投机解码</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">运行时 shape</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">chunked-prefill-size=65536、batch_size=1、已启用 EAGLE（speculative_num_draft_tokens=4）</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">nsys 文件</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sglang_glm52_chunk64_p_kernel_opt_0824.nsys-rep</code></td>
</tr>
</table>
<h1 style="margin:0">2. 分析结果</h1>
<h2 style="margin:0">2.1 整体耗时统计</h2>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">阶段</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">Forward step</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">Target 主模型</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">Draft 模型</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">Token间间隙</th>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">857.06</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">817.27</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">28.57</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">11.22</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">耗时百分比</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">100%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">95.36%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.33%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.31%</td>
</tr>
</table>
<h2 style="margin:0">2.2 Target部分耗时统计</h2>
<p style="margin:0"><b>分析思路</b>：GLM5.2 稀疏 MoE 区按 non-shared(full) Indexer : shared Indexer = 1 : 3 交替，以 <b>4 层为一个分析pattern</b>（1 × Full-Indexer + 3 × Shared-Indexer），pattern耗时 41.51 ms。</p>
<h3 style="margin:0">2.2.1 整体耗时统计</h3>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">环节</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">Target 主模型耗时</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">调度与输入准备</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">MLA-NSA-FullIndexer-MoE 层 × 21</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">MLA-NSA-SharedIndexer-MoE 层 × 57</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">其他</th>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">817.27</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">6.26</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">269.18</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">535.29</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">6.53</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">占 forward step</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">95.36%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.73%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">31.41%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">62.46%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.76%</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">单层耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">12.82</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">9.39</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
</tr>
</table>
<h3 style="margin:0">2.2.2 按功能模块划分统计</h3>
<p style="margin:0">以下口径为<b>一个重复 pattern</b>内、稳定样本逐算子平均耗时之和，pattern 合计 41.51 ms，下表覆盖其中 41.10 ms（99.0%，余量为未归类的零散算子）。</p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">功能模块</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">pattern总耗时</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">Attention 输入与投影</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">NSA Indexer 稀疏索引</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">MLA 输入吸收与 KV Cache</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">MLA 稀疏注意力核心</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">Attention 输出与投影</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">MoE 输入与共享专家</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">MoE 路由与 TopK</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">MoE Experts 计算与输出</th>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">41.51</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.48</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.55</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">5.48</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">12.71</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.35</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.17</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.67</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">11.70</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">耗时百分比</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">100%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.57%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">8.54%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">13.19%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">30.61%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">8.08%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.81%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4.03%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">28.18%</td>
</tr>
</table>
<h3 style="margin:0">2.2.3 按算子大类划分统计</h3>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">算子类型</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">pattern总耗时</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">核心计算</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">通信</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">小算子（辅助算子）</th>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">算子数量</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">184</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">44</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">5</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">135</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">41.51</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">30.26</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.71</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">10.14</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">耗时百分比</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">100%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">72.89%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.71%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">24.42%</td>
</tr>
</table>
<h3 style="margin:0">2.2.4 按算子小类划分统计</h3>
<p style="margin:0">按算子合计耗时从高到低排列，Top 15；同一 kernel 跨多个模块出现时，耗时/次数为跨模块合计，所属模块列出全部（按各自贡献从高到低排序）。</p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">算子名称</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">所属模块</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">耗时(ms)</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">占pattern耗时</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">启动次数</th>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sparse_attn_fwd_kernel_head64</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MLA 稀疏注意力核心</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">12.71</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">30.61%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_fp8_fp4_mega_moe_impl&lt;max_m=8256,H=6144,I=2048,E=256,topk=8&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MoE Experts 计算与输出</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">11.34</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">27.31%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_fp8_fp4_gemm_1d1d_impl&lt;N=6144,K=16384&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">Attention 输出与投影</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.95</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4.69%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>topk_optimized_kernel</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">NSA Indexer 稀疏索引</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.88</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4.52%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>gatherTopK&lt;float,uint,2,false&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MoE 路由与 TopK</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.36</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.27%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">12</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_fp8_mqa_logits&lt;32,128&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">NSA Indexer 稀疏索引</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.12</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.69%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>_dequantize_k_cache_paged_kernel</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MLA 输入吸收与 KV Cache</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.03</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.48%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>per_token_group_quant_8bit_kernel&lt;bf16,fp8_e4m3&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">Attention 输出与投影、NSA Indexer 稀疏索引</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.98</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.37%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">6</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>set_mla_kv_buffer_kernel</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MLA 输入吸收与 KV Cache</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.87</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.09%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>elementwise_kernel&lt;direct_copy&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MLA 输入吸收与 KV Cache、NSA Indexer 稀疏索引</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.77</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.86%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">13</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>_quantize_k_cache_fast_kernel</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MLA 输入吸收与 KV Cache</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.73</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.75%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>concat_mla_absorb_q_kernel</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MLA 输入吸收与 KV Cache</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.72</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.73%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>ncclDevKernel_AllGather_RING_LL</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MLA 输入吸收与 KV Cache、NSA Indexer 稀疏索引</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.71</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.71%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">5</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_fp8_fp4_gemm_1d1d_impl&lt;N=16384,K=2048&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">Attention 输入与投影</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.68</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.63%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>rmsnorm_per_token_quant_kernel&lt;16,128,true,true&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MoE 输入与共享专家、Attention 输入与投影</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.60</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.44%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">8</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">Top 15 累积耗时</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">37.42</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">90.14%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">78</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">pattern总耗时</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">41.51</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">100%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
</tr>
</table>
<h3 style="margin:0">2.2.5 按核心计算统计</h3>
<p style="margin:0">仅统计核心计算类算子，按执行顺序排列；MFU/MBU 为该算子各次出现的均值，缺 shape 证据时留空。</p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">算子名称</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">所属模块</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">shape</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">耗时(ms)</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">占pattern耗时</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">MFU</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">MBU</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">启动次数</th>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_fp8_fp4_gemm_1d1d_impl&lt;N=2624,K=6144&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">Attention 输入与投影</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>(M=8192,N=2624,K=6144)</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.38</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.92%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">61.81%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">14.41%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_fp8_fp4_gemm_1d1d_impl&lt;N=16384,K=2048&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">Attention 输入与投影</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>(M=8192,N=16384,K=2048)</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.68</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.63%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">72.24%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">23.56%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_fp8_fp4_gemm_1d1d_impl&lt;N=4096,K=2048&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">NSA Indexer 稀疏索引</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>(M=8192,N=4096,K=2048)</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.05</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.11%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">65.67%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">24.80%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_fp8_fp4_gemm_1d1d_impl&lt;N=128,K=6144&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">NSA Indexer 稀疏索引</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>(M=8192,N=128,K=6144)</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.01</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.04%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">19.28%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">44.80%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>nvjet_tst_32x64_64x16_1x2_2cta_h_bz_TNN</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">NSA Indexer 稀疏索引</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>(M=8192,N=32,K=6144)</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.02</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.06%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">6.08%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">53.88%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_fp8_mqa_logits&lt;32,128&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">NSA Indexer 稀疏索引</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>—</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.12</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.69%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>nvjet_tst_256x256_64x4_2x1_2cta_v_bz_TNT</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MLA 输入吸收与 KV Cache</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>(M=524288,N=512,K=192)</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.56</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.34%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">32.95%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">66.39%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sparse_attn_fwd_kernel_head64</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MLA 稀疏注意力核心</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>—</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">12.71</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">30.61%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">Attention 输出与投影</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>(M=524288,N=256,K=512)</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.54</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.30%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">45.15%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">74.43%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_fp8_fp4_gemm_1d1d_impl&lt;N=6144,K=16384&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">Attention 输出与投影</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>(M=8192,N=6144,K=16384)</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.95</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4.69%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">75.30%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">8.62%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_fp8_fp4_gemm_1d1d_impl&lt;N=4096,K=6144&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MoE 输入与共享专家</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>(M=8192,N=4096,K=6144)</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.52</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.24%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">71.16%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">13.84%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_fp8_fp4_gemm_1d1d_impl&lt;N=6144,K=2048&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MoE 输入与共享专家</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>(M=8192,N=6144,K=2048)</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.27</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.66%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">67.29%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">23.87%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>nvjet_tst_128x128_64x8_2x2_2cta_h_bz_TNT</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MoE 路由与 TopK</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>(M=8192,N=256,K=6144)</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.13</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">0.30%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">36.49%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">44.69%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>sm100_fp8_fp4_mega_moe_impl&lt;max_m=8256,H=6144,I=2048,E=256,topk=8&gt;</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">MoE Experts 计算与输出</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle"><code>(M=65536,N=6144,K=6144)</code></td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">11.34</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">27.31%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">38.80%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">4</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">核心计算合计</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">30.26</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">72.89%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">44</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">pattern总耗时</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">41.51</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">100%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
</tr>
</table>
<h2 style="margin:0">2.3 Draft部分耗时统计</h2>
<h3 style="margin:0">2.3.1 整体耗时统计</h3>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">环节</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">Draft 模型耗时</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">MLA-NSA-FullIndexer-MoE 层（draft） × 1</th>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#b4c7e7">其他</th>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">28.57</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">9.33</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">19.25</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">占 forward step</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">3.33%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">1.09%</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">2.25%</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:center;vertical-align:middle;background-color:#d9e2f3">单层耗时(ms)</th>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">9.33</td>
<td style="border:1px solid #999;text-align:center;vertical-align:middle">—</td>
</tr>
</table>
<h1 style="margin:0">3. 输出物料</h1>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border:1px solid #999;text-align:left;margin:0">
<tr>
<th style="border:1px solid #999;text-align:left;vertical-align:middle;background-color:#d9e2f3">工具版本</th>
<td style="border:1px solid #999;text-align:left;vertical-align:middle"><code>sglang-nsys-static-analysis</code>，sha256 <code>69c4e23d5a9f8daa82ebf0cef44437a7506a9f15da50d56815aea06cbe985999</code></td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:left;vertical-align:middle;background-color:#d9e2f3">工具启动指令</th>
<td style="border:1px solid #999;text-align:left;vertical-align:middle">nsys/sqlite: /home/users/zhongyuanming/dev_dir/v15.4.7.4/sglang_glm52_chunk64_p_kernel_opt_0824.sqlite<br>model: GLM5.2<br>stage: prefill<br>hardware: Nvidia B200<br>config: /home/users/zhongyuanming/NsysScope/backend/model_configs/GLM5.2.json<br>deployment YAML/script: /home/users/zhongyuanming/dev_dir/v15.4.7.4/p_start.sh<br>model source root: /home/users/zhongyuanming/test_data/aiak_sglang/baidu/hac-aiacc/aiak_sglang/python</td>
</tr>
<tr>
<th style="border:1px solid #999;text-align:left;vertical-align:middle;background-color:#d9e2f3">工具产物</th>
<td style="border:1px solid #999;text-align:left;vertical-align:middle"><code>analysis.json</code>（前端契约）、<code>final_report.md</code>（本报告）、<code>nsysscope-package.json</code>（包清单）、<code>csv/</code>（规范化表）、<code>xlsx/</code>（对应工作簿）、<code>trace/sglang_glm52_chunk64_p_kernel_opt_0824.sqlite</code>（导出的 SQLite trace，原始 nsys 文件：<code>/home/users/zhongyuanming/record_NsysScope_analysis/glm52_prefill_analysis_0829_1/trace/sglang_glm52_chunk64_p_kernel_opt_0824.sqlite</code>）</td>
</tr>
</table>