# 附录 C 面试题反查表

按问题查课号。带箭头的是该课的具体小节。

## 入门与全局

| 问题 | 课号 |
|---|---|
| SGLang 和 vLLM 的区别？ | 01 |
| `sgl.Engine` 和 `sglang serve` 是同一套吗？ | 01 |
| 一条 serve 命令起几个进程？ | 01 → 03 |
| 显存不够先调什么？ | 01、05 |
| 前端 DSL 是什么？为什么不是主路径？ | 02 |
| 一次请求经过哪些组件？ | 03 |

## 请求链路

| 问题 | 课号 |
|---|---|
| 为什么 tokenizer 和 scheduler 分进程？ | 03 |
| overlap 调度重叠了什么？ | 03 |
| 流式输出怎么保证顺序？`[DONE]` 谁加的？ | 03 |
| KV Cache 在哪一步分配？ | 03 → 05 |
| prefill 和 decode 怎么选？ | 03、06 |

## 缓存与调度

| 问题 | 课号 |
|---|---|
| RadixAttention 和 PagedAttention 的区别？ | 04 |
| 前缀匹配为什么是 O(路径长度)？ | 04 |
| 为什么要分裂节点？ | 04 |
| 引用计数为什么加到根？ | 04 |
| 驱逐为什么从叶子开始？ | 04 |
| page_size 怎么选？ | 04 |
| 连续批处理体现在哪？ | 06 |
| batch_is_full 为什么要粘滞？ | 06 |
| 显存不够 retract 还是 evict？ | 06 |
| 撤回顺序按什么排？ | 06 |
| LPM 怎么算优先级？ | 06 |
| chunked prefill 解决什么问题？ | 07 |
| chunk 大小由谁决定？ | 07 |
| chunked 与 CUDA Graph 冲突在哪？ | 07、11 |

## 加载与精度

| 问题 | 课号 |
|---|---|
| 模型加载分几步？ | 05 |
| `mem-fraction-static` 作用在哪一步？ | 05、01 |
| GQA 的 KV 头数怎么算？ | 05、10 |
| 权重怎么切到 TP rank？ | 05、15 |
| FP16 / BF16 / FP8 / FP4 怎么选？ | 08 |
| per-token 与 per-channel 的区别？ | 08 |
| W4A16 和 W8A8 的差别？ | 08、09 |
| KV Cache 也能量化吗？ | 08 |
| AWQ 的 activation-aware 是什么？ | 09 |
| 为什么加载后要 repack？ | 09 |
| AWQ 什么时候自动升级成 marlin？ | 09 |

## 结构

| 问题 | 课号 |
|---|---|
| MHA / GQA / MQA / MLA 的 KV 开销？ | 10 |
| MLA 为什么能省显存？ | 10 |
| MLA 的 prefill 为什么常用 MHA one-shot？ | 10 |
| SWA 和 DSA 的区别？ | 10 |
| 混合注意力难在哪？ | 10 |

## 执行与加速

| 问题 | 课号 |
|---|---|
| CUDA Graph 解决什么瓶颈？ | 11 |
| 为什么要分桶？桶怎么定？ | 11 |
| capture 前为什么要 warmup？ | 11 |
| graph 显存为什么共享 pool？ | 11 |
| 哪些情况回退 eager？ | 11 |
| Dynamo / FX / Inductor 各做什么？ | 12 |
| 为什么必须切图？ | 12 |
| dynamic shape 怎么标？ | 12 |
| guard 失效会怎样？ | 12 |
| Triton 从源码到 cubin 经过什么？ | 13 |
| MoE 为什么用 JSON 表而不是 autotune？ | 13 |
| late kernel load 为什么危险？ | 13 |
| 推测解码为什么等价？ | 14 |
| tree mask 为什么不能用 causal mask？ | 14 |
| accept_length 和 accept_rate 的区别？ | 14 |
| 为什么需要 draft_extend？ | 14 |

## 分布式与 MoE

| 问题 | 课号 |
|---|---|
| TP / PP / DP / EP / SP / CP 各切什么？ | 15 |
| TP 的通信量怎么估？ | 15 |
| PP 的气泡怎么算？ | 15 |
| DP attention 特殊在哪？ | 15 |
| MoE 一层有哪几个阶段？ | 16 |
| EP 和 TP 在 MoE 上的区别？ | 16 |
| all-to-all 通信量怎么估？ | 16 |
| EPLB 解决什么问题？ | 17 |
| 冗余专家怎么起作用？ | 17 |
| 重排后权重怎么恢复？ | 17 |

## 服务化

| 问题 | 课号 |
|---|---|
| 为什么要 PD 分离？代价是什么？ | 18 |
| P/D 的 TP/PP 为什么可以不一样？ | 18 |
| bootstrap_host / port / room 谁填的？ | 18 |
| P 侧为什么有两个队列？ | 18 |
| D 侧 prealloc 在等什么？ | 18 |
| KV 传失败了请求怎么收场？ | 18 |
| 约束解码和 JSON mode 的区别？ | 19 |
| 编译为什么是异步的？缓存怎么算？ | 19 |
| 每步 mask 的开销怎么估？ | 19 |
| 推测解码下文法怎么截断？ | 19 |
