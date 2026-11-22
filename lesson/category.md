# SGLang Course 课程目录
> SGLANG INFERENCE SYSTEMS，一共19节课
## MODULE 01 基础与请求主链
从第一次调用，走通 SGLang 的架构与请求执行路径。（LESSON 01–03）
- 01 介绍、环境搭建与快速上手
  项目起源、安装方式、离线/在线推理与 OpenAI 兼容 API。
- 02 双层体系架构全景
  前端语言层、DSL、解释执行器、Backend Adapter 与 SRT。
- 03 请求端到端生命周期
  TokenizerManager、Scheduler、ModelRunner 与结果返回。

## MODULE 02 缓存、加载与调度
理解高性能推理运行时如何管理请求、Token 与 KV Cache。（LESSON 04–07）
- 04 RadixAttention
  Radix Tree、前缀匹配、引用计数与缓存驱逐。
- 05 模型加载端到端详解
  从 ServerArgs、ModelConfig 到权重加载与后处理。
- 06 Scheduler 核心流程
  连续批处理、Prefill / Decode 决策与 Token 预算。
- 07 Chunked Prefill
  平衡 Decode 延迟、批次预算与 KV Cache 压力。

## MODULE 03 模型结构与精度压缩
把模型表示、精度压缩与推理框架实现放在同一条链中。（LESSON 08–10）
- 08 模型量化基础
  FP32 / FP16 / FP8 / FP4、量化粒度与方案分类。
- 09 AWQ 量化深度解读
  Activation-aware、AWQConfig、Linear 接入与 Marlin。
- 10 LLM Attention 架构及变体
  MHA、GQA、MLA、SWA、DSA 与 Hybrid Attention。

## MODULE 04 图执行、编译器与推测解码
向下理解 Kernel 启动、图捕获、编译流水线与生成加速。（LESSON 11–14）
- 11 SGLang CUDA Graph
  Capture、Warmup、Replay、分桶与 Graph Backend。
- 12 SGLang torch.compile
  TorchDynamo、FX、Inductor Lowering 与运行时守卫。
- 13 Triton 编译全流程
  Python AST、TTIR、IR Lowering 与 GPU Kernel。
- 14 SGLang 推测解码
  Draft-Verify-Accept、Tree Attention 与 EAGLE / MTP。

## MODULE 05 分布式推理与 MoE
从集合通信进入专家并行，再解决真实部署中的负载不均衡。（LESSON 15–17）
- 15 分布式推理原理
  集合通信，以及 TP、PP、DP、EP、SP、CP 的选型。
- 16 分布式推理与 EP 并行
  MoE Router、All-to-All、Expert Compute 与 Combine。
- 17 EPLB 专家负载均衡
  冗余专家、负载统计、专家重排与权重恢复。

---
## MODULE 06 服务化能力
从单机请求链走向真实部署与结构化输出。（LESSON 18–19）
- 18 PD 分离
  Prefill / Decode 两组引擎、bootstrap 配对、KV 传输与失败恢复。
- 19 约束解码
  JSON Schema / Regex / EBNF 编译、位图 mask、FSM 推进与推测解码截断。

---
## 课程目标
把“会调用”升级为 **能解释、能定位、能优化**
1. 独立搭建并调用 SGLang：完成离线 Engine、在线服务与兼容 API 调用。
2. 解释一次请求的完整链路：从入口调度、模型执行到 token 返回。
3. 建立性能分析框架：理解缓存、量化、图执行与推测解码的边界。
4. 为分布式与 MoE 选择方案：分析并行策略、通信代价与专家负载问题。
5. 部署 PD 分离并解释一次 KV 传输的配对与失败路径。
6. 解释约束解码的三段链路：编译、mask、FSM 推进。