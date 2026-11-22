# 附录 A 源码目录地图

只列与课程相关的部分，路径相对于仓库根目录。

```
python/sglang/
  launch_server.py              服务启动入口（sglang serve 的兼容入口）
  lang/                         前端 DSL（第 02 课，非热路径）
    ir.py                       S / System / User / Assistant 语法树
    interpreter.py:271          StreamExecutor，解释执行
    backend/                    外部后端适配（openai / litellm / runtime_endpoint）
  srt/                          运行时主体（全书主线）
    server_args.py:826          PortArgs，三条 ZMQ 通道的定义
    entrypoints/
      engine.py                 Engine，离线入口
      http_server.py            HTTP / OpenAI 兼容层
      engine_info_bootstrap_server.py  transfer engine 注册信息（第 18 课）
    managers/
      tokenizer_manager.py      分词、ReqState、结果聚合
      scheduler.py              调度主循环、组批、KV 分配
      schedule_policy.py        PrefillAdder、优先级策略
      schedule_batch.py         ScheduleBatch / Req（CPU 调度视图）
      tp_worker.py              TP/PP worker，ForwardBatch 构造
      detokenizer_manager.py    token id → 文本
      io_struct.py              跨进程消息定义 + ZMQ 编解码
    model_executor/
      model_runner.py           模型执行、采样
      runner/                   eager / prefill graph / decode graph
    mem_cache/
      radix_cache.py            RadixAttention（第 04 课）
      allocation.py             alloc_for_extend，KV 槽位分配
    model_loader/loader.py      权重加载（第 05 课）
    configs/model_config.py     模型配置与显存预算
    layers/
      attention/                各 attention backend（第 10 课）
      quantization/             量化方案（第 08、09 课）
      moe/                      MoE 与 EP（第 16 课）
    compilation/                torch.compile 接入（第 12 课）
    speculative/                EAGLE / ngram（第 14 课）
    distributed/parallel_state.py  TP/PP/DP/EP 进程组（第 15 课）
    eplb/                       专家负载均衡（第 17 课）
    disaggregation/              PD 分离（第 18 课）
      prefill.py                P 侧 bootstrap / transfer 队列
      decode.py                 D 侧 prealloc / transfer 队列
      common/conn.py            KV 传输抽象与 bootstrap 会合点
    constrained/                约束解码（第 19 课）
      grammar_manager.py        grammar 队列与轮询
      xgrammar_backend.py       默认后端（FSM + 位图）
python/sglang/kernels/          Triton / AOT kernel（第 13 课）
```

## 三个容易找错的位置

| 想找 | 实际位置 |
|---|---|
| `cuda_graph_runner.py` | 已拆分到 `model_executor/runner/` |
| `compile/` | 现名 `compilation/` |
| 量化 kernel 实现 | 多半在 `python/sglang/kernels/` 或第三方包，不在 `layers/quantization/` |
