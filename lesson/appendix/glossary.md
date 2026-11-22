# 附录 B 术语表

| 术语 | 含义 | 出现课程 |
|---|---|---|
| TTFT | Time To First Token，首 token 延迟 | 03、06、07 |
| TPOT | Time Per Output Token，decode 每 token 延迟 | 03、06 |
| Req | Scheduler 侧的请求状态（CPU） | 03、06 |
| ReqState | TokenizerManager 侧的请求状态（输出缓冲 + Event） | 03 |
| ScheduleBatch | 调度视图，一批 Req | 03、06 |
| ForwardBatch | 执行视图，摊平后的张量与元数据 | 03、11 |
| RadixAttention | 用 radix tree 管理前缀 KV 复用 | 04 |
| Chunked Prefill | 把长 prefill 切成多段，与 decode 交替 | 07 |
| CUDA Graph | 捕获固定形状的计算图后重放 | 11 |
| EAGLE | 用草稿模型推测多个 token 再由主模型验证 | 14 |
| EP | Expert Parallel，专家切分到不同 rank | 16 |
| EPLB | Expert Parallel Load Balancer，专家负载均衡 | 17 |
| PD 分离 | Prefill / Decode 拆成两组引擎，中间只传 KV | 18 |
| room | PD 配对的 64 位请求键，由路由器分配 | 18 |
| KVPoll | KV 传输的五态状态机 | 18 |
| 约束解码 | 采样前把非法 token 的 logits 置为 -inf | 19 |
| bitmask | 合法 token 位图，按 32 位打包，每步生成一次 | 19 |
