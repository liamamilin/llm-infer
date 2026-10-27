# 核心术语

| 名词 | 通俗解释 |
|---|---|
| 权重 (Weights) | 训练得到的数值参数，不是完整执行程序 |
| Tokenizer | 把文本转换为 Token ID 的组件 |
| Prefill | 处理输入上下文、建立后续生成所需状态的阶段 |
| Decode | 逐步预测和选择下一个 Token 的阶段 |
| KV Cache | 保存历史注意力 Key/Value 状态以减少重复运算 |
| TTFT | 从发出请求到收到首个生成内容 Token 的时间（需说明计时边界） |
| TPOT | 生成阶段每个输出 Token 平均耗时（定义与统计口径需写清） |
| Throughput | 单位时间完成的请求数或 Token 数，需说明单位 |
| Quantization | 用更低精度表示参数或部分计算数据以降低资源需求 |
| Prefix Caching | 对具有相同前缀的请求复用已经计算的状态（需引擎支持） |
| Continuous Batching | 随请求状态动态组织批次，以改善资源利用率的调度方法 |
