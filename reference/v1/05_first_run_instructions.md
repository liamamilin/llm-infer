# 05｜发送给 Codex 的第一轮执行指令

你是《大模型 Inference 工程实践：从模型文件到可用服务》的工程教材实施者。请完整阅读本目录中的：

- `00_project_brief.md`
- `01_toc.md`
- `02_environment_and_models.md`
- `03_production_spec.md`
- `04_acceptance_criteria.md`

然后遵守以下任务合同。

## A. 此次允许完成的范围

**只实现里程碑 M0、M1：真实环境发现、最小模型运行、第 00 章和第 01 章。**

你可以根据全书目录建立空的后续章节索引，但不能生成未经验证的第 02—12 章内容；不能宣布整本书已完成。

## B. 第一步：只读探测并报告

先在当前执行环境探测 OS、CPU、内存、GPU/加速后端、Python/框架、磁盘空间和网络条件，生成 `evidence/environment.json` 与 `docs/environment_report.md`。**不得使用任何聊天历史中的用户设备配置替代这次真实探测**。

在安装大型依赖、下载模型、请求云资源之前，判断是否已有允许使用的缓存与环境，检查用户提供的权限及资源预算。需要权限或预算决定时提问；不要先做潜在收费或大规模变更。

## C. 第二步：选择一个足够小的模型和一条主线

根据实测环境选择**可执行、公开、许可清晰、体量小**的自回归语言模型及最小可用推理栈。候选包括但不限于 `HuggingFaceTB/SmolLM2-135M-Instruct`，但必须核实许可证、可访问性、架构兼容和真实执行结果；如果不能运行，说明原因并做一次合理替代。不要为了性能排名而优先选大模型。

选择一个主后端；优先保证 CPU/MPS/CUDA 实际可执行，不要强行在无 NVIDIA GPU 的机器上要求运行 CUDA 专用栈。把选择理由写入 `docs/adr/0001-mainline-model-and-runtime.md`。

固定模型 revision commit hash、关键依赖版本，产生锁文件与 `configs/environment.lock.json`。

## D. 第三步：先写并执行实验（不是先写长文）

实现至少四个实验：

- `E00-01`：探测环境并输出机器可读的运行 profile。
- `E00-02`：最小模型加载、生成与解码，断言至少新增 1 个合法 Token。
- `E01-01`：读取真实模型文件清单和配置，并区分权重、配置、Tokenizer 资产。
- `E01-02`：确认模型 revision、核对必要文件及 Tokenizer 编码/解码过程。

每个实验必须有可直接运行的入口与清楚的 PASS/FAIL 断言。不要将自然语言答案完全一致当作必备断言。不要仅用 mock 或网络示例代替真实模型加载。

完整运行全部必选实验，留存 `evidence/runs/` 中的实际 JSON 结果与日志。第二次从新进程重跑关键 smoke test。任何命令失败都必须留记录并修复、重跑；无法修复就真实标 FAIL/BLOCKED。

## E. 第四步：验收通过后再写两章

基于实际实验、日志和型号写出：

- `docs/book/00_environment.md`：真实机器探测、依赖准备、最小生成闭环与故障排查。
- `docs/book/01_model_assets.md`：模型仓库、文件结构、权重/配置/Tokenizer、模型 revision、文件格式、运行逻辑归属，以及对应的可执行检查实验。

章节既要解释必要概念，也要给出读者可直接使用的命令与代码、真实观察、边界、自动验收入口与故障排查。模型文件如有差异，只描述**实际存在**的内容。

执行静态与文档一致性检查。确保正文中的命令能匹配代码仓库，并且对读者没有隐式、未记录的人工操作要求。

## F. 工作规范（不可破坏）

1. **先跑实验，后写教材；运行证据优先于模型自信。**
2. 不得伪造日志、版本、benchmark、截图或 PASS 状态。
3. 禁止泄漏令牌、私人目录内容、设备序列号和其他敏感数据。
4. 不得因某条路径受限便悄悄替换实验目标或删除失败证据。
5. 必选实验有一个失败就不能报告章节验收通过。
6. 不得一次性推进到第 02 章；结束时等待用户批准。
7. 如果当前 Codex 没有可执行终端/没有网络访问，停止于准备阶段，标记 BLOCKED，并明确说明缺少的权限与条件。不要假装已经运行。

## G. 第一轮必须交付的可检查成果

1. 环境报告 + 运行环境锁 + 模型/后端选型 ADR。
2. `experiments/ch00/`、`experiments/ch01/` 和必要验证脚本。
3. 两章中文正文（不是提纲），命令与本地代码真实对应。
4. 带真实执行时间、版本、命令、退出码、断言结果与证据路径的报告。
5. 一个清晰的最终总结：
   - 实际探测到了什么环境（脱敏）；
   - 选了哪个模型和引擎，为什么；
   - 4 个必选实验分别 PASS/FAIL/BLOCKED；
   - 重新执行验证是否成功；
   - 哪些内容暂未验证；
   - 如何从零开始运行；
   - 需要用户决定什么，以及**下一轮准备做什么**。

## H. 可复制的最终验收摘要格式

```text
Milestone: M1
Target profile: <actual>
Model: <actual-ID>@<commit>
Engine: <actual-name-and-version>
E00-01: <PASS|FAIL|BLOCKED>
E00-02: <PASS|FAIL|BLOCKED>
E01-01: <PASS|FAIL|BLOCKED>
E01-02: <PASS|FAIL|BLOCKED>
Repeat run: <PASS|FAIL|BLOCKED>
Chapter 00: <REVIEWED|NOT_READY>
Chapter 01: <REVIEWED|NOT_READY>
Evidence root: <actual-path>
Remaining blockers: <actual-items-or-none>
Next action: Wait for user acceptance before Chapter 02.
```

填入真实值，不能把样例文本直接当成结果报告。

**现在请从环境只读探测开始执行。**
