# 第一部分实验：需求画像、能力画像与基础调度

本项目实现论文第一部分的可运行原型：复杂请求包含预先定义的并行子任务；节点 DT 根据静态能力画像提供推荐；KDN 根据候选推荐和当前队列进行分配；在 EdgeSimPy 中完成输入传输、FIFO 排队、执行、结果返回与汇聚。

本阶段使用合成参数，画像静态且准确，即调度器看到的画像与仿真真实状态一致；不包含状态信息陈旧、模型能力漂移、模型训练、云端卸载或模型层切分。图像分类、目标检测是任务实例标签，没有加载真实视觉模型或图片数据集。`quality_by_input` 是合成统计能力参数，不是实测准确率，也不是单次推理是否正确。

详细定义、公式、方法对比和论文表述边界见 `docs/第一部分实验说明.md`。

## 1. 在哪里运行

在 VS Code 的 **WSL Ubuntu 终端**中运行，项目目录为 `/home/sanyue/edge-infer-task`，使用已经建立的 Conda 环境 **`edge_dt`**。不要改用 Windows Python。

```bash
conda activate edge_dt
cd /home/sanyue/edge-infer-task
python --version
python -c "import sys; print(sys.executable)"
```

当前目标环境为 Python 3.10.22。实验采用官方 EdgeSimPy 提交 `76eb5ead74596bb4240759fa4336f1d6f190c70a`；固定依赖见 `requirements.txt`。后续实际运行版本会写入结果的 `manifest.json`。

## 2. 先检查，再运行

```bash
conda activate edge_dt
cd /home/sanyue/edge-infer-task
python -m pytest
python run.py --suite smoke
```

`smoke` 使用 20 个请求、首个种子、默认中间到达率，运行全部 5 种方法，适合检查流程。它不适合支持正式论文结论。

完整默认实验：

```bash
python run.py --suite standard
```

默认配置是 6 个节点、2 类子任务，每个请求每类恰好一个子任务。标准实验对每种到达率和种子生成 120 个请求：5 个独立种子 × 3 个到达率 × 5 种方法，共 **75 次仿真运行**。同一到达率和种子下，5 种方法复用完全相同的请求轨迹、节点配置和质量画像，属于配对比较。

结果默认写入新的 `/home/sanyue/edge-infer-task/results/stage1_时间戳/`，不会覆盖旧实验；显式指定的输出目录若非空，程序会拒绝写入。

## 3. 常用命令

只比较两个满足 QoS 约束的方法：

```bash
python run.py --suite smoke --methods qos_greedy,global_beam
```

自定义请求数、种子和负载：

```bash
python run.py --suite standard --request-count 120 --seeds 11,22,33,44,55 --arrival-rates 0.5,1,2
```

使用另一份配置或关闭绘图：

```bash
python run.py --config /home/sanyue/edge-infer-task/config/default.json --suite smoke --no-plot
```

重放之前保存的轨迹，将下面路径替换为真实文件：

```bash
python run.py --trace /home/sanyue/edge-infer-task/results/某次实验/traces/rate_1_seed_11.json
```

`--trace` 模式只重放这一份轨迹，忽略请求生成参数；结果中的种子和到达率留空，运行目录为 `trace_replay`，图中标为外部轨迹。原轨迹文件的来源元数据会保留，不能把未知的生成条件写成种子 0 或到达率 0。

独立核查导出的资源时间线、QoS 分母和源码哈希：

```bash
python validate_results.py results/stage1_standard_v2 --expect-runs 75
```

本次交付以 `results/stage1_standard_v2/` 为标准结果，`docs/验证记录.md` 说明已完成的检查；其他结果目录是开发和重放检查记录。

## 4. 五种方法比较什么

| 方法 | 分配规则 | 质量与预计截止时间作为硬约束 |
|---|---|---|
| `round_robin` | 在类型、输入条件和内存兼容的节点中轮询 | 否 |
| `earliest_finish` | 选择预计子任务返回最早的节点 | 否 |
| `quality_first` | 优先较高画像质量，再比较预计返回时间 | 否 |
| `qos_greedy` | 过滤质量/时限不可行节点后，按时延和 DT 推荐分数逐子任务选择 | 是 |
| `global_beam` | 在同一管理 tick 的请求批次中联合分配，使用有限宽度 beam search | 是 |

所有方法看到相同的信息；它们对约束的使用不同，不能把差异解释为某个基线拿不到质量信息。前三种是简单原型基线，后两种的对比用于观察逐任务贪心与联合搜索的差别。这些方法不是 HCIO、CAO 或其他论文算法的完整复现。

`global_beam` 先尽量增加当前批接纳请求数，再比较归一化时延与推荐分数；它是近似搜索，不保证数学意义上的全局最优，也不读取未来到达。每个节点只有一个同时执行的槽位，但可以接纳多个任务排队，并在整次实验中服务多个任务。

## 5. 文件结构

```text
edge-infer-task/
├── run.py                       # 命令行、重复实验、统计、结果导出与绘图
├── validate_results.py          # 独立核对结果、资源时间线及复现哈希
├── requirements.txt             # 固定运行依赖
├── config/default.json          # 场景、节点、任务、网络和负载参数
├── stage1/
│   ├── scenario.py              # 需求/能力画像、参数校验、轨迹生成与重放
│   ├── matching.py              # DT 主动推荐：质量/时延/内存归一化裕量
│   ├── planner.py               # 候选筛选、时间预测、贪心及 beam 分配
│   └── engine.py                # EdgeSimPy 网络、执行 FIFO、汇聚和容量审计
├── tests/test_stage1.py          # 机制和边界验证
└── docs/第一部分实验说明.md       # 建模和实验解释
```

核心源码包含中文注释。阅读顺序建议为 `scenario.py → matching.py → planner.py → engine.py → run.py`。

## 6. 结果怎么看

每次实验目录主要包含：

| 文件 | 内容 |
|---|---|
| `config_snapshot.json` | 实际加载的配置快照 |
| `traces/*.json` | 各负载和种子的固定请求轨迹 |
| `runs/*/方法/requests.csv` | 每个请求的接纳/拒绝/完成、时延和 QoS 判定 |
| `runs/*/方法/tasks.csv` | 已接纳子任务的模型、节点、推荐分数和时间线 |
| `runs/*/方法/nodes.csv` | 节点接纳数、CPU 利用率和执行内存峰值 |
| `metrics.csv` | 每次运行的汇总指标 |
| `summary.csv` | 同负载同方法的种子间均值、样本标准差和有效种子数 |
| `comparison.png` / `comparison.pdf` | QoS、拒绝率、平均和 P95 完成时延对比 |
| `manifest.json` | Python/依赖环境、命令参数、代码/配置/轨迹 SHA-256 |
| `source_snapshot/` | 本次实际运行的源代码副本，配合代码哈希追溯实现 |
| `validation.json` | 运行 validate_results.py 后生成的独立审计结果 |

优先同时看 **QoS 满足率、拒绝率、未完成率和完成时延**。时延统计包含所有已完成请求，包括超时或画像质量不达标的已完成请求；不包含被拒绝或截断未完成请求。方法可能通过多拒绝请求降低完成样本时延，所以不能只比较时延。

`summary.csv` 的 `_std` 和图中误差棒是种子间**样本标准差**，不是置信区间。只有一个有效种子时标准差留空，图中按无误差棒显示。`tasks.csv` 的 `predicted_return_s` 与 `returned_s` 可用于检查预测误差。

## 7. 参数来源与当前结论边界

老师要求对应的内容是：通用黑盒子任务、需求/能力画像、DT 主动推荐、KDN 全局协调，以及可以先做纯仿真。第一部分的范围已经确认为画像匹配与基础调度。

6 个节点、2 类任务、120 个请求、负载、内存/网络/质量数值、匹配权重 `(0.5, 0.4, 0.1)`、`matching_weight=0.15` 和 beam 宽度是本次实现的起步参数，不是老师指定值或真实测量值。修改场景使用 `config/default.json`；两项匹配权重分别由顶层键 `matching_weights` 和 `matching_weight` 配置。

第一部分验证静态画像与调度机制。正式论文仍需补充近年文献基线、参数来源或真实 profiling 依据，以及负载、权重、beam 宽度和 tick 的敏感性实验；状态新鲜度和模型能力变化留待后续部分。

## 8. 使用的开源平台与引用

网络采用官方 EdgeSimPy 的 `Simulator`、`NetworkFlow` 和默认公平分配，CPU 单槽 FIFO 与请求汇聚由本项目扩展。链路容量由 MB/s 换算为 MB/tick；传播时延由适配层显式补齐。平台代码及论文来源见 [EdgeSimPy 官方仓库](https://github.com/EdgeSimPy/EdgeSimPy)和[平台论文](https://doi.org/10.1016/j.future.2023.06.013)。

```bibtex
@article{souza2023edgesimpy,
  author = {Paulo S. Souza and Tiago Ferreto and Rodrigo N. Calheiros},
  title = {EdgeSimPy: Python-Based Modeling and Simulation of Edge Computing Resource Management Policies},
  journal = {Future Generation Computer Systems},
  volume = {148},
  pages = {446--459},
  year = {2023},
  doi = {10.1016/j.future.2023.06.013}
}
```
