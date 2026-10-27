"""论文第一部分的实验入口：固定轨迹、多方法、多负载、多种子与结果导出。"""
import argparse
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import statistics
import sys

from stage1.scenario import load_config, nodes_from_config, generate_requests, save_trace, load_trace
from stage1.engine import Experiment
from stage1.planner import METHODS


def write_csv(path, rows):
    """用 UTF-8 BOM 方便 Windows Excel 阅读中文；缺测值留空，不写成零。"""
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, p):
    """线性插值分位数；空样本返回 None，不能把没有完成任务当成零时延。"""
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def summarize(result):
    rows = result["requests"]
    total = len(rows)
    completed = [r for r in rows if r["status"] == "completed"]
    latencies = [r["latency_s"] for r in completed]
    # 质量为画像统计水平：qos_success_rate 表示“满足所设质量水平和时限的
    # 请求比例”，不能解释成这些请求的所有实际模型输出都正确。
    return dict(request_count=total, completed_count=len(completed),
        qos_success_rate=sum(r["qos_ok"] for r in rows) / total if total else 0,
        rejection_rate=sum(r["status"] == "rejected" for r in rows) / total if total else 0,
        unfinished_rate=sum(r["status"] == "unfinished" for r in rows) / total if total else 0,
        # drain 在所有请求截止之后结束；unfinished 也已经超时，不能遗漏。
        timeout_rate=sum(r["status"] == "unfinished" or (r["status"] == "completed" and not r["on_time"])
                         for r in rows) / total if total else 0,
        quality_failure_rate=sum(not r["quality_ok"] for r in completed) / total if total else 0,
        mean_completed_latency_s=statistics.mean(latencies) if latencies else None,
        p95_completed_latency_s=percentile(latencies, 0.95),
        mean_cpu_utilization=statistics.mean(n["cpu_utilization"] for n in result["nodes"]),
        mean_scheduler_wall_ms=1000 * result["scheduler_wall_s"] / max(1, result["decision_count"]),
        scheduler_wall_s=result["scheduler_wall_s"], wall_runtime_s=result["wall_runtime_s"],
        simulation_duration_s=result["simulation_duration_s"],
        max_network_capacity_ratio=result["max_network_capacity_ratio"],
        network_flow_count=result["network_flow_count"],
        residual_execution_memory_mb=result["residual_execution_memory_mb"])


METRICS = ("qos_success_rate", "rejection_rate", "unfinished_rate", "timeout_rate",
           "quality_failure_rate", "mean_completed_latency_s", "p95_completed_latency_s",
           "mean_cpu_utilization", "mean_scheduler_wall_ms", "wall_runtime_s")


def aggregate(rows):
    result = []
    groups = sorted(set((r["arrival_rate"], r["method"]) for r in rows))
    for rate, method in groups:
        group = [r for r in rows if r["arrival_rate"] == rate and r["method"] == method]
        item = dict(arrival_rate=rate, method=method, seed_count=len(group))
        for metric in METRICS:
            values = [r[metric] for r in group if r[metric] is not None]
            item[metric + "_mean"] = statistics.mean(values) if values else None
            item[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else None
            item[metric + "_valid_seeds"] = len(values)
        result.append(item)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(aggregate_rows, output):
    """标准静态科研图；误差棒为种子间样本标准差，不是置信区间。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = {
        "qos_success_rate": "Request QoS satisfaction",
        "rejection_rate": "Request rejection",
        "mean_completed_latency_s": "Latency of completed requests (s)",
        "p95_completed_latency_s": "P95 latency of completed requests (s)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (metric, title) in zip(axes.flat, titles.items()):
        for method in dict.fromkeys(r["method"] for r in aggregate_rows):
            values = [r for r in aggregate_rows if r["method"] == method
                      and r[metric + "_mean"] is not None]
            ax.errorbar([r["arrival_rate"] if r["arrival_rate"] is not None else 0 for r in values],
                        [r[metric + "_mean"] for r in values],
                        yerr=[r[metric + "_std"] or 0 for r in values],
                        marker="o", capsize=3, label=method)
        ax.set(xlabel="Arrival rate (requests/s)", ylabel=title)
        if any(r["arrival_rate"] is None for r in aggregate_rows):
            ax.set_xlabel("Replayed workload")
            ax.set_xticks([0], ["external trace"])
        if metric in ("qos_success_rate", "rejection_rate"):
            ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Synthetic static profiles | mean ± sample SD across seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/default.json")
    parser.add_argument("--suite", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--request-count", type=int)
    parser.add_argument("--seeds", help="逗号分隔的随机种子，例如 11,22,33")
    parser.add_argument("--arrival-rates", help="逗号分隔的请求到达率，例如 0.5,1,2")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--trace", type=Path, help="可选：重放已有 trace JSON，忽略生成参数")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    nodes = nodes_from_config(config)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else (
        config["seeds"][:1] if args.suite == "smoke" else config["seeds"])
    rates = [float(s) for s in args.arrival_rates.split(",")] if args.arrival_rates else (
        [config["arrival_rates"][1]] if args.suite == "smoke" and len(config["arrival_rates"]) > 1
        else config["arrival_rates"])
    count = args.request_count if args.request_count is not None else (
        min(20, config["request_count"]) if args.suite == "smoke" else config["request_count"])
    methods = args.methods.split(",")
    if count <= 0 or not seeds or not rates or any(not 0 < r < float("inf") for r in rates):
        parser.error("请求数和到达率必须为正，种子和到达率列表不能为空")
    if any(m not in METHODS for m in methods) or len(set(methods)) != len(methods):
        parser.error(f"methods 必须是无重复的可选方法: {METHODS}")
    if len(set(seeds)) != len(seeds) or len(set(rates)) != len(rates):
        parser.error("种子和到达率不能重复，否则会伪造独立重复次数")
    stamp = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y%m%d_%H%M%S")
    output = args.output or Path(__file__).parent / "results" / ("stage1_" + stamp)
    if output.exists() and any(output.iterdir()):
        parser.error(f"输出目录非空，请换一个新目录以保留旧结果: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_snapshot.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    jobs = [(None, None, load_trace(args.trace))] if args.trace else [
        (seed, rate, generate_requests(config, seed, rate, count)) for rate in rates for seed in seeds]
    root = Path(__file__).parent
    # 结果附当次可执行源码副本；以后改算法也能复现这次结果。
    source_files = [root / "run.py", *sorted((root / "stage1").glob("*.py"))]
    for source in source_files:
        destination = output / "source_snapshot" / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for name in ("requirements.txt", "requirements-lock.txt"):
        if (root / name).exists():
            shutil.copyfile(root / name, output / "source_snapshot" / name)
    (output / "source_snapshot" / "config").mkdir(exist_ok=True)
    shutil.copyfile(args.config, output / "source_snapshot" / "config" / "default.json")
    manifest = dict(created_at=datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        framework="EdgeSimPy", framework_commit="76eb5ead74596bb4240759fa4336f1d6f190c70a",
        versions={p: importlib.metadata.version(p) for p in ("edge_sim_py", "Mesa", "networkx", "matplotlib")},
        arguments=vars(args) | {"config": str(args.config), "output": str(output),
                                "trace": str(args.trace) if args.trace else None},
        config_sha256=sha(args.config), code_sha256={str(p.relative_to(root)): sha(p) for p in source_files},
        data_source="synthetic unless externally supplied and explicitly documented",
        uncertainty_scope="static exact profiles; no AoI, drift, training or real inference",
        trace_hashes={})
    all_metrics = []
    for seed, rate, requests in jobs:
        label = "trace_replay" if args.trace else f"rate_{rate:g}_seed_{seed}"
        trace_path = output / "traces" / (label + ".json")
        trace_path.parent.mkdir(exist_ok=True)
        if args.trace:
            shutil.copyfile(args.trace, trace_path)  # 保留外部轨迹的原始来源元数据。
        else:
            save_trace(requests, trace_path)
        manifest["trace_hashes"][label] = sha(trace_path)
        # 所有策略读取同一个不可变请求元组，环境与质量画像也完全相同。
        for method in methods:
            result = Experiment(config, nodes, requests, method).run()
            directory = output / "runs" / label / method
            write_csv(directory / "requests.csv", result["requests"])
            write_csv(directory / "tasks.csv", result["tasks"])
            write_csv(directory / "nodes.csv", result["nodes"])
            metrics = dict(arrival_rate=rate, seed=seed, method=method, **summarize(result))
            all_metrics.append(metrics)
            write_csv(output / "metrics.csv", all_metrics)
            print(f"{label} {method}: QoS={metrics['qos_success_rate']:.3f}, "
                  f"rejected={metrics['rejection_rate']:.3f}, wall={metrics['wall_runtime_s']:.2f}s", flush=True)
    summary = aggregate(all_metrics)
    write_csv(output / "summary.csv", summary)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if not args.no_plot:
        draw(summary, output)
    print(f"实验结束，结果目录：{output.resolve()}")


if __name__ == "__main__":
    main()
