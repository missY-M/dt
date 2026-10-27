"""独立审计已导出的实验数据，防止只看汇总图而遗漏资源或统计错误。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def boolean(value):
    if value not in ("True", "False"):
        raise AssertionError(f"非法布尔值: {value}")
    return value == "True"


def close(actual, expected, context):
    assert math.isclose(float(actual), expected, rel_tol=1e-8, abs_tol=1e-7), (context, actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expect-runs", type=int)
    args = parser.parse_args()
    directory = args.result_dir
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config_snapshot.json").read_text(encoding="utf-8"))
    config_path = directory / "source_snapshot" / "config" / "default.json"
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == manifest["config_sha256"]
    for path, expected in manifest["code_sha256"].items():
        source = directory / "source_snapshot" / path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, f"源码快照不匹配: {path}"
    metrics = read_rows(directory / "metrics.csv")
    if args.expect_runs is not None:
        assert len(metrics) == args.expect_runs, (len(metrics), args.expect_runs)
    specs = {n["node_id"]: n for n in config["nodes"]}
    total_requests = total_tasks = 0
    max_capacity_ratio = 0.0
    for label, expected in manifest["trace_hashes"].items():
        trace_path = directory / "traces" / (label + ".json")
        assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == expected, f"轨迹被改写: {label}"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))["requests"]
        reference = {r["request_id"]: r for r in trace}
        for run_dir in sorted((directory / "runs" / label).iterdir()):
            method = run_dir.name
            requests = read_rows(run_dir / "requests.csv")
            tasks = read_rows(run_dir / "tasks.csv")
            nodes = read_rows(run_dir / "nodes.csv")
            assert len(requests) == len(reference)
            assert {r["request_id"] for r in requests} == set(reference)
            total_requests += len(requests)
            total_tasks += len(tasks)
            by_request = {r["request_id"]: [] for r in requests}
            by_node = {n: [] for n in specs}
            for task in tasks:
                by_request[task["request_id"]].append(task)
                by_node[task["node_id"]].append(task)
                spec = specs[task["node_id"]]
                assert float(task["memory_mb"]) <= spec["memory_mb"] + 1e-9
                if task["start_s"]:
                    assert float(task["start_s"]) >= float(task["input_ready_s"]) - 1e-9
                    assert float(task["cpu_finish_s"]) > float(task["start_s"])
                if task["returned_s"] and task["cpu_finish_s"]:
                    assert float(task["returned_s"]) >= float(task["cpu_finish_s"]) - 1e-9
            for node_id, assigned in by_node.items():
                running = sorted((t for t in assigned if t["start_s"]), key=lambda t: float(t["start_s"]))
                for first, second in zip(running, running[1:]):
                    assert float(first["cpu_finish_s"]) <= float(second["start_s"]) + 1e-9, "执行槽重叠"
                exported = next(n for n in nodes if n["node_id"] == node_id)
                assert int(exported["assignments"]) == len(assigned)
                assert float(exported["max_execution_memory_mb"]) <= specs[node_id]["memory_mb"] + 1e-9
                assert 0 <= float(exported["cpu_utilization"]) <= 1 + 1e-9
            for request in requests:
                request_id = request["request_id"]
                original = reference[request_id]
                children = by_request[request_id]
                if request["status"] == "rejected":
                    assert not children, "整体拒绝的请求仍提交了子任务"
                    assert not request["latency_s"]
                    continue
                assert {t["task_id"] for t in children} == {t["task_id"] for t in original["tasks"]}
                if request["status"] == "completed":
                    assert all(t["status"] == "returned" for t in children)
                    finish = math.ceil((max(float(t["returned_s"]) for t in children)
                                       + original["aggregation_s"]) / config["tick_s"] - 1e-9) * config["tick_s"]
                    close(request["finish_s"], finish, "并行汇聚时刻")
                    close(request["latency_s"], finish - original["arrival_s"], "端到端时延")
                    quality_ok = all(float(t["profile_quality"]) + 1e-9 >= float(t["min_quality"]) for t in children)
                    on_time = finish <= original["deadline_s"] + 1e-9
                    assert boolean(request["quality_ok"]) == quality_ok
                    assert boolean(request["on_time"]) == on_time
                    assert boolean(request["qos_ok"]) == (quality_ok and on_time)
            # 汇总表必须能从每请求明细独立重算，不读取调度器内部状态。
            matched = [m for m in metrics if m["method"] == method and (
                label == "trace_replay" or label == f"rate_{float(m['arrival_rate']):g}_seed_{int(m['seed'])}")]
            assert len(matched) == 1, (label, method)
            metric = matched[0]
            denominator = len(requests)
            close(metric["qos_success_rate"], sum(boolean(r["qos_ok"]) for r in requests) / denominator, "QoS比例")
            close(metric["rejection_rate"], sum(r["status"] == "rejected" for r in requests) / denominator, "拒绝比例")
            close(metric["timeout_rate"], sum(r["status"] == "unfinished" or (
                r["status"] == "completed" and not boolean(r["on_time"])) for r in requests) / denominator, "超时比例")
            max_capacity_ratio = max(max_capacity_ratio, float(metric["max_network_capacity_ratio"]))
            assert float(metric["max_network_capacity_ratio"]) <= 1 + 1e-7
            if all(r["status"] != "unfinished" for r in requests):
                close(metric["residual_execution_memory_mb"], 0, "完整结束的执行内存")
    report = dict(status="passed", run_count=len(metrics), trace_count=len(manifest["trace_hashes"]),
                  request_records=total_requests, admitted_task_records=total_tasks,
                  max_network_capacity_ratio=max_capacity_ratio,
                  checks=["source/config/trace SHA-256", "request denominator and summary rates",
                          "whole-request rejection", "single CPU slot and memory capacity",
                          "input/compute/output ordering", "parallel max aggregation and QoS",
                          "network allocation capacity bound"])
    (directory / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
