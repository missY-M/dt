"""第一阶段实验的物理不变量和手算小例子。

测试使用可手算的合成任务，不把合成质量参数解释为实测准确率。
网络事件允许一拍的离散化误差；CPU 占用、截止边界与接纳回滚单独验证。
EdgeSimPy 使用组件全局表，因此这些测试应串行运行，不使用 pytest-xdist。
"""

from copy import deepcopy
from dataclasses import asdict

import pytest

from stage1.engine import Experiment
from stage1.planner import Planner
from stage1.scenario import (
    Capability,
    NodeSpec,
    Request,
    TaskDemand,
    generate_requests,
    load_trace,
    nodes_from_config,
    save_trace,
)


def _node(node_id="n1", *, cpu=1.0, memory=32.0, bandwidth=100.0,
          latency=0.0, task_types=("A",), quality=0.9):
    return NodeSpec(
        node_id=node_id,
        cpu_gflops=cpu,
        memory_mb=memory,
        bandwidth_mb_s=bandwidth,
        latency_s=latency,
        capabilities=tuple(
            Capability(f"{node_id}_{kind}", kind, {"standard": quality})
            for kind in task_types
        ),
    )


def _task(request_id, task_type="A", *, index=0, quality=0.8, work=1.0,
          memory=8.0, input_mb=0.0, output_mb=0.0, input_class="standard"):
    return TaskDemand(
        task_id=f"{request_id}_{task_type}_{index}",
        request_id=request_id,
        task_type=task_type,
        input_class=input_class,
        min_quality=quality,
        work_gflop=work,
        memory_mb=memory,
        input_mb=input_mb,
        output_mb=output_mb,
    )


def _request(request_id, tasks, *, arrival=0.0, deadline=10.0, aggregation=0.0):
    return Request(request_id, arrival, deadline, aggregation, tuple(tasks))


def _config(nodes, *, tick=0.1, backbone=100.0, latency=0.0, drain=2.0):
    # 保留完整配置，轨迹生成与执行测试使用同一套经过校验的字段。
    node_rows = []
    for node in nodes:
        row = asdict(node)
        row["capabilities"] = [asdict(cap) for cap in node.capabilities]
        node_rows.append(row)
    return {
        "tick_s": tick,
        "backbone_bandwidth_mb_s": backbone,
        "backbone_latency_s": latency,
        "max_drain_s": drain,
        "aggregation_s": 0.2,
        "request_count": 6,
        "beam_width": 32,
        "seeds": [3, 7],
        "arrival_rates": [0.5],
        "request_deadline_s_range": [5.0, 8.0],
        "task_types": [
            {
                "task_type": kind,
                "input_classes": ["standard"],
                "min_quality_values": [0.8],
                "work_gflop_range": [0.2, 1.0],
                "memory_mb": 8.0,
                "input_mb_range": [0.0, 0.5],
                "output_mb": 0.1,
            }
            for kind in ("A", "B")
        ],
        "nodes": node_rows,
    }


def _run(nodes, requests, method="qos_greedy", **config_options):
    return Experiment(_config(nodes, **config_options), tuple(nodes),
                      tuple(requests), method).run()


def test_single_node_queues_cpu_and_releases_execution_memory():
    """两个 1 GFLOP 任务在 1 GFLOP/s 单槽节点上必须串行执行。"""
    node = _node(memory=8.0)
    requests = tuple(
        _request(name, [_task(name, memory=8.0)]) for name in ("r1", "r2")
    )
    result = _run((node,), requests)
    rows = sorted(result["tasks"], key=lambda row: row["start_s"])

    assert [row["status"] for row in result["requests"]] == ["completed", "completed"]
    assert rows[0]["start_s"] == pytest.approx(0.0)
    assert rows[0]["cpu_finish_s"] == pytest.approx(1.0)
    assert rows[1]["start_s"] == pytest.approx(1.0)
    assert rows[1]["cpu_finish_s"] == pytest.approx(2.0)
    assert rows[1]["queue_wait_s"] == pytest.approx(1.0)
    assert result["nodes"][0]["cpu_busy_s"] == pytest.approx(2.0)
    assert result["nodes"][0]["cpu_utilization"] <= 1.0 + 1e-9
    # 两个任务各需要全部内存，却因串行执行不会同时占用 16 MB。
    assert result["nodes"][0]["max_execution_memory_mb"] == pytest.approx(8.0)
    assert result["residual_execution_memory_mb"] == pytest.approx(0.0)


def test_input_output_serialization_and_round_trip_propagation():
    """手算：输入 1 s + 传播 0.3 s + CPU 1 s + 输出 0.5 s + 传播 0.3 s。"""
    node = _node(bandwidth=2.0, latency=0.2)
    task = _task("r", input_mb=2.0, output_mb=1.0)
    result = _run((node,), (_request("r", [task]),), backbone=10.0, latency=0.1)
    row = result["tasks"][0]

    assert row["input_ready_s"] == pytest.approx(1.3, abs=0.1 + 1e-9)
    assert row["start_s"] == pytest.approx(row["input_ready_s"])
    assert row["cpu_finish_s"] - row["start_s"] == pytest.approx(1.0)
    assert row["returned_s"] - row["cpu_finish_s"] == pytest.approx(0.8, abs=0.1 + 1e-9)
    assert result["requests"][0]["finish_s"] == pytest.approx(3.1, abs=0.2 + 1e-9)
    assert result["network_flow_count"] == 2
    assert result["max_network_capacity_ratio"] <= 1.0 + 1e-7


@pytest.mark.parametrize("direction", ["input", "output"])
def test_simultaneous_flows_share_the_backbone(direction):
    """两条各 1 MB 的流共用 1 MB/s 骨干，应花约 2 s，不能各自独占容量。"""
    nodes = (_node("a", task_types=("A",)), _node("b", task_types=("B",)))
    sizes = {"input_mb": 1.0} if direction == "input" else {"output_mb": 1.0}
    tasks = [_task("r", kind, work=0.1, **sizes) for kind in ("A", "B")]
    result = _run(nodes, (_request("r", tasks),), backbone=1.0)

    for row in result["tasks"]:
        duration = (row["input_ready_s"] - row["admitted_s"] if direction == "input"
                    else row["returned_s"] - row["cpu_finish_s"])
        assert duration == pytest.approx(2.0, abs=0.1 + 1e-9)
    assert result["network_flow_count"] == 2
    assert result["max_network_capacity_ratio"] == pytest.approx(1.0)
    assert result["requests"][0]["qos_ok"] is True


def test_parallel_subtasks_aggregate_after_maximum_not_sum():
    """两个独立节点分别执行 1 s、2 s，汇聚 0.3 s，完整请求应为 2.3 s。"""
    nodes = (_node("a", task_types=("A",)), _node("b", task_types=("B",)))
    req = _request("r", [_task("r", "A", work=1.0), _task("r", "B", work=2.0)],
                   aggregation=0.3)
    result = _run(nodes, (req,))

    assert sorted(row["start_s"] for row in result["tasks"]) == pytest.approx([0.0, 0.0])
    assert result["requests"][0]["finish_s"] == pytest.approx(2.3)
    assert result["requests"][0]["latency_s"] == pytest.approx(2.3)
    assert result["residual_execution_memory_mb"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "min_quality,memory,deadline,input_class,expected_reason",
    [
        (0.8, 32.0, 1.2, "standard", ""),
        (0.8001, 32.0, 1.2, "standard", "no_quality_candidate"),
        (0.8, 32.0001, 1.2, "standard", "no_compatible_capacity"),
        (0.8, 32.0, 1.19, "standard", "predicted_deadline"),
        (0.8, 32.0, 1.2, "unseen", "no_compatible_capacity"),
    ],
)
def test_quality_memory_and_absolute_deadline_boundaries(
    min_quality, memory, deadline, input_class, expected_reason,
):
    """等于门槛可以接纳；稍微越界、未支持输入类别则必须明确拒绝。"""
    node = _node(memory=32.0, quality=0.8)
    req = _request("r", [_task("r", quality=min_quality, memory=memory,
                                input_class=input_class)],
                   deadline=deadline, aggregation=0.2)
    result = _run((node,), (req,))
    row = result["requests"][0]

    assert row["rejection_reason"] == expected_reason
    if expected_reason:
        assert row["status"] == "rejected"
        assert row["finish_s"] is None
        assert row["qos_ok"] is False
        assert result["tasks"] == []
    else:
        assert row["status"] == "completed"
        assert row["finish_s"] == pytest.approx(1.2)
        assert row["quality_ok"] is True
        assert row["on_time"] is True
        assert row["qos_ok"] is True
    assert result["residual_execution_memory_mb"] == pytest.approx(0.0)


def test_deadline_is_absolute_and_latency_starts_at_request_arrival():
    """3 s 到达、4.2 s 截止，实际允许 1.2 s，不能再把截止时间当预算相加。"""
    node = _node()
    req = _request("r", [_task("r")], arrival=3.0, deadline=4.2, aggregation=0.2)
    result = _run((node,), (req,))
    row = result["requests"][0]

    assert result["tasks"][0]["start_s"] == pytest.approx(3.0)
    assert row["finish_s"] == pytest.approx(4.2)
    assert row["latency_s"] == pytest.approx(1.2)
    assert row["on_time"] is True
    assert row["qos_ok"] is True


def test_rejected_request_does_not_commit_partial_tasks_or_cpu_reservations():
    """先匹配成功的 A 必须随无候选 B 一起回滚，否则会挤掉后续紧期限任务。"""
    nodes = (_node(memory=8.0),)
    # 两请求同截止时间，按ID先处理失败请求，确实覆盖“先预约A再回滚”。
    bad = _request("a_bad", [_task("a_bad", "A"), _task("a_bad", "B")], deadline=1.0)
    good = _request("b_good", [_task("b_good", memory=8.0)], deadline=1.0)
    config = _config(nodes)
    ready = {"n1": 0.0}
    plans = Planner(config, nodes, "qos_greedy").plan_batch((bad, good), ready, 0.0)

    assert plans[0].assignments == ()
    assert plans[0].rejection_reason == "no_compatible_capacity"
    assert plans[1].rejection_reason == ""
    assert plans[1].assignments[0].predicted_cpu_finish_s == pytest.approx(1.0)
    assert ready == {"n1": 0.0}  # 调用方传入的预约表也不应被修改。

    result = Experiment(config, nodes, (bad, good), "qos_greedy").run()
    assert [row["status"] for row in result["requests"]] == ["rejected", "completed"]
    assert {row["request_id"] for row in result["tasks"]} == {"b_good"}
    assert result["requests"][1]["finish_s"] == pytest.approx(1.0)
    assert result["nodes"][0]["assignments"] == 1
    assert result["residual_execution_memory_mb"] == pytest.approx(0.0)


def test_joint_beam_avoids_greedy_trap_within_one_complex_request():
    """A 可用两节点、B 只能用快节点；给 B 留快节点，两者才能在 2 s 完成。"""
    nodes = (_node("fast", task_types=("A", "B")),
             _node("slow", cpu=0.5, task_types=("A",)))
    req = _request("r", [_task("r", "A", work=1.0), _task("r", "B", work=2.0)],
                   deadline=2.0)
    config = _config(nodes)
    ready = {node.node_id: 0.0 for node in nodes}
    greedy = Planner(config, nodes, "qos_greedy").plan_batch((req,), ready, 0.0)[0]
    beam = Planner(config, nodes, "global_beam").plan_batch((req,), ready, 0.0)[0]

    assert greedy.assignments == ()
    assert greedy.rejection_reason
    assert beam.rejection_reason == ""
    placement = {assignment.task_id: assignment.node_id for assignment in beam.assignments}
    assert placement == {"r_A_0": "slow", "r_B_0": "fast"}
    assert [a.predicted_finish_s for a in beam.assignments] == pytest.approx([2.0, 2.0])


def test_batch_beam_keeps_alternative_placement_for_later_request():
    """同批请求的联合选择要考虑后续请求，不能仅把独立贪心重新命名。"""
    nodes = (_node("fast", task_types=("A", "B")),
             _node("slow", cpu=0.5, task_types=("A",)))
    flexible = _request("a_flexible", [_task("a_flexible", "A", work=1.0)], deadline=2.0)
    rigid = _request("b_rigid", [_task("b_rigid", "B", work=2.0)], deadline=2.0)
    config = _config(nodes)
    ready = {node.node_id: 0.0 for node in nodes}

    greedy = Planner(config, nodes, "qos_greedy").plan_batch((flexible, rigid), ready, 0.0)
    beam = Planner(config, nodes, "global_beam").plan_batch((flexible, rigid), ready, 0.0)
    assert sum(bool(plan.assignments) for plan in greedy) == 1
    assert sum(bool(plan.assignments) for plan in beam) == 2
    assert {plan.request_id: plan.assignments[0].node_id for plan in beam} == {
        "a_flexible": "slow", "b_rigid": "fast",
    }


def test_generated_trace_round_trip_and_repeated_execution_are_reproducible(tmp_path):
    """复现实验比较仿真输出，真实运行/调度耗时不属于确定性比较字段。"""
    nodes = (_node("n1", task_types=("A", "B")),
             _node("n2", cpu=2.0, task_types=("A", "B")))
    config = _config(nodes)
    assert nodes_from_config(config) == nodes
    requests = generate_requests(config, seed=3, arrival_rate=0.5)
    assert generate_requests(config, seed=3, arrival_rate=0.5) == requests
    assert generate_requests(config, seed=7, arrival_rate=0.5) != requests
    before = deepcopy(requests)
    trace_path = tmp_path / "trace.json"
    save_trace(requests, trace_path)
    assert load_trace(trace_path) == requests

    first = Experiment(config, nodes, requests, "earliest_finish").run()
    second = Experiment(config, nodes, load_trace(trace_path), "earliest_finish").run()
    for field in ("requests", "tasks", "nodes", "simulation_duration_s",
                  "max_network_capacity_ratio", "network_flow_count",
                  "residual_execution_memory_mb"):
        assert first[field] == second[field]
    Experiment(config, nodes, requests, "qos_greedy").run()
    assert requests == before  # 更换策略不能改写已生成的任务轨迹。


def test_drain_limit_preserves_unfinished_request_as_failure():
    """截断在运行中的 100 s 任务；必须保留失败样本，不能伪造完成时延。"""
    node = _node()
    req = _request("r", [_task("r", work=100.0)], deadline=0.5)
    # 此基线不做预计截止时间过滤，因此可以真实覆盖运行后的截断分支。
    result = _run((node,), (req,), method="earliest_finish", drain=0.5)
    row = result["requests"][0]

    assert row["status"] == "unfinished"
    assert row["rejection_reason"] == "drain_limit"
    assert row["finish_s"] is None
    assert row["latency_s"] is None
    assert row["qos_ok"] is False
    assert len(result["requests"]) == 1
    assert result["simulation_duration_s"] <= 1.2 + 1e-9
    assert result["tasks"][0]["cpu_finish_s"] > result["simulation_duration_s"]
    # 截断快照中的在途占用应如实保留；这不是已经完成任务的资源泄漏。
    assert result["residual_execution_memory_mb"] == pytest.approx(8.0)
    assert result["nodes"][0]["cpu_busy_s"] <= result["simulation_duration_s"] + 1e-9


def test_zero_byte_zero_propagation_task_is_reported_returned():
    """请求完成时，零字节输出不能在任务明细中仍显示传输中。"""
    node = _node()
    req = _request("r", [_task("r", work=1.0)], deadline=2.0)
    result = _run((node,), (req,))
    assert result["requests"][0]["status"] == "completed"
    assert result["tasks"][0]["status"] == "returned"
    assert result["tasks"][0]["returned_s"] == pytest.approx(1.0)


def test_timeout_rate_counts_late_and_unfinished_but_separates_rejection():
    """4个请求：准时、迟到、截断、拒绝；超时应为2/4，不能漏掉截断。"""
    from run import summarize
    rows = [
        dict(status="completed", latency_s=1.0, on_time=True, quality_ok=True, qos_ok=True),
        dict(status="completed", latency_s=2.0, on_time=False, quality_ok=True, qos_ok=False),
        dict(status="unfinished", latency_s=None, on_time=False, quality_ok=False, qos_ok=False),
        dict(status="rejected", latency_s=None, on_time=False, quality_ok=False, qos_ok=False),
    ]
    result = dict(requests=rows, nodes=[dict(cpu_utilization=0.5)], scheduler_wall_s=0.01,
                  decision_count=4, wall_runtime_s=0.1, simulation_duration_s=10,
                  max_network_capacity_ratio=1, network_flow_count=2, residual_execution_memory_mb=8)
    metrics = summarize(result)
    assert metrics["timeout_rate"] == pytest.approx(0.5)
    assert metrics["rejection_rate"] == pytest.approx(0.25)
    assert metrics["unfinished_rate"] == pytest.approx(0.25)
    assert metrics["mean_completed_latency_s"] == pytest.approx(1.5)


@pytest.mark.parametrize("weights", [[1, 1], [0, 0, 0], [1, -1, 1], [float("nan"), 1, 1]])
def test_invalid_matching_weight_vector_is_rejected(weights):
    from stage1.matching import recommend
    with pytest.raises(ValueError):
        recommend(0.8, 0.9, 1.0, 0.5, 32, 64, weights)
