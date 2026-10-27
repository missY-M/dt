"""第一阶段的静态画像、配置读取和可复现任务轨迹。

本模块不依赖调度策略或仿真器。先生成并保存轨迹，再交给不同策略，
可以避免策略改变随机数消耗顺序而导致不公平的任务输入。

统一单位：秒、GFLOP、GFLOP/s、十进制 MB、MB/s。
质量字段是合成的统计能力参数，不表示某次输出是否正确，也不是实测准确率。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Capability:
    """一个模型的静态能力；compute_factor 是相对计算量系数。"""

    model_id: str
    task_type: str
    quality_by_input: dict[str, float]
    compute_factor: float = 1.0


@dataclass(frozen=True)
class NodeSpec:
    """节点画像；memory_mb 为扣除常驻模型后可供任务使用的内存。"""

    node_id: str
    cpu_gflops: float
    memory_mb: float
    bandwidth_mb_s: float
    latency_s: float
    capabilities: tuple[Capability, ...]

    def capability_for(self, task: TaskDemand) -> Capability | None:
        """按任务类型和输入类别查能力；质量/内存门槛由调度器检查。

        配置校验限制每节点每任务类型只有一个模型，避免暗中假定
        “最高质量模型也最快”。保留 max 写法以明确候选查询的规则。
        """

        candidates = [
            capability
            for capability in self.capabilities
            if capability.task_type == task.task_type
            and task.input_class in capability.quality_by_input
        ]
        return max(
            candidates,
            key=lambda capability: capability.quality_by_input[task.input_class],
            default=None,
        )


@dataclass(frozen=True)
class TaskDemand:
    """一个不可再拆分的子任务需求；输入输出数据量使用十进制 MB。"""

    task_id: str
    request_id: str
    task_type: str
    input_class: str
    min_quality: float
    work_gflop: float
    memory_mb: float
    input_mb: float
    output_mb: float


@dataclass(frozen=True)
class Request:
    """复杂请求；子任务可并行，deadline_s 是绝对截止时刻。"""

    request_id: str
    arrival_s: float
    deadline_s: float
    aggregation_s: float
    tasks: tuple[TaskDemand, ...]


def _object(value: Any, field: str) -> dict[str, Any]:
    """给配置/轨迹错误提供字段路径，而不是迟到的仿真运行异常。"""

    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是 JSON 对象")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} 必须是非空数组")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空字符串")
    return value


def _number(value: Any, field: str, *, positive: bool = True) -> float:
    # bool 是 int 的子类，因此必须额外排除，防止把 true 当作 1。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        requirement = "大于 0" if positive else "大于等于 0"
        raise ValueError(f"{field} 必须是有限数值且{requirement}")
    return result


def _integer(value: Any, field: str, *, positive: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} 必须是整数")
    if value < (1 if positive else 0):
        raise ValueError(f"{field} 必须{'大于 0' if positive else '大于等于 0'}")
    return value


def _quality(value: Any, field: str) -> float:
    result = _number(value, field, positive=False)
    if result > 1:
        raise ValueError(f"{field} 必须处于 [0, 1]")
    return result


def _range(value: Any, field: str, *, positive: bool = True) -> tuple[float, float]:
    values = _list(value, field)
    if len(values) != 2:
        raise ValueError(f"{field} 必须恰好包含下界和上界两个数")
    lower = _number(values[0], f"{field}[0]", positive=positive)
    upper = _number(values[1], f"{field}[1]", positive=positive)
    if lower > upper:
        raise ValueError(f"{field} 下界不能超过上界")
    return lower, upper


def _unique(values: list[Any], field: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field} 不能包含重复值")


def _required(data: dict[str, Any], key: str, field: str) -> Any:
    if key not in data:
        raise ValueError(f"{field} 缺少字段 {key}")
    return data[key]


def _validate_config(config: dict[str, Any]) -> None:
    """检查量纲和画像的关联关系；不要求每个节点都能执行每类任务。"""

    for field in ("tick_s", "backbone_bandwidth_mb_s", "max_drain_s"):
        _number(_required(config, field, "config"), field)
    for field in ("backbone_latency_s", "aggregation_s"):
        _number(_required(config, field, "config"), field, positive=False)
    for field in ("request_count", "beam_width"):
        _integer(_required(config, field, "config"), field)
    if "matching_weight" in config:
        _number(config["matching_weight"], "matching_weight", positive=False)
    if "matching_weights" in config:
        weights = _list(config["matching_weights"], "matching_weights")
        if len(weights) != 3:
            raise ValueError("matching_weights 必须依次提供质量、时延、内存三个权重")
        for index, weight in enumerate(weights):
            _number(weight, f"matching_weights[{index}]", positive=False)
        if sum(weights) <= 0:
            raise ValueError("matching_weights 至少一个权重必须为正")
    seeds = _list(_required(config, "seeds", "config"), "seeds")
    for index, seed in enumerate(seeds):
        _integer(seed, f"seeds[{index}]", positive=False)
    _unique(seeds, "seeds")
    rates = _list(_required(config, "arrival_rates", "config"), "arrival_rates")
    for index, rate in enumerate(rates):
        _number(rate, f"arrival_rates[{index}]")
    _unique(rates, "arrival_rates")
    _range(_required(config, "request_deadline_s_range", "config"), "request_deadline_s_range")

    task_definitions: dict[str, dict[str, Any]] = {}
    for index, raw_task in enumerate(_list(_required(config, "task_types", "config"), "task_types")):
        field = f"task_types[{index}]"
        task = _object(raw_task, field)
        task_type = _string(_required(task, "task_type", field), f"{field}.task_type")
        if task_type in task_definitions:
            raise ValueError(f"{field}.task_type 重复：{task_type}")
        task_definitions[task_type] = task
        classes = _list(_required(task, "input_classes", field), f"{field}.input_classes")
        for class_index, input_class in enumerate(classes):
            _string(input_class, f"{field}.input_classes[{class_index}]")
        _unique(classes, f"{field}.input_classes")
        qualities = _list(_required(task, "min_quality_values", field), f"{field}.min_quality_values")
        for quality_index, quality in enumerate(qualities):
            _quality(quality, f"{field}.min_quality_values[{quality_index}]")
        _unique(qualities, f"{field}.min_quality_values")
        _range(_required(task, "work_gflop_range", field), f"{field}.work_gflop_range")
        _range(_required(task, "input_mb_range", field), f"{field}.input_mb_range", positive=False)
        _number(_required(task, "memory_mb", field), f"{field}.memory_mb")
        _number(_required(task, "output_mb", field), f"{field}.output_mb", positive=False)

    node_ids: list[str] = []
    for index, raw_node in enumerate(_list(_required(config, "nodes", "config"), "nodes")):
        field = f"nodes[{index}]"
        node = _object(raw_node, field)
        node_ids.append(_string(_required(node, "node_id", field), f"{field}.node_id"))
        for name in ("cpu_gflops", "memory_mb", "bandwidth_mb_s"):
            _number(_required(node, name, field), f"{field}.{name}")
        _number(_required(node, "latency_s", field), f"{field}.latency_s", positive=False)
        seen_types: set[str] = set()
        model_ids: list[str] = []
        for cap_index, raw_cap in enumerate(_list(_required(node, "capabilities", field), f"{field}.capabilities")):
            cap_field = f"{field}.capabilities[{cap_index}]"
            cap = _object(raw_cap, cap_field)
            model_ids.append(_string(_required(cap, "model_id", cap_field), f"{cap_field}.model_id"))
            task_type = _string(_required(cap, "task_type", cap_field), f"{cap_field}.task_type")
            if task_type not in task_definitions:
                raise ValueError(f"{cap_field}.task_type 未在 task_types 中定义：{task_type}")
            if task_type in seen_types:
                raise ValueError(f"{field} 每任务类型只允许一个模型：{task_type}")
            seen_types.add(task_type)
            _number(cap.get("compute_factor", 1.0), f"{cap_field}.compute_factor")
            quality_map = _object(_required(cap, "quality_by_input", cap_field), f"{cap_field}.quality_by_input")
            if not quality_map:
                raise ValueError(f"{cap_field}.quality_by_input 不能为空")
            allowed_classes = task_definitions[task_type]["input_classes"]
            for input_class, quality in quality_map.items():
                if input_class not in allowed_classes:
                    raise ValueError(f"{cap_field} 存在未定义的输入类别：{input_class}")
                _quality(quality, f"{cap_field}.quality_by_input.{input_class}")
        _unique(model_ids, f"{field}.capabilities.model_id")
    _unique(node_ids, "nodes.node_id")


def load_config(path: str | Path) -> dict[str, Any]:
    """从 JSON 读取并校验配置；保留 metadata 以便结果记录参数来源。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = _object(json.load(handle), "config")
    _validate_config(config)
    return config


def nodes_from_config(config: dict[str, Any]) -> tuple[NodeSpec, ...]:
    """转换为不可重新赋值的画像对象；本阶段不会在线修改画像。"""

    _validate_config(_object(config, "config"))
    return tuple(
        NodeSpec(
            node_id=node["node_id"],
            cpu_gflops=float(node["cpu_gflops"]),
            memory_mb=float(node["memory_mb"]),
            bandwidth_mb_s=float(node["bandwidth_mb_s"]),
            latency_s=float(node["latency_s"]),
            capabilities=tuple(
                Capability(
                    model_id=cap["model_id"],
                    task_type=cap["task_type"],
                    quality_by_input={key: float(value) for key, value in cap["quality_by_input"].items()},
                    compute_factor=float(cap.get("compute_factor", 1.0)),
                )
                for cap in node["capabilities"]
            ),
        )
        for node in config["nodes"]
    )


def generate_requests(
    config: dict[str, Any],
    seed: int,
    arrival_rate: float,
    request_count: int | None = None,
) -> tuple[Request, ...]:
    """以局部随机数生成泊松到达轨迹，单位为请求/秒。

    首个请求也等待一个指数分布间隔。默认两类任务各生成一个；
    配置增加任务类型后仍按每类一个生成，不需要改生成器。
    质量阈值、输入类别、计算量等均在生成阶段确定，调度时不再采样。
    """

    _validate_config(_object(config, "config"))
    _integer(seed, "seed", positive=False)
    rate = _number(arrival_rate, "arrival_rate")
    count = _integer(config["request_count"] if request_count is None else request_count, "request_count")
    rng = random.Random(seed)
    requests: list[Request] = []
    arrival_s = 0.0
    for index in range(count):
        arrival_s += rng.expovariate(rate)
        request_id = f"request_{index:06d}"
        deadline_budget = rng.uniform(*config["request_deadline_s_range"])
        tasks = tuple(
            TaskDemand(
                task_id=f"{request_id}_task_{task_index:02d}",
                request_id=request_id,
                task_type=task["task_type"],
                input_class=rng.choice(task["input_classes"]),
                min_quality=float(rng.choice(task["min_quality_values"])),
                work_gflop=rng.uniform(*task["work_gflop_range"]),
                memory_mb=float(task["memory_mb"]),
                input_mb=rng.uniform(*task["input_mb_range"]),
                output_mb=float(task["output_mb"]),
            )
            for task_index, task in enumerate(config["task_types"])
        )
        requests.append(
            Request(
                request_id=request_id,
                arrival_s=arrival_s,
                deadline_s=arrival_s + deadline_budget,
                aggregation_s=float(config["aggregation_s"]),
                tasks=tasks,
            )
        )
    return tuple(requests)


def _requests_from_rows(rows: Any) -> tuple[Request, ...]:
    """校验轨迹，不允许错误 ID、负数据量或无穷截止时间悄悄进入实验。"""

    if not isinstance(rows, list):
        raise ValueError("trace.requests 必须是数组")
    requests: list[Request] = []
    request_ids: set[str] = set()
    task_ids: set[str] = set()
    previous_arrival = -1.0
    for index, raw_request in enumerate(rows):
        field = f"trace.requests[{index}]"
        row = _object(raw_request, field)
        request_id = _string(_required(row, "request_id", field), f"{field}.request_id")
        if request_id in request_ids:
            raise ValueError(f"{field}.request_id 重复：{request_id}")
        request_ids.add(request_id)
        arrival = _number(_required(row, "arrival_s", field), f"{field}.arrival_s", positive=False)
        deadline = _number(_required(row, "deadline_s", field), f"{field}.deadline_s")
        if arrival < previous_arrival:
            raise ValueError("轨迹必须按 arrival_s 非递减排列")
        previous_arrival = arrival
        if deadline <= arrival:
            raise ValueError(f"{field}.deadline_s 必须晚于 arrival_s")
        aggregation = _number(_required(row, "aggregation_s", field), f"{field}.aggregation_s", positive=False)
        tasks: list[TaskDemand] = []
        task_types: set[str] = set()
        for task_index, raw_task in enumerate(_list(_required(row, "tasks", field), f"{field}.tasks")):
            task_field = f"{field}.tasks[{task_index}]"
            task = _object(raw_task, task_field)
            task_id = _string(_required(task, "task_id", task_field), f"{task_field}.task_id")
            if task_id in task_ids:
                raise ValueError(f"{task_field}.task_id 重复：{task_id}")
            task_ids.add(task_id)
            task_request_id = _string(_required(task, "request_id", task_field), f"{task_field}.request_id")
            if task_request_id != request_id:
                raise ValueError(f"{task_field}.request_id 与所属请求不一致")
            task_type = _string(_required(task, "task_type", task_field), f"{task_field}.task_type")
            if task_type in task_types:
                raise ValueError(f"{field} 每任务类型应恰好一个子任务：{task_type}")
            task_types.add(task_type)
            tasks.append(
                TaskDemand(
                    task_id=task_id,
                    request_id=request_id,
                    task_type=task_type,
                    input_class=_string(_required(task, "input_class", task_field), f"{task_field}.input_class"),
                    min_quality=_quality(_required(task, "min_quality", task_field), f"{task_field}.min_quality"),
                    work_gflop=_number(_required(task, "work_gflop", task_field), f"{task_field}.work_gflop"),
                    memory_mb=_number(_required(task, "memory_mb", task_field), f"{task_field}.memory_mb"),
                    input_mb=_number(_required(task, "input_mb", task_field), f"{task_field}.input_mb", positive=False),
                    output_mb=_number(_required(task, "output_mb", task_field), f"{task_field}.output_mb", positive=False),
                )
            )
        requests.append(Request(request_id, arrival, deadline, aggregation, tuple(tasks)))
    return tuple(requests)


def save_trace(requests: tuple[Request, ...], path: str | Path) -> None:
    """保存策略无关的轨迹；浮点数不四舍五入，保证往返读取一致。"""

    rows = [asdict(request) for request in requests]
    # asdict 保留 tuple；JSON 写出时会转数组。先转为 JSON 数据结构后统一校验。
    rows = json.loads(json.dumps(rows, ensure_ascii=False, allow_nan=False))
    _requests_from_rows(rows)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "metadata": {
            "data_source": "synthetic",
            "units": "s, GFLOP, decimal MB (1000000 bytes)",
            "deadline_semantics": "deadline_s 为绝对截止时刻",
            "quality_semantics": "min_quality 为统计能力参数的阈值，不是单次正确性",
        },
        "requests": rows,
    }
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_trace(path: str | Path) -> tuple[Request, ...]:
    """读取已保存轨迹；不同策略必须读取同一份或复用同一个返回值。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = _object(json.load(handle), "trace")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("trace.schema_version 必须为整数 1")
    return _requests_from_rows(_required(payload, "requests", "trace"))
