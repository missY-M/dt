"""需求/能力匹配和请求批次调度。

这是一套可复现的算法原型，不能据此宣称已经证明论文创新性。
global_beam 是有限宽度的联合搜索，不是数学意义上的全局最优算法。
所有方法共用类型、输入条件和执行内存约束；qos_greedy 与 global_beam
还共用质量/预计截止时间约束，便于区分画像筛选与联合搜索的作用。
"""
from dataclasses import dataclass, replace
from itertools import product
import math
from .scenario import NodeSpec, Request, TaskDemand
from .matching import recommend

METHODS = ("round_robin", "earliest_finish", "quality_first", "qos_greedy", "global_beam")
EPS = 1e-9


def ceil_tick(seconds: float, tick_s: float) -> float:
    """所有事件向上对齐时钟；减去浮点误差，避免整点多等一拍。"""
    return math.ceil(seconds / tick_s - EPS) * tick_s


@dataclass(frozen=True)
class Assignment:
    task_id: str
    node_id: str
    model_id: str
    quality: float
    predicted_cpu_finish_s: float
    predicted_finish_s: float
    matching_score: float = 0.0


@dataclass(frozen=True)
class RequestPlan:
    request_id: str
    assignments: tuple[Assignment, ...] = ()
    rejection_reason: str = ""


class Planner:
    def __init__(self, config: dict, nodes: tuple[NodeSpec, ...], method: str):
        if method not in METHODS:
            raise ValueError(f"未知方法: {method}；可选 {METHODS}")
        self.config, self.nodes, self.method = config, nodes, method
        self.cursor = 0

    def compatible(self, task: TaskDemand) -> list[NodeSpec]:
        """结构化匹配；本轮不把字段匹配称为自然语言语义理解。"""
        return [n for n in self.nodes if n.capability_for(task) is not None
                and n.memory_mb + EPS >= task.memory_mb]

    def estimate(self, task: TaskDemand, node: NodeSpec, ready: dict[str, float],
                 now: float, share: int) -> Assignment:
        """用当前队列与保守等分带宽近似预计完成时间。

        真实传输由 EdgeSimPy 动态公平分配；未来到达流未知，所以这不是
        截止时间保证。误差在任务 CSV 中保留，不向调度器泄漏未来任务。
        CPU 单槽串行，输出传输不占执行槽；计算量单位 GFLOP、算力 GFLOP/s。
        """
        cfg = self.config
        dt = cfg["tick_s"]
        cap = node.capability_for(task)
        assert cap is not None
        bw = min(node.bandwidth_mb_s, cfg["backbone_bandwidth_mb_s"]) / max(1, share)
        prop = node.latency_s + cfg["backbone_latency_s"]
        inbound = ceil_tick(task.input_mb / bw, dt) if task.input_mb else 0.0
        outbound = ceil_tick(task.output_mb / bw, dt) if task.output_mb else 0.0
        input_ready = ceil_tick(now + inbound + prop, dt)
        cpu_finish = ceil_tick(max(ready[node.node_id], input_ready)
                               + ceil_tick(task.work_gflop * cap.compute_factor / node.cpu_gflops, dt), dt)
        finish = ceil_tick(cpu_finish + outbound + prop, dt)
        return Assignment(task.task_id, node.node_id, cap.model_id,
                          cap.quality_by_input[task.input_class], cpu_finish, finish)

    def _rejection(self, req: Request, ready: dict[str, float], now: float, share: int) -> str:
        for task in req.tasks:
            candidates = self.compatible(task)
            if not candidates:
                return "no_compatible_capacity"
            qualified = [n for n in candidates if n.capability_for(task).quality_by_input[task.input_class]
                         + EPS >= task.min_quality]
            if not qualified:
                return "no_quality_candidate"
            if not any(self.estimate(task, n, ready, now, share).predicted_finish_s
                       + req.aggregation_s <= req.deadline_s + EPS for n in qualified):
                return "predicted_deadline"
        return "joint_deadline_or_search_limit"

    def _joint_options(self, req: Request, ready: dict[str, float], now: float, share: int):
        """枚举同一请求内子任务的联合放置；已分配任务立即预约 CPU 队尾。"""
        options = []
        for task in req.tasks:
            options.append([n for n in self.compatible(task)
                            if n.capability_for(task).quality_by_input[task.input_class]
                            + EPS >= task.min_quality])
        for placement in product(*options):
            book, assigned = dict(ready), []
            for task, node in zip(req.tasks, placement):
                a = self.estimate(task, node, book, now, share)
                a = self._score(task, node, a, req, now)
                book[node.node_id] = a.predicted_cpu_finish_s
                assigned.append(a)
            # 子任务可并行，完整请求用最晚返回时刻再加汇聚开销。
            finish = ceil_tick(max(a.predicted_finish_s for a in assigned) + req.aggregation_s,
                               self.config["tick_s"])
            if finish <= req.deadline_s + EPS:
                budget = max(req.deadline_s - req.arrival_s, EPS)
                cost = (finish - req.arrival_s) / budget - self.config.get("matching_weight", 0.15) * (
                    sum(a.matching_score for a in assigned) / len(assigned))
                yield RequestPlan(req.request_id, tuple(assigned)), book, cost

    def _score(self, task, node, assignment, req, now):
        advice = recommend(task.min_quality, assignment.quality, req.deadline_s - now,
                           assignment.predicted_finish_s + req.aggregation_s - now,
                           task.memory_mb, node.memory_mb,
                           self.config.get("matching_weights", [0.5, 0.4, 0.1]))
        return replace(assignment, matching_score=advice.score)

    def plan_batch(self, requests: tuple[Request, ...], ready: dict[str, float], now: float,
                   active_flows: int = 0) -> tuple[RequestPlan, ...]:
        if not requests:
            return ()
        # 一轮管理回调内已到达请求同时可见；不读取未来到达。
        share = max(1, active_flows + sum(len(r.tasks) for r in requests))
        if self.method == "global_beam":
            # 状态=(已接纳请求数, 累计归一化时延减匹配奖励, CPU预约表, 计划列表)。
            # 字典序目标：先最大接纳数，再最小二级代价；保留拒绝分支才能避免
            # 先接收一个昂贵请求，导致同批更多请求无法满足截止时间。
            states = [(0, 0.0, dict(ready), ())]
            for req in sorted(requests, key=lambda r: (r.deadline_s, r.request_id)):
                expanded = []
                for accepted, latency, book, plans in states:
                    expanded.append((accepted, latency, book, plans + (RequestPlan(
                        req.request_id, rejection_reason="batch_admission_tradeoff"),)))
                    for plan, next_book, cost in self._joint_options(req, book, now, share):
                        expanded.append((accepted + 1, latency + cost, next_book, plans + (plan,)))
                expanded.sort(key=lambda s: (-s[0], s[1], tuple(s[2].values())))
                states = expanded[:self.config["beam_width"]]
            best = states[0]
            result = []
            for plan in best[3]:
                if not plan.assignments:
                    req = next(r for r in requests if r.request_id == plan.request_id)
                    reason = self._rejection(req, ready, now, share)
                    if reason != "joint_deadline_or_search_limit":
                        plan = replace(plan, rejection_reason=reason)
                result.append(plan)
            return tuple(result)

        book, plans = dict(ready), []
        ordered = sorted(requests, key=lambda r: (r.deadline_s, r.request_id)) if self.method == "qos_greedy" else requests
        for req in ordered:
            trial, chosen, reason = dict(book), [], ""
            for task in req.tasks:
                candidates = self.compatible(task)
                if not candidates:
                    reason = "no_compatible_capacity"
                    break
                values = [self._score(task, n, self.estimate(task, n, trial, now, share), req, now)
                          for n in candidates]
                if self.method == "qos_greedy":
                    values = [a for a in values if a.quality + EPS >= task.min_quality
                              and ceil_tick(a.predicted_finish_s + req.aggregation_s,
                                            self.config["tick_s"]) <= req.deadline_s + EPS]
                    if not values:
                        reason = self._rejection(req, trial, now, share)
                        break
                if self.method == "round_robin":
                    a = values[self.cursor % len(values)]
                    self.cursor += 1
                elif self.method == "quality_first":
                    a = min(values, key=lambda x: (-x.quality, x.predicted_finish_s, x.node_id))
                elif self.method == "qos_greedy":
                    a = min(values, key=lambda x: ((x.predicted_finish_s + req.aggregation_s - req.arrival_s)
                        / max(req.deadline_s - req.arrival_s, EPS)
                        - self.config.get("matching_weight", 0.15) * x.matching_score, x.node_id))
                else:
                    a = min(values, key=lambda x: (x.predicted_finish_s, x.node_id))
                trial[a.node_id] = a.predicted_cpu_finish_s
                chosen.append(a)
            # 请求整体接纳/拒绝，失败分支不会提交此前子任务的预约。
            if reason:
                plans.append(RequestPlan(req.request_id, rejection_reason=reason))
            else:
                plans.append(RequestPlan(req.request_id, tuple(chosen)))
                book = trial
        return tuple(plans)
