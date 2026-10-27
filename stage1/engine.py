"""EdgeSimPy 适配层：真实框架时钟、节点、拓扑和共享网络流。

EdgeSimPy 原生面向长期服务，本文件扩展短请求的执行 FIFO 与汇聚。
不使用另一个离散事件模拟器，也不把模型精度当成推理代码测量值。
每节点一个执行槽；memory_mb 为已扣除常驻模型后可用的执行内存。
"""
from collections import deque
from dataclasses import dataclass
import math
import time
from edge_sim_py import Simulator, EdgeServer, NetworkSwitch, NetworkLink, NetworkFlow
from .scenario import NodeSpec, Request, TaskDemand
from .planner import Assignment, Planner, EPS, ceil_tick


@dataclass
class TaskState:
    demand: TaskDemand
    assignment: Assignment
    admitted_s: float
    status: str = "input"
    input_flow: object = None
    output_flow: object = None
    input_ready_s: float | None = None
    start_s: float | None = None
    cpu_finish_s: float | None = None
    returned_s: float | None = None


class Experiment:
    """一次独立实验。EdgeSimPy 有全局组件表，不允许多线程交错多个实例。"""
    def __init__(self, config: dict, nodes: tuple[NodeSpec, ...], requests: tuple[Request, ...], method: str):
        self.config, self.nodes, self.requests = config, nodes, requests
        self.dt = config["tick_s"]
        self.spec = {n.node_id: n for n in nodes}
        self.planner = Planner(config, nodes, method)
        self.method = method
        self.tasks: dict[str, TaskState] = {}
        self.results: dict[str, dict] = {}
        self.pending = deque(sorted(requests, key=lambda r: (r.arrival_s, r.request_id)))
        self.queue = {n.node_id: deque() for n in nodes}
        self.running = {n.node_id: None for n in nodes}
        self.cpu_busy_s = {n.node_id: 0.0 for n in nodes}
        self.assignment_count = {n.node_id: 0 for n in nodes}
        self.max_memory_mb = {n.node_id: 0.0 for n in nodes}
        self.decision_wall_s = []
        self.network_audit = []
        self.max_network_ratio = 0.0
        self.request_by_id = {r.request_id: r for r in requests}
        self.sim = Simulator(tick_duration=self.dt, tick_unit="seconds",
                             resource_management_algorithm=self._manage,
                             resource_management_algorithm_parameters={}, dump_interval=float("inf"))
        # initialize({}) 同时重置官方组件全局表，避免不同基线互相污染。
        self.sim.initialize({})
        self.client = self.sim.initialize_agent(NetworkSwitch())
        self.core = self.sim.initialize_agent(NetworkSwitch())
        self.servers, self.switches = {}, {}
        self.links = []
        self._link(self.client, self.core, config["backbone_bandwidth_mb_s"], config["backbone_latency_s"])
        for node in nodes:
            switch = self.sim.initialize_agent(NetworkSwitch())
            server = self.sim.initialize_agent(EdgeServer(cpu=node.cpu_gflops, memory=node.memory_mb, disk=0))
            server.network_switch = switch
            switch.edge_servers.append(server)
            self.switches[node.node_id], self.servers[node.node_id] = switch, server
            self._link(self.core, switch, node.bandwidth_mb_s, node.latency_s)

    def _link(self, first, second, bw: float, delay: float):
        link = self.sim.initialize_agent(NetworkLink())
        link.topology, link.nodes = self.sim.topology, [first, second]
        # 官方 NetworkFlow 每 tick 直接减 bandwidth，必须从 MB/s 换成 MB/tick。
        link.bandwidth, link.delay = bw * self.dt, delay
        graph = self.sim.topology
        graph.add_edge(first, second)
        # 与官方 initialize 实现一致，让图边和 NetworkLink 共享同一状态。
        graph._adj[first][second] = graph._adj[second][first] = link
        first.links.append(link)
        second.links.append(link)
        self.links.append(link)

    def _new_flow(self, state: TaskState, output: bool, now: float):
        task, node_id = state.demand, state.assignment.node_id
        size = task.output_mb if output else task.input_mb
        prop = self.spec[node_id].latency_s + self.config["backbone_latency_s"]
        if size <= EPS:
            # 零字节也保留显式消息传播时延，不给官方分配器传空路径。
            ready = ceil_tick(now + prop, self.dt)
            if output:
                state.returned_s = ready
                if ready <= now + EPS:
                    state.status = "returned"
            else:
                state.input_ready_s = ready
            return None
        switch = self.switches[node_id]
        path = [switch, self.core, self.client] if output else [self.client, self.core, switch]
        flow = NetworkFlow(topology=self.sim.topology, source=path[0], target=path[-1],
                           start=self.sim.schedule.steps + 1, path=path, data_to_transfer=size,
                           metadata={"type": "task_output" if output else "task_input", "object": task.task_id})
        self.sim.initialize_agent(flow)
        return flow

    def _collect_flow_completion(self, state: TaskState):
        prop = self.spec[state.assignment.node_id].latency_s + self.config["backbone_latency_s"]
        if state.input_ready_s is None and state.input_flow is not None and state.input_flow.status == "finished":
            state.input_ready_s = ceil_tick(state.input_flow.end * self.dt + prop, self.dt)
        if state.returned_s is None and state.output_flow is not None and state.output_flow.status == "finished":
            state.returned_s = ceil_tick(state.output_flow.end * self.dt + prop, self.dt)

    def _ready_hints(self, now: float) -> dict[str, float]:
        """只使用当前可观测状态预测队尾；不查未来任务，也不使用未来随机数。"""
        hints = {}
        active = max(1, sum(f.status == "active" for f in NetworkFlow.all()))
        for node in self.nodes:
            current = self.running[node.node_id]
            tail = max(now, current.cpu_finish_s) if current else now
            bw = min(node.bandwidth_mb_s, self.config["backbone_bandwidth_mb_s"]) / active
            prop = node.latency_s + self.config["backbone_latency_s"]
            for state in self.queue[node.node_id]:
                arrival = state.input_ready_s
                if arrival is None:
                    remaining = max(0.0, state.input_flow.data_to_transfer) if state.input_flow else 0.0
                    arrival = ceil_tick(now + remaining / bw + prop, self.dt)
                cap = node.capability_for(state.demand)
                tail = ceil_tick(max(tail, arrival) + ceil_tick(
                    state.demand.work_gflop * cap.compute_factor / node.cpu_gflops, self.dt), self.dt)
            hints[node.node_id] = tail
        return hints

    def _manage(self, parameters):
        """由 Simulator.step 回调：边界时刻结算事件，再接纳新任务。"""
        now = self.sim.schedule.steps * self.dt
        for state in self.tasks.values():
            if state.status == "returned":
                continue
            self._collect_flow_completion(state)
            if state.returned_s is not None and state.returned_s <= now + EPS:
                state.status = "returned"
        for node in self.nodes:
            node_id = node.node_id
            state = self.running[node_id]
            if state is not None and state.cpu_finish_s <= now + EPS:
                # 执行完成释放执行内存，返回网络与下一任务计算可以重叠。
                self.servers[node_id].memory_demand = 0.0
                self.servers[node_id].cpu_demand = 0.0
                self.running[node_id] = None
                state.status = "output"
                state.output_flow = self._new_flow(state, output=True, now=now)
        for req in self.requests:
            if req.request_id in self.results:
                continue
            states = [self.tasks.get(t.task_id) for t in req.tasks]
            if all(s is not None and s.returned_s is not None for s in states):
                finish = ceil_tick(max(s.returned_s for s in states) + req.aggregation_s, self.dt)
                if finish <= now + EPS:
                    quality_ok = all(s.assignment.quality + EPS >= s.demand.min_quality for s in states)
                    on_time = finish <= req.deadline_s + EPS
                    self.results[req.request_id] = dict(request_id=req.request_id, status="completed",
                        rejection_reason="", arrival_s=req.arrival_s, deadline_s=req.deadline_s,
                        finish_s=finish, latency_s=finish - req.arrival_s, quality_ok=quality_ok,
                        on_time=on_time, qos_ok=quality_ok and on_time)
        arriving = []
        while self.pending and self.pending[0].arrival_s <= now + EPS:
            arriving.append(self.pending.popleft())
        if arriving:
            begin = time.perf_counter()
            plans = self.planner.plan_batch(tuple(arriving), self._ready_hints(now), now,
                sum(f.status == "active" for f in NetworkFlow.all()))
            self.decision_wall_s.append(time.perf_counter() - begin)
            for plan in plans:
                req = self.request_by_id[plan.request_id]
                if plan.rejection_reason:
                    self.results[req.request_id] = dict(request_id=req.request_id, status="rejected",
                        rejection_reason=plan.rejection_reason, arrival_s=req.arrival_s,
                        deadline_s=req.deadline_s, finish_s=None, latency_s=None,
                        quality_ok=False, on_time=False, qos_ok=False)
                    continue
                task_by_id = {t.task_id: t for t in req.tasks}
                for assignment in plan.assignments:
                    state = TaskState(task_by_id[assignment.task_id], assignment, now)
                    self.tasks[state.demand.task_id] = state
                    self.queue[assignment.node_id].append(state)
                    self.assignment_count[assignment.node_id] += 1
                    state.input_flow = self._new_flow(state, output=False, now=now)
        for node in self.nodes:
            node_id = node.node_id
            queue = self.queue[node_id]
            if self.running[node_id] is None and queue:
                state = queue[0]
                # FIFO 按接纳顺序，不让后到小任务越过尚在传输的队头。
                if state.input_ready_s is not None and state.input_ready_s <= now + EPS:
                    queue.popleft()
                    assert state.demand.memory_mb <= node.memory_mb + EPS, "执行内存超容量"
                    state.start_s = now
                    cap = node.capability_for(state.demand)
                    duration = ceil_tick(state.demand.work_gflop * cap.compute_factor / node.cpu_gflops, self.dt)
                    state.cpu_finish_s = ceil_tick(now + duration, self.dt)
                    state.status = "running"
                    self.running[node_id] = state
                    self.servers[node_id].memory_demand = state.demand.memory_mb
                    self.servers[node_id].cpu_demand = node.cpu_gflops
                    self.max_memory_mb[node_id] = max(self.max_memory_mb[node_id], state.demand.memory_mb)

    def _audit_tick(self):
        """保存网络容量审计摘要；CPU 利用率在实际执行的 tick 上积分。"""
        step = self.sim.schedule.steps
        for node_id, state in self.running.items():
            if state is not None:
                self.cpu_busy_s[node_id] += self.dt
        # 每拍只遍历一次流表，避免为每条链路重复扫描历史流。
        allocation = {link.id: 0.0 for link in self.links}
        for flow in NetworkFlow.all():
            # 本拍结束的流也消费了本拍带宽；不能只统计仍 active 的流。
            if flow.status == "active" or flow.end == step:
                used = min(flow.bandwidth.values()) if flow.bandwidth else 0.0
                for link_id in flow.bandwidth:
                    allocation[link_id] += used
        for link in self.links:
            allocated = allocation[link.id]
            ratio = allocated / link.bandwidth if link.bandwidth else 0.0
            self.max_network_ratio = max(self.max_network_ratio, ratio)
            if ratio > 1 + 1e-7:
                raise AssertionError(f"链路 {link.id} 分配超过物理容量: {ratio}")

    def run(self) -> dict:
        wall_start = time.perf_counter()
        latest = max((r.deadline_s for r in self.requests), default=0.0)
        limit = latest + self.config["max_drain_s"]
        max_steps = math.ceil(limit / self.dt) + 1
        while len(self.results) < len(self.requests) and self.sim.schedule.steps < max_steps:
            self.sim.step()  # 框架推进时钟、网络分配以及 NetworkFlow 传输。
            self._audit_tick()
        now = self.sim.schedule.steps * self.dt
        # 截断样本显式计为失败，不能从分母中删除。
        for req in self.requests:
            if req.request_id not in self.results:
                self.results[req.request_id] = dict(request_id=req.request_id, status="unfinished",
                    rejection_reason="drain_limit", arrival_s=req.arrival_s, deadline_s=req.deadline_s,
                    finish_s=None, latency_s=None, quality_ok=False, on_time=False, qos_ok=False)
        task_rows = []
        for state in self.tasks.values():
            a, t = state.assignment, state.demand
            task_rows.append(dict(task_id=t.task_id, request_id=t.request_id, node_id=a.node_id,
                task_type=t.task_type, input_class=t.input_class, model_id=a.model_id,
                min_quality=t.min_quality, profile_quality=a.quality, memory_mb=t.memory_mb,
                dt_matching_score=a.matching_score,
                admitted_s=state.admitted_s, input_ready_s=state.input_ready_s, start_s=state.start_s,
                cpu_finish_s=state.cpu_finish_s, returned_s=state.returned_s,
                queue_wait_s=None if state.start_s is None else state.start_s - state.input_ready_s,
                predicted_return_s=a.predicted_finish_s, status=state.status))
        node_rows = [dict(node_id=n.node_id, assignments=self.assignment_count[n.node_id],
            cpu_busy_s=self.cpu_busy_s[n.node_id], cpu_utilization=self.cpu_busy_s[n.node_id] / now if now else 0,
            max_execution_memory_mb=self.max_memory_mb[n.node_id], memory_capacity_mb=n.memory_mb)
            for n in self.nodes]
        return dict(requests=sorted(self.results.values(), key=lambda r: r["request_id"]),
            tasks=task_rows, nodes=node_rows, simulation_duration_s=now,
            wall_runtime_s=time.perf_counter() - wall_start,
            scheduler_wall_s=sum(self.decision_wall_s), decision_count=len(self.decision_wall_s),
            max_network_capacity_ratio=self.max_network_ratio,
            network_flow_count=len(NetworkFlow.all()),
            residual_execution_memory_mb=sum(s.memory_demand for s in self.servers.values()))
