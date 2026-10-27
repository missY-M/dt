"""DT 的主动推荐：在相同含义的需求/能力维度上计算可解释满足度。

老师提出把需求与能力组织成向量后对齐。第一阶段使用结构化向量，
不训练语义编码器，也不把类型标签之间的余弦相似度伪装成模型能力。
类型、输入条件在 Planner.compatible 中硬匹配；这里对齐质量、时延、
执行内存三维。时延越小越好，不能把时延数值越大错误地视为能力越强。
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DTRecommendation:
    demand_vector: tuple[float, float, float]
    capability_vector: tuple[float, float, float]
    satisfaction_vector: tuple[float, float, float]
    score: float


def recommend(min_quality: float, quality: float, time_budget_s: float,
              predicted_latency_s: float, required_memory_mb: float,
              available_memory_mb: float, weights=(0.5, 0.4, 0.1)) -> DTRecommendation:
    """带方向和量纲归一化的需求满足度，分数在 [0,1]。

    先以硬约束保证可行，再用可用裕量排序。每一维用同维需求归一化，
    不直接把秒、MB 和精度拼起来算欧氏距离。质量在恰好达标时裕量为零，
    但仍然可接纳。权重是可配置实验超参数，不代表老师指定值。
    """
    clip = lambda x: min(1.0, max(0.0, x))
    quality_slack = clip((quality - min_quality) / max(1.0 - min_quality, 1e-9))
    time_slack = clip(1.0 - predicted_latency_s / max(time_budget_s, 1e-9))
    memory_slack = clip(1.0 - required_memory_mb / max(available_memory_mb, 1e-9))
    satisfaction = (quality_slack, time_slack, memory_slack)
    if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("必须提供三个有限非负匹配权重，且至少一个为正")
    total = sum(weights)
    return DTRecommendation((min_quality, time_budget_s, required_memory_mb),
                            (quality, predicted_latency_s, available_memory_mb),
                            satisfaction, sum(w * v for w, v in zip(weights, satisfaction)) / total)
