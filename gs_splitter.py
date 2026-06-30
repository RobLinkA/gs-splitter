# -*- coding: utf-8 -*-
"""
GS-Splitter：3D Gaussian Splatting PLY 大半径异常高斯点后处理拆分工具
=============================================================

核心改进点：
  1) 更稳健的异常筛查：全局分位 + MAD robust z-score + 局部正常邻域半径比。
  2) 自适应拆分数量：按 (父点半径 / 局部目标半径)^3 估算，而不是线性估算。
  3) 有界椭球采样：子点中心限制在母高斯椭球内部，避免正态随机采样飞出边界。
  4) opacity 守恒：默认把父点 alpha 拆成多个子 alpha，避免复制 opacity 导致局部变厚/变亮。
  5) 输出体积保护：通过 max_output_multiplier 控制点数膨胀上限。
  6) 保留所有原始字段：x/y/z、normal、SH、opacity、scale、rotation 及其他字段都会继承。

依赖：
  pip install numpy scipy plyfile
  可选拖拽 GUI：pip install tkinterdnd2

运行：
  GUI:  python gs_split_gui_optimized.py
  CLI:  python gs_split_gui_optimized.py input.ply --profile balanced

注意：
  - 标准 3DGS PLY 的 scale_* 通常是 log scale，本工具会 exp 后处理，再 log 写回。
  - opacity 通常是 logit opacity，本工具默认按 alpha 合成近似守恒后写回 logit。
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import threading
import traceback
import warnings
from collections import OrderedDict
from typing import Callable, Dict, Iterable, Tuple

import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("缺少依赖 plyfile，请运行：pip install plyfile")
    raise

try:
    from scipy.spatial import cKDTree
except ImportError:
    print("缺少依赖 scipy，请运行：pip install scipy")
    raise

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False


# =========================================================
#                         参数
# =========================================================

def default_params(profile: str = "balanced") -> OrderedDict:
    """返回默认参数。profile 只改变默认值，不改变算法逻辑。"""
    presets = {
        # 通用默认：适合大部分存量 3DGS PLY，兼顾效果与体积。
        "balanced": OrderedDict([
            ("global_percentile", 97.5),       # 全局大半径候选阈值百分位
            ("normal_percentile", 85.0),       # 局部参考半径只优先看低于该分位的“正常邻居”
            ("local_ratio", 2.10),             # 候选点 > 局部正常半径 * 该倍数才确认
            ("local_ratio_hard", 3.20),        # 即便没超过全局阈值，超过该局部倍数也强制候选
            ("robust_z", 5.00),                # median + robust_z * MAD_sigma 的全局异常阈值
            ("knn", 32),                       # 局部邻域数量
            ("min_split", 2),                  # 单点最少拆分数
            ("max_split", 64),                 # 单点最多拆分数
            ("target_radius_factor", 1.15),    # 子点目标半径 = 局部正常半径 * 该系数
            ("split_multiplier", 1.00),        # 拆分强度倍率
            ("max_output_multiplier", 1.80),   # 输出点数最多为原始点数的多少倍
            ("spread", 0.70),                  # 子点中心在父椭球内的散布强度
            ("shrink_extra", 1.05),            # 额外缩小，>1 更锐利但更可能欠覆盖
            ("boundary_margin", 1.15),         # 子点中心离父椭球边界至少留出多少倍子半径
            ("min_alpha", 0.010),              # opacity 存在时，低于该 alpha 的点不拆，避免处理几乎不可见噪声
            ("opacity_conserve", 1.0),         # 1=按 alpha 守恒拆分，0=直接复制父 opacity
            ("opacity_gain", 1.00),            # 子点 alpha 额外增益，一般保持 1
            ("rot_noise", 0.0),                # 姿态微扰弧度，默认关闭
            ("color_noise", 0.0),              # f_dc 颜色微扰，默认关闭
            ("min_scale_log", -12.0),          # scale log 下限
            ("random_seed", 12345),            # 固定随机种子，保证可复现
            ("write_when_no_change", 1.0),     # 没有异常时是否仍写出 _split.ply
        ]),
        # 文物/小物体：更积极，允许更多点数增长。
        "artifact": OrderedDict([
            ("global_percentile", 96.5),
            ("normal_percentile", 82.0),
            ("local_ratio", 1.85),
            ("local_ratio_hard", 2.80),
            ("robust_z", 4.50),
            ("knn", 40),
            ("min_split", 2),
            ("max_split", 128),
            ("target_radius_factor", 1.05),
            ("split_multiplier", 1.10),
            ("max_output_multiplier", 2.50),
            ("spread", 0.72),
            ("shrink_extra", 1.08),
            ("boundary_margin", 1.20),
            ("min_alpha", 0.006),
            ("opacity_conserve", 1.0),
            ("opacity_gain", 1.02),
            ("rot_noise", 0.0),
            ("color_noise", 0.0),
            ("min_scale_log", -12.0),
            ("random_seed", 12345),
            ("write_when_no_change", 1.0),
        ]),
        # 建筑/大场景：更保守，重点压边缘毛刺和大球，控制体积。
        "city": OrderedDict([
            ("global_percentile", 98.5),
            ("normal_percentile", 88.0),
            ("local_ratio", 2.35),
            ("local_ratio_hard", 3.60),
            ("robust_z", 5.50),
            ("knn", 32),
            ("min_split", 2),
            ("max_split", 32),
            ("target_radius_factor", 1.25),
            ("split_multiplier", 0.85),
            ("max_output_multiplier", 1.45),
            ("spread", 0.65),
            ("shrink_extra", 1.04),
            ("boundary_margin", 1.15),
            ("min_alpha", 0.012),
            ("opacity_conserve", 1.0),
            ("opacity_gain", 1.00),
            ("rot_noise", 0.0),
            ("color_noise", 0.0),
            ("min_scale_log", -12.0),
            ("random_seed", 12345),
            ("write_when_no_change", 1.0),
        ]),
    }
    if profile not in presets:
        raise ValueError(f"未知 profile: {profile}，可选：{', '.join(presets)}")
    return presets[profile].copy()


PARAM_LABELS = OrderedDict([
    ("global_percentile", "全局半径百分位"),
    ("normal_percentile", "正常邻居百分位"),
    ("local_ratio", "局部异常倍数"),
    ("local_ratio_hard", "强制局部倍数"),
    ("robust_z", "MAD异常强度"),
    ("knn", "KNN邻居数"),
    ("min_split", "最小拆分数"),
    ("max_split", "最大拆分数"),
    ("target_radius_factor", "目标半径系数"),
    ("split_multiplier", "拆分强度倍率"),
    ("max_output_multiplier", "输出点数上限倍数"),
    ("spread", "椭球内散布强度"),
    ("shrink_extra", "额外收缩系数"),
    ("boundary_margin", "边界安全系数"),
    ("min_alpha", "最低可见Alpha"),
    ("opacity_conserve", "Alpha守恒 1/0"),
    ("opacity_gain", "Alpha增益"),
    ("rot_noise", "旋转扰动弧度"),
    ("color_noise", "颜色扰动"),
    ("min_scale_log", "最小scale(log)"),
    ("random_seed", "随机种子"),
    ("write_when_no_change", "无变化仍写出 1/0"),
])


# =========================================================
#                         数学工具
# =========================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """q: (..., 4)，顺序 w, x, y, z。"""
    n = np.sqrt(np.sum(q * q, axis=-1, keepdims=True))
    n[n == 0] = 1.0
    return q / n


def quat_rotate_vectors(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用单位四元数 q(N,4: w,x,y,z) 旋转向量 v(N,3)。"""
    w = q[:, 0]
    qv = q[:, 1:4]
    t = 2.0 * np.cross(qv, v)
    return v + w[:, None] * t + np.cross(qv, t)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法 q1*q2，均为 (N,4)，顺序 w,x,y,z。"""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1)


def sample_unit_ball(m: int, rng: np.random.Generator) -> np.ndarray:
    """均匀采样单位球内部，保证 ||p|| <= 1。"""
    if m <= 0:
        return np.empty((0, 3), dtype=np.float64)
    v = rng.normal(size=(m, 3))
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    direction = v / norm
    radius = rng.random(m) ** (1.0 / 3.0)
    return direction * radius[:, None]


# =========================================================
#                       PLY / 3DGS 工具
# =========================================================

def get_vertex_data(ply: PlyData):
    for el in ply.elements:
        if el.name == "vertex":
            return el, el.data
    return ply.elements[0], ply.elements[0].data


def require_fields(names: Iterable[str], required: Iterable[str]) -> None:
    missing = [f for f in required if f not in names]
    if missing:
        raise ValueError("该 PLY 缺少字段：" + ", ".join(missing) + "。请确认是标准 3DGS PLY。")


def stack_fields(data: np.ndarray, fields: Tuple[str, ...], dtype=np.float64) -> np.ndarray:
    return np.stack([np.asarray(data[f]) for f in fields], axis=1).astype(dtype, copy=False)


def _safe_percentile(values: np.ndarray, p: float) -> float:
    p = float(np.clip(p, 0.0, 100.0))
    return float(np.percentile(values, p))


# =========================================================
#                      核心处理逻辑
# =========================================================

def process_ply(
    input_path: str,
    params: Dict[str, float],
    log_fn: Callable[[str], None] = print,
    progress_fn: Callable[[float], None] = lambda v: None,
) -> str | None:
    """处理 3DGS PLY，返回输出路径；若无变化且不写出，返回 None。"""
    p = default_params("balanced")
    p.update(params or {})

    log_fn("=" * 72)
    log_fn(f"读取文件: {input_path}")
    ply = PlyData.read(input_path)
    el, data = get_vertex_data(ply)
    dtype = data.dtype
    names = list(dtype.names or [])
    n_points = len(data)
    log_fn(f"顶点/高斯点总数: {n_points:,}")
    progress_fn(4)

    if n_points < 3:
        raise ValueError("点数太少，无法进行 KNN 局部判断。")

    require_fields(names, ("x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"))

    # 3DGS 原始 PLY 里 scale_* 通常为 log scale。
    sc_log = stack_fields(data, ("scale_0", "scale_1", "scale_2"))
    scale_lin = np.exp(np.clip(sc_log, -60.0, 60.0))
    # 使用最大轴长抓“刺出来/大球”问题，同时保留几何平均半径用于日志。
    size = np.max(scale_lin, axis=1)
    size_geo = np.cbrt(np.maximum(scale_lin[:, 0] * scale_lin[:, 1] * scale_lin[:, 2], 1e-36))
    log_fn(
        "半径统计(max-axis, linear): "
        f"min={size.min():.6g}, median={np.median(size):.6g}, "
        f"p95={np.percentile(size, 95):.6g}, p99={np.percentile(size, 99):.6g}, max={size.max():.6g}"
    )
    log_fn(
        "半径统计(geo-mean, linear): "
        f"median={np.median(size_geo):.6g}, p99={np.percentile(size_geo, 99):.6g}"
    )
    progress_fn(12)

    xyz = stack_fields(data, ("x", "y", "z"))
    if not np.all(np.isfinite(xyz)):
        raise ValueError("x/y/z 中存在 NaN 或 Inf，请先清理 PLY。")
    if not np.all(np.isfinite(scale_lin)):
        raise ValueError("scale_* 中存在 NaN 或 Inf，请先清理 PLY。")

    # ---------- 局部参考半径 ----------
    knn = int(np.clip(round(float(p["knn"])), 2, max(2, n_points - 1)))
    k_query = min(knn + 1, n_points)
    log_fn(f"构建 KDTree，KNN={knn}，分析局部正常半径 ...")
    tree = cKDTree(xyz)
    dists, idxs = tree.query(xyz, k=k_query)
    if k_query == 1:
        raise ValueError("KNN 查询失败：邻域为空。")
    neigh_idx = idxs[:, 1:]
    neigh_size = size[neigh_idx]
    normal_thr = _safe_percentile(size, p["normal_percentile"])

    normal_mask = neigh_size <= normal_thr
    masked_neigh_size = np.where(normal_mask, neigh_size, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        local_ref = np.nanmedian(masked_neigh_size, axis=1)
    fallback_ref = np.median(neigh_size, axis=1)
    local_ref = np.where(np.isfinite(local_ref), local_ref, fallback_ref)
    local_ref = np.maximum(local_ref, 1e-12)

    # 局部点距仅用于日志，不直接强过滤，避免不同尺度场景误伤。
    local_spacing = np.median(dists[:, 1:], axis=1)
    local_spacing = np.maximum(local_spacing, 1e-12)
    progress_fn(34)

    # ---------- 异常筛查 ----------
    global_thr = _safe_percentile(size, p["global_percentile"])
    med = float(np.median(size))
    mad = float(np.median(np.abs(size - med)))
    robust_sigma = max(1.4826 * mad, 1e-12)
    robust_thr = med + float(p["robust_z"]) * robust_sigma

    ratio_local = size / local_ref
    candidate = (size >= global_thr) | (size >= robust_thr) | (ratio_local >= float(p["local_ratio_hard"]))
    confirm = candidate & (ratio_local >= float(p["local_ratio"]))

    if "opacity" in names and float(p["min_alpha"]) > 0:
        alpha = sigmoid(np.asarray(data["opacity"], dtype=np.float64))
        visible_mask = alpha >= float(p["min_alpha"])
        confirm &= visible_mask
        log_fn(f"opacity 可见性过滤: alpha >= {float(p['min_alpha']):.4g}, 保留可见点 {visible_mask.sum():,}/{n_points:,}")
    else:
        alpha = None

    sel_idx = np.where(confirm)[0]
    log_fn(
        "异常筛查: "
        f"global_thr(p{float(p['global_percentile']):.2f})={global_thr:.6g}, "
        f"robust_thr={robust_thr:.6g}, normal_ref_thr(p{float(p['normal_percentile']):.1f})={normal_thr:.6g}"
    )
    log_fn(
        f"候选 {int(candidate.sum()):,} 个；局部确认异常 {len(sel_idx):,} 个 "
        f"({len(sel_idx) / n_points * 100:.4f}%)"
    )
    if len(sel_idx) > 0:
        log_fn(
            "异常点局部倍数统计: "
            f"median={np.median(ratio_local[sel_idx]):.3f}, "
            f"p90={np.percentile(ratio_local[sel_idx], 90):.3f}, "
            f"max={np.max(ratio_local[sel_idx]):.3f}"
        )
    progress_fn(45)

    if len(sel_idx) == 0:
        log_fn("未检测到符合条件的大半径异常点。")
        if float(p.get("write_when_no_change", 1.0)) >= 0.5:
            return write_ply(data.copy(), input_path, log_fn, progress_fn)
        progress_fn(100)
        return None

    # ---------- 自适应拆分数 ----------
    min_split = int(max(2, round(float(p["min_split"]))))
    max_split = int(max(min_split, round(float(p["max_split"]))))
    target_radius = np.maximum(local_ref[sel_idx] * float(p["target_radius_factor"]), math.exp(float(p["min_scale_log"])))
    split_ratio = np.maximum(size[sel_idx] / target_radius, 1.0)
    n_split = np.ceil((split_ratio ** 3.0) * float(p["split_multiplier"])).astype(np.int64)
    n_split = np.clip(n_split, min_split, max_split)

    # 输出点数预算：final = 原点数 - 被替换父点 + 子点数。
    max_output_multiplier = max(1.0, float(p["max_output_multiplier"]))
    max_out_points = int(math.floor(n_points * max_output_multiplier))
    allowed_children = max_out_points - (n_points - len(sel_idx))
    allowed_children = max(0, allowed_children)

    if int(n_split.sum()) > allowed_children:
        severity = ratio_local[sel_idx]
        if allowed_children < min_split * len(sel_idx):
            keep_parent_count = max(1, allowed_children // min_split)
            order = np.argsort(severity)[::-1][:keep_parent_count]
            sel_idx = sel_idx[order]
            severity = severity[order]
            target_radius = np.maximum(local_ref[sel_idx] * float(p["target_radius_factor"]), math.exp(float(p["min_scale_log"])))
            split_ratio = np.maximum(size[sel_idx] / target_radius, 1.0)
            n_split = np.ceil((split_ratio ** 3.0) * float(p["split_multiplier"])).astype(np.int64)
            n_split = np.clip(n_split, min_split, max_split)
            allowed_children = max_out_points - (n_points - len(sel_idx))
            allowed_children = max(min_split * len(sel_idx), allowed_children)
            log_fn(
                "点数预算触发：异常点数量过多，优先保留最严重异常点 "
                f"{len(sel_idx):,} 个参与拆分。"
            )

        min_total = min_split * len(sel_idx)
        extra = np.maximum(n_split - min_split, 0)
        allowed_extra = max(0, allowed_children - min_total)
        if extra.sum() > allowed_extra:
            if extra.sum() > 0 and allowed_extra > 0:
                raw_extra = extra.astype(np.float64) * (allowed_extra / float(extra.sum()))
                floored = np.floor(raw_extra).astype(np.int64)
                remain = int(allowed_extra - floored.sum())
                if remain > 0:
                    # 把剩余配额给小数部分更大、且异常更严重的点。
                    frac = raw_extra - floored
                    order = np.lexsort((-severity, -frac))[::-1]
                    floored[order[:remain]] += 1
                n_split = min_split + floored
            else:
                n_split = np.full(len(sel_idx), min_split, dtype=np.int64)
            log_fn(
                "点数预算触发：已压缩单点拆分数，"
                f"max_output_multiplier={max_output_multiplier:.2f}。"
            )

    total_children = int(n_split.sum())
    final_points = n_points - len(sel_idx) + total_children
    log_fn(
        f"拆分计划: 父点 {len(sel_idx):,} 个，子点 {total_children:,} 个，"
        f"输出总点数 {final_points:,} / 原始 {n_points:,} = {final_points / n_points:.3f}x"
    )
    log_fn(
        f"单点拆分数: min={n_split.min()}, median={np.median(n_split):.1f}, "
        f"p90={np.percentile(n_split, 90):.1f}, max={n_split.max()}"
    )
    progress_fn(55)

    # ---------- 构造输出 ----------
    keep_mask = np.ones(n_points, dtype=bool)
    keep_mask[sel_idx] = False
    kept = data[keep_mask]
    out = np.empty(final_points, dtype=dtype)
    out[:len(kept)] = kept

    rng = np.random.default_rng(int(round(float(p["random_seed"]))))
    rot_all = stack_fields(data, ("rot_0", "rot_1", "rot_2", "rot_3"))
    cursor = len(kept)
    parent_chunk = 8192

    for start in range(0, len(sel_idx), parent_chunk):
        end = min(start + parent_chunk, len(sel_idx))
        parents = sel_idx[start:end]
        counts = n_split[start:end]
        parent_rep = np.repeat(parents, counts)
        per_child_n = np.repeat(counts, counts).astype(np.float64)
        m = len(parent_rep)

        children = out[cursor:cursor + m]
        for name in names:
            children[name] = data[name][parent_rep]

        # scale：按拆分数量立方根缩小，并额外向局部目标半径靠拢。
        div = np.power(per_child_n, 1.0 / 3.0) * max(float(p["shrink_extra"]), 1.0)
        new_scale_lin = scale_lin[parent_rep] / div[:, None]
        target_rep = (local_ref[parent_rep] * float(p["target_radius_factor"]))[:, None]
        new_scale_lin = np.minimum(new_scale_lin, target_rep)
        min_scale = math.exp(float(p["min_scale_log"]))
        new_scale_lin = np.maximum(new_scale_lin, min_scale)

        # 有界椭球采样：center offset 不超过父尺度 - 子尺度安全边界。
        unit = sample_unit_ball(m, rng)
        boundary_margin = max(float(p["boundary_margin"]), 1.0)
        fit_radii = scale_lin[parent_rep] - new_scale_lin * boundary_margin
        fit_radii = np.maximum(fit_radii, scale_lin[parent_rep] * 0.05)
        local_off = unit * fit_radii * np.clip(float(p["spread"]), 0.0, 1.0)
        q_parent = quat_normalize(rot_all[parent_rep])
        world_off = quat_rotate_vectors(q_parent, local_off)
        children["x"] = children["x"] + world_off[:, 0]
        children["y"] = children["y"] + world_off[:, 1]
        children["z"] = children["z"] + world_off[:, 2]

        new_scale_log = np.log(np.maximum(new_scale_lin, min_scale))
        new_scale_log = np.maximum(new_scale_log, float(p["min_scale_log"]))
        children["scale_0"] = new_scale_log[:, 0]
        children["scale_1"] = new_scale_log[:, 1]
        children["scale_2"] = new_scale_log[:, 2]

        # rotation：默认继承并归一化；可选微扰。
        q_child = quat_normalize(stack_fields(children, ("rot_0", "rot_1", "rot_2", "rot_3")))
        if float(p["rot_noise"]) > 0:
            axis = rng.normal(size=(m, 3))
            axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
            ang = rng.normal(0.0, float(p["rot_noise"]), m)
            half = ang / 2.0
            dq = np.concatenate([np.cos(half)[:, None], axis * np.sin(half)[:, None]], axis=1)
            q_child = quat_normalize(quat_mul(dq, q_child))
        children["rot_0"] = q_child[:, 0]
        children["rot_1"] = q_child[:, 1]
        children["rot_2"] = q_child[:, 2]
        children["rot_3"] = q_child[:, 3]

        # opacity：父 alpha 分解为 n 个子 alpha，近似保持合成透明度。
        if "opacity" in names and float(p["opacity_conserve"]) >= 0.5:
            parent_alpha = sigmoid(np.asarray(data["opacity"][parent_rep], dtype=np.float64))
            child_alpha = 1.0 - np.power(np.maximum(1.0 - parent_alpha, 1e-9), 1.0 / per_child_n)
            child_alpha = np.clip(child_alpha * float(p["opacity_gain"]), 1e-6, 1.0 - 1e-6)
            children["opacity"] = logit(child_alpha).astype(children["opacity"].dtype, copy=False)

        # 颜色微扰默认关闭；打开时只扰动 DC，不动 SH 高频项。
        if float(p["color_noise"]) > 0:
            sigma = float(p["color_noise"])
            for c in ("f_dc_0", "f_dc_1", "f_dc_2"):
                if c in names:
                    children[c] = children[c] + rng.normal(0.0, sigma, m).astype(children[c].dtype, copy=False)

        cursor += m
        if len(sel_idx) > 0:
            progress_fn(55 + 35 * (end / len(sel_idx)))

    log_fn(f"输出组成: 保留正常点 {len(kept):,} + 新生子点 {total_children:,} = {len(out):,}")
    progress_fn(94)
    return write_ply(out, input_path, log_fn, progress_fn)


def write_ply(out: np.ndarray, input_path: str, log_fn: Callable[[str], None], progress_fn: Callable[[float], None]) -> str:
    base, _ = os.path.splitext(input_path)
    output_path = base + "_split.ply"
    log_fn(f"写出文件: {output_path}")
    el_out = PlyElement.describe(out, "vertex")
    PlyData([el_out], text=False).write(output_path)
    progress_fn(100)
    log_fn("处理完成。")
    log_fn(f"结果已保存: {output_path}")
    log_fn("=" * 72)
    return output_path


# =========================================================
#                            GUI
# =========================================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("3DGS 大半径异常高斯点拆分优化工具")
        self.root.geometry("880x780")
        self.msg_q = queue.Queue()
        self.busy = False
        self.input_path = None

        top = ttk.LabelFrame(root, text="① 文件：拖入或点击选择 3DGS .ply")
        top.pack(fill="x", padx=10, pady=8)
        self.drop = tk.Label(
            top,
            text="把 .ply 文件拖到这里\n或点击选择文件",
            relief="ridge",
            bd=2,
            height=4,
            bg="#f3f6fb",
            fg="#333",
            cursor="hand2",
        )
        self.drop.pack(fill="x", padx=8, pady=8)
        self.drop.bind("<Button-1>", lambda e: self.choose_file())
        if DND_AVAILABLE:
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind("<<Drop>>", self.on_drop)
        else:
            self.drop.config(text="点击选择 .ply 文件\n未安装 tkinterdnd2，拖拽不可用")

        self.path_var = tk.StringVar(value="未选择文件")
        ttk.Label(top, textvariable=self.path_var, foreground="#0a5").pack(anchor="w", padx=8, pady=(0, 6))

        profile_frame = ttk.Frame(root)
        profile_frame.pack(fill="x", padx=10, pady=2)
        ttk.Label(profile_frame, text="参数预设:").pack(side="left")
        self.profile_var = tk.StringVar(value="balanced")
        self.profile_box = ttk.Combobox(
            profile_frame,
            textvariable=self.profile_var,
            values=["balanced", "artifact", "city"],
            width=14,
            state="readonly",
        )
        self.profile_box.pack(side="left", padx=6)
        ttk.Button(profile_frame, text="应用预设", command=self.apply_profile).pack(side="left")
        ttk.Label(profile_frame, text="balanced=通用；artifact=文物高精；city=建筑大场景保守").pack(side="left", padx=10)

        pf = ttk.LabelFrame(root, text="② 参数")
        pf.pack(fill="x", padx=10, pady=6)
        self.params_vars: Dict[str, tk.StringVar] = {}
        self.grid = ttk.Frame(pf)
        self.grid.pack(fill="x", padx=8, pady=6)
        self._build_param_grid(default_params("balanced"))

        cf = ttk.Frame(root)
        cf.pack(fill="x", padx=10, pady=8)
        self.btn = ttk.Button(cf, text="开始处理", command=self.start)
        self.btn.pack(side="left")
        self.pb = ttk.Progressbar(cf, length=620, mode="determinate", maximum=100)
        self.pb.pack(side="left", padx=10)
        self.pct = ttk.Label(cf, text="0%")
        self.pct.pack(side="left")

        lf = ttk.LabelFrame(root, text="③ 日志")
        lf.pack(fill="both", expand=True, padx=10, pady=8)
        self.log = tk.Text(lf, height=18, wrap="word", bg="#101418", fg="#cfe6ff")
        self.log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lf, command=self.log.yview)
        sb.pack(side="right", fill="y")
        self.log.config(yscrollcommand=sb.set)
        self.root.after(100, self.poll_queue)

    def _build_param_grid(self, params: OrderedDict):
        for w in self.grid.winfo_children():
            w.destroy()
        self.params_vars.clear()
        for i, (key, val) in enumerate(params.items()):
            r, c = divmod(i, 2)
            ttk.Label(self.grid, text=PARAM_LABELS.get(key, key) + ":", width=18).grid(
                row=r, column=c * 2, sticky="w", padx=4, pady=3
            )
            var = tk.StringVar(value=str(val))
            ttk.Entry(self.grid, textvariable=var, width=14).grid(
                row=r, column=c * 2 + 1, sticky="w", padx=4, pady=3
            )
            self.params_vars[key] = var

    def apply_profile(self):
        if self.busy:
            return
        self._build_param_grid(default_params(self.profile_var.get()))
        self.log_fn(f"已应用预设: {self.profile_var.get()}")

    def choose_file(self):
        if self.busy:
            return
        p = filedialog.askopenfilename(title="选择 PLY 文件", filetypes=[("PLY 文件", "*.ply"), ("所有文件", "*.*")])
        if p:
            self.set_file(p)

    def on_drop(self, event):
        if self.busy:
            return
        raw = event.data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        path = raw.split("} {")[0].strip().strip("{}")
        self.set_file(path)

    def set_file(self, p):
        if not p.lower().endswith(".ply"):
            messagebox.showwarning("提示", "请选择 .ply 文件")
            return
        if not os.path.isfile(p):
            messagebox.showwarning("提示", "文件不存在")
            return
        self.input_path = p
        self.path_var.set(p)
        self.drop.config(text="已选择文件\n" + os.path.basename(p))

    def log_fn(self, text):
        self.msg_q.put(("log", str(text)))

    def progress_fn(self, v):
        self.msg_q.put(("prog", float(v)))

    def poll_queue(self):
        try:
            while True:
                kind, val = self.msg_q.get_nowait()
                if kind == "log":
                    self.log.insert("end", val + "\n")
                    self.log.see("end")
                elif kind == "prog":
                    self.pb["value"] = val
                    self.pct.config(text=f"{int(val)}%")
                elif kind == "done":
                    self.busy = False
                    self.btn.config(state="normal")
                elif kind == "error":
                    self.busy = False
                    self.btn.config(state="normal")
                    messagebox.showerror("处理出错", val)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_queue)

    def start(self):
        if self.busy:
            return
        if not self.input_path:
            messagebox.showwarning("提示", "请先选择一个 .ply 文件")
            return
        try:
            params = {key: float(var.get()) for key, var in self.params_vars.items()}
        except ValueError:
            messagebox.showerror("参数错误", "请检查参数，必须为数字")
            return

        self.busy = True
        self.btn.config(state="disabled")
        self.pb["value"] = 0
        self.pct.config(text="0%")
        t = threading.Thread(target=self._worker, args=(self.input_path, params), daemon=True)
        t.start()

    def _worker(self, path, params):
        try:
            process_ply(path, params, self.log_fn, self.progress_fn)
            self.msg_q.put(("done", ""))
        except Exception:
            err = traceback.format_exc()
            self.log_fn("出错:\n" + err)
            self.msg_q.put(("error", err.splitlines()[-1] if err else "未知错误"))


def run_gui():
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    App(root)
    root.mainloop()


# =========================================================
#                            CLI
# =========================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="3DGS PLY 大半径异常高斯点后处理拆分工具")
    parser.add_argument("input", nargs="?", help="输入 .ply 文件。不传则启动 GUI。")
    parser.add_argument("--profile", choices=["balanced", "artifact", "city"], default="balanced", help="参数预设")
    parser.add_argument("--set", action="append", default=[], help="覆盖参数，例如 --set max_split=96 --set global_percentile=97")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.input:
        run_gui()
        return
    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)
    params = default_params(args.profile)
    for item in args.set:
        if "=" not in item:
            raise ValueError("--set 格式应为 key=value")
        key, value = item.split("=", 1)
        if key not in params:
            raise KeyError(f"未知参数 {key}，可选：{', '.join(params.keys())}")
        params[key] = float(value)
    process_ply(args.input, params, print, lambda v: None)


if __name__ == "__main__":
    main()
