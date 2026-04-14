#!/usr/bin/env python3
"""
生成 track_5_long_horizon.json

在 vlabench conda 环境下运行：
    conda activate vlabench
    cd /datasets/code/VLABench
    python generate_track5_long_horizon.py \
        --n-episodes 50 \
        --output VLABench/configs/evaluation/tracks/track_5_long_horizon.json

长期任务列表（来自 dim2task.json 的 Complex 类别）：
    cook_dishes, book_rearrange, texas_holdem, cool_drink,
    make_juice, hammer_nail_and_hang_picture, take_chemistry_experiment
"""

import argparse
import json
import os
import random
import types


class _ClassNameEncoder(json.JSONEncoder):
    """将 ABCMeta / type 类对象序列化为其类名字符串。"""
    def default(self, obj):
        if isinstance(obj, type):
            return obj.__name__
        return super().default(obj)

os.environ.setdefault("VLABENCH_ROOT", os.path.join(os.path.dirname(__file__), "VLABench"))
os.environ.setdefault("MUJOCO_GL", "egl")

from VLABench.tasks import *


LONG_HORIZON_TASKS = [
    "cook_dishes",
    "book_rearrange",
    "texas_holdem",
    "cool_drink",
    "make_juice",
    "hammer_nail_and_hang_picture",
    "take_chemistry_experiment",
]


def generate_episodes(task_name: str, n_episodes: int) -> list:
    from VLABench.utils.register import register

    if task_name not in register._config_managers:
        print(f"  [警告] {task_name} 未在 register.config_manager 中找到，跳过")
        return []

    mgr_cls = register._config_managers[task_name]
    mgr = mgr_cls(task_name)

    episodes = []
    for i in range(n_episodes):
        try:
            cfg = mgr.get_seen_task_config()
            episodes.append({"task": cfg["task"]})
        except Exception as e:
            print(f"  [警告] {task_name} 第 {i} 条生成失败: {e}")
            continue

    print(f"  {task_name}: 生成 {len(episodes)}/{n_episodes} 条 episode")
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=50,
                        help="每个任务生成的 episode 数量")
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.environ["VLABENCH_ROOT"],
                                             "configs/evaluation/tracks/track_5_long_horizon.json"),
                        help="输出 JSON 路径")
    parser.add_argument("--tasks", nargs="+", default=LONG_HORIZON_TASKS,
                        help="要生成的任务列表")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    track = {}
    for task in args.tasks:
        print(f"生成 {task} ...")
        episodes = generate_episodes(task, args.n_episodes)
        if episodes:
            track[task] = episodes

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(track, f, indent=2, cls=_ClassNameEncoder)

    print(f"\n已保存到 {args.output}")
    print(f"任务数: {len(track)}, 总 episode 数: {sum(len(v) for v in track.values())}")


if __name__ == "__main__":
    main()
