"""日本語指示 -> Teacher -> Waypoint -> Genesis CPU -> 実測判定 -> 自動修復 -> 成功デモ収集（3タスク版）。

実行例:
  python multi_pipeline.py run '赤・緑・青のキューブを同じ色のトレイに仕分けて'
  python multi_pipeline.py run '青いキューブを赤いキューブの上に積んで' --mode place_bias
  python multi_pipeline.py collect --episodes 20 --seed 100   # ランダム配置で成功デモを収集
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import uuid

import numpy as np

from pipeline import write_json
from tasks import (COLORS, Layout, MultiTeacher, TaskSpec, evaluate_task, make_waypoints, parse_task,
                   sample_layout)

ROOT = Path(__file__).resolve().parent
MODES = ("normal", "bad_start", "weak_force", "place_bias", "perturbed", "timing_jitter")
MAX_ITERATIONS = 6


def _relative(path: Path, base: Path) -> str:
    """公開リポジトリに絶対パスを残さないよう、出力フォルダ相対で記録する。"""
    path = Path(path).resolve()
    return str(path.relative_to(base)) if path.is_relative_to(base) else path.name


def run_task(instruction: str, sim, *, layout: Layout | None = None, mode: str = "normal",
             max_iterations: int = MAX_ITERATIONS, seed: int = 0, teacher: MultiTeacher | None = None,
             output_dir: Path = ROOT / "runs", run_name: str = "demo", video_path: Path | None = None,
             video_every: int = 4, quiet: bool = False, noise_std: float = 0.0) -> dict:
    if type(max_iterations) is not int or not 1 <= max_iterations <= MAX_ITERATIONS:
        raise ValueError(f"max_iterations must be an integer in [1, {MAX_ITERATIONS}]")
    task = parse_task(instruction)
    layout = (layout or Layout()).validate()
    teacher = teacher or MultiTeacher()
    params, decision = teacher.propose(task, layout, [], mode, seed)
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", run_name)
    run_dir = Path(output_dir).resolve() / f"{safe}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    history: list[dict] = []
    result = {"schema_version": 2, "run_id": run_dir.name, "task": asdict(task), "layout": layout.cubes,
              "mode": mode, "seed": seed, "noise_std": noise_std, "max_iterations": max_iterations, "success": False,
              "history": history, "trajectory_file": None, "video_file": None}
    frames: list[np.ndarray] = []
    for iteration in range(1, max_iterations + 1):
        if iteration > 1:
            params, decision = teacher.propose(task, layout, history, mode, seed)
        prompt_file = run_dir / f"teacher_{iteration:02d}.txt"
        prompt_file.write_text(decision.pop("prompt"), encoding="utf-8")
        segments = make_waypoints(task, layout, params)
        frame_sink = frames.append if video_path is not None else None
        trajectory = sim.rollout(layout, segments, params.F, video_every=video_every if frame_sink else 0,
                                 on_frame=frame_sink, noise_std=noise_std, noise_seed=seed * 100 + iteration)
        metrics = evaluate_task(trajectory, task, layout)
        physics_seconds, rollout_seconds = trajectory["timing"].tolist()
        metrics.update(physics_seconds=physics_seconds, rollout_seconds=rollout_seconds,
                       steps_per_second=metrics["steps"] / physics_seconds)
        history.append({"iteration": iteration, "parameters": asdict(params), "teacher": decision,
                        "prompt_file": _relative(prompt_file, run_dir.parent), "metrics": metrics})
        stages = ",".join(f"{c}={info.get('stage')}/{'ok' if info['success'] else 'NG'}"
                          for c, info in metrics["per_cube"].items())
        line = (f"[{run_name}] iteration={iteration}/{max_iterations} {task.kind} F={params.F:.2f} "
                f"grasp_dz={params.grasp_dz:.3f} cubes[{stages}] steps={metrics['steps']} "
                f"steps/s={metrics['steps_per_second']:.0f} {'PASS' if metrics['success'] else 'FAIL'} "
                f"repair={decision['reason']}")
        if not quiet:
            print(line, flush=True)
        with (run_dir / "iterations.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        if metrics["success"]:
            trajectory_file = run_dir / "trajectory.npz"
            np.savez_compressed(trajectory_file, **{k: v for k, v in trajectory.items() if k != "timing"},
                                task_kind=np.array(task.kind), layout_xy=np.array([layout.cubes[c] for c in COLORS]))
            from PIL import Image
            Image.fromarray(trajectory["final_rgb"]).save(run_dir / "final_rgb.png")
            result.update(success=True, trajectory_file=_relative(trajectory_file, run_dir.parent))
            write_json(run_dir / "summary.json", result)
            break
        if len(history) >= 2 and history[-1]["parameters"] == history[-2]["parameters"]:
            result["stopped"] = "no_progress: same parameters failed twice"
            write_json(run_dir / "summary.json", result)
            break
        write_json(run_dir / "summary.json", result)
    if video_path is not None and frames:
        import imageio.v2 as imageio
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=int(round(1 / (0.01 * video_every))), macro_block_size=8)
        result["video_file"] = _relative(video_path, run_dir.parent)
        write_json(run_dir / "summary.json", result)
    result["summary_file"] = _relative(run_dir / "summary.json", run_dir.parent)
    return result


COLLECT_INSTRUCTIONS = (
    "赤・緑・青のキューブを同じ色のトレイに仕分けて",
    "青いキューブを赤いキューブの上に積んで",
    "緑のキューブを青いキューブの上に積み重ねて",
    "赤いキューブを緑のキューブの上に重ねて",
)


def collect(sim, episodes: int, seed: int, output_dir: Path, instructions=COLLECT_INSTRUCTIONS,
            mode: str = "normal", noise_std: float = 0.0) -> dict:
    """ランダム配置で run_task を繰り返し、成功軌道（自動修復後を含む）だけをデモとして集める。
    方策学習用には mode="timing_jitter", noise_std>0（DART）で多様な状態を含める。"""
    rng = np.random.default_rng(seed)
    rows = []
    for episode in range(episodes):
        instruction = instructions[episode % len(instructions)]
        layout = sample_layout(rng)
        result = run_task(instruction, sim, layout=layout, output_dir=output_dir, seed=seed + episode, mode=mode,
                          run_name=f"collect{episode:03d}", quiet=True, noise_std=noise_std)
        rows.append({"episode": episode, "instruction": instruction, "success": result["success"],
                     "iterations": len(result["history"]), "trajectory_file": result["trajectory_file"],
                     "layout": layout.cubes})
        print(f"collect {episode + 1}/{episodes}: {'PASS' if result['success'] else 'FAIL'} "
              f"iters={len(result['history'])} {instruction}", flush=True)
    summary = {"episodes": episodes, "seed": seed, "mode": mode, "noise_std": noise_std, "successes": sum(r["success"] for r in rows), "rows": rows}
    write_json(Path(output_dir) / "collect_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("instruction")
    run.add_argument("--mode", choices=MODES, default="normal")
    run.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--random-layout", action="store_true", help="seedからcube配置をランダム生成")
    run.add_argument("--video", type=Path, help="全試行を連結したmp4を書き出す")
    run.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    col = sub.add_parser("collect")
    col.add_argument("--episodes", type=int, default=20)
    col.add_argument("--seed", type=int, default=100)
    col.add_argument("--output-dir", type=Path, default=ROOT / "data" / "demos")
    col.add_argument("--mode", choices=MODES, default="normal")
    col.add_argument("--noise-std", type=float, default=0.0, help="DART型ノイズの標準偏差[m]")
    args = parser.parse_args()
    if args.command == "run":
        parse_task(args.instruction)  # 重いscene build前に不正指示を拒否
    from sim import MultiCubeSim
    sim = MultiCubeSim(seed=getattr(args, "seed", 0))
    try:
        if args.command == "run":
            layout = sample_layout(np.random.default_rng(args.seed)) if args.random_layout else None
            result = run_task(args.instruction, sim, layout=layout, mode=args.mode, seed=args.seed,
                              max_iterations=args.max_iterations, output_dir=args.output_dir, video_path=args.video)
            print(f"{'PASS' if result['success'] else 'FAIL'} summary={result['summary_file']}")
            return 0 if result["success"] else 1
        summary = collect(sim, args.episodes, args.seed, args.output_dir, mode=args.mode, noise_std=args.noise_std)
        print(f"collected {summary['successes']}/{summary['episodes']}")
        return 0
    finally:
        sim.close()


if __name__ == "__main__":
    sys.exit(main())
