"""改善版の実物理E2E検証（Genesis CPU）。結果を results/e2e.json と VALIDATION_IMPROVED.md に書き出す。

実行: python validate_improved.py [--videos]
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np

from multi_pipeline import run_task
from pipeline import write_json
from tasks import Layout, sample_layout

ROOT = Path(__file__).resolve().parent

CASES = [
    # name, instruction, mode, layout_seed(None=既定配置), expected_min_iterations, video
    ("sort_normal", "赤・緑・青のキューブを同じ色のトレイに仕分けて", "normal", None, 1, True),
    ("stack_normal", "青いキューブを赤いキューブの上に積んで", "normal", None, 1, True),
    ("lift_green_25cm", "緑のキューブを高さ25cmまで持ち上げて", "normal", None, 1, False),
    ("sort_auto_repair", "色ごとに仕分けして", "bad_start", None, 2, True),
    ("stack_auto_repair", "青いキューブを赤いキューブの上に積んで", "place_bias", None, 2, True),
    ("stack_bad_start_random", "緑のキューブを青いキューブの上に積み重ねて", "bad_start", 11, 2, False),
    ("sort_perturbed_random", "色ごとに仕分けして", "perturbed", 12, 2, False),
    ("sort_careful_random", "ゆっくり丁寧に色ごとに仕分けて", "normal", 13, 1, False),
    ("lift_original_task", "赤いブロックを掴んで高さ30cmまで持ち上げて", "normal", None, 1, False),
]


def write_report(path: Path, suite: dict):
    lines = [
        "# 改善版 E2E 検証レポート（Genesis CPU 実物理）", "",
        f"実行日時（日本時間）: {suite['timestamp']}。総合結果: **{'PASS' if suite['success'] else 'FAIL'}**"
        f"（{sum(c['passed'] for c in suite['cases'])}/{len(suite['cases'])} ケース + 上限到達テスト "
        f"{'PASS' if suite['exhaustion_passed'] else 'FAIL'}）。",
        f"環境: {suite['platform']} / Python {suite['python']} / Genesis {suite['versions']['genesis-world']} / "
        f"Torch {suite['versions']['torch']} / NumPy {suite['versions']['numpy']}。gs.cpu, dt=0.01s, substeps=4, seed=0。",
        f"scene build: {suite['build_seconds']:.1f}s。総実行時間: {suite['elapsed_seconds']:.1f}s。", "",
        "成功条件（すべて最後の40ステップ=0.4秒の全フレームで判定）:",
        "- 仕分け: 各cube中心がトレイ中心から3cm以内、トレイ上の高さ±8mm、hand原点から0.14m以上離れている、速度<0.05m/s。対象外cubeの移動≤1.5cm。",
        "- 積み重ね: 上cubeと土台cubeの水平ずれ≤12mm、高さ差0.04±0.006m、土台の移動≤1.5cm、手が離れている、静止。",
        "- 持ち上げ: 目標高さ±2.5cm、hand-cube距離≤0.14m。", "",
        "| ケース | 指示 | モード | 配置 | 試行数 | 結果 | 修復ログ（2試行目以降） |", "|---|---|---|---|---:|---|---|",
    ]
    for case in suite["cases"]:
        r = case.get("result")
        repairs = " / ".join(h["teacher"]["reason"] for h in r["history"][1:]) if r else case.get("error", "")[:80]
        lines.append(f"| {case['name']} | {case['instruction']} | {case['mode']} | "
                     f"{'既定' if case['layout_seed'] is None else 'random seed=' + str(case['layout_seed'])} | "
                     f"{len(r['history']) if r else '-'} | {'PASS' if case['passed'] else 'FAIL'} | {repairs or '-'} |")
    lines += ["", "## 試行ごとの実測値", "", "| ケース/試行 | cube | 段階 | 判定 | 最終xy誤差[mm] | 高さ誤差[mm] | 積み重ねずれ[mm] | steps | physics steps/s | e2e steps/s |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for case in suite["cases"]:
        for h in (case.get("result") or {}).get("history", []):
            m = h["metrics"]
            for color, info in m["per_cube"].items():
                stack = f"{info['stack_xy_error'] * 1000:.1f}" if "stack_xy_error" in info else "-"
                lines.append(f"| {case['name']}/{h['iteration']} | {color} | {info.get('stage')} | "
                             f"{'ok' if info['success'] else 'NG'} | {info['final_xy_error'] * 1000:.1f} | "
                             f"{info['final_height_error'] * 1000:+.1f} | {stack} | {m['steps']} | "
                             f"{m['steps_per_second']:.0f} | {m['steps'] / m['rollout_seconds']:.0f} |")
    lines += ["", "## 実行ログ", ""]
    for case in suite["cases"]:
        r = case.get("result")
        if r:
            log = (ROOT / "runs" / r["summary_file"]).parent / "iterations.log"
            lines += [f"### {case['name']}", "", "```text", log.read_text(encoding="utf-8").rstrip(), "```", ""]
        if case.get("error"):
            lines += [f"### {case['name']}（例外）", "", "```text", case["error"], "```", ""]
    lines += ["## 上限到達テスト", "",
              "bad_start を max_iterations=1 で実行し、FAIL・1試行のみ・trajectory_file=null・NPZ未生成を確認: "
              f"**{'PASS' if suite['exhaustion_passed'] else 'FAIL'}**", "",
              "## 動画", ""]
    for case in suite["cases"]:
        r = case.get("result")
        if r and r.get("video_file"):
            lines.append(f"- {case['name']}: `runs/{r['video_file']}`")
    lines += ["", "e2e steps/s は IK・状態取得・最終RGB描画を含む rollout 全体の速度（動画描画ありのケースは遅くなる）。"
              "元版（毎ステップRGB描画）の e2e は約24 steps/s。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", action="store_true", help="主要ケースのmp4を書き出す（遅くなる）")
    parser.add_argument("--report", type=Path, default=ROOT / "VALIDATION_IMPROVED.md")
    args = parser.parse_args()
    start = time.perf_counter()
    from sim import MultiCubeSim
    sim = MultiCubeSim(seed=0)
    out = ROOT / "runs"
    exhaustion_passed = False
    cases = []
    try:
        for name, instruction, mode, layout_seed, min_iters, video in CASES:
            case = {"name": name, "instruction": instruction, "mode": mode, "layout_seed": layout_seed, "passed": False}
            cases.append(case)
            try:
                layout = Layout() if layout_seed is None else sample_layout(np.random.default_rng(layout_seed))
                video_path = out / "videos" / f"{name}.mp4" if (args.videos and video) else None
                result = run_task(instruction, sim, layout=layout, mode=mode, output_dir=out, run_name=name,
                                  video_path=video_path)
                case["result"] = result
                assert result["success"], "6試行以内に成功しなかった"
                assert len(result["history"]) >= min_iters, "初期誤差を注入したのに1回目で成功した"
                if min_iters > 1:
                    assert not result["history"][0]["metrics"]["success"]
                with np.load(out / result["trajectory_file"], allow_pickle=False) as data:
                    assert data["cube_pos"].shape == (result["history"][-1]["metrics"]["steps"], 3, 3)
                    assert np.isfinite(data["cube_pos"]).all() and np.isfinite(data["qpos"]).all()
                case["passed"] = True
            except Exception:
                case["error"] = traceback.format_exc()
            print(f"TEST {name}: {'PASS' if case['passed'] else 'FAIL'}", flush=True)
        try:
            r = run_task("色ごとに仕分けして", sim, mode="bad_start", max_iterations=1, output_dir=out,
                         run_name="expected_failure")
            folder = (out / r["summary_file"]).parent
            assert not r["success"] and len(r["history"]) == 1 and r["trajectory_file"] is None
            assert (folder / "summary.json").is_file() and not list(folder.glob("*.npz"))
            exhaustion_passed = True
        except Exception:
            print(traceback.format_exc(), flush=True)
        print(f"TEST expected_failure: {'PASS' if exhaustion_passed else 'FAIL'}", flush=True)
    finally:
        sim.close()
    import importlib.metadata
    suite = {"timestamp": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
             "success": all(c["passed"] for c in cases) and exhaustion_passed,
             "cases": cases, "exhaustion_passed": exhaustion_passed, "build_seconds": sim.build_seconds,
             "elapsed_seconds": time.perf_counter() - start, "platform": platform.platform(),
             "python": platform.python_version(),
             "versions": {n: importlib.metadata.version(n) for n in ("genesis-world", "torch", "numpy")}}
    (ROOT / "results").mkdir(exist_ok=True)
    write_json(ROOT / "results" / "e2e.json", suite)
    write_report(args.report, suite)
    print(f"{'ALL E2E PASS' if suite['success'] else 'E2E FAILED'} report={args.report.name}", flush=True)
    return 0 if suite["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
