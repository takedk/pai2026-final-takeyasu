"""日本語指示 -> Teacher -> Genesis CPU -> Auto-Repair -> 成功デモ収集。

実行例: .venv/bin/python pipeline.py '赤いブロックを掴んで高さ30cmまで持ち上げて'
外部サービス不要。Teacher(proposer=...) に任意の LLM 呼び出しを注入可能。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Callable
import unicodedata
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parent
DT = 0.01
CUBE_START = (0.65, 0.0, 0.02)
HAND_OFFSET = 0.11
HEIGHT_TOLERANCE = 0.025
DISTANCE_LIMIT = 0.14
HOLD_WINDOW = 40
PHASES = ("approach", "descend", "grasp", "lift", "hold")


@dataclass(frozen=True)
class Task:
    instruction: str
    target_cube_z: float = 0.30
    careful: bool = False
    object_name: str = "red_cube"


NEGATION = r"ないで|しないこと|禁止|やめて|不要"
UNIT_PATTERN = r"([+-]?\d+(?:\.\d+)?)\s*(センチメートル|センチ|ミリメートル|ミリ|メートル|cm|mm|m)(?![a-z])"


def parse_height(text: str, default: float = 0.30) -> float:
    """「高さ30cm」等をキューブ中心の絶対高さ[m]へ。複数指定・漢数字・単位なしは拒否。"""
    heights = re.findall(UNIT_PATTERN, text, re.I)
    if len(heights) > 1:
        raise ValueError("目標高さは1つだけ指定してください。")
    if not heights:
        if re.search(r"高さ\s*\d|\d\s*(?:センチ|cm)|[一二三四五六七八九十百]+(?:センチ|メートル)|高さ[一二三四五六七八九十百]", text):
            raise ValueError("高さはアラビア数字と cm / m / mm で指定してください。")
        return default
    value, unit = heights[0]
    unit = unit.lower()
    scale = 0.01 if unit in ("cm", "センチ", "センチメートル") else 0.001 if unit in ("mm", "ミリ", "ミリメートル") else 1.0
    return float(value) * scale


def parse_instruction(instruction: str) -> Task:
    text = unicodedata.normalize("NFKC", instruction).strip()
    if re.search(NEGATION, text):
        # 「持ち上げないで」を持ち上げ指示として実行しない。
        raise ValueError("否定・禁止を含む指示は実行しません。")
    if not text or not re.search(r"持ち上|持上|リフト|上げ", text):
        raise ValueError("キューブを把持して持ち上げる日本語指示を指定してください。")
    if not re.search(r"ブロック|キューブ|立方体", text):
        raise ValueError("対象はブロック / キューブ / 立方体で指定してください。")
    if re.search(r"青|緑|黄色|複数|二つ|2個|置いて|投げ", text):
        raise ValueError("この最小環境の対象は赤いキューブ1個の把持・リフトです。")
    target = parse_height(text)
    if not 0.15 <= target <= 0.40:
        raise ValueError("対応するキューブ中心の絶対高さは0.15～0.40mです。")
    return Task(text, target, bool(re.search(r"慎重|ゆっくり|丁寧", text)))


@dataclass(frozen=True)
class Parameters:
    x: float = 0.65
    y: float = 0.0
    approach_z: float = 0.25
    H: float = 0.13  # hand 原点の降下先の床からの絶対高さ [m]
    F: float = -5.0  # 各指へ送る閉方向の力 [N]; 負値が閉方向
    lift_z: float = 0.41  # hand 原点の目標高さ [m]
    approach_steps: int = 150
    descend_steps: int = 100
    grasp_steps: int = 100
    lift_steps: int = 200
    hold_steps: int = 60

    def validate(self) -> Parameters:
        bounds = {
            "x": (0.50, 0.72), "y": (-0.10, 0.10),
            "approach_z": (0.20, 0.40), "H": (0.105, 0.24),
            "F": (-20.0, -0.01), "lift_z": (0.25, 0.55),
        }
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} must be a finite number in [{low}, {high}]")
        for phase in PHASES:
            value = getattr(self, f"{phase}_steps")
            if type(value) is not int or not 60 <= value <= 500:
                raise ValueError(f"{phase}_steps must be an integer in [60, 500]")
        if self.approach_z < self.H + 0.02 or self.lift_z < self.H + 0.05:
            raise ValueError("approach_z / lift_z must be above grasp height H")
        return self


TEACHER_PROMPT = """あなたはPandaの把持・リフト用Waypoint Generatorです。
ユーザー指示: {instruction}
目標はキューブ中心の床からの絶対高さ {target_cube_z} m。
赤い4cm立方体の初期中心は [0.65, 0, 0.02] m。hand原点は指先より約0.11m上。
Hはhand原点の降下高さ[m]、Fは各指の閉方向の力[N]（負）。
実行フェーズ: approach -> descend -> grasp -> lift -> hold。
ルールベース推奨パラメータ: {recommendation}
これまでの実際のシミュレーション結果: {feedback}
空振りならHとxy、滑りならFと移動時間、保持できて高さ不足ならlift_zを修正。
出力は推奨と同じキーを持つJSONオブジェクトのみ。コードは生成しない。
制約: x=[0.50,0.72], y=[-0.10,0.10], approach_z=[0.20,0.40],
H=[0.105,0.24], F=[-20,-0.01], lift_z=[0.25,0.55]。
各フェーズのstepsは整数60～500。approach_z>=H+0.02, lift_z>=H+0.05。
"""


def extract_json(text: str) -> dict:
    """LLM出力から最初のJSONオブジェクトを取り出す。```json フェンスや前後の説明文を許容。"""
    if isinstance(text, dict):
        return text
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("LLM出力にJSONオブジェクトが見つかりません")


def merge_proposal(base, proposal: dict):
    """部分的なLLM提案をルール推奨へ上書きする。未知キーは拒否（既定値への暗黙リセットを防ぐ）。"""
    known = set(asdict(base))
    unknown = set(proposal) - known
    if unknown:
        raise ValueError(f"unknown keys: {sorted(unknown)}")
    return replace(base, **proposal)


class Teacher:
    """LLMプロンプト + 検証済み提案、または決定論的ルールによる修復。"""

    def __init__(self, proposer: Callable[[str], str | dict] | None = None):
        self.proposer = proposer

    def propose(self, task: Task, history: list[dict], mode: str, seed: int):
        reason = "initial_rule"
        if not history:
            params = Parameters(lift_z=task.target_cube_z + HAND_OFFSET)
            if task.careful:
                params = replace(params, approach_steps=200, descend_steps=150, lift_steps=300)
            if mode == "bad_start":
                params = replace(params, H=0.21, F=-0.05)
            elif mode == "weak_force":
                params = replace(params, F=-0.05)
            elif mode == "perturbed":
                rng = np.random.default_rng(seed)
                params = replace(params, x=float(rng.uniform(0.61, 0.69)),
                                 y=float(rng.uniform(-0.025, 0.025)),
                                 H=float(rng.uniform(0.15, 0.21)), F=-0.1)
            elif mode != "normal":
                raise ValueError(f"Unknown initial mode: {mode}")
        else:
            last = history[-1]
            params = Parameters(**last["parameters"])
            metrics = last["metrics"]
            grasp = metrics["phase_end"]["grasp"]
            # 診断は実測の閉じた位置・cube位置・持ち上がり量から行う。
            xy_error = np.linalg.norm(np.array(grasp["ee_pos"][:2]) - grasp["cube_pos"][:2])
            gap = grasp["ee_pos"][2] - grasp["cube_pos"][2]
            if xy_error > 0.018 or gap > 0.145 or gap < 0.085:
                reason = "missed_cube: correct_xy_and_H"
                # 既知の初期位置ではなく、把持直前に実測したcube位置へ寄せる。
                cube = grasp["cube_pos"]
                params = replace(params, x=float(np.clip(cube[0], 0.50, 0.72)), y=float(np.clip(cube[1], -0.10, 0.10)),
                                 H=float(np.clip(cube[2] + HAND_OFFSET, 0.105, 0.24)))
            elif not metrics["grasp_success"]:
                reason = "slip_or_weak_grasp: increase_force_and_duration"
                params = replace(params, F=-min(20.0, max(5.0, abs(params.F) * 2)),
                                 H=CUBE_START[2] + HAND_OFFSET,
                                 grasp_steps=min(500, params.grasp_steps + 50),
                                 lift_steps=min(500, params.lift_steps + 50))
            else:
                reason = "target_height_error: compensate_lift_z"
                correction = task.target_cube_z - metrics["cube_z"]
                params = replace(params, lift_z=float(np.clip(params.lift_z + correction, 0.25, 0.55)),
                                 lift_steps=min(500, params.lift_steps + 50))
        params.validate()
        prompt = TEACHER_PROMPT.format(
            instruction=task.instruction, target_cube_z=task.target_cube_z,
            recommendation=json.dumps(asdict(params), ensure_ascii=False),
            feedback=json.dumps(history, ensure_ascii=False),
        )
        decision = {"source": "offline_rules", "reason": reason, "prompt": prompt}
        if self.proposer is not None:
            try:
                proposal = extract_json(self.proposer(prompt))
                params = merge_proposal(params, proposal).validate()
                decision["source"] = "llm_proposal"
            except Exception as exc:
                # LLMがオフライン・タイムアウト・不正JSONでもルール案を使用。
                decision["fallback_error"] = f"{type(exc).__name__}: {exc}"
        return params, decision


def array(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def evaluate(trajectory: dict[str, np.ndarray], target_z: float) -> dict:
    """最後の0.4秒すべてで保持・高さを確認し、瞬間的な跳ね上がりを除外。"""
    cubes, hands = trajectory["cube_pos"], trajectory["ee_pos"]
    distances = np.linalg.norm(cubes - hands, axis=1)
    tail = slice(-HOLD_WINDOW, None)
    grasp_success = bool(np.all(cubes[tail, 2] >= 0.15) and np.all(distances[tail] <= DISTANCE_LIMIT))
    height_success = bool(np.all(np.abs(cubes[tail, 2] - target_z) <= HEIGHT_TOLERANCE))
    return {
        "success": grasp_success and height_success,
        "grasp_success": grasp_success, "target_height_success": height_success,
        "cube_z": float(cubes[-1, 2]), "ee_z": float(hands[-1, 2]),
        "ee_cube_distance": float(distances[-1]), "target_cube_z": target_z,
        "height_error": float(cubes[-1, 2] - target_z),
        "max_cube_z": float(cubes[:, 2].max()),
        "hold_min_cube_z": float(cubes[tail, 2].min()),
        "hold_max_distance": float(distances[tail].max()),
        "hold_max_height_error": float(np.abs(cubes[tail, 2] - target_z).max()),
    }


class GenesisEvaluator:
    """1つのCPU sceneを再利用し、各試行前に物理状態をリセットする。"""

    def __init__(self, seed: int = 0):
        cache = Path(tempfile.gettempdir()) / "pai_genesis_cache"
        os.environ.setdefault("XDG_CACHE_HOME", str(cache))
        os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        os.environ.setdefault("LP_NUM_THREADS", "1")
        os.environ.setdefault("QD_NUM_THREADS", "1")
        import genesis as gs
        import torch

        torch.set_num_threads(1)
        self.gs = gs
        self.seed = seed
        print("Building Genesis CPU scene (first compilation may take several minutes)...", flush=True)
        start = time.perf_counter()
        gs.init(backend=gs.cpu, seed=seed, logging_level="error")
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, substeps=4), show_viewer=False)
        self.scene.add_entity(gs.morphs.Plane())
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=CUBE_START),
            surface=gs.surfaces.Default(color=(0.85, 0.04, 0.03)),
        )
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        self.camera = self.scene.add_camera(res=(96, 72), pos=(1.8, -1.0, 1.2),
                                            lookat=(0.5, 0, 0.2), fov=40, GUI=False)
        self.scene.build()
        self.robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        limits = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])
        self.robot.set_dofs_force_range(-limits, limits)
        self.ee = self.robot.get_link("hand")
        self.build_seconds = time.perf_counter() - start
        print(f"Scene ready: {self.build_seconds:.2f}s", flush=True)

    def close(self):
        self.gs.destroy()

    def rollout(self, params: Parameters, task: Task):
        params.validate()
        self.scene.reset()
        # resetは物理状態を戻す。IK内の乱数も試行ごとに同じseedへ戻す。
        self.gs.set_random_seed(self.seed)
        initial_qpos = array(self.robot.get_qpos())
        initial_cube = array(self.cube.get_pos())
        rows = {key: [] for key in ("qpos", "target_action", "action_mode", "ee_pos", "cube_pos", "ee_target", "phase", "rgb_mean", "rgb_std")}
        phase_end = {}
        physics_seconds = 0.0
        render_seconds = 0.0
        started = time.perf_counter()
        for phase_idx, phase in enumerate(PHASES):
            z = params.approach_z if phase == "approach" else params.H if phase in ("descend", "grasp") else params.lift_z
            target = np.array([params.x, params.y, z])
            q = array(self.robot.inverse_kinematics(link=self.ee, pos=target, quat=np.array([0, 1, 0, 0])))
            if not np.isfinite(q).all():
                raise RuntimeError("IK returned non-finite joint positions")
            self.robot.control_dofs_position(q[:7], np.arange(7))
            force_mode = phase in ("grasp", "lift", "hold")
            finger_action = np.full(2, params.F if force_mode else 0.04)
            if force_mode:
                self.robot.control_dofs_force(finger_action, np.arange(7, 9))
            else:
                self.robot.control_dofs_position(finger_action, np.arange(7, 9))
            action = np.concatenate((q[:7], finger_action))
            for _ in range(getattr(params, f"{phase}_steps")):
                step_start = time.perf_counter()
                self.scene.step()
                physics_seconds += time.perf_counter() - step_start
                rows["qpos"].append(array(self.robot.get_qpos()))
                rows["target_action"].append(action.copy())
                rows["action_mode"].append(int(force_mode))
                rows["ee_target"].append(target.copy())
                rows["ee_pos"].append(array(self.ee.get_pos()))
                rows["cube_pos"].append(array(self.cube.get_pos()))
                rows["phase"].append(phase_idx)
                render_start = time.perf_counter()
                rgb = np.asarray(self.camera.render()[0])
                if rgb.shape != (72, 96, 3) or not np.isfinite(rgb).all():
                    raise RuntimeError(f"Invalid real camera RGB: {rgb.shape}")
                rows["rgb_mean"].append(rgb.mean(axis=(0, 1)))
                rows["rgb_std"].append(rgb.std(axis=(0, 1)))
                render_seconds += time.perf_counter() - render_start
            phase_end[phase] = {"cube_pos": rows["cube_pos"][-1].tolist(),
                                "ee_pos": rows["ee_pos"][-1].tolist(),
                                "finger_qpos": rows["qpos"][-1][-2:].tolist()}
        rollout_seconds = time.perf_counter() - started
        trajectory = {key: np.asarray(value) for key, value in rows.items()}
        count = len(trajectory["qpos"])
        trajectory.update(time=np.arange(1, count + 1) * DT, initial_qpos=initial_qpos,
                          initial_cube_pos=initial_cube, final_rgb=rgb)
        metrics = evaluate(trajectory, task.target_cube_z)
        metrics.update(steps=count, physics_seconds=physics_seconds, render_seconds=render_seconds,
                       rollout_seconds=rollout_seconds, steps_per_second=count / physics_seconds,
                       end_to_end_steps_per_second=count / rollout_seconds, phase_end=phase_end)
        return metrics, trajectory


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_pipeline(instruction: str, *, evaluator: GenesisEvaluator | None = None,
                 output_dir: Path = ROOT / "trajectories", mode: str = "normal",
                 max_iterations: int = 6, seed: int = 0, teacher: Teacher | None = None,
                 run_name: str = "demo") -> dict:
    if type(max_iterations) is not int or not 1 <= max_iterations <= 6:
        raise ValueError("max_iterations must be an integer in [1, 6]")
    task = parse_instruction(instruction)
    teacher = teacher or Teacher()
    # 不正な設定は、重いscene build前に拒否する。
    params, decision = teacher.propose(task, [], mode, seed)
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", run_name)
    run_id = f"{safe_name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    owned = evaluator is None
    history = []
    result = {
        "schema_version": 1, "run_id": run_id, "task": asdict(task), "mode": mode,
        "seed": seed, "max_iterations": max_iterations, "success": False,
        "backend": "gs.cpu", "dt": DT, "substeps": 4,
        "versions": {name: importlib.metadata.version(name) for name in ("genesis-world", "numpy", "torch")},
        "python": sys.version, "thresholds": {"min_cube_z": 0.15, "distance_limit": DISTANCE_LIMIT,
                                              "target_height_tolerance": HEIGHT_TOLERANCE, "hold_window": HOLD_WINDOW},
        "history": history, "trajectory_file": None, "summary_file": str(run_dir / "summary.json"),
        "trajectory_schema": {
            "qpos": "(T,9): post-step arm radians (7) and finger meters (2)",
            "target_action": "(T,9): command applied before this step; first 7=position radians, last 2 depend on action_mode",
            "action_mode": "(T,): 0=position meters for fingers; 1=force newtons for fingers",
            "ee_pos/cube_pos/ee_target": "(T,3): world coordinates [m]; EE=hand link origin",
            "time": "(T,): post-step simulation time [s]",
            "phase": "(T,): 0=approach, 1=descend, 2=grasp, 3=lift, 4=hold",
            "rgb_mean/rgb_std": "(T,3): actual camera RGB channel mean/std, 0..255, rendered every step",
            "final_rgb": "(72,96,3): final actual RGB image", "initial_qpos": "(9,): reset state",
            "initial_cube_pos": "(3,): reset state [m]",
        },
    }
    try:
        evaluator = evaluator or GenesisEvaluator(seed)
        if evaluator.seed != seed:
            raise ValueError("evaluator.seed and pipeline seed must agree")
        result["scene_build_seconds"] = evaluator.build_seconds
        for iteration in range(1, max_iterations + 1):
            if iteration > 1:
                params, decision = teacher.propose(task, history, mode, seed)
            prompt_file = run_dir / f"teacher_{iteration:02d}.txt"
            prompt_file.write_text(decision.pop("prompt"), encoding="utf-8")
            metrics, trajectory = evaluator.rollout(params, task)
            history.append({"iteration": iteration, "parameters": asdict(params), "teacher": decision,
                            "prompt_file": str(prompt_file), "metrics": metrics,
                            "initial_cube_pos": trajectory["initial_cube_pos"].tolist()})
            line = (f"[{run_name}] iteration={iteration}/{max_iterations} H={params.H:.3f} F={params.F:.2f} "
                    f"lift_z={params.lift_z:.3f} cube_z={metrics['cube_z']:.4f} ee_z={metrics['ee_z']:.4f} "
                    f"distance={metrics['ee_cube_distance']:.4f} steps/s={metrics['steps_per_second']:.1f} "
                    f"e2e_steps/s={metrics['end_to_end_steps_per_second']:.1f} "
                    f"{'PASS' if metrics['success'] else 'FAIL'} repair={decision['reason']}")
            print(line, flush=True)
            with (run_dir / "iterations.log").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
            if metrics["success"]:
                trajectory_file = run_dir / "trajectory.npz"
                np.savez_compressed(trajectory_file, **trajectory)
                from PIL import Image
                Image.fromarray(trajectory["final_rgb"]).save(run_dir / "final_rgb.png")
                result.update(success=True, trajectory_file=str(trajectory_file))
            write_json(run_dir / "summary.json", result)
            if result["success"]:
                break
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        write_json(run_dir / "summary.json", result)
        raise
    finally:
        if owned and evaluator is not None:
            evaluator.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction")
    parser.add_argument("--mode", choices=("normal", "bad_start", "weak_force", "perturbed"), default="normal")
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "trajectories")
    args = parser.parse_args()
    result = run_pipeline(args.instruction, mode=args.mode, max_iterations=args.max_iterations,
                          seed=args.seed, output_dir=args.output_dir)
    print(f"{'PASS' if result['success'] else 'FAIL'} summary={result['summary_file']}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
