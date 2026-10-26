"""Genesis CPU 上の3色cube＋3トレイscene。1回buildし、試行ごとにresetしてcube配置だけ差し替える。"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
from typing import Callable

import numpy as np

from tasks import (COLORS, CUBE_HALF, RGB, TRAY_SIZE, TRAYS, DEFAULT_CUBES, Layout, Segment,
                   interpolate_targets)

DT = 0.01
SUBSTEPS = 4
QUAT_DOWN = np.array([0, 1, 0, 0])
OPEN_WIDTH = 0.04


def array(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy() if hasattr(tensor, "detach") else np.array(tensor, copy=True)


def dart_noise(segments: list[Segment], std: float, seed: int, corr: float = 0.97) -> np.ndarray:
    """移動区間の前半だけに入るOUノイズ(T,3)。区間の90%時点で0へ減衰させ、把持・解放時の位置は正確に保つ。"""
    rng = np.random.default_rng(seed)
    rows, state = [], np.zeros(3)
    for seg in segments:
        moving = seg.name in ("approach", "descend", "lift", "carry", "lower", "retreat")
        progress = np.arange(1, seg.steps + 1) / seg.steps
        envelope = np.clip((0.9 - progress) / 0.3, 0, 1) if moving else np.zeros(seg.steps)
        for k in range(seg.steps):
            state = corr * state + np.sqrt(1 - corr ** 2) * rng.normal(0, std, 3)
            rows.append(state * envelope[k] * np.array([1, 1, 0.5]))
    return np.asarray(rows)


class MultiCubeSim:
    """rollout(segments) は実測の状態列を返すだけ。成功判定は tasks.evaluate_task に分離。"""

    def __init__(self, seed: int = 0, video_res: tuple[int, int] = (640, 360)):
        cache = Path(tempfile.gettempdir()) / "pai_genesis_cache"
        os.environ.setdefault("XDG_CACHE_HOME", str(cache))
        os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        import genesis as gs
        import torch

        torch.set_num_threads(1)
        self.gs, self.seed = gs, seed
        print("Building Genesis CPU multi-cube scene...", flush=True)
        start = time.perf_counter()
        gs.init(backend=gs.cpu, seed=seed, logging_level="error")
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS), show_viewer=False)
        self.scene.add_entity(gs.morphs.Plane())
        for color in COLORS:
            tint = tuple(0.55 + 0.45 * v for v in RGB[color])
            self.scene.add_entity(gs.morphs.Box(size=TRAY_SIZE, pos=(*TRAYS[color], TRAY_SIZE[2] / 2), fixed=True),
                                  surface=gs.surfaces.Default(color=tint))
        self.cubes = {c: self.scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(*DEFAULT_CUBES[c], CUBE_HALF)),
                                               surface=gs.surfaces.Default(color=RGB[c])) for c in COLORS}
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        self.camera = self.scene.add_camera(res=(96, 72), pos=(1.6, -0.9, 1.1), lookat=(0.5, 0, 0.1), fov=40, GUI=False)
        self.video_camera = self.scene.add_camera(res=video_res, pos=(1.10, -0.55, 0.55), lookat=(0.52, 0.0, 0.08),
                                                  fov=42, GUI=False)
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

    def reset(self, layout: Layout):
        self.scene.reset()
        self.gs.set_random_seed(self.seed)
        for color, (x, y) in layout.validate().cubes.items():
            self.cubes[color].set_pos(np.array([x, y, CUBE_HALF]))
        self._q = None

    def observe(self) -> dict:
        return {"qpos": array(self.robot.get_qpos()), "ee_pos": array(self.ee.get_pos()),
                "cube_pos": np.stack([array(self.cubes[c].get_pos()) for c in COLORS])}

    def apply(self, hand_target: np.ndarray, grip: int, force: float, solve_ik: bool = True) -> np.ndarray:
        """hand目標(xyz, 下向き姿勢)をIKで関節目標へ。前回解を初期値にして連続性を保つ。"""
        if solve_ik or self._q is None:
            q = array(self.robot.inverse_kinematics(link=self.ee, pos=np.asarray(hand_target, dtype=float),
                                                    quat=QUAT_DOWN, init_qpos=self._q))
            if not np.isfinite(q).all():
                raise RuntimeError("IK returned non-finite joint positions")
            self._q = q
            self.robot.control_dofs_position(q[:7], np.arange(7))
        if grip:
            self.robot.control_dofs_force(np.full(2, force), np.arange(7, 9))
        else:
            self.robot.control_dofs_position(np.full(2, OPEN_WIDTH), np.arange(7, 9))
        return self._q[:7].copy()

    def rollout(self, layout: Layout, segments: list[Segment], force: float, *,
                video_every: int = 0, on_frame: Callable[[np.ndarray], None] | None = None,
                noise_std: float = 0.0, noise_seed: int = 0) -> dict:
        """noise_std>0 で DART 型のデータ収集: 実際の指令にだけ滑らかなノイズを足し、記録する ee_target は
        クリーンな目標のまま。学習ラベル(ee_target - ee_pos)が「ずれた状態からの復帰」を含むようになる。"""
        self.reset(layout)
        start_obs = self.observe()
        targets = interpolate_targets(segments, start_obs["ee_pos"])
        seg_index = np.concatenate([np.full(s.steps, k) for k, s in enumerate(segments)])
        noise = dart_noise(segments, noise_std, noise_seed) if noise_std > 0 else np.zeros_like(targets)
        rows = {k: [] for k in ("qpos", "ee_pos", "cube_pos", "ee_target", "q_target", "grip", "segment",
                                "cube_goal", "active_cube", "release")}
        physics_seconds, started = 0.0, time.perf_counter()
        previous_target = None
        for t, target in enumerate(targets):
            seg = segments[seg_index[t]]
            # 目標が静止している区間ではIKを解き直さない（計算量削減）。
            command = target + noise[t]
            moving = previous_target is None or not np.allclose(command, previous_target)
            q_target = self.apply(command, seg.grip, force, solve_ik=moving)
            previous_target = command
            step_start = time.perf_counter()
            self.scene.step()
            physics_seconds += time.perf_counter() - step_start
            obs = self.observe()
            for key in ("qpos", "ee_pos", "cube_pos"):
                rows[key].append(obs[key])
            rows["ee_target"].append(target)
            rows["q_target"].append(q_target)
            rows["grip"].append(seg.grip)
            rows["segment"].append(seg_index[t])
            rows["cube_goal"].append(seg.cube_goal)
            rows["active_cube"].append(COLORS.index(seg.cube))
            rows["release"].append(int(seg.release))
            if video_every and t % video_every == 0 and on_frame is not None:
                on_frame(np.asarray(self.video_camera.render()[0]))
        trajectory = {k: np.asarray(v) for k, v in rows.items()}
        rgb = np.asarray(self.camera.render()[0])
        count = len(targets)
        trajectory.update(
            time=np.arange(1, count + 1) * DT, initial_qpos=start_obs["qpos"], initial_cube_pos=start_obs["cube_pos"],
            final_rgb=rgb.astype(np.uint8), segment_names=np.array([s.name for s in segments]),
            segment_cubes=np.array([s.cube for s in segments]), colors=np.array(COLORS),
            force=np.array(force, dtype=float))
        rollout_seconds = time.perf_counter() - started
        trajectory["timing"] = np.array([physics_seconds, rollout_seconds])
        return trajectory

    def render_video_frame(self) -> np.ndarray:
        return np.asarray(self.video_camera.render()[0])
