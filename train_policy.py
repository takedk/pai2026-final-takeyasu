"""成功デモ(.npz)から軽量な模倣学習方策（Behavior Cloning）を学習・評価する。

方策は「1個のcubeを掴んで目標位置へ置く／持ち上げる」ゴール条件付きprimitive。
  観測 15次元: hand位置(3), 指の開き幅(1), cube-hand(3), goal-hand(3), goal-cube(3), release flag(1), 位相(1)
  行動 4次元: hand目標の差分 Δxyz(3) と グリッパ閉(1, 確率; 20ステップ先読みラベル)
仕分け＝色ごとにprimitiveを順に呼ぶ、積み重ね＝goalを土台cubeの上にする、で両タスクを1つの方策で扱う。
位相 = primitive開始からの経過ステップ / 660（デモの標準所要時間）。--no-phase でアブレーション可能。

実行例:
  python train_policy.py --demos data/demos --out results/policy            # 学習＋オフライン評価
  python train_policy.py --demos data/demos --out results/policy --closed-loop 10   # Genesisで閉ループ評価
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np

from tasks import COLORS, Layout, PickPlaceParams, STEP_KEYS, TaskSpec, evaluate_task, make_waypoints, sample_layout

ROOT = Path(__file__).resolve().parent
PRIMITIVE_STEPS = sum(getattr(PickPlaceParams(), k) for k in STEP_KEYS if k != "hold_steps")  # 660
DELTA_CLIP = 0.08  # [m] 1ステップのhand目標差分の上限（学習・実行の両方で使用）
# グリッパの先読みラベル: 「kステップ後の指令」を教師にする。デモは把持姿勢で開いたまま待ってから閉じるため、
# そのままでは「開き幅0.08で静止」=開 と学習され、方策が自分から閉じ始められない（閉ループで実測した失敗）。
GRIP_LOOKAHEAD = 20


def lookahead(grip: np.ndarray, k: int = GRIP_LOOKAHEAD) -> np.ndarray:
    """primitive内でグリッパ指令をkステップ前倒しする（末尾は最後の値で埋める）。"""
    if k <= 0:
        return np.asarray(grip, dtype=np.float32)
    return np.r_[grip[k:], np.repeat(grip[-1:], min(k, len(grip)))].astype(np.float32)
FEATURE_NAMES = ("ee_x", "ee_y", "ee_z", "width", "cube_dx", "cube_dy", "cube_dz", "goal_dx", "goal_dy", "goal_dz",
                 "goal_cube_dx", "goal_cube_dy", "goal_cube_dz", "release", "phase")


def features(ee_pos, width, cube_pos, goal, release, phase, use_phase: bool = True) -> np.ndarray:
    """学習時と実行時で共有する観測ベクトル。配列は(..., 3)でバッチ可。"""
    ee_pos, cube_pos, goal = (np.asarray(v, dtype=np.float32) for v in (ee_pos, cube_pos, goal))
    parts = [ee_pos, np.asarray(width, np.float32)[..., None], cube_pos - ee_pos, goal - ee_pos, goal - cube_pos,
             np.asarray(release, np.float32)[..., None]]
    if use_phase:
        parts.append(np.clip(np.asarray(phase, np.float32), 0, 1.2)[..., None])
    return np.concatenate(parts, axis=-1)


# ---------------------------------------------------------------- データセット
def load_demo(path: Path, use_phase: bool = True, k: int = GRIP_LOOKAHEAD) -> list[dict]:
    """新形式(schema v2)のデモをprimitive（同じactive_cubeの連続区間）ごとの(obs, act)へ分割。"""
    with np.load(path, allow_pickle=False) as data:
        if "active_cube" not in data:
            return load_legacy_demo(path, use_phase, k)
        active, cubes = data["active_cube"], data["cube_pos"]
        ee, width = data["ee_pos"], data["qpos"][:, 7:9].sum(axis=1)
        target, grip, goal, release = data["ee_target"], data["grip"], data["cube_goal"], data["release"]
    chunks = []
    bounds = np.flatnonzero(np.diff(active)) + 1
    for start, stop in zip(np.r_[0, bounds], np.r_[bounds, len(active)]):
        idx = np.arange(start, stop)
        cube = cubes[idx, active[start]]
        phase = (idx - start) / PRIMITIVE_STEPS
        obs = features(ee[idx], width[idx], cube, goal[idx], release[idx], phase, use_phase)
        delta = np.clip(target[idx] - ee[idx], -DELTA_CLIP, DELTA_CLIP)
        act = np.concatenate([delta, lookahead(grip[idx], k)[:, None]], axis=1).astype(np.float32)
        chunks.append({"obs": obs, "act": act, "source": str(path), "kind": "pick_place" if release[start] else "lift"})
    return chunks


def load_legacy_demo(path: Path, use_phase: bool = True, k: int = GRIP_LOOKAHEAD) -> list[dict]:
    """元の pipeline.py の赤cube持ち上げデモ(schema v1)を同じ観測空間へ変換（hindsight goal = 最終cube位置）。"""
    with np.load(path, allow_pickle=False) as data:
        ee, cube, width = data["ee_pos"], data["cube_pos"], data["qpos"][:, 7:9].sum(axis=1)
        target, mode = data["ee_target"], data["action_mode"]
    goal = np.repeat(cube[-1:], len(cube), axis=0)
    phase = np.arange(len(cube)) / PRIMITIVE_STEPS
    obs = features(ee, width, cube, goal, np.zeros(len(cube)), phase, use_phase)
    act = np.concatenate([np.clip(target - ee, -DELTA_CLIP, DELTA_CLIP), lookahead(mode, k)[:, None]], axis=1).astype(np.float32)
    return [{"obs": obs, "act": act, "source": str(path), "kind": "lift"}]


def find_demos(folder: Path) -> list[Path]:
    return sorted(Path(folder).rglob("trajectory.npz"))


def split_by_episode(paths: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    """同じエピソードのprimitiveが学習と検証に跨らないよう、ファイル単位で分割。"""
    order = np.random.default_rng(seed).permutation(len(paths))
    n_val = max(1, int(round(len(paths) * val_fraction))) if len(paths) > 1 else 0
    val = {int(i) for i in order[:n_val]}
    return [p for i, p in enumerate(paths) if i not in val], [p for i, p in enumerate(paths) if i in val]


def stack_chunks(chunks: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([c["obs"] for c in chunks]), np.concatenate([c["act"] for c in chunks])


# ---------------------------------------------------------------- モデル
@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        return cls(x.mean(axis=0), np.maximum(x.std(axis=0), 1e-4))

    def __call__(self, x):
        return (x - self.mean) / self.std


class LinearPolicy:
    """ベースライン: 正規化観測からのリッジ回帰（グリッパはロジットではなく値を0.5で二値化）。"""

    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge

    def fit(self, obs, act):
        self.norm = Normalizer.fit(obs)
        x = np.c_[self.norm(obs), np.ones(len(obs))]
        self.w = np.linalg.solve(x.T @ x + self.ridge * np.eye(x.shape[1]), x.T @ act)
        return self

    def predict(self, obs):
        out = np.c_[self.norm(obs), np.ones(len(obs))] @ self.w
        out[:, 3] = np.clip(out[:, 3], 0, 1)
        return out


class MLPPolicy:
    """PyTorch 2層MLP。Δxyz はMSE（正規化後）、グリッパはBCEで学習。"""

    def __init__(self, hidden: int = 128, epochs: int = 30, lr: float = 2e-3, batch: int = 512, seed: int = 0):
        self.hidden, self.epochs, self.lr, self.batch, self.seed = hidden, epochs, lr, batch, seed
        self.history: list[dict] = []

    def _build(self, n_in):
        import torch
        torch.manual_seed(self.seed)
        return torch.nn.Sequential(torch.nn.Linear(n_in, self.hidden), torch.nn.ReLU(),
                                   torch.nn.Linear(self.hidden, self.hidden), torch.nn.ReLU(),
                                   torch.nn.Linear(self.hidden, 4))

    def fit(self, obs, act, val_obs=None, val_act=None):
        import torch
        self.norm = Normalizer.fit(obs)
        self.act_scale = np.maximum(np.abs(act[:, :3]).std(axis=0), 1e-4).astype(np.float32)
        self.net = self._build(obs.shape[1])
        x = torch.tensor(self.norm(obs), dtype=torch.float32)
        y = torch.tensor(np.c_[act[:, :3] / self.act_scale, act[:, 3:]], dtype=torch.float32)
        optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.epochs)
        gen = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(x), generator=gen)
            total = 0.0
            for i in range(0, len(x), self.batch):
                b = perm[i:i + self.batch]
                out = self.net(x[b])
                loss = (torch.nn.functional.mse_loss(out[:, :3], y[b, :3])
                        + torch.nn.functional.binary_cross_entropy_with_logits(out[:, 3], y[b, 3]))
                optim.zero_grad()
                loss.backward()
                optim.step()
                total += loss.item() * len(b)
            sched.step()
            row = {"epoch": epoch + 1, "train_loss": total / len(x)}
            if val_obs is not None and len(val_obs):
                row.update({f"val_{k}": v for k, v in offline_metrics(self.predict(val_obs), val_act).items()})
            self.history.append(row)
        return self

    def predict(self, obs):
        import torch
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(self.norm(np.atleast_2d(obs)), dtype=torch.float32)).numpy()
        return np.c_[out[:, :3] * self.act_scale, 1 / (1 + np.exp(-out[:, 3]))]

    def save(self, path: Path, meta: dict):
        import torch
        torch.save({"state_dict": self.net.state_dict(), "hidden": self.hidden, "mean": self.norm.mean,
                    "std": self.norm.std, "act_scale": self.act_scale, "meta": meta}, path)

    @classmethod
    def load(cls, path: Path) -> "MLPPolicy":
        import torch
        blob = torch.load(path, weights_only=False)
        policy = cls(hidden=blob["hidden"])
        policy.norm = Normalizer(blob["mean"], blob["std"])
        policy.act_scale = blob["act_scale"]
        policy.net = policy._build(len(blob["mean"]))
        policy.net.load_state_dict(blob["state_dict"])
        return policy


def offline_metrics(pred: np.ndarray, act: np.ndarray) -> dict:
    err = pred[:, :3] - act[:, :3]
    return {"delta_rmse_mm": float(np.sqrt((err ** 2).mean()) * 1000),
            "delta_mae_mm": float(np.abs(err).mean() * 1000),
            "grip_accuracy": float(((pred[:, 3] > 0.5) == (act[:, 3] > 0.5)).mean())}


# ---------------------------------------------------------------- 閉ループ評価（Genesis）
def rollout_policy(sim, policy, task: TaskSpec, layout: Layout, *, use_phase: bool = True, force: float = -5.0,
                   on_frame=None, video_every: int = 4, budget: int = 800) -> dict:
    """方策だけでhand目標とグリッパを毎ステップ決める。各primitiveに budget ステップ（デモ標準660の約1.2倍）。
    区間ラベルは判定用（制御には使わない）。"""
    schedule = make_waypoints(task, layout, PickPlaceParams())
    seg_index = np.concatenate([np.full(s.steps, k) for k, s in enumerate(schedule)])
    goals = task.goals(layout)
    sim.reset(layout)
    rows = {k: [] for k in ("qpos", "ee_pos", "cube_pos", "segment", "grip")}
    obs = sim.observe()
    t_global = 0  # 動画フレーム間引き用
    for p_idx, color in enumerate(task.colors):
        steps = budget + (schedule[-1].steps if color == task.colors[-1] else 0)
        for t in range(steps):
            cube = obs["cube_pos"][COLORS.index(color)]
            x = features(obs["ee_pos"], obs["qpos"][7:9].sum(), cube, goals[color], float(task.kind != "lift"),
                         t / PRIMITIVE_STEPS, use_phase)
            action = policy.predict(x[None])[0]
            grip = int(action[3] > 0.5)
            target = obs["ee_pos"] + np.clip(action[:3], -DELTA_CLIP, DELTA_CLIP)
            target[2] = max(target[2], 0.105)  # 床へ突っ込む指令だけは安全側でクリップ
            sim.apply(target, grip, force)
            sim.scene.step()
            obs = sim.observe()
            for key in ("qpos", "ee_pos", "cube_pos"):
                rows[key].append(obs[key])
            # 判定用ラベル: primitive内の経過時間を標準スケジュールへ伸縮して対応付ける（近似）。
            local = min(int(t * PRIMITIVE_STEPS / budget), PRIMITIVE_STEPS - 1) if t < budget else len(seg_index)
            rows["segment"].append(seg_index[min(p_idx * PRIMITIVE_STEPS + local, len(seg_index) - 1)])
            rows["grip"].append(grip)
            if on_frame is not None and t_global % video_every == 0:
                on_frame(np.asarray(sim.video_camera.render()[0]))
            t_global += 1
    trajectory = {k: np.asarray(v) for k, v in rows.items()}
    trajectory.update(segment_names=np.array([s.name for s in schedule]),
                      segment_cubes=np.array([s.cube for s in schedule]))
    return trajectory


def closed_loop_eval(policy, episodes: int, seed: int, use_phase: bool, video_path: Path | None = None) -> dict:
    from sim import MultiCubeSim
    from tasks import parse_task
    tasks = [parse_task("赤・緑・青のキューブを同じ色のトレイに仕分けて"), parse_task("青いキューブを赤いキューブの上に積んで")]
    rng = np.random.default_rng(seed)
    sim = MultiCubeSim(seed=0)
    rows, frames = [], []
    try:
        for episode in range(episodes):
            layout = sample_layout(rng)
            for task in tasks:
                sink = frames.append if (video_path is not None and episode == 0) else None
                start = time.perf_counter()
                traj = rollout_policy(sim, policy, task, layout, use_phase=use_phase, on_frame=sink)
                metrics = evaluate_task(traj, task, layout)
                rows.append({"episode": episode, "kind": task.kind, "success": metrics["success"],
                             "cubes_ok": sum(i["success"] for i in metrics["per_cube"].values()),
                             "cubes": len(task.colors),
                             "stages": {c: i.get("stage") for c, i in metrics["per_cube"].items()},
                             "final_xy_error": {c: round(i["final_xy_error"], 4) for c, i in metrics["per_cube"].items()},
                             "seconds": time.perf_counter() - start})
                print(f"closed-loop {episode + 1}/{episodes} {task.kind}: {'PASS' if metrics['success'] else 'FAIL'} "
                      f"cubes_ok={rows[-1]['cubes_ok']}/{rows[-1]['cubes']} {rows[-1]['stages']}", flush=True)
    finally:
        sim.close()
    if video_path is not None and frames:
        import imageio.v2 as imageio
        imageio.mimsave(video_path, frames, fps=25, macro_block_size=8)
    summary = {"episodes": episodes, "seed": seed, "rows": rows}
    for kind in ("sort", "stack"):
        sel = [r for r in rows if r["kind"] == kind]
        summary[f"{kind}_success_rate"] = sum(r["success"] for r in sel) / max(1, len(sel))
        summary[f"{kind}_cube_success_rate"] = sum(r["cubes_ok"] for r in sel) / max(1, sum(r["cubes"] for r in sel))
    return summary


# ---------------------------------------------------------------- CLI
def train_and_evaluate(demo_paths: list[Path], out: Path, *, use_phase: bool = True, epochs: int = 30,
                       hidden: int = 128, seed: int = 0, val_fraction: float = 0.2, legacy: list[Path] = (),
                       grip_lookahead: int = GRIP_LOOKAHEAD) -> tuple[dict, MLPPolicy]:
    out.mkdir(parents=True, exist_ok=True)
    train_paths, val_paths = split_by_episode(demo_paths, val_fraction, seed)
    train_chunks = [c for p in list(train_paths) + list(legacy) for c in load_demo(p, use_phase, grip_lookahead)]
    val_chunks = [c for p in val_paths for c in load_demo(p, use_phase, grip_lookahead)]
    if not train_chunks:
        raise ValueError("学習デモがありません")
    obs, act = stack_chunks(train_chunks)
    val_obs, val_act = stack_chunks(val_chunks) if val_chunks else (obs[:0], act[:0])
    start = time.perf_counter()
    mlp = MLPPolicy(hidden=hidden, epochs=epochs, seed=seed).fit(obs, act, val_obs, val_act)
    train_seconds = time.perf_counter() - start
    linear = LinearPolicy().fit(obs, act)
    zero = np.c_[np.zeros((len(val_act), 3)), np.full(len(val_act), act[:, 3].mean())]
    report = {
        "use_phase": use_phase, "grip_lookahead": grip_lookahead, "seed": seed, "epochs": epochs, "hidden": hidden,
        "train_episodes": len(train_paths), "val_episodes": len(val_paths), "legacy_demos": len(legacy),
        "train_primitives": len(train_chunks), "val_primitives": len(val_chunks),
        "train_samples": int(len(obs)), "val_samples": int(len(val_obs)), "train_seconds": train_seconds,
        "parameters": int(sum(p.numel() for p in mlp.net.parameters())),
        "features": list(FEATURE_NAMES[:len(obs[0])]),
        "val": {"mlp": offline_metrics(mlp.predict(val_obs), val_act) if len(val_obs) else None,
                "linear": offline_metrics(linear.predict(val_obs), val_act) if len(val_obs) else None,
                "zero_action": offline_metrics(zero, val_act) if len(val_obs) else None},
        "history": mlp.history,
    }
    mlp.save(out / "policy.pt", {"use_phase": use_phase, "features": report["features"]})
    return report, mlp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demos", type=Path, default=ROOT / "data" / "demos")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "policy")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-phase", action="store_true", help="位相特徴を除くアブレーション")
    parser.add_argument("--grip-lookahead", type=int, default=GRIP_LOOKAHEAD, help="グリッパ先読みラベルのステップ数（0で無効）")
    parser.add_argument("--include-legacy", type=Path, nargs="*", default=[], help="元pipelineのlift NPZも学習に混ぜる")
    parser.add_argument("--closed-loop", type=int, default=0, help="Genesisでの閉ループ評価エピソード数（各sort+stack）")
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument("--video", type=Path, help="閉ループ1エピソード目のmp4")
    args = parser.parse_args()
    demos = find_demos(args.demos)
    print(f"demos: {len(demos)} files in {args.demos}", flush=True)
    report, policy = train_and_evaluate(demos, args.out, use_phase=not args.no_phase, epochs=args.epochs,
                                        hidden=args.hidden, seed=args.seed, legacy=args.include_legacy,
                                        grip_lookahead=args.grip_lookahead)
    print(json.dumps({k: report[k] for k in ("train_primitives", "val_primitives", "train_samples", "train_seconds", "val")},
                     ensure_ascii=False, indent=1), flush=True)
    if args.closed_loop:
        report["closed_loop"] = closed_loop_eval(policy, args.closed_loop, args.eval_seed, not args.no_phase, args.video)
        print({k: v for k, v in report["closed_loop"].items() if k.endswith("rate")}, flush=True)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
