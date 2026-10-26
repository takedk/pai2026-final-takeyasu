"""複数タスク（持ち上げ / 色ごとの仕分け / 積み重ね）の定義・Waypoint生成・成功判定・自動修復。

Genesisに依存しない純粋なNumPyロジックだけを置く。物理実行は sim.py、ループは multi_pipeline.py。
軌道は「各セグメントのhand目標・グリッパ指令」の列として生成し、判定と修復は実測の
cube位置・hand位置・指の開き幅だけから行う（スクリプト側の意図は判定に使わない）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
import re
from typing import Callable
import unicodedata

import numpy as np

from pipeline import NEGATION, extract_json, merge_proposal, parse_height

HAND_OFFSET = 0.11  # hand原点 - 把持中のcube中心 [m]
CUBE_SIZE = 0.04
CUBE_HALF = CUBE_SIZE / 2
TRAY_SIZE = (0.10, 0.10, 0.01)
TRAY_TOP = TRAY_SIZE[2]
COLORS = ("red", "green", "blue")
COLOR_JA = {"red": "赤", "green": "緑", "blue": "青"}
RGB = {"red": (0.85, 0.04, 0.03), "green": (0.05, 0.70, 0.10), "blue": (0.05, 0.15, 0.85)}
DEFAULT_CUBES = {"red": (0.62, -0.12), "green": (0.62, 0.0), "blue": (0.62, 0.12)}
TRAYS = {"red": (0.42, -0.22), "green": (0.40, 0.0), "blue": (0.42, 0.22)}
HOLD_WINDOW = 40
SORT_XY_TOLERANCE = 0.03     # トレイ中心からのcube中心ずれ（トレイ半幅0.05 - 余裕0.02）
STACK_XY_TOLERANCE = 0.012   # 上下cube中心の水平ずれ
LIFT_HEIGHT_TOLERANCE = 0.025
GRIPPER_CLEAR_DISTANCE = 0.14  # 置いた後、hand原点とcube中心がこれ以上離れていること（把持中は約0.11）


# ---------------------------------------------------------------- 指示解析
@dataclass(frozen=True)
class TaskSpec:
    kind: str                       # "lift" | "sort" | "stack"
    instruction: str
    colors: tuple[str, ...]         # 操作順のcube色
    base: str | None = None         # stackの土台
    target_cube_z: float | None = None  # liftの目標cube中心高さ
    careful: bool = False

    def goals(self, layout: "Layout") -> dict[str, tuple[float, float, float]]:
        """各操作cubeの最終的なcube中心の目標座標。"""
        if self.kind == "lift":
            x, y = layout.cubes[self.colors[0]]
            return {self.colors[0]: (x, y, self.target_cube_z)}
        if self.kind == "sort":
            return {c: (*TRAYS[c], TRAY_TOP + CUBE_HALF) for c in self.colors}
        x, y = layout.cubes[self.base]
        return {self.colors[0]: (x, y, CUBE_HALF + CUBE_SIZE)}


def _colors_in(text: str) -> list[str]:
    found = [(m.start(), c) for c, ja in COLOR_JA.items() for m in re.finditer(ja, text)]
    return [c for _, c in sorted(found)]


def parse_task(instruction: str) -> TaskSpec:
    """日本語指示を lift / sort / stack のいずれかへ。曖昧・否定・未対応は ValueError。"""
    text = unicodedata.normalize("NFKC", instruction).strip()
    if not text:
        raise ValueError("指示が空です。")
    if re.search(NEGATION, text):
        raise ValueError("否定・禁止を含む指示は実行しません。")
    if re.search(r"黄|白|黒|紫|オレンジ|ピンク", text):
        raise ValueError("対応色は赤・緑・青のみです。")
    careful = bool(re.search(r"慎重|ゆっくり|丁寧", text))
    colors = _colors_in(text)
    is_sort = bool(re.search(r"仕分|分類|分け|ソート|同じ色の(?:トレイ|箱|皿)", text))
    is_stack = bool(re.search(r"積|重ね|の上に(?:置|乗)", text))
    is_lift = bool(re.search(r"持ち上|持上|リフト", text))
    if is_sort + is_stack + is_lift != 1:
        raise ValueError("仕分け・積み重ね・持ち上げのうち1つを指示してください。")
    if is_sort:
        chosen = tuple(dict.fromkeys(colors)) or COLORS
        return TaskSpec("sort", text, chosen, careful=careful)
    if is_stack:
        # 「青を赤の上に」→ top=青, base=赤。色省略時は青を赤の上へ。
        match = re.search(r"(赤|緑|青)[^赤緑青]*?を[^赤緑青]*?(赤|緑|青)[^赤緑青]*?の上", text)
        if match:
            top, base = (next(c for c, ja in COLOR_JA.items() if ja == g) for g in match.groups())
        elif not colors:
            top, base = "blue", "red"
        else:
            raise ValueError("積み重ねは「青いキューブを赤いキューブの上に積んで」の形で指定してください。")
        if top == base:
            raise ValueError("同じcubeを自分自身の上には積めません。")
        return TaskSpec("stack", text, (top,), base=base, careful=careful)
    if len(set(colors)) > 1:
        raise ValueError("持ち上げる対象の色は1つにしてください。")
    target = parse_height(text)
    if not 0.15 <= target <= 0.40:
        raise ValueError("対応するキューブ中心の絶対高さは0.15～0.40mです。")
    return TaskSpec("lift", text, (colors[0] if colors else "red",), target_cube_z=target, careful=careful)


# ---------------------------------------------------------------- 配置
@dataclass(frozen=True)
class Layout:
    cubes: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_CUBES))

    def validate(self) -> "Layout":
        if set(self.cubes) != set(COLORS):
            raise ValueError("cubes must contain red/green/blue")
        points = [np.array(self.cubes[c], dtype=float) for c in COLORS]
        for c, p in zip(COLORS, points):
            if not (np.isfinite(p).all() and 0.52 <= p[0] <= 0.70 and -0.18 <= p[1] <= 0.18):
                raise ValueError(f"{c} cube out of workspace: {p}")
        for i in range(3):
            for j in range(i + 1, 3):
                if np.linalg.norm(points[i] - points[j]) < 0.10:
                    raise ValueError("cubes must be >= 0.10 m apart (finger clearance)")
        return self


def sample_layout(rng: np.random.Generator, max_tries: int = 1000) -> Layout:
    """指が隣のcubeに当たらない間隔（>=0.10m）でcube位置を一様サンプル。トレイは固定。"""
    for _ in range(max_tries):
        cubes = {c: (float(rng.uniform(0.54, 0.68)), float(rng.uniform(-0.17, 0.17))) for c in COLORS}
        try:
            return Layout(cubes).validate()
        except ValueError:
            continue
    raise RuntimeError("could not sample a valid layout")


# ---------------------------------------------------------------- パラメータ
STEP_KEYS = ("move_steps", "descend_steps", "grasp_steps", "lift_steps", "carry_steps",
             "lower_steps", "release_steps", "retreat_steps", "hold_steps")


@dataclass(frozen=True)
class PickPlaceParams:
    grasp_dz: float = 0.0            # 把持高さの補正（hand原点 = cube中心 + 0.11 + grasp_dz）
    F: float = -5.0                  # 各指の閉方向の力 [N]
    approach_z: float = 0.25         # cube上空のhand高さ
    carry_z: float = 0.25            # 運搬中のhand高さ
    release_clearance: float = 0.005  # 置く位置の何m上で離すか
    lift_dz: float = 0.0             # lift: hand目標 = 目標cube高さ + 0.11 + lift_dz
    grasp_offset: dict = field(default_factory=dict)  # 色 -> [dx, dy] 把持xy補正
    place_offset: dict = field(default_factory=dict)  # 色 -> [dx, dy] 配置xy補正
    move_steps: int = 120
    descend_steps: int = 80
    grasp_steps: int = 60
    lift_steps: int = 100
    carry_steps: int = 120
    lower_steps: int = 80
    release_steps: int = 40
    retreat_steps: int = 60
    hold_steps: int = 60

    def validate(self) -> "PickPlaceParams":
        bounds = {"grasp_dz": (-0.02, 0.12), "F": (-20.0, -0.01), "approach_z": (0.18, 0.35),
                  "carry_z": (0.18, 0.35), "release_clearance": (0.0, 0.08), "lift_dz": (-0.10, 0.10)}
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} must be a finite number in [{low}, {high}]")
        for key in ("grasp_offset", "place_offset"):
            table = getattr(self, key)
            if not isinstance(table, dict) or set(table) - set(COLORS):
                raise ValueError(f"{key} must map red/green/blue to [dx, dy]")
            for value in table.values():
                arr = np.asarray(value, dtype=float)
                if arr.shape != (2,) or not np.isfinite(arr).all() or np.abs(arr).max() > 0.08:
                    raise ValueError(f"{key} entries must be [dx, dy] with |d| <= 0.08")
        for key in STEP_KEYS:
            value = getattr(self, key)
            if type(value) is not int or not 20 <= value <= 500:
                raise ValueError(f"{key} must be an integer in [20, 500]")
        if self.approach_z < CUBE_HALF + HAND_OFFSET + self.grasp_dz + 0.03:
            raise ValueError("approach_z must be above the grasp height")
        return self


def initial_params(task: TaskSpec, mode: str = "normal", seed: int = 0) -> PickPlaceParams:
    """ルールベースの初期案。mode は自動修復のデモ用に意図的な誤差を注入する。"""
    params = PickPlaceParams()
    if task.careful:
        params = replace(params, move_steps=180, descend_steps=120, carry_steps=180, lower_steps=120)
    if mode == "normal":
        return params
    if mode == "bad_start":       # 空振り（高すぎる把持）+ 弱すぎる把持力
        return replace(params, grasp_dz=0.08, F=-0.05)
    if mode == "weak_force":
        return replace(params, F=-0.05)
    if mode == "place_bias":      # 配置位置が系統的に+x方向へ4cmずれる（カメラ較正ずれ想定）
        return replace(params, place_offset={c: [0.04, 0.0] for c in task.colors})
    if mode == "timing_jitter":   # デモ収集用: 各区間のステップ数を±20%ランダム化（方策が時間だけに頼らないように）
        rng = np.random.default_rng(seed)
        return replace(params, **{k: int(round(getattr(params, k) * rng.uniform(0.8, 1.2))) for k in STEP_KEYS})
    if mode == "perturbed":
        rng = np.random.default_rng(seed)
        return replace(params, grasp_dz=float(rng.uniform(0.0, 0.06)), F=-0.1,
                       place_offset={c: [float(rng.uniform(-0.03, 0.03)), float(rng.uniform(-0.03, 0.03))]
                                     for c in task.colors})
    raise ValueError(f"Unknown initial mode: {mode}")


# ---------------------------------------------------------------- Waypoint
@dataclass(frozen=True)
class Segment:
    name: str
    cube: str                          # このセグメントで扱うcube
    hand_goal: tuple[float, float, float]
    steps: int
    grip: int                          # 0=開（位置制御 0.04m）, 1=閉（力制御 F）
    cube_goal: tuple[float, float, float]  # このprimitiveでのcube中心の最終目標（方策の条件付けに使用）
    release: bool                      # このprimitiveが最後に手放すか（liftは保持し続ける）


def make_waypoints(task: TaskSpec, layout: Layout, params: PickPlaceParams) -> list[Segment]:
    params.validate()
    goals = task.goals(layout)
    segments: list[Segment] = []
    for color in task.colors:
        cx, cy = np.add(layout.cubes[color], params.grasp_offset.get(color, (0.0, 0.0)))
        goal = goals[color]
        grasp_z = CUBE_HALF + HAND_OFFSET + params.grasp_dz
        release = task.kind != "lift"

        def add(name, xyz, steps, grip):
            segments.append(Segment(name, color, tuple(float(v) for v in xyz), steps, grip, goal, release))

        add("approach", (cx, cy, params.approach_z), params.move_steps, 0)
        add("descend", (cx, cy, grasp_z), params.descend_steps, 0)
        add("grasp", (cx, cy, grasp_z), params.grasp_steps, 1)
        if task.kind == "lift":
            add("lift", (cx, cy, goal[2] + HAND_OFFSET + params.lift_dz), params.lift_steps, 1)
            add("hold", (cx, cy, goal[2] + HAND_OFFSET + params.lift_dz), params.hold_steps, 1)
            continue
        px, py = np.add(goal[:2], params.place_offset.get(color, (0.0, 0.0)))
        place_z = goal[2] + HAND_OFFSET + params.release_clearance
        add("lift", (cx, cy, params.carry_z), params.lift_steps, 1)
        add("carry", (px, py, params.carry_z), params.carry_steps, 1)
        add("lower", (px, py, place_z), params.lower_steps, 1)
        add("release", (px, py, place_z), params.release_steps, 0)
        add("retreat", (px, py, params.carry_z), params.retreat_steps, 0)
    if task.kind != "lift":
        last = segments[-1]
        segments.append(replace(last, name="hold", steps=params.hold_steps))
    return segments


def interpolate_targets(segments: list[Segment], start: np.ndarray, ratio: float = 0.7) -> np.ndarray:
    """各セグメントの最初の ratio 部分で前の目標から線形補間し、残りは目標で静止（急な関節目標の飛びを防ぐ）。"""
    rows, previous = [], np.asarray(start, dtype=float)
    for seg in segments:
        goal = np.asarray(seg.hand_goal)
        alpha = np.minimum(1.0, np.arange(1, seg.steps + 1) / max(1.0, ratio * seg.steps))
        rows.append(previous + (goal - previous) * alpha[:, None])
        previous = goal
    return np.concatenate(rows)


# ---------------------------------------------------------------- 成功判定
def _segment_end(trajectory: dict, color: str, name: str) -> int | None:
    names, cubes, index = trajectory["segment_names"], trajectory["segment_cubes"], trajectory["segment"]
    for k in range(len(names)):
        if names[k] == name and cubes[k] == color:
            idx = np.flatnonzero(index == k)
            return int(idx[-1]) if len(idx) else None
    return None


def evaluate_task(trajectory: dict, task: TaskSpec, layout: Layout) -> dict:
    """保存軌道（実測値）だけで判定。最後の HOLD_WINDOW ステップすべてで条件を満たすことを要求。"""
    cubes = np.asarray(trajectory["cube_pos"])  # (T, 3色, 3)
    hands = np.asarray(trajectory["ee_pos"])
    width = np.asarray(trajectory["qpos"])[:, 7:9].sum(axis=1)
    tail = slice(-HOLD_WINDOW, None)
    order = {c: i for i, c in enumerate(COLORS)}
    goals = task.goals(layout)
    per_cube, success = {}, True
    for color in task.colors:
        pos = cubes[:, order[color]]
        goal = np.asarray(goals[color])
        info = _diagnose_cube(trajectory, task, color, pos, hands, width, goal)
        xy_err = np.linalg.norm(pos[tail, :2] - goal[:2], axis=1)
        dist = np.linalg.norm(pos[tail] - hands[tail], axis=1)
        speed = np.linalg.norm(np.diff(pos[tail], axis=0), axis=1) / 0.01 if len(pos) > 1 else np.zeros(1)
        if task.kind == "lift":
            ok = bool(np.all(np.abs(pos[tail, 2] - goal[2]) <= LIFT_HEIGHT_TOLERANCE) and np.all(dist <= 0.14)
                      and np.all(pos[tail, 2] >= 0.10))
        elif task.kind == "sort":
            ok = bool(np.all(xy_err <= SORT_XY_TOLERANCE) and np.all(np.abs(pos[tail, 2] - goal[2]) <= 0.008)
                      and np.all(dist >= GRIPPER_CLEAR_DISTANCE) and speed.max() < 0.05)
        else:
            base = cubes[:, order[task.base]]
            rel = pos[tail] - base[tail]
            base_shift = float(np.linalg.norm(base[-1, :2] - base[0, :2]))
            info.update(stack_xy_error=float(np.linalg.norm(rel[-1, :2])), stack_dz=float(rel[-1, 2]),
                        base_shift=base_shift)
            ok = bool(np.all(np.linalg.norm(rel[:, :2], axis=1) <= STACK_XY_TOLERANCE)
                      and np.all(np.abs(rel[:, 2] - CUBE_SIZE) <= 0.006) and base_shift <= 0.015
                      and np.all(dist >= GRIPPER_CLEAR_DISTANCE) and speed.max() < 0.05)
        info.update(success=ok, final_pos=pos[-1].tolist(), final_xy_error=float(xy_err[-1]),
                    final_height_error=float(pos[-1, 2] - goal[2]), hold_max_xy_error=float(xy_err.max()),
                    hold_min_hand_distance=float(dist.min()), hold_max_speed=float(speed.max()))
        per_cube[color] = info
        success &= ok
    # 対象外cubeを倒した・押し出した場合も失敗（仕分けでの巻き込み事故を検出）。
    disturbed = {}
    for color in COLORS:
        if color in task.colors:
            continue
        shift = float(np.linalg.norm(cubes[-1, order[color], :2] - cubes[0, order[color], :2]))
        disturbed[color] = shift
        if shift > 0.015 and not (task.kind == "stack" and color == task.base):
            success = False
    return {"success": bool(success), "kind": task.kind, "per_cube": per_cube, "other_cube_shift": disturbed,
            "steps": int(len(cubes))}


def _diagnose_cube(trajectory, task, color, pos, hands, width, goal) -> dict:
    """primitive内のどの段階で失敗したかを、実測値で分類する（修復ルールの入力）。"""
    grasp_end = _segment_end(trajectory, color, "grasp")
    lift_end = _segment_end(trajectory, color, "lift")
    carry_end = _segment_end(trajectory, color, "carry")
    lower_end = _segment_end(trajectory, color, "lower")
    approach_end = _segment_end(trajectory, color, "approach")
    info: dict = {}
    if grasp_end is None or lift_end is None:
        info["stage"] = "not_executed"
        return info
    pre = pos[approach_end] if approach_end is not None else pos[0]
    info.update(pre_grasp_cube_pos=pre.tolist(), grasp_hand_pos=hands[grasp_end].tolist(),
                grasp_xy_error=float(np.linalg.norm(hands[grasp_end, :2] - pos[grasp_end, :2])),
                grasp_gap=float(hands[grasp_end, 2] - pos[grasp_end, 2]),
                grasp_width=float(width[grasp_end]),
                lift_rise=float(pos[lift_end, 2] - pre[2]),
                lift_hand_distance=float(np.linalg.norm(hands[lift_end] - pos[lift_end])))
    lifted = info["lift_rise"] > 0.05 and info["lift_hand_distance"] < 0.14
    if not lifted:
        missed = info["grasp_xy_error"] > 0.018 or not 0.085 <= info["grasp_gap"] <= 0.135 or info["grasp_width"] < 0.01
        info["stage"] = "missed_grasp" if missed else "weak_grasp"
        return info
    if task.kind == "lift":
        info["stage"] = "lifted"
        return info
    carried = carry_end is not None and np.linalg.norm(hands[carry_end] - pos[carry_end]) < 0.14
    if not carried:
        info["stage"] = "dropped_in_transport"
        return info
    held = pos[lower_end]
    info.update(pre_release_cube_pos=held.tolist(),
                pre_release_xy_error=[float(goal[0] - held[0]), float(goal[1] - held[1])])
    info["stage"] = "placed"
    return info


# ---------------------------------------------------------------- 自動修復
def repair(task: TaskSpec, params: PickPlaceParams, metrics: dict) -> tuple[PickPlaceParams, list[str]]:
    """失敗cubeごとに原因を分類し、すべての原因へ同時に修正をかける（1試行で複数原因を直す）。"""
    reasons: list[str] = []
    grasp_offset = {k: list(v) for k, v in params.grasp_offset.items()}
    place_offset = {k: list(v) for k, v in params.place_offset.items()}
    updates: dict = {}
    tol = STACK_XY_TOLERANCE if task.kind == "stack" else SORT_XY_TOLERANCE
    margin: list[str] = []
    for color in task.colors:
        info = metrics["per_cube"][color]
        if info["success"]:
            # 合格でも許容の半分を超えるずれは、どうせ再試行するので同時に補正しておく（ぎりぎり合格の放置を防ぐ）。
            dx, dy = info.get("pre_release_xy_error", (0.0, 0.0))
            if not metrics["success"] and info.get("stage") == "placed" and math.hypot(dx, dy) > 0.5 * tol:
                current = np.asarray(place_offset.get(color, [0.0, 0.0]))
                place_offset[color] = np.clip(current + [dx, dy], -0.08, 0.08).tolist()
                margin.append(f"{color}:margin->shift_place_xy({dx:+.3f},{dy:+.3f})")
            continue
        stage = info.get("stage")
        if stage == "missed_grasp":
            # 把持直前に実測したcube位置へxyを合わせ、高さ補正を0へ戻す。
            current = np.asarray(grasp_offset.get(color, [0.0, 0.0]))
            delta = (np.subtract(info["pre_grasp_cube_pos"][:2], info["grasp_hand_pos"][:2])
                     if info["grasp_xy_error"] > 0.018 else np.zeros(2))
            grasp_offset[color] = np.clip(current + delta, -0.08, 0.08).tolist()
            updates["grasp_dz"] = 0.0
            # 指がほぼ全閉 = 何も挟めていない。弱すぎる力も同時に疑う。
            if abs(params.F) < 1.0:
                updates["F"] = -5.0
            reasons.append(f"{color}:missed_grasp->correct_xy_and_height")
        elif stage == "weak_grasp":
            updates["F"] = -min(20.0, max(5.0, abs(params.F) * 2))
            updates["grasp_steps"] = min(500, params.grasp_steps + 30)
            reasons.append(f"{color}:weak_grasp->increase_force")
        elif stage == "dropped_in_transport":
            updates["F"] = -min(20.0, max(5.0, abs(params.F) * 1.5))
            updates["carry_steps"] = min(500, params.carry_steps + 60)
            reasons.append(f"{color}:dropped->stronger_and_slower_transport")
        elif stage == "placed":
            dx, dy = info["pre_release_xy_error"]
            if math.hypot(dx, dy) > 0.5 * tol:
                current = np.asarray(place_offset.get(color, [0.0, 0.0]))
                place_offset[color] = np.clip(current + [dx, dy], -0.08, 0.08).tolist()
                reasons.append(f"{color}:place_error->shift_place_xy({dx:+.3f},{dy:+.3f})")
            else:
                # 位置は合っているのに倒れた・揺れた → 低い位置からゆっくり離す。
                updates["release_clearance"] = round(params.release_clearance / 2, 4)
                updates["lower_steps"] = min(500, params.lower_steps + 40)
                updates["release_steps"] = min(500, params.release_steps + 20)
                reasons.append(f"{color}:unstable_release->lower_and_slower")
        elif stage == "lifted":
            error = task.target_cube_z - info["final_pos"][2]
            updates["lift_dz"] = float(np.clip(params.lift_dz + error, -0.10, 0.10))
            updates["lift_steps"] = min(500, params.lift_steps + 50)
            reasons.append(f"{color}:height_error->compensate_lift({error:+.3f})")
    if not metrics["success"] and not reasons:
        # 操作対象は成功したが他cubeを動かした等。慎重に動かす。
        updates.update(move_steps=min(500, params.move_steps + 40), carry_steps=min(500, params.carry_steps + 40))
        reasons.append("side_effect->slower_motion")
    reasons += margin
    new = replace(params, grasp_offset=grasp_offset, place_offset=place_offset, **updates).validate()
    return new, reasons


# ---------------------------------------------------------------- Teacher（LLM差し込み口）
TEACHER_PROMPT = """あなたはFranka Pandaのpick&place用Waypoint Generatorです。
ユーザー指示: {instruction}
解析済みタスク: {task}
cube（4cm立方体）配置 [x, y] m: {layout}
トレイ中心 [x, y] m（上面高さ0.01m）: {trays}
hand原点は把持中cube中心の約0.11m上。Fは各指の閉方向の力[N]（負）。
セグメント: approach -> descend -> grasp -> lift -> carry -> lower -> release -> retreat（liftタスクは lift -> hold）。
ルールベース推奨パラメータ: {recommendation}
直前までの実測フィードバック（cubeごとの失敗段階と誤差）: {feedback}
missed_grasp ならgrasp_offset/grasp_dz、weak_grasp/dropped ならFと時間、placed（位置ずれ）なら place_offset、
倒れたなら release_clearance を小さく。出力は変更したいキーだけを含むJSONオブジェクトのみ。
制約: grasp_dz=[-0.02,0.12], F=[-20,-0.01], approach_z/carry_z=[0.18,0.35], release_clearance=[0,0.08],
lift_dz=[-0.1,0.1], offsetは色ごとの[dx,dy]で各|d|<=0.08, *_stepsは整数20～500。
"""


def compact_feedback(history: list[dict]) -> list[dict]:
    """プロンプト肥大化を避け、各試行の要点（成否・段階・誤差）だけを渡す。"""
    rows = []
    for item in history:
        cubes = {c: {k: v for k, v in info.items() if k in ("success", "stage", "final_xy_error", "final_height_error",
                                                               "pre_release_xy_error", "grasp_xy_error", "grasp_gap",
                                                               "grasp_width", "stack_xy_error", "base_shift")}
                 for c, info in item["metrics"]["per_cube"].items()}
        rows.append({"iteration": item["iteration"], "success": item["metrics"]["success"],
                     "repair": item["teacher"]["reason"], "cubes": cubes})
    return rows


class MultiTeacher:
    """ルールで修復案を作り、LLM（任意）がそれを部分上書きする。LLM失敗時はルール案を使う。"""

    def __init__(self, proposer: Callable[[str], str | dict] | None = None):
        self.proposer = proposer

    def propose(self, task: TaskSpec, layout: Layout, history: list[dict], mode: str = "normal", seed: int = 0):
        if not history:
            params, reasons = initial_params(task, mode, seed), ["initial_rule"]
        else:
            last = history[-1]
            params, reasons = repair(task, params_from_dict(last["parameters"]), last["metrics"])
        params.validate()
        prompt = TEACHER_PROMPT.format(
            instruction=task.instruction, task=json.dumps(asdict(task), ensure_ascii=False),
            layout=json.dumps(layout.cubes), trays=json.dumps(TRAYS),
            recommendation=json.dumps(asdict(params), ensure_ascii=False),
            feedback=json.dumps(compact_feedback(history), ensure_ascii=False))
        decision = {"source": "offline_rules", "reason": ";".join(reasons), "prompt": prompt}
        if self.proposer is not None:
            try:
                proposal = extract_json(self.proposer(prompt))
                params = merge_proposal(params, proposal).validate()
                decision["source"] = "llm_proposal"
            except Exception as exc:
                decision["fallback_error"] = f"{type(exc).__name__}: {exc}"
        return params, decision


def params_from_dict(value: dict) -> PickPlaceParams:
    return merge_proposal(PickPlaceParams(), value).validate()
