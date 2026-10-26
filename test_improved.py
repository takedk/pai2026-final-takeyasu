"""improved/ の単体テスト。重い物理ループは使わず、運動学だけの FakeSim で
指示解析 → Waypoint → 判定 → 自動修復 → デモ保存 → BC学習 までを数秒で検証する。

Genesisが import できる環境では、最後に1件だけ実物理スモーク（積み重ね720ステップ）を行う。
省略したい場合は PAI_SKIP_GENESIS=1。
実行: python -m unittest -v test_improved   または   python test_improved.py
"""

from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline import Parameters, Teacher, extract_json, merge_proposal, parse_instruction
from multi_pipeline import run_task
from tasks import (COLORS, CUBE_HALF, CUBE_SIZE, HAND_OFFSET, HOLD_WINDOW, TRAYS, TRAY_TOP, Layout, MultiTeacher,
                   PickPlaceParams, compact_feedback, evaluate_task, initial_params, interpolate_targets,
                   make_waypoints, parse_task, repair, sample_layout)
import train_policy as tp
from test_pipeline import ContractTests  # noqa: F401  元の契約テストも回帰として実行

ROOT = Path(__file__).resolve().parent


class FakeSim:
    """運動学だけの代替シミュレータ。把持条件を満たした時だけcubeがhandに付いてくる。

    handは目標へ1次遅れで追従。把持成立: grasp区間の終わりに xyずれ<0.018, 上下間隔0.085～0.135, |F|>=1N。
    離した時: 他cubeの上(水平ずれ<=0.012)なら積まれ、0.012～0.04なら倒れて床へ、トレイ上なら0.03、他は床0.02。
    """

    def __init__(self):
        self.calls = 0

    def rollout(self, layout, segments, force, video_every=0, on_frame=None, noise_std=0.0, noise_seed=0):
        self.calls += 1
        cubes = np.array([[*layout.cubes[c], CUBE_HALF] for c in COLORS], dtype=float)
        start = np.array([0.3, 0.0, 0.6])
        targets = interpolate_targets(segments, start)
        seg_index = np.concatenate([np.full(s.steps, k) for k, s in enumerate(segments)])
        hand, held, touching, rows = start.copy(), None, False, {k: [] for k in ("qpos", "ee_pos", "cube_pos", "ee_target", "grip", "segment",
                                            "cube_goal", "active_cube", "release")}
        for t, target in enumerate(targets):
            seg = segments[seg_index[t]]
            hand += 0.3 * (target - hand)  # PD追従の遅れを1次遅れで近似
            last_of_segment = t + 1 == len(targets) or seg_index[t + 1] != seg_index[t]
            i = COLORS.index(seg.cube)
            if seg.name == "grasp" and last_of_segment and held is None:
                gap = hand[2] - cubes[i, 2]
                touching = bool(np.linalg.norm(hand[:2] - cubes[i, :2]) < 0.018 and 0.085 <= gap <= 0.135)
                held = i if touching and abs(force) >= 1 else None  # 力が弱いと指は触れるが持ち上がらない
            if seg.grip == 0:
                touching = False
                if held is not None:
                    cubes[held] = self._drop(cubes, held)
                    held = None
            if held is not None:
                cubes[held] = hand - [0, 0, HAND_OFFSET]
            width = (0.039 if touching else 0.0) if seg.grip else 0.08
            rows["qpos"].append(np.r_[np.zeros(7), width / 2, width / 2])
            rows["ee_pos"].append(hand.copy())
            rows["cube_pos"].append(cubes.copy())
            rows["ee_target"].append(target.copy())
            rows["grip"].append(seg.grip)
            rows["segment"].append(seg_index[t])
            rows["cube_goal"].append(seg.cube_goal)
            rows["active_cube"].append(i)
            rows["release"].append(int(seg.release))
            if video_every and on_frame and t % video_every == 0:
                on_frame(np.zeros((8, 8, 3), np.uint8))
        traj = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
        for key in ("grip", "segment", "active_cube", "release"):
            traj[key] = traj[key].astype(int)
        traj.update(final_rgb=np.zeros((72, 96, 3), np.uint8), segment_names=np.array([s.name for s in segments]),
                    segment_cubes=np.array([s.cube for s in segments]), colors=np.array(COLORS),
                    force=np.array(force), timing=np.array([0.1, 0.2]), initial_qpos=np.zeros(9),
                    initial_cube_pos=np.asarray(rows["cube_pos"][0]), time=np.arange(len(targets)) * 0.01)
        return traj

    @staticmethod
    def _drop(cubes, i):
        pos = cubes[i].copy()
        for j in range(len(cubes)):
            if j == i:
                continue
            offset = np.linalg.norm(pos[:2] - cubes[j, :2])
            if offset <= 0.012:
                return np.r_[cubes[j, :2] + (pos[:2] - cubes[j, :2]), cubes[j, 2] + CUBE_SIZE]
            if offset < 0.04:
                return np.r_[pos[:2] + (pos[:2] - cubes[j, :2]), CUBE_HALF]  # 縁に載って倒れる
        for tray in TRAYS.values():
            if np.all(np.abs(pos[:2] - tray) <= 0.05):
                return np.r_[pos[:2], TRAY_TOP + CUBE_HALF]
        return np.r_[pos[:2], CUBE_HALF]


class InstructionTests(unittest.TestCase):
    def test_task_kinds_and_colors(self):
        sort = parse_task("赤・緑・青のキューブを同じ色のトレイに仕分けて")
        self.assertEqual((sort.kind, sort.colors), ("sort", ("red", "green", "blue")))
        self.assertEqual(parse_task("色ごとに分けて").colors, COLORS)
        self.assertEqual(parse_task("青と赤だけ仕分けして").colors, ("blue", "red"))
        stack = parse_task("緑のキューブを青いキューブの上に積み重ねて")
        self.assertEqual((stack.kind, stack.colors, stack.base), ("stack", ("green",), "blue"))
        self.assertEqual(parse_task("キューブを積んで").colors, ("blue",))
        lift = parse_task("青いブロックを高さ２５センチまで持ち上げて")
        self.assertEqual((lift.kind, lift.colors), ("lift", ("blue",)))
        self.assertAlmostEqual(lift.target_cube_z, 0.25)
        self.assertTrue(parse_task("ゆっくり色ごとに仕分けて").careful)

    def test_rejections(self):
        for text in ("", "赤いキューブを持ち上げないで", "黄色のキューブを仕分けて", "青を青の上に積んで",
                     "赤と青を持ち上げて", "仕分けてから積んで", "キューブを90cmまで持ち上げて",
                     "こんにちは", "赤いキューブを高さ三十センチまで持ち上げて"):
            with self.assertRaises(ValueError, msg=text):
                parse_task(text)

    def test_original_parser_fixes(self):
        with self.assertRaises(ValueError):
            parse_instruction("キューブを持ち上げないでください")
        self.assertAlmostEqual(parse_instruction("キューブを3秒かけて高さ20cmまで持ち上げる").target_cube_z, 0.20)


class LLMParsingTests(unittest.TestCase):
    def test_extract_json_variants(self):
        self.assertEqual(extract_json('```json\n{"F": -6.0}\n```'), {"F": -6.0})
        self.assertEqual(extract_json('修正案です: {"H": 0.13, "note": {"a": 1}} 以上'), {"H": 0.13, "note": {"a": 1}})
        self.assertEqual(extract_json('[1, 2] {bad} {"F": -3}'), {"F": -3})
        for text in ("", "no json", "[1, 2]"):
            with self.assertRaises(ValueError):
                extract_json(text)

    def test_partial_llm_proposal_keeps_rule_values(self):
        task = parse_instruction("キューブを高さ20cmまで持ち上げる")
        params, decision = Teacher(lambda p: '```json\n{"F": -7.5}\n```').propose(task, [], "normal", 0)
        self.assertEqual(decision["source"], "llm_proposal")
        self.assertEqual(params.F, -7.5)
        self.assertAlmostEqual(params.lift_z, 0.31)  # 旧実装では既定の0.41へ戻っていた
        params, decision = Teacher(lambda p: {"F": -7.5, "speed": 3}).propose(task, [], "normal", 0)
        self.assertEqual(decision["source"], "offline_rules")
        self.assertIn("unknown keys", decision["fallback_error"])
        with self.assertRaises(ValueError):
            merge_proposal(Parameters(), {"bogus": 1})

    def test_multi_teacher_llm_and_fallback(self):
        task, layout = parse_task("色ごとに仕分けて"), Layout()
        params, decision = MultiTeacher(lambda p: 'OK {"place_offset": {"red": [0.01, 0.0]}}').propose(task, layout, [])
        self.assertEqual(decision["source"], "llm_proposal")
        self.assertEqual(params.place_offset, {"red": [0.01, 0.0]})
        self.assertIn("トレイ中心", decision["prompt"])
        for bad in ('{"F": 5}', '{"place_offset": {"red": [0.5, 0]}}', '{"grasp_offset": {"yellow": [0, 0]}}',
                    '{"move_steps": 10.5}', "timeout"):
            params, decision = MultiTeacher(lambda p, bad=bad: bad).propose(task, layout, [])
            self.assertEqual(params, PickPlaceParams(), bad)
            self.assertEqual(decision["source"], "offline_rules")


class GeometryTests(unittest.TestCase):
    def test_layout_and_sampling(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            layout = sample_layout(rng)
            pts = np.array([layout.cubes[c] for c in COLORS])
            self.assertGreaterEqual(min(np.linalg.norm(pts[i] - pts[j]) for i in range(3) for j in range(i + 1, 3)), 0.10)
        with self.assertRaises(ValueError):
            Layout({"red": (0.6, 0.0), "green": (0.6, 0.05), "blue": (0.6, 0.15)}).validate()
        with self.assertRaises(ValueError):
            Layout({"red": (0.9, 0.0), "green": (0.6, 0.0), "blue": (0.6, 0.15)}).validate()

    def test_params_validation(self):
        PickPlaceParams().validate()
        for bad in ({"F": 1.0}, {"F": float("nan")}, {"move_steps": 10}, {"move_steps": 100.0},
                    {"place_offset": {"red": [0.2, 0]}}, {"grasp_offset": {"pink": [0, 0]}}, {"approach_z": 0.18,
                                                                                              "grasp_dz": 0.12}):
            with self.assertRaises(ValueError, msg=bad):
                replace(PickPlaceParams(), **bad).validate()
        with self.assertRaises(ValueError):
            initial_params(parse_task("色ごとに仕分けて"), "unknown")

    def test_waypoints(self):
        layout = Layout()
        sort = make_waypoints(parse_task("色ごとに仕分けて"), layout, PickPlaceParams())
        self.assertEqual(len(sort), 3 * 8 + 1)
        self.assertEqual([s.name for s in sort[:8]],
                         ["approach", "descend", "grasp", "lift", "carry", "lower", "release", "retreat"])
        grasp = sort[2]
        np.testing.assert_allclose(grasp.hand_goal, (*layout.cubes["red"], CUBE_HALF + HAND_OFFSET))
        lower = sort[5]
        np.testing.assert_allclose(lower.hand_goal, (*TRAYS["red"], TRAY_TOP + CUBE_HALF + HAND_OFFSET + 0.005))
        stack = make_waypoints(parse_task("青いキューブを赤いキューブの上に積んで"), layout, PickPlaceParams())
        np.testing.assert_allclose(stack[0].cube_goal, (*layout.cubes["red"], CUBE_HALF + CUBE_SIZE))
        lift = make_waypoints(parse_task("緑のキューブを30cmまで持ち上げて"), layout, PickPlaceParams())
        self.assertEqual([s.name for s in lift], ["approach", "descend", "grasp", "lift", "hold"])
        self.assertFalse(any(s.release for s in lift))
        careful = make_waypoints(parse_task("慎重に色ごとに仕分けて"), layout, initial_params(parse_task("慎重に仕分けて")))
        self.assertGreater(sum(s.steps for s in careful), sum(s.steps for s in sort))

    def test_interpolation_is_continuous(self):
        segs = make_waypoints(parse_task("色ごとに仕分けて"), Layout(), PickPlaceParams())
        targets = interpolate_targets(segs, np.array([0.3, 0.0, 0.6]))
        self.assertEqual(len(targets), sum(s.steps for s in segs))
        self.assertLess(np.linalg.norm(np.diff(targets, axis=0), axis=1).max(), 0.01)  # 1ステップ1cm未満
        ends = np.cumsum([s.steps for s in segs]) - 1
        np.testing.assert_allclose(targets[ends], [s.hand_goal for s in segs])


class EvaluationAndRepairTests(unittest.TestCase):
    def setUp(self):
        self.sim, self.layout = FakeSim(), Layout()

    def rollout(self, text, params=None):
        task = parse_task(text)
        params = params or PickPlaceParams()
        traj = self.sim.rollout(self.layout, make_waypoints(task, self.layout, params), params.F)
        return task, traj, evaluate_task(traj, task, self.layout)

    def test_success_paths(self):
        for text in ("色ごとに仕分けて", "青いキューブを赤いキューブの上に積んで", "赤いキューブを高さ20cmまで持ち上げて"):
            task, traj, metrics = self.rollout(text)
            self.assertTrue(metrics["success"], (text, metrics))

    def test_false_positive_rejections(self):
        task, traj, metrics = self.rollout("色ごとに仕分けて")
        traj = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in traj.items()}
        traj["cube_pos"][-2, 0, 0] += 0.2  # 保持窓の途中だけ外れていてもFAIL
        self.assertFalse(evaluate_task(traj, task, self.layout)["success"])
        task, traj, _ = self.rollout("色ごとに仕分けて")
        traj["ee_pos"][-HOLD_WINDOW:] = traj["cube_pos"][-HOLD_WINDOW:, 2] + [0, 0, 0.11]  # 手が離れていない
        self.assertFalse(evaluate_task(traj, task, self.layout)["success"])
        task, traj, _ = self.rollout("青いキューブを赤いキューブの上に積んで")
        traj["cube_pos"][:, 1, :2] += np.linspace(0, 0.05, len(traj["cube_pos"]))[:, None]  # 対象外の緑を押した
        self.assertFalse(evaluate_task(traj, task, self.layout)["success"])
        task, traj, _ = self.rollout("青いキューブを赤いキューブの上に積んで")
        traj["cube_pos"][-HOLD_WINDOW:, 0, :2] += 0.03  # 土台ごとずれた（相対は合っていても土台移動でFAIL）
        traj["cube_pos"][-HOLD_WINDOW:, 2, :2] += 0.03
        self.assertFalse(evaluate_task(traj, task, self.layout)["success"])

    def test_diagnosis_stages(self):
        _, _, m = self.rollout("色ごとに仕分けて", replace(PickPlaceParams(), grasp_dz=0.08))
        self.assertEqual({i["stage"] for i in m["per_cube"].values()}, {"missed_grasp"})
        _, _, m = self.rollout("色ごとに仕分けて", replace(PickPlaceParams(), F=-0.05))
        self.assertEqual({i["stage"] for i in m["per_cube"].values()}, {"weak_grasp"})
        _, _, m = self.rollout("青いキューブを赤いキューブの上に積んで", replace(PickPlaceParams(), place_offset={"blue": [0.03, 0]}))
        self.assertEqual(m["per_cube"]["blue"]["stage"], "placed")
        self.assertAlmostEqual(m["per_cube"]["blue"]["pre_release_xy_error"][0], -0.03, places=6)
        self.assertFalse(m["success"])

    def test_repair_rules(self):
        task = parse_task("色ごとに仕分けて")
        _, _, m = self.rollout("色ごとに仕分けて", replace(PickPlaceParams(), grasp_dz=0.08, F=-0.05))
        new, reasons = repair(task, replace(PickPlaceParams(), grasp_dz=0.08, F=-0.05), m)
        self.assertEqual((new.grasp_dz, new.F), (0.0, -5.0))
        self.assertEqual(len(reasons), 3)
        _, _, m = self.rollout("色ごとに仕分けて", replace(PickPlaceParams(), F=-0.05))
        new, _ = repair(task, replace(PickPlaceParams(), F=-0.05), m)
        self.assertEqual(new.F, -5.0)
        self.assertGreater(new.grasp_steps, PickPlaceParams().grasp_steps)
        stack = parse_task("青いキューブを赤いキューブの上に積んで")
        biased = replace(PickPlaceParams(), place_offset={"blue": [0.03, -0.02]})
        _, _, m = self.rollout("青いキューブを赤いキューブの上に積んで", biased)
        new, reasons = repair(stack, biased, m)
        np.testing.assert_allclose(new.place_offset["blue"], [0.0, 0.0], atol=1e-6)
        # 合成メトリクス: 位置は合っているのに倒れた / 持ち上げ高さ不足 / 運搬中に落とした / 副作用のみ
        info = {"success": False, "stage": "placed", "pre_release_xy_error": [0.001, 0.0]}
        new, reasons = repair(stack, PickPlaceParams(), {"success": False, "per_cube": {"blue": info}})
        self.assertLess(new.release_clearance, PickPlaceParams().release_clearance)
        self.assertIn("unstable_release", reasons[0])
        lift = parse_task("赤いキューブを高さ30cmまで持ち上げて")
        info = {"success": False, "stage": "lifted", "final_pos": [0.62, -0.12, 0.27]}
        new, _ = repair(lift, PickPlaceParams(), {"success": False, "per_cube": {"red": info}})
        self.assertAlmostEqual(new.lift_dz, 0.03, places=6)
        info = {"success": False, "stage": "dropped_in_transport"}
        new, _ = repair(stack, PickPlaceParams(), {"success": False, "per_cube": {"blue": info}})
        self.assertGreater(abs(new.F), 5.0)
        self.assertGreater(new.carry_steps, PickPlaceParams().carry_steps)
        new, reasons = repair(stack, PickPlaceParams(), {"success": False, "per_cube": {"blue": {"success": True}}})
        self.assertEqual(reasons, ["side_effect->slower_motion"])
        # 他cubeの失敗で再試行するとき、ぎりぎり合格（許容の半分超）のcubeも同時に補正する。
        ok_margin = {"success": True, "stage": "placed", "pre_release_xy_error": [0.02, 0.0]}
        ok_center = {"success": True, "stage": "placed", "pre_release_xy_error": [0.002, 0.0]}
        failed = {"success": False, "stage": "placed", "pre_release_xy_error": [0.0, 0.04]}
        new, reasons = repair(task, PickPlaceParams(), {"success": False, "per_cube": {
            "red": ok_margin, "green": failed, "blue": ok_center}})
        np.testing.assert_allclose(new.place_offset["red"], [0.02, 0.0])
        np.testing.assert_allclose(new.place_offset["green"], [0.0, 0.04])
        self.assertNotIn("blue", new.place_offset)
        self.assertTrue(any(r.startswith("red:margin") for r in reasons))


class PipelineLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_case(self, text, mode="normal", **kw):
        return run_task(text, FakeSim(), mode=mode, output_dir=self.out, quiet=True, **kw)

    def test_normal_and_repairs(self):
        for text, mode, iterations in (("色ごとに仕分けて", "normal", 1), ("色ごとに仕分けて", "bad_start", 2),
                                        ("色ごとに仕分けて", "weak_force", 2),
                                        ("青いキューブを赤いキューブの上に積んで", "place_bias", 2),
                                        ("緑のキューブを青いキューブの上に積んで", "bad_start", 2),
                                        ("赤いキューブを持ち上げて", "bad_start", 2)):
            result = self.run_case(text, mode)
            self.assertTrue(result["success"], (text, mode))
            self.assertEqual(len(result["history"]), iterations, (text, mode))
            run_dir = self.out / Path(result["summary_file"]).parent
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["success"])
            self.assertFalse(Path(summary["trajectory_file"]).is_absolute())  # 公開用に相対パス
            self.assertEqual(len(list(run_dir.glob("teacher_*.txt"))), iterations)
            with np.load(run_dir / "trajectory.npz", allow_pickle=False) as data:
                self.assertEqual(data["cube_pos"].shape[1:], (3, 3))
                self.assertEqual(str(data["task_kind"]), parse_task(text).kind)

    def test_iteration_cap_and_no_progress(self):
        result = self.run_case("色ごとに仕分けて", "bad_start", max_iterations=1)
        self.assertFalse(result["success"])
        self.assertIsNone(result["trajectory_file"])
        self.assertFalse(list(self.out.rglob("*.npz")))
        for limit in (0, 7, True):
            with self.assertRaises(ValueError):
                self.run_case("色ごとに仕分けて", max_iterations=limit)
        stuck = MultiTeacher(lambda p: {"grasp_dz": 0.08})  # LLMが毎回同じ誤った案を返す
        result = self.run_case("色ごとに仕分けて", "bad_start", teacher=stuck)
        self.assertFalse(result["success"])
        self.assertIn("no_progress", result["stopped"])
        self.assertLess(len(result["history"]), 6)

    def test_compact_feedback_is_small(self):
        result = self.run_case("色ごとに仕分けて", "bad_start")
        rows = compact_feedback(result["history"])
        self.assertEqual(rows[0]["cubes"]["red"]["stage"], "missed_grasp")
        self.assertLess(len(json.dumps(rows)), 4000)


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name)
        rng = np.random.default_rng(3)
        for i in range(4):
            text = "色ごとに仕分けて" if i % 2 == 0 else "青いキューブを赤いキューブの上に積んで"
            run_task(text, FakeSim(), layout=sample_layout(rng), output_dir=out, quiet=True, run_name=f"d{i}")
        cls.paths = tp.find_demos(out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_dataset_shapes(self):
        self.assertEqual(len(self.paths), 4)
        chunks = [c for p in self.paths for c in tp.load_demo(p)]
        self.assertEqual(len(chunks), 3 + 1 + 3 + 1)  # sortは3 primitive、stackは1
        obs, act = tp.stack_chunks(chunks)
        self.assertEqual(obs.shape[1], len(tp.FEATURE_NAMES))
        self.assertEqual(act.shape[1], 4)
        self.assertTrue(set(np.unique(act[:, 3])) <= {0.0, 1.0})
        self.assertLessEqual(np.abs(act[:, :3]).max(), tp.DELTA_CLIP)
        no_phase = tp.load_demo(self.paths[0], use_phase=False)[0]["obs"]
        self.assertEqual(no_phase.shape[1], len(tp.FEATURE_NAMES) - 1)
        train, val = tp.split_by_episode(self.paths, 0.25, 0)
        self.assertEqual((len(train), len(val)), (3, 1))
        self.assertFalse(set(train) & set(val))

    def test_grip_lookahead(self):
        grip = np.r_[np.zeros(30), np.ones(30), np.zeros(10)]
        out = tp.lookahead(grip, 20)
        self.assertEqual(len(out), len(grip))
        self.assertEqual(int(np.argmax(out)), 10)          # 閉じ始めが20ステップ前倒し
        self.assertEqual(int(np.flatnonzero(out)[-1]), 39)  # 開き始めも前倒し
        np.testing.assert_array_equal(tp.lookahead(np.ones(5), 20), np.ones(5))

    def test_legacy_demo_adapter(self):
        legacy = sorted((ROOT / "data" / "legacy_demos").glob("*.npz"))
        if not legacy:
            self.skipTest("legacy demos not copied")
        chunk = tp.load_demo(legacy[0])[0]
        self.assertEqual(chunk["obs"].shape[1], len(tp.FEATURE_NAMES))
        self.assertEqual(chunk["kind"], "lift")
        self.assertTrue(np.isfinite(chunk["obs"]).all())

    def test_train_save_load_and_beats_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, policy = tp.train_and_evaluate(self.paths, Path(tmp), epochs=8, hidden=64, val_fraction=0.25)
            self.assertLess(report["history"][-1]["train_loss"], report["history"][0]["train_loss"])
            val = report["val"]
            self.assertLess(val["mlp"]["delta_rmse_mm"], val["zero_action"]["delta_rmse_mm"])
            self.assertGreater(val["mlp"]["grip_accuracy"], 0.8)
            loaded = tp.MLPPolicy.load(Path(tmp) / "policy.pt")
            obs = tp.load_demo(self.paths[0])[0]["obs"][:50]
            np.testing.assert_allclose(loaded.predict(obs), policy.predict(obs), atol=1e-6)
        linear = tp.LinearPolicy().fit(*tp.stack_chunks(tp.load_demo(self.paths[0])))
        self.assertEqual(linear.predict(obs).shape, (50, 4))


@unittest.skipIf(os.environ.get("PAI_SKIP_GENESIS") == "1" or importlib.util.find_spec("genesis") is None,
                 "Genesis未導入、またはPAI_SKIP_GENESIS=1")
class GenesisSmokeTest(unittest.TestCase):
    """実物理は1件だけ: 積み重ね（720ステップ, 約1秒 + scene build）。"""

    def test_stack_real_physics(self):
        from sim import MultiCubeSim
        sim = MultiCubeSim(seed=0)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = run_task("青いキューブを赤いキューブの上に積んで", sim, output_dir=Path(tmp), quiet=True,
                                  max_iterations=2)
                self.assertTrue(result["success"], result["history"][-1]["metrics"])
                info = result["history"][-1]["metrics"]["per_cube"]["blue"]
                self.assertLess(info["stack_xy_error"], 0.012)
                self.assertAlmostEqual(info["stack_dz"], CUBE_SIZE, delta=0.006)
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
