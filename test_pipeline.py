"""実際のGenesis CPUでE2E検証し、日本語レポートを実測値から生成する。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import platform
import time
import traceback
import unittest
from zoneinfo import ZoneInfo

import numpy as np

from pipeline import (ROOT, CUBE_START, DT, HOLD_WINDOW, PHASES, GenesisEvaluator,
                      Parameters, Teacher, evaluate, parse_instruction, run_pipeline, write_json)


class ContractTests(unittest.TestCase):
    def test_japanese_heights_and_careful(self):
        for text, height in (("赤いブロックを掴んで高さ３０cmまで持ち上げて", 0.30),
                             ("キューブを高さ0.2メートルまでリフト", 0.20),
                             ("立方体を250mmまで持ち上げる", 0.25),
                             ("目の前のキューブを慎重に把持して上に持ち上げる", 0.30)):
            self.assertAlmostEqual(parse_instruction(text).target_cube_z, height)
        self.assertTrue(parse_instruction("キューブをゆっくり持ち上げる").careful)
        for text in ("", "キューブを90cmまで持ち上げる", "キューブを-20cmまで持ち上げる",
                     "青いブロックを持ち上げる", "キューブを高さ三十センチまで持ち上げる"):
            with self.assertRaises(ValueError):
                parse_instruction(text)

    def test_offline_and_llm_validation(self):
        task = parse_instruction("キューブを持ち上げる")
        def offline(prompt):
            raise ConnectionError("offline test")
        for proposer in (offline, lambda p: "not JSON", lambda p: {"F": 1000}, lambda p: {"F": float("nan")}):
            params, decision = Teacher(proposer).propose(task, [], "normal", 0)
            self.assertEqual(params, Parameters())
            self.assertEqual(decision["source"], "offline_rules")
            self.assertIn("fallback_error", decision)
        params, decision = Teacher(lambda p: asdict(Parameters(F=-6.0))).propose(task, [], "normal", 0)
        self.assertEqual(params.F, -6.0)
        self.assertEqual(decision["source"], "llm_proposal")
        a = Teacher().propose(task, [], "perturbed", 10)[0]
        b = Teacher().propose(task, [], "perturbed", 10)[0]
        self.assertEqual(a, b)

    def test_evaluator_rejects_false_positives(self):
        cubes = np.tile([0.65, 0.0, 0.30], (HOLD_WINDOW + 1, 1))
        hands = cubes + [0, 0, 0.11]
        data = {"cube_pos": cubes, "ee_pos": hands}
        self.assertTrue(evaluate(data, 0.30)["success"])
        self.assertFalse(evaluate(data, 0.20)["success"])
        hands[:, 0] += 0.30
        self.assertFalse(evaluate(data, 0.30)["success"])
        hands[:, 0] -= 0.30
        cubes[-2, 2] = 0.02  # 最終フレームだけ成功でもPASSさせない。
        self.assertFalse(evaluate(data, 0.30)["success"])

    def test_iteration_limit_validation(self):
        for limit in (0, 7, -1, True):
            with self.assertRaises(ValueError):
                run_pipeline("キューブを持ち上げる", max_iterations=limit)


def check_trajectory(result: dict):
    assert result["success"], "6イテレーション以内に成功しませんでした"
    assert 1 <= len(result["history"]) <= 6
    path = Path(result["trajectory_file"])
    assert path.is_file() and path.stat().st_size > 0
    summary = json.loads(Path(result["summary_file"]).read_text(encoding="utf-8"))
    assert summary["success"] and summary["history"] == result["history"]
    last = result["history"][-1]
    with np.load(path, allow_pickle=False) as data:
        count = last["metrics"]["steps"]
        for key, shape in {"qpos": (count, 9), "target_action": (count, 9),
                           "ee_pos": (count, 3), "cube_pos": (count, 3), "ee_target": (count, 3),
                           "rgb_mean": (count, 3), "rgb_std": (count, 3),
                           "time": (count,), "phase": (count,), "action_mode": (count,),
                           "final_rgb": (72, 96, 3), "initial_qpos": (9,), "initial_cube_pos": (3,)}.items():
            assert data[key].shape == shape, (key, data[key].shape, shape)
            assert np.isfinite(data[key]).all(), key
        np.testing.assert_allclose(data["initial_cube_pos"], CUBE_START, atol=1e-6)
        np.testing.assert_allclose(np.diff(data["time"]), DT, atol=1e-10)
        np.testing.assert_allclose(data["rgb_mean"][-1], data["final_rgb"].mean(axis=(0, 1)))
        np.testing.assert_allclose(data["rgb_std"][-1], data["final_rgb"].std(axis=(0, 1)))
        assert data["final_rgb"].dtype == np.uint8
        assert np.min(data["rgb_mean"]) >= 0 and np.max(data["rgb_mean"]) <= 255
        assert np.max(np.ptp(data["rgb_mean"], axis=0)) > 0.1, "カメラ画像が時間で変化しない"
        assert np.max(data["rgb_std"]) > 1, "カメラ画像が単色"
        assert np.ptp(data["qpos"][:, 0:7]) > 0.1
        assert data["cube_pos"][-1, 2] - data["initial_cube_pos"][2] > 0.10
        assert evaluate(data, result["task"]["target_cube_z"])["success"]
        for phase_idx, phase in enumerate(PHASES):
            mask = data["phase"] == phase_idx
            assert mask.sum() == last["parameters"][f"{phase}_steps"]
            expected_mode = int(phase_idx >= 2)
            assert np.all(data["action_mode"][mask] == expected_mode)
            np.testing.assert_allclose(data["target_action"][mask, -2:], last["parameters"]["F"] if expected_mode else 0.04)
        np.testing.assert_allclose(data["cube_pos"][-1, 2], last["metrics"]["cube_z"], atol=1e-7)
    for item in result["history"]:
        np.testing.assert_allclose(item["initial_cube_pos"], CUBE_START, atol=1e-6)
        assert item["metrics"]["steps_per_second"] > 0
        assert item["metrics"]["end_to_end_steps_per_second"] > 0
        assert Path(item["prompt_file"]).is_file()


def write_report(path: Path, suite: dict):
    def relative(value):
        return str(Path(value).relative_to(ROOT)) if Path(value).is_relative_to(ROOT) else value
    lines = [
        "# PAI応用編1 最終課題：週末マイルストーン検証レポート", "",
        f"実行日時（日本時間）: {suite['timestamp']}。総合結果: **{'PASS' if suite['success'] else 'FAIL'}**。",
        f"実行環境: {platform.platform()} / Python {platform.python_version()} / Genesis {suite['versions']['genesis-world']} / Torch {suite['versions']['torch']} / NumPy {suite['versions']['numpy']}。",
        "物理バックエンドは全試行 `gs.cpu`、dt=0.01秒、substeps=4、seed=0。GUIなし、EGLで実カメラRGBを描画。",
        "", "## アーキテクチャ", "", "```mermaid", "flowchart TD",
        '    A[日本語指示] --> B[指示解析：対象・目標高さ・慎重さ]',
        '    B --> C[Teacher：ルール推奨とLLMプロンプト生成]',
        '    C --> D[パラメータ検証とWaypoint生成]',
        '    D --> E[Genesis CPU物理シミュレーション]',
        '    E --> F{高さ・距離・継続保持を評価}',
        '    F -->|成功| G[成功デモ収集：NPZとJSON]',
        '    F -->|失敗かつ試行数6未満| H[定量フィードバックから自動修復]',
        '    H --> C', '    F -->|6試行失敗| I[FAILと診断ログ保存]', "```", "",
        "`pipeline.py` に指示解析、Teacher、Waypoint制御、Evaluator、保存処理を実装。`test_pipeline.py` は実物理シミュレーションを実行して軌道の内容も検査し、このレポートを再生成する。",
        "LLM未接続時も決定論的ルールで動く。`Teacher(proposer=callable)` にプロンプトからJSONを返すLLMを接続できる。例外・不正JSON・範囲外出力はルール案へフォールバックする。各試行の実際のプロンプトを `teacher_XX.txt` に保存。今回のE2E実行はオフラインルールのみであり、外部LLMの精度・接続性は検証していない。",
        "", "## 指示・パラメータ・成功基準", "",
        "対象は座標(0.65, 0, 0.02)mに置いた赤い4cmキューブ1個。ブロック／キューブ／立方体の言い換え、cm・m・mm・全角数字、慎重／ゆっくり／丁寧に対応。高さ省略時は0.30m、対応範囲は0.15～0.40m。高さはキューブ中心の床からの絶対高さとして解釈する。未対応色や範囲外高さは明示的に拒否する。",
        "Hはhand原点の降下高さ[m]、Fは各指に指令する力[N]（負が閉方向）。リフト先hand高さは目標cube高さ+0.11mを初期値とする。`smoke2.py` のPDゲイン、力制限、IK姿勢、4 substepsを利用した。",
        "フェーズは approach → descend → grasp → lift → hold。慎重指示では移動ステップ数を増やす。各試行の前にsceneを初期状態に戻し、IK乱数seedもリセットする。",
        "**成功条件**: 最後の40ステップ（0.4秒）すべてで、cube_z ≥ 0.15m、handとcube中心の距離 ≤ 0.14m、指定目標高さとの誤差 ≤ 0.025m。最終フレームだけの跳ね上がりは成功にしない。",
        "修復規則: 把持終了時の実測XYずれ・hand/cubeの上下間隔から空振りを検出してXYとHを修正。保持失敗時はFの絶対値を増やし、把持・リフトのステップ数を増加。保持成功で目標高さが違う場合は実測誤差でlift_zを補正。最大6試行で終了。",
        "", "## 実行コマンド", "", "```bash", "cd pai2026-final-takeyasu",
        ".venv/bin/python -u test_pipeline.py",
        ".venv/bin/python -u pipeline.py '赤いブロックを掴んで高さ30cmまで持ち上げて'",
        ".venv/bin/python -u pipeline.py '目の前のキューブを慎重に把持して上に持ち上げる' --mode bad_start",
        ".venv/bin/python -u pipeline.py 'キューブを高さ20cmまで持ち上げる' --mode perturbed --seed 0",
        "```", "",
        "追加モード `weak_force` は力だけを弱くする。`--max-iterations` は1～6。未成功時はCLI終了コード1、成功時は0。外部APIキーや追加pipインストールは不要。キャッシュは書き込み可能な `/tmp/pai_genesis_cache` を既定値にし、既存の環境変数指定があれば優先する。初回はカーネルコンパイルに数分かかる場合がある。",
        "", "## 実測テスト結果", "",
        f"軽量テスト: {suite['unit_tests']}件。E2E成功ケース: {sum(c['passed'] for c in suite['cases'])}/{len(suite['cases'])}。上限到達・未保存の実シミュレーション検査: {'PASS' if suite['exhaustion_passed'] else 'FAIL'}。総実行時間（import後から）: {suite['elapsed_seconds']:.2f}秒。",
        "",
        "`steps/s` は scene.step() の合計実時間に対する物理ステップ数。`e2e steps/s` はIK・観測取得・毎ステップRGB描画を含むrollout全体の速度。scene build、reset、ファイル保存はどちらにも含めない。最初の試行は遅延コンパイルを含み得るため速度を定常性能と解釈しない。",
        "", "| ケース | モード | 目標cube高さ | 試行数 | 結果 |", "|---|---|---:|---:|---|",
    ]
    for case in suite["cases"]:
        result = case.get("result")
        lines.append(f"| {case['name']} | {case['mode']} | {result['task']['target_cube_z'] if result else '-'} m | {len(result['history']) if result else '-'} | {'PASS' if case['passed'] else 'FAIL'} |")
    lines += ["", "| ケース/試行 | H [m] | F [N/指] | lift_z [m] | cube_z [m] | ee_z [m] | 距離 [m] | steps/s | e2e steps/s | 結果 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for case in suite["cases"]:
        result = case.get("result")
        if not result:
            continue
        for item in result["history"]:
            p, m = item["parameters"], item["metrics"]
            lines.append(f"| {case['name']}/{item['iteration']} | {p['H']:.3f} | {p['F']:.2f} | {p['lift_z']:.3f} | {m['cube_z']:.4f} | {m['ee_z']:.4f} | {m['ee_cube_distance']:.4f} | {m['steps_per_second']:.1f} | {m['end_to_end_steps_per_second']:.1f} | {'PASS' if m['success'] else 'FAIL'} |")
    for case in suite["cases"]:
        lines += ["", f"### {case['name']}: {case['instruction']}", ""]
        if case.get("result"):
            result = case["result"]
            log = Path(result["summary_file"]).parent / "iterations.log"
            lines += ["```text", log.read_text(encoding="utf-8").rstrip(), "```"]
        if case.get("error"):
            lines += ["```text", case["error"], "```"]
    lines += ["", "## 生成された軌道と検証済み内容", ""]
    for case in suite["cases"]:
        result = case.get("result")
        if result and result["trajectory_file"]:
            traj = Path(result["trajectory_file"])
            lines.append(f"- {case['name']}: [{relative(str(traj))}]({relative(str(traj))})（{traj.stat().st_size:,} bytes、{result['history'][-1]['metrics']['steps']}ステップ）、[summary.json]({relative(result['summary_file'])})、[最終RGB]({relative(str(traj.parent / 'final_rgb.png'))})。")
    lines += ["", "NPZは `allow_pickle=False` で読み込める。各ステップのqpos(9)、target_action(9)、action_mode、EE座標、cube座標、EE目標、時刻、フェーズ、実カメラRGBのチャンネル平均・標準偏差を保存。加えて初期qpos、初期cube座標、最終RGB画像を保存する。actionはそのステップに適用した指令、状態はstep後。指の指令単位はaction_mode=0でm、1でN、アーム7軸はrad。",
              "テストはNPZ/JSONの存在だけでなく、配列形状、全値有限、時刻間隔、フェーズ長と指令の整合、初期状態リセット、最終画像から再計算した統計量、画像の時間変化、cubeの実際の上昇、保存軌道からの成功判定再計算を検査する。初回失敗ケースでは1回目の失敗、H/F変更、修復後成功を必須とする。",
              "上限テストはbad_startをmax_iterations=1で実行し、実際の失敗、1回だけの実行、trajectory_file=null、NPZ未生成、診断JSON保存を確認する。",
              f"機械可読のテスト結果: [{relative(suite['result_file'])}]({relative(suite['result_file'])})。実行コンソールログ: `test_results/test_run.log`（今回の実行時に保存）。",
              "", "## 検証範囲と残課題", "",
              "これは固定位置の単一キューブを対象とする最小パイプライン。色や指示からの対象検出、複数物体、障害物回避、実機転移、任意物体に対する成功率は未検証。XY修復は既知の初期位置を使う。RGBは実測だが物体認識には使用していない。今回の成功は記載した指示・seed・環境の結果であり、すべての摂動や対応高さ範囲全体を保証するものではない。関節目標は各フェーズで切り替える方式で、速度制約付きの滑らかな経路計画は今後の課題。",
              "", "## 参照", "",
              "- 既存の `smoke.py` / `smoke2.py` およびインストール済みGenesis 1.4.5のscene.reset、rigid_entity、Panda MJCFを確認。",
              "- ヘッドレスカメラの使用法は [Genesis公式Renderingドキュメント](https://genesis-world.readthedocs.io/en/latest/user_guide/rendering/) を参照。実装はインストール済みAPIで動作確認。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "trajectories")
    parser.add_argument("--report", type=Path, default=ROOT / "VALIDATION_REPORT.md")
    args = parser.parse_args()
    start = time.perf_counter()
    unit_result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    cases = [
        {"name": "standard_30cm", "instruction": "赤いブロックを掴んで高さ30cmまで持ち上げて", "mode": "normal"},
        {"name": "auto_repair", "instruction": "目の前のキューブを慎重に把持して上に持ち上げる", "mode": "bad_start"},
        {"name": "different_20cm", "instruction": "目の前の立方体を把持して高さ20cmまで持ち上げる", "mode": "normal"},
    ]
    evaluator = GenesisEvaluator(seed=0)
    exhaustion_passed = False
    exhaustion_result = None
    try:
        for case in cases:
            case["passed"] = False
            try:
                result = run_pipeline(case["instruction"], evaluator=evaluator, mode=case["mode"],
                                      output_dir=args.output_dir, run_name=case["name"])
                case["result"] = result
                check_trajectory(result)
                if case["mode"] == "bad_start":
                    assert not result["history"][0]["metrics"]["success"], "初回失敗を確認できない"
                    assert 2 <= len(result["history"]) <= 6
                    first, last = result["history"][0]["parameters"], result["history"][-1]["parameters"]
                    assert last["H"] != first["H"] and last["F"] != first["F"]
                case["passed"] = True
                print(f"TEST {case['name']}: PASS", flush=True)
            except Exception:
                case["error"] = traceback.format_exc()
                print(f"TEST {case['name']}: FAIL\n{case['error']}", flush=True)
        try:
            exhaustion_result = run_pipeline("キューブを持ち上げる", evaluator=evaluator, mode="bad_start",
                                             max_iterations=1, output_dir=args.output_dir, run_name="expected_failure")
            assert not exhaustion_result["success"]
            assert len(exhaustion_result["history"]) == 1
            assert exhaustion_result["trajectory_file"] is None
            folder = Path(exhaustion_result["summary_file"]).parent
            assert (folder / "summary.json").is_file() and not list(folder.glob("*.npz"))
            exhaustion_passed = True
            print("TEST iteration_cap_and_failure_artifacts: PASS", flush=True)
        except Exception:
            print(traceback.format_exc(), flush=True)
    finally:
        evaluator.close()
    result_dir = ROOT / "test_results"
    result_dir.mkdir(exist_ok=True)
    import importlib.metadata
    suite = {"timestamp": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
             "success": unit_result.wasSuccessful() and all(case["passed"] for case in cases) and exhaustion_passed,
             "unit_tests": unit_result.testsRun, "unit_success": unit_result.wasSuccessful(),
             "cases": cases, "exhaustion_passed": exhaustion_passed, "exhaustion_result": exhaustion_result,
             "elapsed_seconds": time.perf_counter() - start,
             "versions": {name: importlib.metadata.version(name) for name in ("genesis-world", "torch", "numpy")},
             "result_file": str(result_dir / "results.json")}
    write_json(result_dir / "results.json", suite)
    write_report(args.report, suite)
    print(f"{'ALL TESTS PASS' if suite['success'] else 'TESTS FAILED'}: {sum(case['passed'] for case in cases)}/3 E2E; "
          f"report={args.report}", flush=True)
    return 0 if suite["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
