# 改善版 E2E 検証レポート（Genesis CPU 実物理）

実行日時（日本時間）: 2026-10-26T10:00:20+09:00。総合結果: **PASS**（9/9 ケース + 上限到達テスト PASS）。
環境: Linux-6.12.94+-x86_64-with-glibc2.41 / Python 3.13.5 / Genesis 1.4.5 / Torch 2.14.1+cpu / NumPy 2.5.4。gs.cpu, dt=0.01s, substeps=4, seed=0。
scene build: 7.3s。総実行時間: 128.4s。

成功条件（すべて最後の40ステップ=0.4秒の全フレームで判定）:
- 仕分け: 各cube中心がトレイ中心から3cm以内、トレイ上の高さ±8mm、hand原点から0.14m以上離れている、速度<0.05m/s。対象外cubeの移動≤1.5cm。
- 積み重ね: 上cubeと土台cubeの水平ずれ≤12mm、高さ差0.04±0.006m、土台の移動≤1.5cm、手が離れている、静止。
- 持ち上げ: 目標高さ±2.5cm、hand-cube距離≤0.14m。

| ケース | 指示 | モード | 配置 | 試行数 | 結果 | 修復ログ（2試行目以降） |
|---|---|---|---|---:|---|---|
| sort_normal | 赤・緑・青のキューブを同じ色のトレイに仕分けて | normal | 既定 | 1 | PASS | - |
| stack_normal | 青いキューブを赤いキューブの上に積んで | normal | 既定 | 1 | PASS | - |
| lift_green_25cm | 緑のキューブを高さ25cmまで持ち上げて | normal | 既定 | 1 | PASS | - |
| sort_auto_repair | 色ごとに仕分けして | bad_start | 既定 | 2 | PASS | red:missed_grasp->correct_xy_and_height;green:missed_grasp->correct_xy_and_height;blue:missed_grasp->correct_xy_and_height |
| stack_auto_repair | 青いキューブを赤いキューブの上に積んで | place_bias | 既定 | 2 | PASS | blue:place_error->shift_place_xy(-0.041,-0.000) |
| stack_bad_start_random | 緑のキューブを青いキューブの上に積み重ねて | bad_start | random seed=11 | 2 | PASS | green:missed_grasp->correct_xy_and_height |
| sort_perturbed_random | 色ごとに仕分けして | perturbed | random seed=12 | 3 | PASS | red:missed_grasp->correct_xy_and_height;green:missed_grasp->correct_xy_and_height;blue:missed_grasp->correct_xy_and_height / green:place_error->shift_place_xy(+0.026,-0.019) |
| sort_careful_random | ゆっくり丁寧に色ごとに仕分けて | normal | random seed=13 | 1 | PASS | - |
| lift_original_task | 赤いブロックを掴んで高さ30cmまで持ち上げて | normal | 既定 | 1 | PASS | - |

## 試行ごとの実測値

| ケース/試行 | cube | 段階 | 判定 | 最終xy誤差[mm] | 高さ誤差[mm] | 積み重ねずれ[mm] | steps | physics steps/s | e2e steps/s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| sort_normal/1 | red | placed | ok | 1.9 | -0.0 | - | 2040 | 849 | 90 |
| sort_normal/1 | green | placed | ok | 0.8 | -0.0 | - | 2040 | 849 | 90 |
| sort_normal/1 | blue | placed | ok | 0.6 | -0.0 | - | 2040 | 849 | 90 |
| stack_normal/1 | blue | placed | ok | 1.6 | -0.1 | 1.6 | 720 | 857 | 92 |
| lift_green_25cm/1 | green | lifted | ok | 2.7 | -2.1 | - | 420 | 924 | 564 |
| sort_auto_repair/1 | red | missed_grasp | NG | 223.6 | -10.0 | - | 2040 | 992 | 96 |
| sort_auto_repair/1 | green | missed_grasp | NG | 220.0 | -10.0 | - | 2040 | 992 | 96 |
| sort_auto_repair/1 | blue | missed_grasp | NG | 223.6 | -10.0 | - | 2040 | 992 | 96 |
| sort_auto_repair/2 | red | placed | ok | 1.9 | -0.0 | - | 2040 | 809 | 92 |
| sort_auto_repair/2 | green | placed | ok | 0.8 | -0.0 | - | 2040 | 809 | 92 |
| sort_auto_repair/2 | blue | placed | ok | 0.6 | -0.0 | - | 2040 | 809 | 92 |
| stack_auto_repair/1 | blue | placed | NG | 63.5 | -40.0 | 63.6 | 720 | 839 | 88 |
| stack_auto_repair/2 | blue | placed | ok | 2.1 | -0.1 | 2.1 | 720 | 817 | 88 |
| stack_bad_start_random/1 | green | missed_grasp | NG | 312.4 | -40.0 | 312.4 | 720 | 880 | 548 |
| stack_bad_start_random/2 | green | placed | ok | 0.9 | -0.1 | 0.9 | 720 | 900 | 559 |
| sort_perturbed_random/1 | red | missed_grasp | NG | 231.7 | -10.0 | - | 2040 | 973 | 590 |
| sort_perturbed_random/1 | green | missed_grasp | NG | 292.1 | -10.0 | - | 2040 | 973 | 590 |
| sort_perturbed_random/1 | blue | missed_grasp | NG | 238.5 | -10.0 | - | 2040 | 973 | 590 |
| sort_perturbed_random/2 | red | placed | ok | 28.9 | -0.0 | - | 2040 | 884 | 557 |
| sort_perturbed_random/2 | green | placed | NG | 34.1 | -0.0 | - | 2040 | 884 | 557 |
| sort_perturbed_random/2 | blue | placed | ok | 24.6 | -0.0 | - | 2040 | 884 | 557 |
| sort_perturbed_random/3 | red | placed | ok | 28.9 | -0.0 | - | 2040 | 874 | 547 |
| sort_perturbed_random/3 | green | placed | ok | 2.4 | -0.0 | - | 2040 | 874 | 547 |
| sort_perturbed_random/3 | blue | placed | ok | 24.6 | -0.0 | - | 2040 | 874 | 547 |
| sort_careful_random/1 | red | placed | ok | 1.7 | -0.0 | - | 2640 | 871 | 547 |
| sort_careful_random/1 | green | placed | ok | 0.9 | -0.0 | - | 2640 | 871 | 547 |
| sort_careful_random/1 | blue | placed | ok | 2.0 | -0.0 | - | 2640 | 871 | 547 |
| lift_original_task/1 | red | lifted | ok | 2.9 | -2.3 | - | 420 | 872 | 528 |

## 実行ログ

### sort_normal

```text
[sort_normal] iteration=1/6 sort F=-5.00 grasp_dz=0.000 cubes[red=placed/ok,green=placed/ok,blue=placed/ok] steps=2040 steps/s=849 PASS repair=initial_rule
```

### stack_normal

```text
[stack_normal] iteration=1/6 stack F=-5.00 grasp_dz=0.000 cubes[blue=placed/ok] steps=720 steps/s=857 PASS repair=initial_rule
```

### lift_green_25cm

```text
[lift_green_25cm] iteration=1/6 lift F=-5.00 grasp_dz=0.000 cubes[green=lifted/ok] steps=420 steps/s=924 PASS repair=initial_rule
```

### sort_auto_repair

```text
[sort_auto_repair] iteration=1/6 sort F=-0.05 grasp_dz=0.080 cubes[red=missed_grasp/NG,green=missed_grasp/NG,blue=missed_grasp/NG] steps=2040 steps/s=992 FAIL repair=initial_rule
[sort_auto_repair] iteration=2/6 sort F=-5.00 grasp_dz=0.000 cubes[red=placed/ok,green=placed/ok,blue=placed/ok] steps=2040 steps/s=809 PASS repair=red:missed_grasp->correct_xy_and_height;green:missed_grasp->correct_xy_and_height;blue:missed_grasp->correct_xy_and_height
```

### stack_auto_repair

```text
[stack_auto_repair] iteration=1/6 stack F=-5.00 grasp_dz=0.000 cubes[blue=placed/NG] steps=720 steps/s=839 FAIL repair=initial_rule
[stack_auto_repair] iteration=2/6 stack F=-5.00 grasp_dz=0.000 cubes[blue=placed/ok] steps=720 steps/s=817 PASS repair=blue:place_error->shift_place_xy(-0.041,-0.000)
```

### stack_bad_start_random

```text
[stack_bad_start_random] iteration=1/6 stack F=-0.05 grasp_dz=0.080 cubes[green=missed_grasp/NG] steps=720 steps/s=880 FAIL repair=initial_rule
[stack_bad_start_random] iteration=2/6 stack F=-5.00 grasp_dz=0.000 cubes[green=placed/ok] steps=720 steps/s=900 PASS repair=green:missed_grasp->correct_xy_and_height
```

### sort_perturbed_random

```text
[sort_perturbed_random] iteration=1/6 sort F=-0.10 grasp_dz=0.038 cubes[red=missed_grasp/NG,green=missed_grasp/NG,blue=missed_grasp/NG] steps=2040 steps/s=973 FAIL repair=initial_rule
[sort_perturbed_random] iteration=2/6 sort F=-5.00 grasp_dz=0.000 cubes[red=placed/ok,green=placed/NG,blue=placed/ok] steps=2040 steps/s=884 FAIL repair=red:missed_grasp->correct_xy_and_height;green:missed_grasp->correct_xy_and_height;blue:missed_grasp->correct_xy_and_height
[sort_perturbed_random] iteration=3/6 sort F=-5.00 grasp_dz=0.000 cubes[red=placed/ok,green=placed/ok,blue=placed/ok] steps=2040 steps/s=874 PASS repair=green:place_error->shift_place_xy(+0.026,-0.019)
```

### sort_careful_random

```text
[sort_careful_random] iteration=1/6 sort F=-5.00 grasp_dz=0.000 cubes[red=placed/ok,green=placed/ok,blue=placed/ok] steps=2640 steps/s=871 PASS repair=initial_rule
```

### lift_original_task

```text
[lift_original_task] iteration=1/6 lift F=-5.00 grasp_dz=0.000 cubes[red=lifted/ok] steps=420 steps/s=872 PASS repair=initial_rule
```

## 上限到達テスト

bad_start を max_iterations=1 で実行し、FAIL・1試行のみ・trajectory_file=null・NPZ未生成を確認: **PASS**

## 動画

- sort_normal: `runs/videos/sort_normal.mp4`
- stack_normal: `runs/videos/stack_normal.mp4`
- sort_auto_repair: `runs/videos/sort_auto_repair.mp4`
- stack_auto_repair: `runs/videos/stack_auto_repair.mp4`

e2e steps/s は IK・状態取得・最終RGB描画を含む rollout 全体の速度（動画描画ありのケースは遅くなる）。元版（毎ステップRGB描画）の e2e は約24 steps/s。
