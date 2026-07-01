# Experiment Suite 2 — ReID Model Sweep (Report)

Analysis based only on files under `outputs/experiments_2/`, `configs/experiments_2/all_exps.yaml`, and `outputs/experiments_2/experiments_summary.json`. There is **no ground-truth annotation**; metrics are tracking-stability proxies derived from `vid_short_details.json`.

---

## What Was Run

From `run.sh` (experiments_2 block):

```bash
python scripts/run_experiments.py \
  --all-exps configs/experiments_2/all_exps.yaml \
  --source inputs/vid_short.mp4 \
  --no-result-video \
  --save-frame \
  --output-root outputs/experiments_2
```

| Setting | Value |
|---------|-------|
| Video | `inputs/vid_short.mp4` (1499 frames, 1280×720, 29.97 fps) |
| Detection model | `models/player.pt` |
| Detection conf | **0.5** (all experiments) |
| NMS IoU | 0.35 |
| Tracker | **exp_a baseline** BoT-SORT settings (same as `configs/experiments_1/exp_a_baseline.yaml`) |
| Variable | **ReID encoder only** |
| Output | `outputs/experiments_2/exp_*/` |

Summary generated at: **2026-07-01T07:08:00Z**

---

## Hardware (test machine)

All runtimes below are **specific to this laptop** — they will differ on other GPUs or machines.

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 596.08                 Driver Version: 596.08         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX A2000 Laptop GPU  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   63C    P3             17W /   35W |       0MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

**GPU:** NVIDIA RTX A2000 Laptop GPU (4 GB VRAM)

---

## Run Outcome

| Exp ID | Name | ReID model | Status | Frames saved | Wall time |
|--------|------|------------|--------|--------------|-----------|
| **exp_auto** | reid_auto | `auto` | ✅ success | 1499 | 14.7 min |
| **exp_reid_n** | reid_yolo26n | `yolo26n-reid.onnx` | ✅ success | 1499 | 52.1 min |
| **exp_reid_s** | reid_yolo26s | `yolo26s-reid.onnx` | ✅ success | 1499 | 29.1 min |
| **exp_reid_m** | reid_yolo26m | `yolo26m-reid.onnx` | ✅ success | 1499 | 119.2 min |
| **exp_reid_l** | reid_yolo26l | `yolo26l-reid.onnx` | ✅ success | 1499 | ~40 min (est.)¹ |
| **exp_reid_x** | reid_yolo26x | `yolo26x-reid.onnx` | ✅ success | 1499 | 61.3 min |

**Overall: 6/6 experiments completed successfully.**

Wall times are from `started_at` / `finished_at` in `experiments_summary.json`, except where noted.

¹ **exp_reid_l:** JSON timestamps show **859.8 min (~14.3 h)** because the laptop was **put to sleep** during the run. Estimated active runtime on this machine: **~40 min**.

---

## Experiment Design

All six runs share identical tracker parameters:

| Parameter | Value |
|-----------|-------|
| track_high_thresh | 0.25 |
| track_low_thresh | 0.1 |
| new_track_thresh | 0.25 |
| track_buffer | 30 |
| match_thresh | 0.8 |
| proximity_thresh | 0.5 |
| appearance_thresh | 0.8 |
| fuse_score | true |
| gmc_method | sparseOptFlow |

Only the ReID `model` field changes:

| Exp | ReID model | Description |
|-----|------------|-------------|
| exp_auto | `auto` | Native YOLO detector features |
| exp_reid_n | `yolo26n-reid.onnx` | Smallest dedicated encoder |
| exp_reid_s | `yolo26s-reid.onnx` | Small |
| exp_reid_m | `yolo26m-reid.onnx` | Medium |
| exp_reid_l | `yolo26l-reid.onnx` | Large |
| exp_reid_x | `yolo26x-reid.onnx` | Extra-large |

---

## Metrics (from `vid_short_details.json`)

| Metric | Meaning |
|--------|---------|
| **total_track_instances** | Sum of per-frame track counts — detection recall proxy |
| **unique_ids** | Distinct track IDs over the video — lower = less fragmentation |
| **short_lived (<30f)** | IDs present fewer than 30 frames |
| **swap_heuristic_full** | Frames where an ID dies and a new ID is born within 50px |
| **range_25_110_*** | Same heuristics limited to crowded segment frames 25–110 |

### Full-video comparison

| Metric | exp_auto | exp_reid_n | exp_reid_s | exp_reid_m | **exp_reid_l** | exp_reid_x |
|--------|:--------:|:----------:|:----------:|:----------:|:--------------:|:----------:|
| avg tracks/frame | 7.06 | 7.06 | 7.06 | 7.06 | 7.06 | 7.06 |
| min tracks (any frame) | 0 | 0 | 0 | 0 | 0 | 0 |
| max tracks (any frame) | 11 | 11 | 11 | 11 | 11 | 11 |
| total_track_instances | 10,584 | 10,586 | 10,584 | 10,585 | 10,586 | 10,583 |
| **unique_ids** ↓ | 82 | 82 | 83 | 83 | **82** | 83 |
| max_id ↓ | 110 | 109 | 110 | 110 | **109** | 111 |
| short_lived (<30f) ↓ | 24 | 24 | 25 | 24 | **24** | 25 |
| avg lifespan ↑ | 129.1 | 129.1 | 127.5 | 127.5 | **129.1** | 127.5 |
| id_gaps | 205 | 200 | 199 | 198 | **197** | 198 |
| **swap_heuristic** ↓ | **15** | 10 | **9** | **9** | **7** | 10 |
| suspicious_jumps | 10 | 10 | 10 | 10 | 10 | 10 |
| total births ↓ | 287 | 282 | 282 | 281 | **279** | 281 |
| total deaths ↓ | 280 | 275 | 275 | 274 | **272** | 274 |

### Crowded segment (frames 25–110)

| Metric | exp_auto | exp_reid_n | exp_reid_s | exp_reid_m | **exp_reid_l** | exp_reid_x |
|--------|:--------:|:----------:|:----------:|:----------:|:--------------:|:----------:|
| avg tracks | 9.15 | 9.15 | 9.15 | 9.15 | 9.15 | 9.14 |
| births ↓ | 19 | 19 | 18 | 18 | **18** | 18 |
| deaths ↓ | 21 | 21 | 20 | 20 | **20** | 20 |
| **swap heuristic** ↓ | **4** | 4 | **3** | **3** | **3** | 4 |
| close bursts | 9 | 9 | 9 | 9 | 9 | 9 |
| high IoU pairs (>0.15) | 51 | 52 | 51 | 51 | 51 | 50 |

### Frames 1–9 (track counts — identical across all six runs)

| Frame | Track count |
|-------|-------------|
| 1 | 11 |
| 2–7 | 10 |
| 8 | 11 |
| 9 | 10 |

All six experiments produce **identical per-frame track counts** in frames 1–9 and nearly identical counts full video (10,583–10,586 instances). ReID model choice affects **ID assignment and stability**, not how many players are detected at conf 0.5.

---

## Per-Experiment Findings

### exp_auto — `auto` ReID ✅
- **Fastest run:** 14.7 min.
- **Highest swap-heuristic count:** 15 full video, 4 in frames 25–110 (worst among suite).
- Same detection output as all other runs; native YOLO features are fastest but least stable on swap proxy.

### exp_reid_n — `yolo26n-reid.onnx` ✅
- 52.1 min (~3.5× slower than auto).
- Swap-heuristic: 10 (full video) — better than auto, same as x.
- Unique IDs and track instances match auto closely.

### exp_reid_s — `yolo26s-reid.onnx` ✅
- 29.1 min — good speed/quality balance among ONNX encoders.
- Swap-heuristic: **9** (tied best with m).
- Range 25–110 swaps: **3** (down from 4 on auto).

### exp_reid_m — `yolo26m-reid.onnx` ✅
- 119.2 min (~2 h).
- Swap-heuristic: **9** — tied with s.
- Metrics nearly identical to s; no clear gain over s for ~4× longer runtime.

### exp_reid_l — `yolo26l-reid.onnx` ✅ — **best ID stability in suite**
- **~40 min** on this laptop (859.8 min in JSON includes sleep time — see Run Outcome footnote).
- **Lowest swap-heuristic:** **7** full video (auto: 15, −53%).
- **Lowest id_gaps:** 197; lowest births/deaths (279/272).
- Range 25–110 swaps: **3**; tied best short-lived ID count (24).
- Best stability metrics; runtime comparable to x (~61 min) on active compute.

### exp_reid_x — `yolo26x-reid.onnx` ✅
- 61.3 min.
- Swap-heuristic: 10 — better than auto, not as good as s/m/l.
- Slightly fewer close-proximity observations in 25–110 (93 vs 95); marginal.

---

## ID Assignment Differences (frames 25–110)

Track **counts** match across runs, but **track ID numbers** diverge once dedicated encoders are used:

| Comparison vs exp_auto | Frames with different IDs or counts (25–110) |
|------------------------|-----------------------------------------------|
| exp_reid_n | 58 / 86 |
| exp_reid_s | 58 / 86 |
| exp_reid_m | 58 / 86 |
| exp_reid_l | 58 / 86 |
| exp_reid_x | 59 / 86 |

Dedicated ReID models reassign IDs in the crowded segment but keep the same number of visible tracks. Aggregate stability metrics (swaps, births, gaps) still improve for larger encoders.

---

## Detection Recall at conf 0.5

At conf **0.5** (vs conf **0.25** in experiment suite 1):

- **10,583–10,586** total track-instances (suite 2) vs **11,092** for exp_a baseline (suite 1, documented in `exp-1_report.md`).
- Frames 2–7 show **10 tracks** (not 11) across all suite-2 runs — one fewer player tracked in the opening segment at the higher conf threshold.
- Some frames still have **0 tracks** (`min_tracks: 0`) in every run.

Higher conf reduces detections uniformly; ReID model does not recover missed detections.

---

## Runtime vs Stability Trade-off

Runtimes below are for **this laptop (RTX A2000, 4 GB)**; exp_reid_l uses corrected ~40 min estimate.

```
ReID model          Wall time    Swap heuristic (full video)
────────────────────────────────────────────────────────────
auto                14.7 min     15  ← fastest, least stable
yolo26n-reid.onnx   52.1 min     10
yolo26s-reid.onnx   29.1 min      9  ← good speed/stability balance
yolo26m-reid.onnx  119.2 min      9
yolo26l-reid.onnx   ~40 min       7  ← most stable (est.; JSON inflated by sleep)
yolo26x-reid.onnx   61.3 min     10
```

---

## Output Files

| File | Purpose |
|------|---------|
| `outputs/experiments_2/experiments_summary.json` | Master run log |
| `outputs/experiments_2/exp_*/experiment_meta.json` | Per-run metadata |
| `outputs/experiments_2/exp_*/vid_short_details.json` | Per-frame tracking data |
| `outputs/experiments_2/exp_*/images/frame*.jpg` | Annotated frames (1499 each) |

---

## Conclusions

1. **All six ReID configurations completed** on the full 1499-frame video at conf 0.5.

2. **Detection output is effectively identical** across ReID models (same avg tracks, same frames 1–9 counts). The sweep isolates **association / identity stability**, not detection recall.

3. **`auto` ReID is fastest but worst on swap-heuristic** (15 events vs 7–10 for ONNX encoders).

4. **`yolo26l-reid.onnx` achieves the best ID stability** (fewest swaps, gaps, births/deaths) at **~40 min** on this laptop (JSON showed 14.3 h only because the machine slept mid-run).

5. **`yolo26s-reid.onnx` is a strong practical choice:** swap-heuristic 9 (same as m), range 25–110 swaps 3, runtime 29.1 min (~2× auto).

6. **Larger encoders:** m took 119.2 min with no gain over s on swap metrics; l beats s on stability (~7 vs 9 swaps) at ~40 min estimated on this hardware.

7. **conf 0.5 reduces track instances** vs suite-1 baseline at conf 0.25; frames 2–7 consistently show 10 tracks instead of 11.

8. **No labeled ground truth** — rankings reflect internal heuristics only. Visual review of crowded frames (25–110) in `images/` remains necessary to confirm perceived ID switch improvement.

---

## Practical Recommendation (from data only)

| Priority | Config | ReID | conf |
|----------|--------|------|------|
| **Best balance (speed + stability)** | `configs/experiments_2/exp_reid_s.yaml` | `yolo26s-reid.onnx` | 0.5 |
| **Best stability** | `configs/experiments_2/exp_reid_l.yaml` | `yolo26l-reid.onnx` | 0.5 (~40 min on this laptop) |
| **Fastest (accept more swaps)** | `configs/experiments_2/exp_auto.yaml` | `auto` | 0.5 |

If detection recall in frames 1–9 matters (11 vs 10 tracks), consider lowering conf toward 0.25 (as in suite 1) while keeping `yolo26s-reid.onnx` — that combination was **not tested** in this suite.
