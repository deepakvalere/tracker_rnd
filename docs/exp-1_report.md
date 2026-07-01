# Experiment Results Analysis

Analysis is based only on files under `outputs/` and the experiment definitions in `configs/experiments/`. There is **no ground-truth annotation** in these outputs, so this is a **tracking-stability analysis** (ID churn, lifespan, heuristics for swaps/occlusion), not a formal MOTA/IDF1 evaluation.

---

## What Was Run

From `run.sh` line 16:

```bash
python scripts/run_experiments.py --source inputs/vid_short.mp4 --no-result-video --save-frame
```

That runs all 6 experiments in `configs/experiments/all_exps.yaml` on **`inputs/vid_short.mp4`**, with:

| Setting | Value |
|---------|-------|
| Model | `models/player.pt` |
| ReID | `auto` |
| NMS IoU | `0.35` (default) |
| Output root | `outputs/` |
| Result video | **discarded** (`--no-result-video`) |
| Annotated frames | **saved** to each `outputs/exp_*/images/` |

Video properties (from every `vid_short_details.json` header): **1499 frames**, **1280×720**, **29.97 fps** (~50 seconds).

---

## Run Outcome (from `outputs/experiments_summary.json`)

| Exp | Name | Status | Duration (UTC timestamps) | Frames in JSON | Frames saved (images/) |
|-----|------|--------|---------------------------|----------------|------------------------|
| **exp_a** | baseline | ✅ success | ~21 min | 1499 | 1499 |
| **exp_b** | tuned_full | ✅ success | ~107 min | 1499 | 1499 |
| **exp_c** | proximity | ✅ success | ~16 min | 1499 | 1499 |
| **exp_d** | new_track | ✅ success | ~14 min | 1499 | 1499 |
| **exp_e** | buffer | ❌ **failed** | ~2 min | **442 only** | **442** (partial) |
| **exp_f** | appearance | ✅ success | ~15 min | 1499 | 1499 |

**Overall: 5/6 experiments completed successfully.**  
Generated at: `2026-06-30T10:47:21Z`.

**exp_e failure:** `experiment_meta.json` records `"error": "track_players exited with code 1"`. The partial `vid_short_details.json` stops at **frame 442** (`total_frames: 442`). No error log file exists in `outputs/exp_e/`. **exp_e cannot be compared on full-video metrics.**

---

## What Each Experiment Changed

Each experiment isolates (or combines) BoT-SORT tracker parameters. Detection `conf` is also part of the experiment definition:

| Exp | Variable tested | Key changes vs baseline (exp_a) | Detection `conf` |
|-----|-----------------|-------------------------------|------------------|
| **exp_a** | baseline | default BoT-SORT ReID settings | 0.25 |
| **exp_b** | combined tuning | high_thresh 0.42, new_track 0.50, buffer 55, match 0.75, proximity 0.32, appearance 0.72 | **0.42** |
| **exp_c** | proximity only | `proximity_thresh: 0.32` (was 0.5) | 0.25 |
| **exp_d** | new-track thresholds | `track_high_thresh: 0.42`, `new_track_thresh: 0.50` | 0.25 |
| **exp_e** | track buffer | `track_buffer: 55` (was 30) | 0.25 |
| **exp_f** | appearance | `appearance_thresh: 0.72` (was 0.8) | 0.25 |

**Important:** `exp_b` changes **both** tracker settings **and** detection confidence (0.42 vs 0.25). It is not a pure tracker-only comparison.

---

## Metrics Derived from `vid_short_details.json`

Each successful run writes per-frame track data: `track_id`, bounding box, confidence, etc. Metrics below were computed from those files across the full video (1499 frames), plus the **frames 25–110** range (used by existing analysis scripts `analyze_id_switches.py` / `analyze_id_switches_deep.py` in this repo).

| Metric | Meaning |
|--------|---------|
| **unique_ids** | Total distinct track IDs assigned over the video — lower suggests less ID fragmentation |
| **max_id** | Highest ID number reached — lower suggests less ID inflation |
| **short_lived_ids (<30 frames)** | IDs that appear fewer than 30 frames — lower is better |
| **avg_lifespan_frames** | Mean frames per ID — higher suggests more stable identities |
| **id_gaps** | Times an ID disappears then reappears (same ID, non-consecutive frames) |
| **swap_heuristic_full** | Frames where an ID dies and a new ID is born within 50px (swap proxy) |
| **range_25_110_swaps** | Same swap heuristic, limited to frames 25–110 |
| **suspicious_jumps** | Consecutive frames where same ID has IoU < 0.3 or speed > 80px |
| **close_bursts (25–110)** | Sustained proximity events (pairs < 60px for ≥3 consecutive frames) |

---

## Full-Video Comparison (5 completed runs only)

| Metric | exp_a baseline | exp_b tuned_full | exp_c proximity | **exp_d new_track** | exp_f appearance |
|--------|:-:|:-:|:-:|:-:|:-:|
| avg tracks/frame | 7.40 | 7.16 | 7.40 | 7.18 | 7.40 |
| min tracks (any frame) | 0 | 0 | 0 | 0 | 0 |
| max tracks (any frame) | 11 | 11 | 12 | 11 | 11 |
| **unique_ids** ↓ | 122 | 92 | 124 | **82** | 125 |
| **max_id** ↓ | 216 | 123 | 208 | **101** | 211 |
| **short_lived (<30f)** ↓ | 64 | 34 | 65 | **26** | 66 |
| **avg lifespan** ↑ | 90.9 | 116.6 | 89.5 | **131.3** | 88.8 |
| id_gaps | 137 | **201** | 141 | 170 | **132** |
| swap_heuristic ↓ | 19 | **26** | 25 | **13** | 18 |
| suspicious_jumps | 10 | 9 | 10 | 10 | 10 |

### Crowded segment (frames 25–110)

| Metric | exp_a | exp_b | exp_c | **exp_d** | exp_f |
|--------|:-:|:-:|:-:|:-:|:-:|
| avg tracks | 9.62 | 9.34 | 9.60 | 9.38 | 9.63 |
| births | 15 | 18 | 16 | 15 | 15 |
| deaths | 17 | 20 | 18 | 17 | 17 |
| **swap heuristic** ↓ | 4 | 3 | 4 | **2** | 4 |
| close bursts | **6** | 10 | 7 | 8 | **6** |
| high IoU pairs (>0.15) | 78 | 60 | 78 | 60 | 80 |

---

## Per-Experiment Findings

### exp_a — baseline ✅
Reference run. Matches `configs/botsort_reid.yaml` / `exp_a_baseline.yaml`.

- 122 unique IDs over 1499 frames; max ID 216.
- 64 short-lived IDs (< 30 frames).
- In frames 25–110: 4 swap-heuristic events, 6 sustained close-proximity bursts.
- Some frames have **0 tracks** (`min_tracks: 0`).

### exp_d — new_track ✅ — best ID stability, **poor detection recall**
Changes only `track_high_thresh` (0.42) and `new_track_thresh` (0.50); detection conf stays **0.25** (same as baseline).

**Improvements vs baseline (ID stability metrics only):**
- **33% fewer unique IDs** (82 vs 122)
- **59% fewer short-lived IDs** (26 vs 64)
- **44% longer average track lifespan** (131.3 vs 90.9 frames)
- **32% fewer swap-heuristic events** full video (13 vs 19)
- **Fewest swap events in frames 25–110** (2 vs 4)
- **Lowest max ID** among completed runs (101 vs 216)

**Critical trade-off — missed detections:**
- exp_d output contains **zero tracks with conf < 0.42** across the full video. The raised `track_high_thresh` / `new_track_thresh` act as a second filter on top of detection conf 0.25.
- **291 frames** have fewer tracks than baseline (**326 total missing track-instances**; 11,092 vs 10,768 track-instances full video).
- **Frames 1–9:** baseline and exp_c both show **11 tracks every frame**; exp_d drops to **10 on frames 2–7** because **baseline ID 9** (conf 0.33–0.38) is rejected. ID 9 reappears from frame 8 when conf rises above 0.42, then drops again on frames 14–15 (conf 0.37–0.38).
- **exp_c matches baseline recall** in frames 1–9 and overall (11,098 vs 11,092 track-instances).

**Other trade-offs vs baseline:**
- More ID gaps (170 vs 137) — tracks disappear and return under the same ID more often.
- Slightly more close-proximity bursts in 25–110 (8 vs 6).

**Interpretation:** exp_d's improved ID metrics come largely from **not tracking** borderline detections, not from better association. Not suitable when missing visible players is unacceptable.

### exp_b — tuned_full ✅ — mixed / confounded
Combines all tuning changes **and** raises detection conf to **0.42**.

**Pros vs baseline:**
- Fewer unique IDs (92), fewer short-lived IDs (34), longer avg lifespan (116.6).
- Fewer high-IoU overlap pairs in 25–110 (60 vs 78) — less box merging in crowds.

**Cons vs baseline:**
- **Most ID gaps** (201) among completed runs.
- **Most swap-heuristic events** (26 full video, 10 close bursts in 25–110).
- **Slowest run** (~107 min vs ~14–21 min for others) — cause not recorded in output files.

**Interpretation:** Lower detection conf + combined tuning reduces raw ID count but **does not** improve swap/gap behavior vs baseline. Cannot attribute effects to tracker alone because conf also changed.

### exp_c — proximity ✅ — no ID improvement, **same detection recall as baseline**
Only `proximity_thresh: 0.32` (ReID engages earlier when boxes are close).

- **Worse** than baseline on ID metrics: 124 unique IDs, 65 short-lived, 25 swaps.
- In 25–110, IDs diverge from baseline mainly from frame 54 onward (different ID assignments in crowded cluster).
- **Same track count as baseline in frames 1–9** (11 tracks every frame); overall track-instances essentially matched (11,098 vs 11,092).

**Interpretation:** Lower proximity threshold did not help ID stability on this video. Unlike exp_d, it does **not** drop low-confidence detections.

### exp_f — appearance ✅ — no improvement
Only `appearance_thresh: 0.72` (stricter ReID matching).

- **Worst** short-lived ID count (66).
- **Most** unique IDs (125) and high max ID (211).
- Swap count similar to baseline (18 vs 19).

**Interpretation:** Loosening appearance matching strictness (0.72 vs 0.8) did not reduce ID problems here; if anything, it increased fragmentation.

### exp_e — buffer ❌ — incomplete / failed
Only change: `track_buffer: 55` (longer occlusion memory).

- Crashed at **frame 442 / 1499** (~29.5% of video).
- 442 annotated frames exist in `images/`; `experiment_meta.json` reports `frames_saved: 0` because the runner only counts frames on `status: success`.
- Partial metrics for 442 frames look superficially good, but **are not comparable** to full runs.

**Interpretation:** No valid conclusion for buffer-only tuning. The run must be re-run to test this hypothesis.

---

## Detection Recall vs ID Stability (frames 1–9 and full video)

Visual review flagged missed players in **exp_d** at the start of the video. JSON data confirms:

| Frame | baseline (exp_a) | exp_c | exp_d |
|-------|------------------|-------|-------|
| 1 | 11 | 11 | 11 |
| 2–7 | 11 | 11 | **10** |
| 8–9 | 11 | 11 | 11 |

The missing player is **baseline track ID 9** at ~cx 521–542, cy 369–373. YOLO detects it (conf ≥ 0.25) but exp_d drops it whenever score is **0.33–0.38** (below `track_high_thresh: 0.42`).

| Experiment | Total track-instances (1499 frames) | vs baseline |
|------------|-------------------------------------|-------------|
| exp_a baseline | 11,092 | — |
| exp_c | 11,098 | same recall |
| exp_d | 10,768 | **−324 missed** |
| exp_b | 10,730 | −362 missed |

**Takeaway:** Optimizing only for ID-stability metrics (unique IDs, lifespan, swap heuristics) can reward configs that silently filter real detections. Both metrics must be evaluated together.

---

## Focus Region: Frames 25–110 (Crowd / ID-Switch Zone)

Existing repo analysis scripts treat **frames 25–110** as the critical crowded segment. Birth/death events there:

**exp_a (baseline)** — 25 ID appear/disappear events, e.g.:
- Frame 46: ID 13 born (crowding begins)
- Frames 48–52: IDs 10, 8, 1 churn
- Frame 81: ID 19 born
- Frames 98–104: IDs 7, 5, 20 flicker

**exp_d (new_track)** — also 28 events but **fewer high-number IDs** in the same region (e.g. uses IDs 11–15 instead of 13–20 in several frames). Swap-heuristic count is **half** of baseline (2 vs 4).

**exp_b** — most births/deaths in this range (18/20) and most close bursts (10).

---

## Output Files Produced

| File | Purpose |
|------|---------|
| `outputs/experiments_summary.json` | Master run log — status, configs, paths, timing |
| `outputs/exp_*/experiment_meta.json` | Per-run metadata (duplicate of summary entry) |
| `outputs/exp_*/vid_short_details.json` | Full per-frame tracking data (~400k+ lines each) |
| `outputs/exp_*/images/frame*.jpg` | Annotated frames (1499 each except exp_e: 442) |

Also present: `outputs/video_on_first_run/` — an **older** partial run from **2026-06-29** with only **exp_a**, with video kept. Not part of this latest batch.

---

## Conclusions

1. **The experiment suite ran successfully for 5/6 configs** on `vid_short.mp4` (1499 frames each).

2. **exp_e (buffer-only) failed** at frame 442. No error details beyond exit code 1 are in the outputs.

3. **exp_d (new_track) wins on ID-stability metrics but loses on detection recall.**  
   Stricter `track_high_thresh` (0.42) and `new_track_thresh` (0.50) reduce ID churn but drop real players whenever detection confidence is below 0.42 (e.g. frames 2–7, 14–15). **Not recommended** when missing players is unacceptable.

4. **exp_c (proximity) does not improve ID metrics** but **preserves baseline detection recall** (including frames 1–9).

5. **Single-knob changes that did not help ID stability:**
   - **exp_c** (proximity 0.32): slightly worse ID churn, same recall
   - **exp_f** (appearance 0.72): worse than baseline

6. **exp_b (full tuning + conf 0.42)** reduces raw ID count but **increases** ID gaps and swap events vs baseline — not a clear win, and confounding prevents isolating tracker effects.

7. **No experiment eliminated tracking problems entirely.** All configs still show ID gaps, swap-heuristic events, and sustained close-proximity bursts in the crowded segment.

8. **There is no labeled ground truth** in these outputs, so rankings reflect **internal stability proxies plus track-count recall**, not verified identity accuracy.

---

## Practical Recommendation (from data only)

If choosing a config to carry forward on this video:

- **Best balance (recall + acceptable ID behavior):** **baseline (`exp_a`)** at conf 0.25 — full detection recall; moderate ID churn
- **Same recall as baseline; proximity tuning did not help IDs:** `exp_c` at conf 0.25
- **Do not use for recall-sensitive work:** `exp_d_new_track.yaml` — drops sub-0.42 detections system-wide
- **Avoid (ID metrics):** `exp_f_appearance.yaml`
- **Re-run required:** `exp_e_buffer.yaml` (failed run)
- **Use caution with:** `exp_b_tuned_full.yaml` — mixed metrics and conf 0.42 confounds tracker evaluation

**Next step (experiment suite 2):** sweep dedicated ReID encoders at **conf 0.5** with baseline tracker settings — see `configs/experiments_2/all_exps.yaml`.