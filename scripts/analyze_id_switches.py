#!/usr/bin/env python3
"""Analyze tracking JSON for ID switches in a frame range."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def dist(t1: dict, t2: dict) -> float:
    return math.hypot(t1["cx"] - t2["cx"], t1["cy"] - t2["cy"])


def main() -> None:
    path = Path(__file__).resolve().parent.parent / "outputs" / "vid_short_details.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    fps = data["fps"]
    frames_all = {f["frame_number"]: f for f in data["frames"]}
    start, end = 25, 110
    sel_nums = list(range(start, end + 1))

    print("=== METADATA ===")
    for k in ("video_name", "fps", "width", "height", "conf_threshold", "iou_threshold"):
        print(f"  {k}: {data[k]}")
    print(f"  frame_range: {start}-{end}")

    # --- per-frame summary ---
    print("\n=== TRACKS PER FRAME ===")
    for fn in sel_nums:
        f = frames_all[fn]
        ids = sorted(t["track_id"] for t in f["tracks"])
        confs = [t["conf"] for t in f["tracks"]]
        print(
            f"  frame {fn:3d}: n={f['num_tracks']:2d} "
            f"ids={ids} conf=[{min(confs):.2f}-{max(confs):.2f}]"
            if confs
            else f"  frame {fn:3d}: n=0"
        )

    # --- build per-id timeline ---
    id_timeline: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for fn in sel_nums:
        for t in frames_all[fn]["tracks"]:
            id_timeline[t["track_id"]].append((fn, t))

    print("\n=== TRACK LIFESPAN IN RANGE (first/last frame, span) ===")
    for tid in sorted(id_timeline):
        frames_present = [fn for fn, _ in id_timeline[tid]]
        print(
            f"  ID {tid:2d}: frames {min(frames_present)}-{max(frames_present)} "
            f"({len(frames_present)} frames present)"
        )

    # --- detect ID gaps within range (track disappears then reappears same id) ---
    print("\n=== GAPS WITHIN SAME ID (disappeared then returned) ===")
    for tid in sorted(id_timeline):
        present = sorted(fn for fn, _ in id_timeline[tid])
        gaps = []
        for i in range(len(present) - 1):
            if present[i + 1] - present[i] > 1:
                gaps.append((present[i], present[i + 1]))
        if gaps:
            print(f"  ID {tid:2d}: gaps {gaps}")

    # --- motion per track: speed between consecutive frames ---
    print("\n=== PER-TRACK MOTION (max speed_px between consecutive frames) ===")
    track_speeds: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for tid in sorted(id_timeline):
        entries = id_timeline[tid]
        max_spd = 0.0
        max_frame = entries[0][0]
        for i in range(1, len(entries)):
            f_prev, t_prev = entries[i - 1]
            f_curr, t_curr = entries[i]
            if f_curr != f_prev + 1:
                continue
            dx = t_curr["cx"] - t_prev["cx"]
            dy = t_curr["cy"] - t_prev["cy"]
            spd = math.hypot(dx, dy)
            track_speeds[tid].append((f_curr, spd, t_curr["conf"]))
            if spd > max_spd:
                max_spd = spd
                max_frame = f_curr
        if track_speeds[tid]:
            avg = sum(s for _, s, _ in track_speeds[tid]) / len(track_speeds[tid])
            print(f"  ID {tid:2d}: avg={avg:.1f}px max={max_spd:.1f}px @ frame {max_frame}")

    # --- sudden jumps (possible switch): large displacement or low iou with prev ---
    print("\n=== SUSPICIOUS JUMPS (iou_prev<0.3 OR speed>80px, consecutive frames) ===")
    for tid in sorted(id_timeline):
        entries = id_timeline[tid]
        for i in range(1, len(entries)):
            f_prev, t_prev = entries[i - 1]
            f_curr, t_curr = entries[i]
            if f_curr != f_prev + 1:
                continue
            spd = math.hypot(t_curr["cx"] - t_prev["cx"], t_curr["cy"] - t_prev["cy"])
            iou_v = iou(t_prev, t_curr)
            if iou_v < 0.3 or spd > 80:
                print(
                    f"  ID {tid:2d} frame {f_prev}->{f_curr}: "
                    f"speed={spd:.1f}px iou_prev={iou_v:.3f} "
                    f"conf {t_prev['conf']:.3f}->{t_curr['conf']:.3f}"
                )

    # --- pairwise proximity events ---
    print("\n=== CLOSE PAIRS (center dist < 60px) ===")
    close_events: list[tuple[int, int, int, float, float]] = []
    for fn in sel_nums:
        tracks = frames_all[fn]["tracks"]
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                d = dist(tracks[i], tracks[j])
                if d < 60:
                    close_events.append(
                        (
                            fn,
                            tracks[i]["track_id"],
                            tracks[j]["track_id"],
                            d,
                            iou(tracks[i], tracks[j]),
                        )
                    )
    # group consecutive frames with same pair
    if close_events:
        prev = close_events[0]
        burst = [prev]
        bursts = []
        for ev in close_events[1:]:
            if ev[1:3] == prev[1:3] and ev[0] == prev[0] + 1:
                burst.append(ev)
            else:
                if len(burst) >= 3:
                    bursts.append(burst)
                burst = [ev]
            prev = ev
        if len(burst) >= 3:
            bursts.append(burst)

        print(f"  total close-pair observations: {len(close_events)}")
        print(f"  sustained close bursts (>=3 consecutive frames, same pair):")
        for b in bursts:
            fn0, id_a, id_b, _, _ = b[0]
            fn1 = b[-1][0]
            min_d = min(x[3] for x in b)
            max_iou = max(x[4] for x in b)
            print(
                f"    frames {fn0}-{fn1}: IDs {id_a} & {id_b} "
                f"min_dist={min_d:.1f}px max_iou={max_iou:.3f} ({len(b)} frames)"
            )

    # --- ID birth/death in range ---
    print("\n=== ID APPEAR / DISAPPEAR IN RANGE ===")
    prev_ids = set(frames_all[start - 1]["tracks"][i]["track_id"] for i in range(len(frames_all[start - 1]["tracks"]))) if start - 1 in [f["frame_number"] for f in data["frames"]] else set()
    if start > 1:
        prev_frame = frames_all[start - 1]
        prev_ids = {t["track_id"] for t in prev_frame["tracks"]}
    else:
        prev_ids = set()

    for fn in sel_nums:
        curr_ids = {t["track_id"] for t in frames_all[fn]["tracks"]}
        born = curr_ids - prev_ids
        died = prev_ids - curr_ids
        if born or died:
            print(f"  frame {fn}: +{sorted(born) if born else []} -{sorted(died) if died else []}")
        prev_ids = curr_ids

    # --- swap detection: A close to B, then positions cross / ids remap ---
    print("\n=== POTENTIAL ID SWAPS (heuristic) ===")
    print("  Looking for: ID lost, new ID born same frame, new bbox near lost bbox location...")
    prev_tracks = {t["track_id"]: t for t in frames_all[start - 1]["tracks"]}
    for fn in range(start, end + 1):
        curr_list = frames_all[fn]["tracks"]
        curr = {t["track_id"]: t for t in curr_list}
        prev_ids_set = set(prev_tracks)
        curr_ids_set = set(curr)
        died = prev_ids_set - curr_ids_set
        born = curr_ids_set - prev_ids_set
        if died and born:
            for d_id in died:
                for b_id in born:
                    d_prev = prev_tracks[d_id]
                    # where did dead id go? check if born id is near dead id's PREV position
                    d_near_born = dist(d_prev, curr[b_id])
                    # check if any surviving id jumped to dead id's old spot
                    for s_id in curr_ids_set & prev_ids_set:
                        s_prev = prev_tracks[s_id]
                        s_curr = curr[s_id]
                        iou_dead_to_survivor = iou(d_prev, s_curr)
                        iou_survivor_self = iou(s_prev, s_curr)
                        if iou_dead_to_survivor > 0.4 and iou_survivor_self < 0.2:
                            print(
                                f"  frame {fn}: ID {d_id} lost, ID {s_id} bbox jumped to "
                                f"dead position (iou_dead->surv={iou_dead_to_survivor:.3f}, "
                                f"iou_self={iou_survivor_self:.3f}), born={sorted(born)}"
                            )
                        if d_near_born < 50 and b_id != d_id:
                            print(
                                f"  frame {fn}: ID {d_id} died, ID {b_id} born "
                                f"{d_near_born:.1f}px from dead position "
                                f"(conf {d_prev['conf']:.3f}->{curr[b_id]['conf']:.3f})"
                            )
        prev_tracks = curr

    # --- confidence drops during close events ---
    print("\n=== CONF DROPS (>0.15 drop vs prev frame) DURING CLOSE EVENTS ===")
    for fn in sel_nums:
        tracks = frames_all[fn]["tracks"]
        if fn - 1 not in frames_all:
            continue
        prev_by_id = {t["track_id"]: t for t in frames_all[fn - 1]["tracks"]}
        for t in tracks:
            tid = t["track_id"]
            if tid not in prev_by_id:
                continue
            drop = prev_by_id[tid]["conf"] - t["conf"]
            if drop > 0.15:
                # is this track close to anyone?
                others = [o for o in tracks if o["track_id"] != tid]
                min_d = min((dist(t, o) for o in others), default=999)
                if min_d < 80:
                    print(
                        f"  frame {fn} ID {tid}: conf {prev_by_id[tid]['conf']:.3f}->"
                        f"{t['conf']:.3f} (drop {drop:.3f}) nearest_other={min_d:.1f}px"
                    )

    # --- overlap / merged boxes ---
    print("\n=== HIGH IOU PAIRS (iou > 0.15, possible merge/occlusion) ===")
    for fn in sel_nums:
        tracks = frames_all[fn]["tracks"]
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                iv = iou(tracks[i], tracks[j])
                if iv > 0.15:
                    d = dist(tracks[i], tracks[j])
                    print(
                        f"  frame {fn}: ID {tracks[i]['track_id']} & "
                        f"{tracks[j]['track_id']} iou={iv:.3f} dist={d:.1f}px "
                        f"conf={tracks[i]['conf']:.2f}/{tracks[j]['conf']:.2f}"
                    )


if __name__ == "__main__":
    main()
