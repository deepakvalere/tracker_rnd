#!/usr/bin/env python3
"""Deep trajectory analysis for ID switch events."""

from __future__ import annotations

import json
import math
from pathlib import Path


def iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    ab = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (aa + ab - inter)


def pos(t: dict) -> tuple[float, float]:
    return t["cx"], t["cy"]


def main() -> None:
    path = Path(__file__).resolve().parent.parent / "outputs" / "vid_short_details.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    frames = {f["frame_number"]: {t["track_id"]: t for t in f["tracks"]} for f in data["frames"]}
    start, end = 25, 110

    def trail(tid: int, f0: int, f1: int) -> None:
        print(f"\n--- Trail ID {tid} frames {f0}-{f1} ---")
        prev = None
        for fn in range(f0, f1 + 1):
            if tid not in frames.get(fn, {}):
                print(f"  frame {fn:3d}: ABSENT")
                prev = None
                continue
            t = frames[fn][tid]
            cx, cy = t["cx"], t["cy"]
            line = f"  frame {fn:3d}: cx={cx:6.1f} cy={cy:6.1f} conf={t['conf']:.3f} area={t['area']:.0f}"
            if prev:
                spd = math.hypot(cx - prev[0], cy - prev[1])
                iou_p = iou(prev[2], t)
                line += f"  d={spd:5.1f}px iou_prev={iou_p:.3f}"
            print(line)
            prev = (cx, cy, t)

    # Key event regions identified from first pass
    print("=" * 60)
    print("EVENT A: frames 28-45 — IDs 9, 10, 11 cluster (early crowd)")
    for tid in [9, 10, 11]:
        trail(tid, 28, 45)

    print("\n" + "=" * 60)
    print("EVENT B: frames 46-52 — ID 13 born, ID 1 lost, ID 8 gap")
    for tid in [1, 8, 10, 13]:
        trail(tid, 46, 54)

    print("\n" + "=" * 60)
    print("EVENT C: frames 54-62 — IDs 8, 9, 13 crowded; ID 9->10 at 61")
    for tid in [8, 9, 10, 13]:
        trail(tid, 54, 62)

    print("\n" + "=" * 60)
    print("EVENT D: frames 63-70 — IDs 8, 14, 15; ID 8 gap 67")
    for tid in [7, 8, 14, 15]:
        trail(tid, 63, 70)

    print("\n" + "=" * 60)
    print("EVENT E: frames 71-82 — ID 17 brief; ID 19 born; IDs 7 & 13 close")
    for tid in [7, 8, 13, 17, 19]:
        trail(tid, 71, 85)

    print("\n" + "=" * 60)
    print("EVENT F: frames 88-110 — IDs 7, 8, 13, 19 sustained proximity")
    for tid in [7, 8, 13, 19]:
        trail(tid, 88, 110)

    # Crossover test: for pair (A,B), does A move toward B's prev position when close?
    print("\n" + "=" * 60)
    print("CROSSOVER TEST: when two IDs are close, does bbox follow the OTHER id's path?")
    pairs = [(7, 13), (8, 9), (8, 19), (9, 13), (10, 13), (1, 8)]
    for id_a, id_b in pairs:
        swaps = []
        for fn in range(start + 1, end + 1):
            prev = frames.get(fn - 1, {})
            curr = frames.get(fn, {})
            if id_a not in prev or id_b not in prev or id_a not in curr or id_b not in curr:
                continue
            a_prev, b_prev = prev[id_a], prev[id_b]
            a_curr, b_curr = curr[id_a], curr[id_b]
            d_close = math.hypot(a_prev["cx"] - b_prev["cx"], a_prev["cy"] - b_prev["cy"])
            if d_close > 55:
                continue
            # if A jumped toward where B was
            d_a_to_b_prev = math.hypot(a_curr["cx"] - b_prev["cx"], a_curr["cy"] - b_prev["cy"])
            d_a_to_a_prev = math.hypot(a_curr["cx"] - a_prev["cx"], a_curr["cy"] - a_prev["cy"])
            d_b_to_a_prev = math.hypot(b_curr["cx"] - a_prev["cx"], b_curr["cy"] - a_prev["cy"])
            d_b_to_b_prev = math.hypot(b_curr["cx"] - b_prev["cx"], b_curr["cy"] - b_prev["cy"])
            # swap signal: A ends closer to B's old spot than B does, and vice versa
            if d_a_to_b_prev < 25 and d_b_to_a_prev < 25 and d_close < 50:
                iou_ab = iou(a_curr, b_curr)
                swaps.append(
                    (fn, d_close, d_a_to_b_prev, d_b_to_a_prev, iou_ab,
                     a_prev["conf"], b_prev["conf"], a_curr["conf"], b_curr["conf"])
                )
        if swaps:
            print(f"\n  Pair ({id_a},{id_b}): {len(swaps)} crossover-like frames")
            for s in swaps[:15]:
                print(
                    f"    frame {s[0]}: dist_prev={s[1]:.1f} "
                    f"A->B_old={s[2]:.1f} B->A_old={s[3]:.1f} iou={s[4]:.3f} "
                    f"conf A {s[5]:.2f}->{s[7]:.2f} B {s[6]:.2f}->{s[8]:.2f}"
                )
            if len(swaps) > 15:
                print(f"    ... +{len(swaps)-15} more")

    # When ID disappears, who takes its position next frame?
    print("\n" + "=" * 60)
    print("ID HANDOFF: when an ID vanishes, which surviving ID occupies its bbox?")
    for fn in range(start, end + 1):
        prev = frames.get(fn, {})
        curr = frames.get(fn + 1, {})
        if not prev or not curr:
            continue
        died = set(prev) - set(curr)
        if not died:
            continue
        for d_id in died:
            d_box = prev[d_id]
            best = []
            for s_id, s_box in curr.items():
                iv = iou(d_box, s_box)
                d = math.hypot(s_box["cx"] - d_box["cx"], s_box["cy"] - d_box["cy"])
                best.append((iv, d, s_id, s_box["conf"]))
            best.sort(reverse=True)
            top = best[:3]
            print(f"  frame {fn}->{fn+1}: ID {d_id} lost (conf={d_box['conf']:.3f} cx={d_box['cx']:.0f} cy={d_box['cy']:.0f})")
            for iv, d, sid, conf in top:
                print(f"    -> ID {sid}: iou={iv:.3f} dist={d:.1f}px conf={conf:.3f}")

    # Low conf tracks in crowd
    print("\n" + "=" * 60)
    print("LOW CONF TRACKS (<0.45) in range — often failed/occluded detections")
    for fn in range(start, end + 1):
        low = [t for t in frames.get(fn, {}).values() if t["conf"] < 0.45]
        if low:
            parts = [f"ID{t['track_id']}={t['conf']:.2f}" for t in low]
            print(f"  frame {fn}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
