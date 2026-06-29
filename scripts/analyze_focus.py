#!/usr/bin/env python3
import json, math
from pathlib import Path

path = Path(r"c:\Users\ATM\Desktop\pshi\research\tracker_rnd\outputs\vid_short_details.json")
with path.open(encoding="utf-8") as f:
    data = json.load(f)
frames = {f["frame_number"]: {t["track_id"]: t for t in f["tracks"]} for f in data["frames"]}

def iou(a,b):
    ix1,iy1=max(a["x1"],b["x1"]),max(a["y1"],b["y1"])
    ix2,iy2=min(a["x2"],b["x2"]),min(a["y2"],b["y2"])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter<=0: return 0
    aa=(a["x2"]-a["x1"])*(a["y2"]-a["y1"])
    ab=(b["x2"]-b["x1"])*(b["y2"]-b["y1"])
    return inter/(aa+ab-inter)

print("=== ID1 vs ID8 proximity frames 45-54 ===")
for fn in range(45,55):
    f=frames.get(fn,{})
    if 1 in f and 8 in f:
        d=math.hypot(f[1]["cx"]-f[8]["cx"], f[1]["cy"]-f[8]["cy"])
        print(f"frame {fn}: dist={d:.1f}px iou={iou(f[1],f[8]):.3f} ID1=({f[1]['cx']:.0f},{f[1]['cy']:.0f}) conf={f[1]['conf']:.2f} ID8=({f[8]['cx']:.0f},{f[8]['cy']:.0f}) conf={f[8]['conf']:.2f}")
    else:
        ids=list(f.keys())
        print(f"frame {fn}: present={sorted(ids)}")

print("\n=== ID7 vs ID13 proximity every 5 frames 55-110 ===")
for fn in range(55,111,5):
    f=frames.get(fn,{})
    if 7 in f and 13 in f:
        d=math.hypot(f[7]["cx"]-f[13]["cx"], f[7]["cy"]-f[13]["cy"])
        print(f"frame {fn}: dist={d:.1f}px iou={iou(f[7],f[13]):.3f} ID7=({f[7]['cx']:.0f},{f[7]['cy']:.0f}) ID13=({f[13]['cx']:.0f},{f[13]['cy']:.0f})")

print("\n=== ID8 vs ID19 proximity frames 81-110 ===")
for fn in range(81,111):
    f=frames.get(fn,{})
    if 8 in f and 19 in f:
        d=math.hypot(f[8]["cx"]-f[19]["cx"], f[8]["cy"]-f[19]["cy"])
        if d < 70 or fn in (92,110):
            print(f"frame {fn}: dist={d:.1f}px iou={iou(f[8],f[19]):.3f} ID8 conf={f[8]['conf']:.2f} area={f[8]['area']:.0f} ID19 conf={f[19]['conf']:.2f}")

print("\n=== Spatial spread at peak crowd frame 57 (all tracks cx,cy) ===")
f=frames[57]
for tid in sorted(f):
    t=f[tid]
    print(f"  ID{tid:2d}: cx={t['cx']:6.1f} cy={t['cy']:6.1f} conf={t['conf']:.3f} w={t['width']:.0f} h={t['height']:.0f}")

print("\n=== Unique IDs born in range with lifespan ===")
# already have from first script
