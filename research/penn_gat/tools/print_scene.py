#!/usr/bin/env python3
"""Print a scene's obstacles in human-measurable form.

Usage: python3 print_scene.py <scene.json>
"""
import json
import sys

path = sys.argv[1]
d = json.load(open(path))
print(f"{path}: {len(d)//4} obstacles")
for i in range(0, len(d), 4):
    x, y, z, diam = d[i:i + 4]
    print(f"  obstacle {i//4}: x={x:+.2f}m  y={y:.2f}m  z(center)={z:.2f}m  "
          f"diameter={diam:.2f}m  radius={diam/2:.2f}m")
