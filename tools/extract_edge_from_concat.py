from PIL import Image
import sys
import os

src = sys.argv[1]
dst = sys.argv[2]

img = Image.open(src).convert("RGB")
w, h = img.size
edge = img.crop((0, 0, w // 2, h))
edge.save(dst)

print(f"saved edge to {dst}")

