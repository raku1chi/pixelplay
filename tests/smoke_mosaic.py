from pathlib import Path
import sys
from PIL import Image

root = Path(__file__).resolve().parents[1]
# Ensure project root is importable when running via uv
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from tiler import build_mosaic_from_pil, load_tiles_with_config  # noqa: E402

input_img_path = root / "tiler" / "images" / "cake_lego.png"
tiles_dir = root / "tiler" / "tiles" / "plus" / "gen_plus"

assert input_img_path.exists(), f"Input image not found: {input_img_path}"
assert tiles_dir.exists(), f"Tiles dir not found: {tiles_dir}"

img = Image.open(input_img_path)
# keep it small to run fast
img = img.resize((256, 256))

# Load tiles once
tiles = load_tiles_with_config(
    [str(tiles_dir)], resizing_scales=[0.4, 0.3], color_depth=16
)

out = build_mosaic_from_pil(
    img,
    tiles=tiles,
    color_depth=16,
    image_scale=0.6,
    resizing_scales=[0.4, 0.3],
    pool_size=1,  # single-threaded for stability
    overlap_tiles=False,
)

out_dir = root / "tests" / "_outputs"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "smoke_out.png"
out.save(out_path)
print(f"OK: saved {out_path}")
