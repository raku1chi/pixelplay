import cv2
import numpy as np
import os
import sys
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import math
import pickle
from . import conf
from time import sleep


# Defaults from conf; functions below allow overriding via parameters
COLOR_DEPTH = conf.COLOR_DEPTH
IMAGE_SCALE = conf.IMAGE_SCALE
RESIZING_SCALES = conf.RESIZING_SCALES
PIXEL_SHIFT = conf.PIXEL_SHIFT
POOL_SIZE = conf.POOL_SIZE
OVERLAP_TILES = conf.OVERLAP_TILES


# reduces the number of colors in an image
def color_quantization(img, n_colors):
    return np.round(img / 255 * n_colors) / n_colors * 255


# returns an image given its path
def read_image(path, mainImage=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = color_quantization(img.astype("float"), COLOR_DEPTH)
    # scale the image according to IMAGE_SCALE, if this is the main image
    if mainImage:
        img = cv2.resize(img, (0, 0), fx=IMAGE_SCALE, fy=IMAGE_SCALE)
    return img.astype("uint8")


# New: read an image from a numpy array (BGRA or RGBA) with configurable params
def read_image_from_array(arr, mainImage=False, *, color_depth=None, image_scale=None):
    cd = COLOR_DEPTH if color_depth is None else color_depth
    iscale = IMAGE_SCALE if image_scale is None else image_scale
    img = arr
    # ensure BGRA for internal processing
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    else:
        # assume RGBA -> convert to BGRA for consistency
        # detect if provided is RGBA by heuristic: no certain flag; assume RGBA and swap
        # This function will be used with data we convert ourselves, so it's fine.
        pass
    img = color_quantization(img.astype("float"), cd)
    if mainImage and iscale != 1:
        img = cv2.resize(img, (0, 0), fx=iscale, fy=iscale)
    return img.astype("uint8")


# scales an image
def resize_image(img, ratio):
    img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
    return img


# the most frequent color in an image and its relative frequency
def mode_color(img, ignore_alpha=False):
    counter = defaultdict(int)
    total = 0
    for y in img:
        for x in y:
            if len(x) < 4 or ignore_alpha or x[3] != 0:
                counter[tuple(x[:3])] += 1
            else:
                counter[(-1, -1, -1)] += 1
            total += 1

    if total > 0:
        mode_color = max(counter, key=counter.get)
        if mode_color == (-1, -1, -1):
            return None, None
        else:
            return mode_color, counter[mode_color] / total
    else:
        return None, None


# displays an image
def show_image(img, wait=True):
    cv2.imshow("img", img)
    if wait:
        cv2.waitKey(0)
    else:
        cv2.waitKey(1)


# load and process the tiles
def load_tiles(paths):
    print("Loading tiles")
    tiles = defaultdict(list)

    for path in paths:
        if os.path.isdir(path):
            for tile_name in tqdm(os.listdir(path)):
                tile = read_image(os.path.join(path, tile_name))
                mode, rel_freq = mode_color(tile, ignore_alpha=True)
                if mode is not None:
                    for scale in RESIZING_SCALES:
                        t = resize_image(tile, scale)
                        res = tuple(t.shape[:2])
                        tiles[res].append(
                            {"tile": t, "mode": mode, "rel_freq": rel_freq}
                        )

            with open("tiles.pickle", "wb") as f:
                pickle.dump(tiles, f)

        # load pickle with tiles (one file only)
        else:
            with open(path, "rb") as f:
                tiles = pickle.load(f)

    return tiles


# New: config-aware tiles loading (does not rely on module globals)
def load_tiles_with_config(paths, resizing_scales=None, color_depth=None):
    print("Loading tiles (configurable)")
    rs = RESIZING_SCALES if resizing_scales is None else resizing_scales
    cd = COLOR_DEPTH if color_depth is None else color_depth
    tiles = defaultdict(list)

    for path in paths:
        if os.path.isdir(path):
            for tile_name in tqdm(os.listdir(path)):
                tile_path = os.path.join(path, tile_name)
                tile = cv2.imread(tile_path, cv2.IMREAD_UNCHANGED)
                if tile is None:
                    continue
                if tile.shape[2] == 3:
                    tile = cv2.cvtColor(tile, cv2.COLOR_BGR2BGRA)
                tile = color_quantization(tile.astype("float"), cd).astype("uint8")
                mode, rel_freq = mode_color(tile, ignore_alpha=True)
                if mode is not None:
                    for scale in rs:
                        t = resize_image(tile, scale)
                        res = tuple(t.shape[:2])
                        tiles[res].append(
                            {"tile": t, "mode": mode, "rel_freq": rel_freq}
                        )
        else:
            with open(path, "rb") as f:
                tiles = pickle.load(f)

    return tiles


# returns the boxes (image and start pos) from an image, with 'res' resolution
def image_boxes(img, res):
    if not PIXEL_SHIFT:
        shift = np.flip(res)
    else:
        shift = PIXEL_SHIFT

    boxes = []
    for y in range(0, img.shape[0], shift[1]):
        for x in range(0, img.shape[1], shift[0]):
            boxes.append({"img": img[y : y + res[0], x : x + res[1]], "pos": (x, y)})

    return boxes


# euclidean distance between two colors
def color_distance(c1, c2):
    c1_int = [int(x) for x in c1]
    c2_int = [int(x) for x in c2]
    return math.sqrt(
        (c1_int[0] - c2_int[0]) ** 2
        + (c1_int[1] - c2_int[1]) ** 2
        + (c1_int[2] - c2_int[2]) ** 2
    )


# returns the most similar tile to a box (in terms of color)
def most_similar_tile(box_mode_freq, tiles):
    if not box_mode_freq[0]:
        return (0, np.zeros(shape=tiles[0]["tile"].shape))
    else:
        min_distance = None
        min_tile_img = None
        for t in tiles:
            dist = (1 + color_distance(box_mode_freq[0], t["mode"])) / box_mode_freq[1]
            if min_distance is None or dist < min_distance:
                min_distance = dist
                min_tile_img = t["tile"]
        return (min_distance, min_tile_img)


# builds the boxes and finds the best tile for each one
def get_processed_image_boxes(image_path, tiles):
    print("Getting and processing boxes")
    img = read_image(image_path, mainImage=True)
    pool = Pool(POOL_SIZE)
    all_boxes = []

    for res, ts in tqdm(sorted(tiles.items(), reverse=True)):
        boxes = image_boxes(img, res)
        modes = pool.map(mode_color, [x["img"] for x in boxes])
        most_similar_tiles = pool.starmap(
            most_similar_tile, zip(modes, [ts for x in range(len(modes))])
        )

        i = 0
        for min_dist, tile in most_similar_tiles:
            boxes[i]["min_dist"] = min_dist
            boxes[i]["tile"] = tile
            i += 1

        all_boxes += boxes

    return all_boxes, img.shape


# New: config-aware boxes from preprocessed image
def get_processed_image_boxes_from_img(img, tiles, *, pool_size=None, pixel_shift=None):
    print("Getting and processing boxes (configurable)")
    ps = POOL_SIZE if pool_size is None else pool_size
    # temporarily override global PIXEL_SHIFT behavior for this call
    global PIXEL_SHIFT
    old_shift = PIXEL_SHIFT
    # Allow special sentinel 'auto' to force auto shift (tile size)
    if pixel_shift == "auto":
        PIXEL_SHIFT = None
    elif pixel_shift is not None:
        PIXEL_SHIFT = pixel_shift
    pool = Pool(ps) if ps and ps > 1 else None
    all_boxes = []

    try:
        items = sorted(tiles.items(), reverse=True)
        for res, ts in tqdm(items):
            boxes = image_boxes(img, res)
            imgs = [x["img"] for x in boxes]
            if pool is not None:
                modes = pool.map(mode_color, imgs)
                most_similar_tiles = pool.starmap(
                    most_similar_tile, zip(modes, [ts for _ in range(len(modes))])
                )
            else:
                modes = list(map(mode_color, imgs))
                most_similar_tiles = [most_similar_tile(m, ts) for m in modes]

            for i, (min_dist, tile) in enumerate(most_similar_tiles):
                boxes[i]["min_dist"] = min_dist
                boxes[i]["tile"] = tile

            all_boxes += boxes
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        PIXEL_SHIFT = old_shift

    return all_boxes, img.shape


# places a tile in the image
def place_tile(img, box):
    p1 = np.flip(box["pos"])
    p2 = p1 + box["img"].shape[:2]
    img_box = img[p1[0] : p2[0], p1[1] : p2[1]]
    mask = box["tile"][:, :, 3] != 0
    mask = mask[: img_box.shape[0], : img_box.shape[1]]
    if OVERLAP_TILES or not np.any(img_box[mask]):
        img_box[mask] = box["tile"][: img_box.shape[0], : img_box.shape[1], :][mask]


# tiles the image
def create_tiled_image(boxes, res, render=False):
    print("Creating tiled image")
    img = np.zeros(shape=(res[0], res[1], 4), dtype=np.uint8)

    for box in tqdm(sorted(boxes, key=lambda x: x["min_dist"], reverse=OVERLAP_TILES)):
        place_tile(img, box)
        if render:
            show_image(img, wait=False)
            sleep(0.025)

    return img


# New: config-aware tiling (overlap override)
def create_tiled_image_config(boxes, res, *, overlap_tiles=None, render=False):
    print("Creating tiled image (configurable)")
    img = np.zeros(shape=(res[0], res[1], 4), dtype=np.uint8)
    ov = OVERLAP_TILES if overlap_tiles is None else overlap_tiles
    for box in tqdm(sorted(boxes, key=lambda x: x["min_dist"], reverse=ov)):
        place_tile(img, box)
        if render:
            show_image(img, wait=False)
            sleep(0.025)
    return img


# High-level API: build mosaic from PIL image
def build_mosaic_from_pil(
    pil_image,
    tiles_paths=None,
    *,
    tiles=None,
    color_depth=None,
    image_scale=None,
    resizing_scales=None,
    pixel_shift=None,
    pool_size=None,
    overlap_tiles=None,
    render=False,
):
    """Build a photomosaic from a PIL image.

    Either provide tiles_paths (list of folders/pickle) or a preloaded `tiles` dict.
    Returns a PIL Image (RGBA).
    """
    if tiles is None:
        if not tiles_paths:
            raise ValueError("tiles_paths or tiles must be provided")
        tiles = load_tiles_with_config(tiles_paths, resizing_scales, color_depth)

    # Convert PIL (any mode) to RGBA numpy and then to BGRA for internal
    from PIL import Image

    im_rgba = pil_image.convert("RGBA")
    arr = np.array(im_rgba)
    # Convert RGBA -> BGRA
    arr_bgra = arr[:, :, [2, 1, 0, 3]]
    # Quantize/scale main image
    cd = COLOR_DEPTH if color_depth is None else color_depth
    iscale = IMAGE_SCALE if image_scale is None else image_scale
    img_proc = read_image_from_array(
        arr_bgra, mainImage=True, color_depth=cd, image_scale=iscale
    )

    boxes, original_res = get_processed_image_boxes_from_img(
        img_proc, tiles, pool_size=pool_size, pixel_shift=pixel_shift
    )
    img_bgra = create_tiled_image_config(
        boxes, original_res, overlap_tiles=overlap_tiles, render=render
    )
    # BGRA -> RGBA
    img_rgba = img_bgra[:, :, [2, 1, 0, 3]]
    return Image.fromarray(img_rgba, mode="RGBA")


# main
def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = conf.IMAGE_TO_TILE

    if len(sys.argv) > 2:
        tiles_paths = sys.argv[2:]
    else:
        tiles_paths = conf.TILES_FOLDER.split(" ")

    if not os.path.exists(image_path):
        print("Image not found")
        exit(-1)
    for path in tiles_paths:
        if not os.path.exists(path):
            print("Tiles folder not found")
            exit(-1)

    tiles = load_tiles(tiles_paths)
    boxes, original_res = get_processed_image_boxes(image_path, tiles)
    img = create_tiled_image(boxes, original_res, render=conf.RENDER)
    cv2.imwrite(conf.OUT, img)


if __name__ == "__main__":
    main()
