"""Tiler package integration for PixelPlay.

Exposes a simple API `build_mosaic_from_pil` to create a photomosaic
from a PIL image using the tiles in this package.
"""

from .tiler import build_mosaic_from_pil, load_tiles_with_config  # re-export helpers

__all__ = [
    "build_mosaic_from_pil",
    "load_tiles_with_config",
]
