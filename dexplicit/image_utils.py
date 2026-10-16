# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np


def resize_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Match DiffSynth ImageCropAndResize: aspect-preserving resize + center crop."""
    frame = np.asarray(frame, dtype=np.uint8)
    source_h, source_w = frame.shape[:2]
    if (source_h, source_w) == (height, width):
        return frame
    scale = max(width / source_w, height / source_h)
    resized_w = max(width, round(source_w * scale))
    resized_h = max(height, round(source_h * scale))
    interpolation = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)
    x0 = (resized_w - width) // 2
    y0 = (resized_h - height) // 2
    return resized[y0 : y0 + height, x0 : x0 + width].astype(np.uint8, copy=False)

def resize_mask(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Apply the RGB spatial transform with nearest-neighbor mask resampling."""
    frame = np.asarray(frame, dtype=np.uint8)
    source_h, source_w = frame.shape[:2]
    if (source_h, source_w) == (height, width):
        return frame
    scale = max(width / source_w, height / source_h)
    resized_w = max(width, round(source_w * scale))
    resized_h = max(height, round(source_h * scale))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
    x0 = (resized_w - width) // 2
    y0 = (resized_h - height) // 2
    return resized[y0 : y0 + height, x0 : x0 + width].astype(np.uint8, copy=False)
