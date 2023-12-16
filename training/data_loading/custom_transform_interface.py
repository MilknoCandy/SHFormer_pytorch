"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-01 20:43:16
❤LastEditTime: 2022-12-02 22:21:39
❤Github: https://github.com/MilknoCandy
"""
from typing import Any, Callable, Dict

import cv2
import numpy as np
from albumentations import BasicTransform


class MaskOnlyTransform(BasicTransform):
    """Transform applied to mask only."""

    @property
    def targets(self) -> Dict[str, Callable]:
        return {"mask": self.apply_to_mask}

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        return self.apply(img, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})