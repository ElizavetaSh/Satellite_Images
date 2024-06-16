from typing import Any

import cv2

class SIFT(object):
    def __init__(self) -> None:
        self.method = cv2.SIFT_create()

    def __call__(self, img) -> Any:
        keypoints, descriptors = self.method.detectAndCompute(img, None)
        return keypoints, descriptors