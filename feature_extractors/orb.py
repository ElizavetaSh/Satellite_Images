from typing import Any

import cv2

class ORB(object):
    def __init__(self) -> None:
        self.method = cv2.ORB_create()

    def __call__(self, img) -> Any:
        keypoints, descriptors = self.method.detectAndCompute(img, None)
        return keypoints, descriptors