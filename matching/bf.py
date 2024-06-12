from typing import Any

import cv2

class BFMatcher(object):
    def __init__(self) -> None:
        self.method = cv2.BFMatcher()

    def __call__(self, keypoints, descriptors) -> Any:
        matches = self.method.match(keypoints, descriptors)
        return matches