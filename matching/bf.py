from typing import Any

import cv2

class BFMatcher(object):
    def __init__(self, *args, **kwargs) -> None:
        self.method = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
        # self.method = cv2.BFMatcher_create()
    def __call__(self, keypoints, descriptors) -> Any:
        matches = self.method.match(keypoints, descriptors)
        return matches