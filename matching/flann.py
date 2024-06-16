from typing import Any

import cv2

class FLANNMatcher(object):
    def __init__(self, norm=cv2.NORM_L2, FLANN_INDEX=1, k=2) -> None:
        self.k = k
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm=FLANN_INDEX, trees=5) # FLANN_INDEX = 1
        else:
            flann_params = dict(algorithm=FLANN_INDEX, # FLANN_INDEX = 6
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2

        self.method = cv2.FlannBasedMatcher(flann_params)

    def __call__(self, keypoints, descriptors) -> Any:
        matches = self.method.knnMatch(keypoints, descriptors, k=self.k)
        return matches