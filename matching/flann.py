from typing import Any

import cv2

class FLANNMatcher(object):
    def __init__(self, norm=cv2.NORM_L2, FLANN_INDEX_KDTREE=1, FLANN_INDEX_LSH=6) -> None:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2

        self.method = cv2.FlannBasedMatcher(flann_params)

    def __call__(self, keypoints, descriptors) -> Any:
        # matches = self.method.match(keypoints, descriptors)
        matches = self.method.knnMatch(keypoints, descriptors, k=2)
        return matches