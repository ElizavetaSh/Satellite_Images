from abc import ABC, abstractmethod
from typing import Literal
import cv2
import numpy as np


class Pipline(ABC):
    '''
    preprocessing -> feature extractor -> matching -> postprocessing
    '''


    def __init__(
            self,
            preprocessing,
            feature_extractor,
            matching,
            postprocessing,
            clahe = None,
            clahe_mode: Literal["DYN", "ON", "OFF"] = "OFF"
        ) -> None:
        super().__init__()
        self.preprocessing = preprocessing
        self.feature_extractor = feature_extractor
        self.matching = matching
        self.postprocessing = postprocessing
        self.clahe = clahe
        self.clahe_mode = clahe_mode

        self.pipline = [
            self.preprocessing,
            self.feature_extractor,
            self.matching,
            self.postprocessing
        ]

    def compute(self, shared_objects, ij):
        layout, crop = self.preprocessing(shared_objects, ij)

        crop_keypoints, crop_descriptors = self.feature_extractor(crop)
        layout_keypoints, layout_descriptors = self.feature_extractor(layout)

        if len(layout_keypoints) < 2:
            # print("layout don't have keypoints")
            return None, None, None, None
        if len(crop_keypoints) < 2:
            # print("crop don't have keypoints")
            return None, None, None, None
        # matches = self.matching(layout_descriptors, crop_descriptors)
        matches = self.matching(crop_descriptors, layout_descriptors)

        output = self.postprocessing(shared_objects, ij, layout_keypoints, crop_keypoints, matches)

        return output#, (layout_keypoints, layout_descriptors, crop_keypoints, crop_descriptors)

    def compute_mp(self, shared_objects, ij):
        layout, crop = self.preprocessing(shared_objects, ij)

        crop_keypoints, crop_descriptors = self.feature_extractor(crop)
        layout_keypoints, layout_descriptors = self.feature_extractor(layout)

        recompute = False
        if self.clahe:
            if (self.clahe_mode == "DYN" and len(layout_keypoints) < 10000) or self.clahe_mode == "ON":
                layout = self.clahe.apply(layout)
            
                # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                # layout = cv2.filter2D(layout, -1, sharpen_kernel)
                # layout = cv2.Sobel(layout,cv2.CV_8U,0,1, ksize=5) 
                # layout = cv2.Laplacian(layout, cv2.CV_8U) 
                # (thresh, crop) = cv2.threshold(crop, 128, 255, cv2.THRESH_BINARY)
                recompute = True

            if (self.clahe_mode == "DYN" and len(crop_keypoints) < 10000) or self.clahe_mode == "ON":
                
                crop = self.clahe.apply(crop)
                # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                # crop = cv2.filter2D(crop, -1, sharpen_kernel)
                # crop = cv2.Sobel(crop,cv2.CV_8U,0,1, ksize=5) 
                # crop = cv2.Laplacian(crop, cv2.CV_8U) 

                # (thresh, crop) = cv2.threshold(crop, 128, 255, cv2.THRESH_BINARY_INV)
                recompute = True
        if recompute:
            crop_keypoints, crop_descriptors = self.feature_extractor(crop)
            layout_keypoints, layout_descriptors = self.feature_extractor(layout)

        if len(layout_keypoints) < 2:
            # print("layout don't have keypoints")
            return (ij, 0, 0)
        if len(crop_keypoints) < 2:
            # print("crop don't have keypoints")
            return (ij, 0, 0)

        # matches = self.matching(layout_descriptors, crop_descriptors)
        matches = self.matching(crop_descriptors, layout_descriptors)

        output = self.postprocessing(shared_objects, ij, layout_keypoints, crop_keypoints, matches)

        return output#, (layout_keypoints, layout_descriptors, crop_keypoints, crop_descriptors)
