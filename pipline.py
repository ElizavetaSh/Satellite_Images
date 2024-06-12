from abc import ABC, abstractmethod


class Pipline(ABC):
    '''
    preprocessing -> feature extractor -> matching -> postprocessing
    '''


    def __init__(
            self,
            preprocessing,
            feature_extractor,
            matching,
            postprocessing
        ) -> None:
        super().__init__()
        self.preprocessing = preprocessing
        self.feature_extractor = feature_extractor
        self.matching = matching
        self.postprocessing = postprocessing

        self.pipline = [
            self.preprocessing,
            self.feature_extractor,
            self.matching,
            self.postprocessing
        ]

    # @abstractmethod
    # def preprocessing(self, crop, layout):
    #     raise NotImplementedError

    # @abstractmethod
    # def feature_extractor(self, crop, layout):
    #     raise NotImplementedError

    # @abstractmethod
    # def matching(self, crop, layout):
    #     raise NotImplementedError

    # @abstractmethod
    # def postprocessing(self, crop, layout):
    #     raise NotImplementedError

    # @abstractmethod
    def compute(self, layout, crop):
        # x = (crop, layout)
        # for stage in self.pipline:
        #     x = stage(x)

        layout, crop = self.preprocessing(layout, crop)

        crop_keypoints, crop_descriptors = self.feature_extractor(crop)
        layout_keypoints, layout_descriptors = self.feature_extractor(layout)

        if len(layout_keypoints) < 2:
            # print("layout don't have keypoints")
            return (), (crop_keypoints, crop_descriptors, layout_keypoints, layout_descriptors)
        if len(crop_keypoints) < 2:
            # print("crop don't have keypoints")
            return (), (crop_keypoints, crop_descriptors, layout_keypoints, layout_descriptors)
        matches = self.matching(layout_descriptors, crop_descriptors)

        matches = self.postprocessing(matches)

        return matches, (layout_keypoints, layout_descriptors, crop_keypoints, crop_descriptors)
