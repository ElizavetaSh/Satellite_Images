import argparse
import copy
import math
import os
import tempfile

from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import tqdm
from osgeo import gdal
from mpire import WorkerPool

from pipline import Pipline
from feature_extractors import FEATURE_EXTRACTORS
from matching import MATCHERS
from utils.postprocessing_utils import filter_matches



# tif_file_crop = "/mnt/e/WORK_DL/datasets/18. Sitronics/1_20/crop_0_0_0000.tif"
tif_file_crop = "/mnt/e/WORK_DL/datasets/18. Sitronics/layouts/layout_2021-06-15.tif"
tif_file_layout = "/mnt/e/WORK_DL/datasets/18. Sitronics/layouts/layout_2021-06-15.tif"

n_jobs = 8

step = 1000

crop_w = 250
crop_h = 250
angle = 45
center_x = 500
center_y = 500
center = (center_x, center_y)

clahe = cv2.createCLAHE(clipLimit=128, tileGridSize=(3,3))


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument()
    return parser.parse_args()

def read_test_data():
    crop_img = cv2.imread(tif_file_crop, cv2.IMREAD_UNCHANGED)
    # crop_img = cv2.normalize(crop_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    crop_img = crop_img[
        center_y - crop_h // 2:center_y + crop_h // 2,
        center_x - crop_w // 2:center_x + crop_w // 2
    ]
    height, width = crop_img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((crop_w // 2, crop_h // 2), angle, 1.0)
    crop_img = cv2.warpAffine(crop_img, rot_mat,(width, height))
    # crop_img = cv2.normalize(crop_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    layout_img = cv2.imread(tif_file_layout, cv2.IMREAD_UNCHANGED)

    return crop_img, layout_img

def preprocessing_v2(layout_img, crop_img):
    layout_img = cv2.normalize(layout_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    crop_img = cv2.normalize(crop_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    layout_img = clahe.apply(layout_img)
    crop_img = clahe.apply(crop_img)
    return layout_img, crop_img

def preprocessing_v1(shared_objects, ij):
    tmp_crops_dir, crop_img = shared_objects
    i, j = ij
    layout_img = cv2.imread(os.path.join(tmp_crops_dir, f"{i}_{j}.tif"), cv2.IMREAD_UNCHANGED)
    layout_img = cv2.normalize(layout_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    crop_img = cv2.normalize(crop_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    layout_img = clahe.apply(layout_img)
    crop_img = clahe.apply(crop_img)
    return layout_img, crop_img

def read_layout_by_ij(tmp_dir, ij):
    i, j = ij
    layout_img = cv2.imread(os.path.join(tmp_dir, f"{i}_{j}.tif"), cv2.IMREAD_UNCHANGED)
    layout_img = cv2.normalize(layout_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    layout_img = clahe.apply(layout_img)
    return layout_img

def postprocessing_v1(shared_objects, ij, layout_keypoints, crop_keypoints, matches):
    layout_keypoints, crop_keypoints, kp_pairs, matches = filter_matches(layout_keypoints, crop_keypoints, matches, ratio=0.7)

    H = None
    if len(kp_pairs) >= 4:
        H, status = cv2.findHomography(layout_keypoints, crop_keypoints, cv2.RANSAC, 100.0)
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
        # matches = [m for m, flag in zip(matches, status) if flag]
        return (ij, len(kp_pairs), len(crop_keypoints))
    return (ij, 0, 0)
        # layout_crops.append(((i, j), len(kp_pairs)))

def postprocessing_v2(shared_objects, ij, layout_keypoints, crop_keypoints, matches):
    layout_keypoints, crop_keypoints, kp_pairs, matches = filter_matches(layout_keypoints, crop_keypoints, matches, ratio=0.7)
    H = None
    if len(kp_pairs) >= 4:
        H, status = cv2.findHomography(layout_keypoints, crop_keypoints, cv2.RANSAC, 8.0)
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
        matches = [m for m, flag in zip(matches, status) if flag]
    return H, status, kp_pairs, matches

def save_crop(shared_objects, ij):
    tmp_dir, img = shared_objects
    i, j = ij
    layout_crop = img[
        i * step:(i + 1) * step,
        j * step:(j + 1) * step
    ]
    cv2.imwrite(os.path.join(tmp_dir, f"{i}_{j}.tif"), layout_crop)


def main():
    pipline_1 = Pipline(
        preprocessing=preprocessing_v1,
        feature_extractor=FEATURE_EXTRACTORS["ASIFT"],
        matching=MATCHERS["FLANNMatcher"],
        postprocessing=postprocessing_v1
    )
    pipline_2 = Pipline(
        preprocessing=preprocessing_v1,
        feature_extractor=FEATURE_EXTRACTORS["ASIFT"],
        matching=MATCHERS["FLANNMatcher"],
        postprocessing=postprocessing_v2
    )
    # clahe = cv2.createCLAHE(clipLimit=128, tileGridSize=(3,3))

    crop_img, layout_img = read_test_data()
    layout_h, layout_w = layout_img.shape[:2]

    with tempfile.TemporaryDirectory(dir="./tmp") as tmp_crops_dir:
        print("Подготовока кропов подложки ...")
        ij_list = [[[i, j]] for i in range(layout_h // step + int(layout_h % step != 0)) for j in range(layout_w // step + int(layout_w % step != 0))]
        shared_objects = tmp_crops_dir, layout_img
        with WorkerPool(n_jobs=n_jobs, shared_objects=shared_objects) as pool:
            pool.map_unordered(save_crop, ij_list, progress_bar=True)
        print("Подготовока кропов подложки ... Выполнено")

        print("Матчинг кропа и кропов подложки ...")
        ij_list = [[[i, j]] for i in range(layout_h // step) for j in range(layout_w // step)]
        shared_objects = tmp_crops_dir, crop_img
        with WorkerPool(n_jobs=n_jobs, shared_objects=shared_objects) as pool:
            layout_crops = pool.map_unordered(pipline_1.compute_mp, ij_list, progress_bar=True)
        #     matches, (layout_keypoints, layout_descriptors, crop_keypoints, crop_descriptors) = pipline.compute(layout_img, crop_img)
        #     recompute = False
        #     if len(layout_keypoints) < 10000:
        #         layout_img = clahe.apply(layout_img)
        #         recompute = True
        #     if len(layout_keypoints) < 10000:
        #         crop_img = clahe.apply(crop_img)
        #         recompute = True
        #     if recompute:
        #         matches, (layout_keypoints, layout_descriptors, crop_keypoints, crop_descriptors) = pipline.compute(layout_img, crop_img)

        print("Матчинг кропа и кропов подложки ... Выполнено")
        if len(layout_crops) == 0:
            print("ничего не подошло")
            return

        matches = sorted(layout_crops, key=lambda x: x[1], reverse=True)[0]
        print('max matches: ', matches)

        # crop_keypoints = [p.pt for p in crop_keypoints]

        if (matches[1] > 250) or (matches[1] > matches[2] * 0.5):
            print("Матчинг ...")
            MAX_SIZE = 1024

            i, j = matches[0]
            shared_objects = tmp_crops_dir, crop_img

            ratio_1 = 1
            ratio_2 = 1

            H, status, kp_pairs, matches = pipline_2.compute(shared_objects, [i, j])
            # recompute = False
            # if len(layout_keypoints) < 10000:
            #     layout_img = clahe.apply(layout_img)
            #     recompute = True
            # if len(layout_keypoints) < 10000:
            #     crop_img = clahe.apply(crop_img)
            #     recompute = True
            # if recompute:
            #     matches, (layout_keypoints, layout_descriptors, crop_keypoints, crop_descriptors) = pipline_2.compute(layout_img, crop_img)

            # layout_keypoints, crop_keypoints, kp_pairs, matches = filter_matches(layout_keypoints, crop_keypoints, matches, ratio=0.7)

            # for index in range(len(crop_keypoints)):
            #     pt = crop_keypoints[index]
            #     crop_keypoints[index] = pt / ratio_1

            # for index in range(len(layout_keypoints)):
            #     pt = layout_keypoints[index]
            #     layout_keypoints[index] = pt / ratio_2

            # for index in range(len(kp_pairs)):
            #     element = kp_pairs[index]
            #     kp1, kp2 = element

            #     new_kp1 = cv2.KeyPoint(kp1.pt[0] / ratio_1, kp1.pt[1] / ratio_1, kp1.size)
            #     new_kp2 = cv2.KeyPoint(kp2.pt[0] / ratio_2, kp2.pt[1] / ratio_2, kp2.size)

            #     kp_pairs[index] = (new_kp1, new_kp2)

            for index in range(len(kp_pairs)):
                element = kp_pairs[index]
                kp1, kp2 = element

                new_kp1 = cv2.KeyPoint(kp1.pt[0], kp1.pt[1], kp1.size)
                new_kp2 = cv2.KeyPoint(kp2.pt[0], kp2.pt[1], kp2.size)

                kp_pairs[index] = (new_kp1, new_kp2)

            if H is not None:
                layout_img = read_layout_by_ij(tmp_crops_dir, [i, j])
                h1, w1 = layout_img.shape[:2]
                h2, w2 = crop_img.shape[:2]

                # Create visualized result image
                vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
                vis[:h1, :w1] = layout_img
                vis[:h2, w1:w1 + w2] = crop_img
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
                corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
                cv2.polylines(vis, [corners], True, (255, 255, 255), thickness=10)

                if status is None:
                    status = np.ones(len(kp_pairs), np.bool_)
                p1, p2 = [], []  # python 2 / python 3 change of zip unpacking

                for kpp in kp_pairs:
                    p1.append(np.int32(kpp[0].pt))
                    p2.append(np.int32(np.array(kpp[1].pt) * ratio_2 + [w1, 0]))

                green = (0, 255, 0)

                for (x1, y1), (x2, y2) in [(p1[i], p2[i]) for i in range(len(p1))][:30]:
                    #if inlier:
                    cv2.line(vis, (x1, y1), (x2, y2), green)

                cv2.imwrite(f"./preds/{i}_{j}.png", vis)


                new_cx = corners[0][0]*0.5 + corners[2][0]*0.5 - w1
                new_cy = corners[0][1]*0.5 + corners[2][1]*0.5
                new_cx2 = corners[1][0]*0.5 + corners[2][0]*0.5 - w1
                new_cy2 = corners[1][1]*0.5 + corners[2][1]*0.5
                rot_mat = cv2.getRotationMatrix2D((w2*0.5, h2*0.5), -angle, 1.0)
                cntr = np.matmul(rot_mat, np.array([new_cx, new_cy, 1]))
                cntr2 = np.matmul(rot_mat, np.array([new_cx2, new_cy2, 1]))
                da = math.atan2(cntr2[1] - cntr[1], cntr2[0] - cntr[0]) / math.pi * 180
                if da < 360:
                    da += 360
                if da > 360:
                    da -= 360
                xn1,yn1,an1 = int(center_x) + cntr[0] - w2*0.5, int(center_y) + cntr[1] - h2*0.5, da

                print(f"markup: center_x = {center_x}, center_y = {center_y}, angle = {angle}")
                print(f"prediction: center_x = {xn1}, center_y = {yn1}, angle = {an1}")

            print("Матчинг ... Выполнено")


if __name__ == "__main__":
    main()