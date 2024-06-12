import argparse
import copy
import math
import os
import tempfile

import cv2
import numpy as np
from osgeo import gdal

from pipline import Pipline
from feature_extractors import FEATURE_EXTRACTORS
from matching import MATCHERS
from utils.postprocessing_utils import filter_matches


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument()
    return parser.parse_args()

def main():
    pipline = Pipline(
        preprocessing=lambda *args: args,
        feature_extractor=FEATURE_EXTRACTORS["ASIFT"],
        matching=MATCHERS["FLANNMatcher"],
        postprocessing=lambda x: x#[:20]
    )

    tif_file_crop = "/mnt/e/WORK_DL/datasets/18. Sitronics/1_20/crop_0_0_0000.tif"
    tif_file_layout = "/mnt/e/WORK_DL/datasets/18. Sitronics/layouts/layout_2021-06-15.tif"

    ratio_1 = 1
    ratio_2 = 1
    clahe = cv2.createCLAHE(clipLimit=128, tileGridSize=(3,3))

    # crop_raster = gdal.Open(tif_file_crop)
    # layout_raster = gdal.Open(tif_file_layout)

    # x,y,a = data_df.loc[data_df['id'] == matches[0][0],['center_x','center_y','angle']].values[0]

    # crop_img = crop_raster.ReadAsArray().transpose(1, 2, 0)
    # layout_img = layout_raster.ReadAsArray().transpose(1, 2, 0)

    # crop_img = cv2.imread(tif_file_crop, cv2.IMREAD_GRAYSCALE)
    layout_img = cv2.imread(tif_file_layout, cv2.IMREAD_UNCHANGED)
    layout_img = cv2.resize(layout_img, (11000, 11000))
    # crop_img = cv2.resize(crop_img, (500, 500))
    layout_h, layout_w = layout_img.shape
 
    step = 1000
    with tempfile.TemporaryDirectory(dir="./tmp") as tmp_crops_dir:
        for i in range(layout_h // step):
            for j in range(layout_w // step):
                layout_crop = layout_img[
                    i * step:(i + 1) * step,
                    j * step:(j + 1) * step
                ]
                cv2.imwrite(os.path.join(tmp_crops_dir, f"{i}_{j}.tif"), layout_crop)

        del layout_img
        del layout_crop

        for i in range(layout_h // step):
            for j in range(layout_w // step):
                layout_img = cv2.imread(os.path.join(tmp_crops_dir, f"{i}_{j}.tif"), cv2.IMREAD_UNCHANGED)
                layout_img = cv2.normalize(layout_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # crop_img = cv2.imread(tif_file_crop, cv2.IMREAD_GRAYSCALE)
                crop_img = cv2.imread(tif_file_crop, cv2.IMREAD_UNCHANGED)
                layout_img = cv2.resize(layout_img, (step, step))
                crop_img = cv2.normalize(crop_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

                matches, (layout_keypoints, layout_descriptors, crop_keypoints, crop_descriptors) = pipline.compute(layout_img, crop_img)
                recompute = False
                # if len(layout_keypoints) < 10000:
                layout_img = clahe.apply(layout_img)
                recompute = True
                # if len(layout_keypoints) < 5000:
                crop_img = clahe.apply(crop_img)
                recompute = True
                if recompute:
                    matches, (layout_keypoints, layout_descriptors, crop_keypoints, crop_descriptors) = pipline.compute(layout_img, crop_img)

                layout_keypoints, crop_keypoints, kp_pairs, matches = filter_matches(layout_keypoints, crop_keypoints, matches, ratio=0.7)

                for index in range(len(crop_keypoints)):
                    pt = crop_keypoints[index]
                    crop_keypoints[index] = pt / ratio_1

                for index in range(len(layout_keypoints)):
                    pt = layout_keypoints[index]
                    layout_keypoints[index] = pt / ratio_2

                for index in range(len(kp_pairs)):
                    element = kp_pairs[index]
                    kp1, kp2 = element

                    new_kp1 = cv2.KeyPoint(kp1.pt[0] / ratio_1, kp1.pt[1] / ratio_1, kp1.size)
                    new_kp2 = cv2.KeyPoint(kp2.pt[0] / ratio_2, kp2.pt[1] / ratio_2, kp2.size)

                    kp_pairs[index] = (new_kp1, new_kp2)

                H = None
                if len(kp_pairs) >= 4:
                    H, status = cv2.findHomography(layout_keypoints, crop_keypoints, cv2.RANSAC, 100.0)
                    kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
                    matches = [m for m, flag in zip(matches, status) if flag]

                # final_img = cv2.drawMatches(layout_img, crop_keypoints[:50], crop_img, layout_keypoints[:50], matches[:20],None)

                # final_img = cv2.resize(final_img, (1000,650))

                if H is not None:
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

                    for (x1, y1), (x2, y2) in [(p1[i], p2[i]) for i in range(len(p1))]:
                        #if inlier:
                        cv2.line(vis, (x1, y1), (x2, y2), green)

                    cv2.imwrite(f"./preds/{i}_{j}.png", vis)

                    # Show the final image
                    # if len(matches) > 0:
                    #     print(len(matches))
                    #     crop_keypoints = crop_keypoints.tolist()
                    #     layout_keypoints = layout_keypoints.tolist()

                    #     for index in range(len(crop_keypoints)):
                    #         pt = crop_keypoints[index]
                    #         crop_keypoints[index] = cv2.KeyPoint(pt[0], pt[1], 2)

                    #     for index in range(len(layout_keypoints)):
                    #         pt = layout_keypoints[index]
                    #         layout_keypoints[index] = cv2.KeyPoint(pt[0], pt[1], 2)
                    #     # img_matches = np.empty((max(layout_img.shape[0],crop_img.shape[0]),
                    #     #                 layout_img.shape[1] + crop_img.shape[1], 1), dtype=np.uint8)

                    #     # final_img = cv2.drawMatches(layout_img, crop_keypoints, crop_img, layout_keypoints, matches,img_matches)
                    #     # layout_img = cv2.normalize(layout_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    #     # crop_img = cv2.imread(tif_file_crop, cv2.IMREAD_UNCHANGED)
                    #     # crop_img = cv2.normalize(crop_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    #     # layout_img = cv2.applyColorMap(layout_img, cv2.COLORMAP_JET)
                    #     # crop_img = cv2.applyColorMap(crop_img, cv2.COLORMAP_JET)
                    #     draw_params = dict(matchColor = (0,255,0),
                    #         singlePointColor = (255,0,0),
                    #         flags = 0)
                    #     final_img = cv2.drawMatchesKnn(layout_img, crop_keypoints, crop_img, layout_keypoints, matches, None, **draw_params)
                    #     cv2.imwrite(f"./preds/{i}_{j}.png", final_img)
                    # h1, w1 = crop_img.shape[:2]
                    # h2, w2 = layout_img.shape[:2]
                    # corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
                    # corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
                    # new_cx = corners[0][0]*0.5 + corners[2][0]*0.5 - w1
                    # new_cy = corners[0][1]*0.5 + corners[2][1]*0.5
                    # new_cx2 = corners[1][0]*0.5 + corners[2][0]*0.5 - w1
                    # new_cy2 = corners[1][1]*0.5 + corners[2][1]*0.5
                    # rot_mat = cv2.getRotationMatrix2D((w2*0.5, h2*0.5), -a, 1.0)
                    # cntr = np.matmul(rot_mat, np.array([new_cx, new_cy, 1])) 
                    # cntr2 = np.matmul(rot_mat, np.array([new_cx2, new_cy2, 1])) 
                    # da = math.atan2(cntr2[1] - cntr[1], cntr2[0] - cntr[0]) / math.pi * 180
                    # if da < 360:
                    #     da += 360
                    # if da > 360:
                    #     da -= 360
                    # xn1,yn1,an1 = int(x) + cntr[0] - w2*0.5, int(y) + cntr[1] - h2*0.5, da
                    pass
                    # new_preds[i_img] = (preds[i_img][0],xn1,yn1,an1)
                    # statuses[i_img] = True
                    # print(new_preds[i_img])
                    # d['new_preds'] = new_preds
                    # d['statuses'] = statuses
                    # joblib.dump(d, 'preds.joblib')

    print(len(matches))


if __name__ == "__main__":
    main()