import copy
import os

import cv2
import numpy as np


def filter_matches(kp1, kp2, matches, ratio=0.7):
    mkp1, mkp2, matches_ = [], [], []

    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches_.append(m)
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])

    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)

    return p1, p2, list(kp_pairs), matches_


def read_test_data(tif_file_crop, tif_file_layout, crop_w, crop_h, center_x, center_y, angle):
    crop_img = cv2.imread(tif_file_crop, cv2.IMREAD_UNCHANGED)
    height, width = crop_img.shape[:2]
    # crop_img = cv2.normalize(crop_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    rot_mat = cv2.getRotationMatrix2D((crop_w // 2, crop_h // 2), angle, 1.0)
    crop_img = cv2.warpAffine(crop_img, rot_mat,(width, height))
    crop_img = crop_img[
        center_y - crop_h // 2:center_y + crop_h // 2,
        center_x - crop_w // 2:center_x + crop_w // 2
    ]
    height, width = crop_img.shape[:2]
    # rot_mat = cv2.getRotationMatrix2D((crop_w // 2, crop_h // 2), angle, 1.0)
    # crop_img = cv2.warpAffine(crop_img, rot_mat,(width, height))
    # crop_img = cv2.normalize(crop_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    layout_img = cv2.imread(tif_file_layout, cv2.IMREAD_UNCHANGED)

    return crop_img, layout_img

def read_tif(path, norm=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if norm:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img


def preprocessing_v2(layout_img, crop_img):
    layout_img = cv2.normalize(layout_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    crop_img = cv2.normalize(crop_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # layout_img = clahe.apply(layout_img)
    # crop_img = clahe.apply(crop_img)
    return layout_img, crop_img

def preprocessing_v1(shared_objects, ij):
    tmp_crops_dir, crop_img = shared_objects
    i, j = ij
    layout_img = cv2.imread(os.path.join(tmp_crops_dir, f"{i}_{j}.tif"), cv2.IMREAD_UNCHANGED)
    layout_img = cv2.normalize(layout_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    crop_img = cv2.normalize(crop_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # layout_img = clahe.apply(layout_img)
    # crop_img = clahe.apply(crop_img)
    return layout_img, crop_img

def read_layout_by_ij(tmp_dir, ij):
    i, j = ij
    layout_img = cv2.imread(os.path.join(tmp_dir, f"{i}_{j}.tif"), cv2.IMREAD_UNCHANGED)
    layout_img = cv2.normalize(layout_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # layout_img = clahe.apply(layout_img)
    return layout_img

def postprocessing_v1(shared_objects, ij, layout_keypoints, crop_keypoints, matches):
    crop_keypoints, layout_keypoints, kp_pairs, matches = filter_matches(crop_keypoints, layout_keypoints, matches, ratio=0.7)

    H = None
    if len(kp_pairs) >= 4:
        # H, status = cv2.findHomography(layout_keypoints, crop_keypoints, cv2.RANSAC, 100.0)
        H, status = cv2.findHomography(crop_keypoints, layout_keypoints, cv2.RANSAC, 100.0)
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
        # matches = [m for m, flag in zip(matches, status) if flag]
        return (ij, len(kp_pairs), len(crop_keypoints))
    return (ij, 0, 0)
        # layout_crops.append(((i, j), len(kp_pairs)))

def postprocessing_v2(shared_objects, ij, layout_keypoints, crop_keypoints, matches):
    crop_keypoints, layout_keypoints, kp_pairs, matches = filter_matches(crop_keypoints, layout_keypoints, matches, ratio=0.7)
    H = None
    if len(kp_pairs) >= 4:
        # H, status = cv2.findHomography(layout_keypoints, crop_keypoints, cv2.RANSAC, 8.0)
        H, status = cv2.findHomography(crop_keypoints, layout_keypoints, cv2.RANSAC, 8.0)
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
        matches = [m for m, flag in zip(matches, status) if flag]
        return H, status, kp_pairs, matches
    return None, None, None, None

def save_crop(shared_objects, ij, step):
    tmp_dir, img = shared_objects
    i, j = ij
    layout_crop = img[
        i * step:(i + 1) * step,
        j * step:(j + 1) * step
    ]
    cv2.imwrite(os.path.join(tmp_dir, f"{i}_{j}.tif"), layout_crop)


def save_coordinates(coords, path):
    with open(path, "w") as fw:
        for x,y in coords:
            fw.write(f"{x};{y}\n")
    return True