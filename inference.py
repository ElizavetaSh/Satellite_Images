import argparse
import copy
import math
import os
import tempfile
import glob

from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import tqdm
from osgeo import gdal
from mpire import WorkerPool

from pipline import Pipline
from feature_extractors import FEATURE_EXTRACTORS
from matching import MATCHERS
from utils.processing_utils import (
    filter_matches,
    preprocessing_v1,
    postprocessing_v1,
    postprocessing_v2,
    read_test_data,
    save_crop,
    read_layout_by_ij,
    read_tif,
    save_coordinates
)



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument()
    return parser.parse_args()


def main(CFG):
    pipline_1 = Pipline(
        preprocessing=preprocessing_v1,
        feature_extractor=FEATURE_EXTRACTORS[CFG.FEATURE_EXTRACTOR["name"]](**CFG.FEATURE_EXTRACTOR["params"]),
        matching=MATCHERS[CFG.MATCHER["name"]](**CFG.MATCHER["params"]),
        postprocessing=postprocessing_v1,
        clahe=cv2.createCLAHE(**CFG.CLAHE) if CFG.CLAHE_MODE != "OFF" else None,
        clahe_mode=CFG.CLAHE_MODE
    )
    pipline_2 = Pipline(
        preprocessing=preprocessing_v1,
        feature_extractor=FEATURE_EXTRACTORS[CFG.FEATURE_EXTRACTOR["name"]](**CFG.FEATURE_EXTRACTOR["params"]),
        matching=MATCHERS[CFG.MATCHER["name"]](**CFG.MATCHER["params"]),
        postprocessing=postprocessing_v2,
        clahe=cv2.createCLAHE(**CFG.CLAHE) if CFG.CLAHE_MODE != "OFF" else None,
        clahe_mode=CFG.CLAHE_MODE
    )

    # crop_img, layout_img = read_test_data(tif_file_crop, tif_file_layout, CFG.CROP_W, CFG.CROP_H, center_x, center_y, angle)
    # layout_h, layout_w = layout_img.shape[:2]

    layout_paths = glob.glob(os.path.join(CFG.LAYOUT_DIR, "*"))
    crop_paths = glob.glob(os.path.join(CFG.INPUT_IMG_DIR, "*"))
    for layout_path in layout_paths:
        layout_img = read_tif(layout_path, norm=False, size_scale=CFG.LAYOUT_SCALE)
        layout_h, layout_w = layout_img.shape[:2]
        print("Матчинг для подложки: ", os.path.basename(layout_path))
        with tempfile.TemporaryDirectory(dir="./tmp") as tmp_crops_dir:
            print("Подготовока кропов подложки ...")
            ij_list = [[[i, j], CFG.STEP] for i in range(layout_h // CFG.STEP + int(layout_h % CFG.STEP != 0)) for j in range(layout_w // CFG.STEP + int(layout_w % CFG.STEP != 0))]
            shared_objects = tmp_crops_dir, layout_img
            with WorkerPool(n_jobs=CFG.N_WORKERS, shared_objects=shared_objects) as pool:
                pool.map_unordered(save_crop, ij_list, progress_bar=True)
            print("Подготовока кропов подложки ... Выполнено")

            for crop_path in crop_paths:
                crop_img = read_tif(crop_path, norm=False)
                print(f"Матчинг входной картинки {os.path.basename(crop_path)} и подложки ...")
                ij_list = [[[i, j]] for i in range(layout_h // CFG.STEP) for j in range(layout_w // CFG.STEP)]
                shared_objects = tmp_crops_dir, crop_img
                with WorkerPool(n_jobs=CFG.N_WORKERS, shared_objects=shared_objects) as pool:
                    layout_crops = pool.map_unordered(pipline_1.compute_mp, ij_list, progress_bar=True)

                print("Матчинг входной картинки {os.path.basename(crop_path)} и подложки ... Выполнено")
                if len(layout_crops) == 0:
                    print("ничего не подошло")
                    return

                matches = sorted(layout_crops, key=lambda x: x[1], reverse=True)[0]
                print('max matches: ', matches)

                if (matches[1] > CFG.NUM_MATCHES_THRESH) or (matches[1] > matches[2] * CFG.KP_PAIRS_TO_KP_CROP):
                    print("Определение координат ...")
                    i, j = matches[0]
                    shared_objects = tmp_crops_dir, crop_img
                    H = None
                    H, status, kp_pairs, matches = pipline_2.compute(shared_objects, [i, j])

                    if H is not None:
                        layout_img = read_layout_by_ij(tmp_crops_dir, [i, j])
                        h1, w1 = crop_img.shape[:2]
                        h2, w2 = layout_img.shape[:2]

                        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
                        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))

                        print("координаты", corners)
                        offset_x = j * CFG.STEP
                        offset_y = i * CFG.STEP
                        save_coordinates(
                            corners + (offset_x, offset_y),
                            os.path.join(
                                CFG.OUTPUT_DIR,
                                os.path.basename(crop_path).rsplit(".", 1)[0] + "_" + os.path.basename(layout_path).rsplit(".", 1)[0] + ".txt"
                            )
                        )


                        if CFG.VIS_RESULT:
                            for index in range(len(kp_pairs)):
                                element = kp_pairs[index]
                                kp1, kp2 = element

                                new_kp1 = cv2.KeyPoint(kp1.pt[0], kp1.pt[1], kp1.size)
                                new_kp2 = cv2.KeyPoint(kp2.pt[0], kp2.pt[1], kp2.size)

                                kp_pairs[index] = (new_kp1, new_kp2)
                            # DEBUG
                            crop_img = cv2.normalize(crop_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            layout_img = read_layout_by_ij(tmp_crops_dir, [i, j])
                            assert h2 == layout_img.shape[0] and w2 == layout_img.shape[1]

                            # Create visualized result image
                            vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)

                            vis[:h1, :w1] = crop_img
                            vis[:h2, w1:w1 + w2] = layout_img
                            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                            cv2.polylines(vis, [corners + (w1, 0)], True, (255, 255, 255), thickness=10) # TODO чет не работает

                            if status is None:
                                status = np.ones(len(kp_pairs), np.bool_)
                            p1, p2 = [], []

                            for kpp in kp_pairs:
                                p1.append(np.int32(kpp[0].pt))
                                p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

                            green = (0, 255, 0)

                            for (x1, y1), (x2, y2) in [(p1[i], p2[i]) for i in range(len(p1))]:
                                cv2.line(vis, (x1, y1), (x2, y2), green)

                            cv2.imwrite(
                                os.path.join(
                                    CFG.VIS_DIR,
                                    os.path.basename(crop_path).rsplit(".", 1)[0] + "_" + os.path.basename(layout_path).rsplit(".", 1)[0] + ".png"
                                ),
                                vis
                            )

                        print("Определение координат ... Выполнено")
                    else:
                        print("Определение координат ... Не найдены")

                else:
                    print("Определение координат ... Мало точек")
