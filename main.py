import argparse
import shutil
import os

from utils.read_cfg import read_yaml
from inference import main


def parse():
    parser = argparse.ArgumentParser()
     
    parser.add_argument("--crop_name", type=str, default="./18.Sitronics/1_20/crop_0_0_0000.tif")
    parser.add_argument("--layout_name", type=str, default="./18.Sitronics/layouts/layout_2021-06-15.tif")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    CFG = read_yaml("./cfg/base.yaml")
    CFG.LAYOUT_DIR = args.layout_name
    CFG.INPUT_IMG_DIR = args.crop_name
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    os.makedirs(CFG.VIS_DIR, exist_ok=True)
    os.makedirs(CFG.TMP_DIR, exist_ok=True)
    main(CFG)
    try:
        shutil.rmtree(CFG.TMP_DIR)
    except:
        pass