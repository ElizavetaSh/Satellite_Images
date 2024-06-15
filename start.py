import argparse
import shutil
import os

from utils.read_cfg import read_yaml
from inference import main


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./cfg/base.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    CFG = read_yaml(args.config_path)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    os.makedirs(CFG.VIS_DIR, exist_ok=True)
    os.makedirs(CFG.TMP_DIR, exist_ok=True)
    main(CFG)
    try:
        shutil.rmtree(CFG.TMP_DIR)
    except:
        pass