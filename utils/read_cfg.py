import yaml
from argparse import Namespace


def read_yaml(path="./cfg/base.yaml"):
    with open(path, "r") as fr:
        cfg = yaml.load(fr, Loader=yaml.FullLoader)

    return Namespace(**cfg)