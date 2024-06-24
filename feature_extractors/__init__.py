from .mock import MOCK
from .orb import ORB
from .sift import SIFT, SURF
from .asift import ASIFT



FEATURE_EXTRACTORS = {
    "MOCK": MOCK,
    "ORB": ORB,
    "SIFT": SIFT,
    "ASIFT": ASIFT,
    "SURF":SURF
}