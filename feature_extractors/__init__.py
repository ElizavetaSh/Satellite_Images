from .mock import MOCK
from .orb import ORB
from .sift import SIFT
from .asift import ASIFT



FEATURE_EXTRACTORS = {
    "MOCK": MOCK,
    "ORB": ORB,
    "SIFT": SIFT,
    "ASIFT": ASIFT
}