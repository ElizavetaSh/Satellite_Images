from .bf import BFMatcher
from .mock import MOCK
from .flann import FLANNMatcher

MATCHERS = {
    "BFMatcher": BFMatcher(),
    "MOCK": MOCK(),
    "FLANNMatcher": FLANNMatcher()
}