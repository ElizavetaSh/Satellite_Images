from typing import Any

class MOCK(object):
    def __init__(self) -> None:
        self.method = None

    def __call__(self, img) -> Any:
        return img