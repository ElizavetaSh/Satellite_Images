import copy

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
