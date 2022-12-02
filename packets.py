########################################
# Code by Benjamin Tilbury             #
#       KTP Associate with UWS/Kibble  #
########################################

from dataclasses import dataclass, field
from typing import List

import numpy as np

@dataclass
class FPtoIR_Packet:
    kps: np.ndarray = None
    vels: np.ndarray = None
    vis: np.ndarray = None
    normals: np.ndarray = None
    plane: np.ndarray = None
    good_vis: bool = False
    # scale_coeff: float = 0
 