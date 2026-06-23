"""
Decompose MediaPipe's facial transformation matrix into head rotation + position.

FaceLandmarker, when created with output_facial_transformation_matrixes=True,
returns a 4x4 matrix per face that maps the canonical face model into camera
space. We split that into Euler rotation (degrees) and translation so the Godot
viewer can drive the head node's orientation and position.
"""

from __future__ import annotations

import math

# A pose with no rotation and no translation.
ZERO_POSE = {"rot": [0.0, 0.0, 0.0], "pos": [0.0, 0.0, 0.0]}


def decompose(matrix) -> dict:
    """Return {"rot": [x, y, z] degrees, "pos": [x, y, z]} from a 4x4 matrix.

    *matrix* is anything indexable as matrix[row][col] (numpy array or nested
    list). Rotation is extracted as an XYZ Euler triple in degrees.
    """
    r00, r01, r02 = matrix[0][0], matrix[0][1], matrix[0][2]
    r10, r11, r12 = matrix[1][0], matrix[1][1], matrix[1][2]
    r20, r21, r22 = matrix[2][0], matrix[2][1], matrix[2][2]
    tx, ty, tz = matrix[0][3], matrix[1][3], matrix[2][3]

    sy = math.sqrt(r00 * r00 + r10 * r10)
    if sy > 1e-6:
        x = math.atan2(r21, r22)
        y = math.atan2(-r20, sy)
        z = math.atan2(r10, r00)
    else:  # gimbal lock
        x = math.atan2(-r12, r11)
        y = math.atan2(-r20, sy)
        z = 0.0

    return {
        "rot": [math.degrees(x), math.degrees(y), math.degrees(z)],
        "pos": [float(tx), float(ty), float(tz)],
    }
