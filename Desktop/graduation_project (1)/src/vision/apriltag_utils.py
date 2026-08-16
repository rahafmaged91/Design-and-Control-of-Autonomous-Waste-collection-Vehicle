"""
vision/apriltag_utils.py
==========================
Camera-geometry helpers for the AprilTag centering state (STATE 2a):
building/scaling the calibrated camera matrix, undistort maps, CLAHE
contrast + denoise preprocessing, a multi-pass detector call (raw ->
CLAHE -> CLAHE+denoise), and pose/yaw extraction from a detected tag.
"""

import math

import cv2
import numpy as np

from config import (
    TAG_FOCAL_LENGTH_X, TAG_FOCAL_LENGTH_Y, TAG_CENTER_X, TAG_CENTER_Y,
    TAG_CLAHE_CLIP_LIMIT, TAG_CLAHE_GRID_SIZE, TAG_DENOISE_D,
    TAG_DENOISE_SIGMA_COLOR, TAG_DENOISE_SIGMA_SPACE, TAG_MIN_MARGIN,
    TAG_SIZE_METERS,
)

# ─── AprilTag camera geometry helpers ────────────────────────────────────────

def build_base_tag_camera_matrix():
    return np.array([
        [TAG_FOCAL_LENGTH_X,               0.0, TAG_CENTER_X],
        [               0.0, TAG_FOCAL_LENGTH_Y, TAG_CENTER_Y],
        [               0.0,               0.0,          1.0],
    ], dtype=np.float64)


TAG_DIST_COEFFS = np.array(
    [0.144962, -0.153111, 0.000437, 0.000722, 0.031279], dtype=np.float64)


def scale_tag_camera_matrix(K, from_size, to_size):
    if from_size == to_size:
        return K.copy()
    fw, fh = from_size
    tw, th = to_size
    sx, sy = tw / fw, th / fh
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2


def build_tag_undistort_maps(cam_mat, dist, frame_size):
    new_cam, _ = cv2.getOptimalNewCameraMatrix(cam_mat, dist, frame_size, alpha=0.0)
    m1, m2 = cv2.initUndistortRectifyMap(cam_mat, dist, None, new_cam,
                                         frame_size, cv2.CV_16SC2)
    return m1, m2, new_cam


def apply_tag_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=TAG_CLAHE_CLIP_LIMIT, tileGridSize=TAG_CLAHE_GRID_SIZE)
    return clahe.apply(gray)


def apply_tag_denoise(gray):
    return cv2.bilateralFilter(gray, TAG_DENOISE_D, TAG_DENOISE_SIGMA_COLOR,
                               TAG_DENOISE_SIGMA_SPACE)


def run_tag_detector_multipass(detector, gray):
    """Try progressively harder image enhancements until a confident
    detection appears."""
    results = detector.detect(gray)
    if any(r.decision_margin >= TAG_MIN_MARGIN for r in results):
        return results
    enhanced = apply_tag_clahe(gray)
    results2 = detector.detect(enhanced)
    if any(r.decision_margin >= TAG_MIN_MARGIN for r in results2):
        return results2
    denoised = apply_tag_clahe(apply_tag_denoise(gray))
    results3 = detector.detect(denoised)
    if any(r.decision_margin >= TAG_MIN_MARGIN for r in results3):
        return results3
    return max([results, results2, results3],
               key=lambda r: (len(r), max([d.decision_margin for d in r], default=0)))


def get_tag_pose(image_points, camera_matrix):
    half = TAG_SIZE_METERS / 2.0
    object_points = np.array([
        [-half, -half, 0], [half, -half, 0],
        [half, half, 0], [-half, half, 0],
    ], dtype=np.float64)
    image_pts = np.array(image_points, dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(object_points, image_pts, camera_matrix, TAG_DIST_COEFFS)
    return success, rvec, tvec


def get_tag_yaw(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return math.degrees(math.atan2(R[1][0], R[0][0]))


class TagObservation:
    """Everything a mission state needs to know about the best tag seen in
    the current 8080-stream frame."""
    __slots__ = ("tag_id", "margin", "corners_px", "center_px",
                "distance_mm", "yaw_deg", "pose_ok")

    def __init__(self):
        self.tag_id = None
        self.margin = 0.0
        self.corners_px = None
        self.center_px = None
        self.distance_mm = None
        self.yaw_deg = None
        self.pose_ok = False


