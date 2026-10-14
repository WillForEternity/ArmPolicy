"""Box pose from the overhead camera by excluding the dark tablecloth.

Everything brighter than the cloth is a candidate; the black arm is excluded along with the cloth.
Blobs touching the image border (cardboard base, glare at the edge) are dropped, and the largest
remaining roughly-rectangular blob is the box. Bright area that touches the border but wasn't there
when the episode started is the person's hand reaching in, which tells the policy when to release
the box at handover. The same code labels recorded demos offline and runs live on the UNO Q, so the
policy sees identical features in training and deployment.

    python -m policy.perception IMAGE [--out annotated.jpg]
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WORK_WIDTH = 480  # frames are downscaled to this width before thresholding
BOX_FEATURES = ("x", "y", "cos2a", "sin2a", "long", "short")
ENV_FEATURES = BOX_FEATURES + ("hand",)


@dataclass
class BoxPose:
    cx: float  # center, full-resolution pixels
    cy: float
    long: float  # side lengths, full-resolution pixels
    short: float
    angle: float  # direction of the long side, degrees in [-90, 90)
    area: float  # blob area, full-resolution pixels^2

    def corners(self) -> np.ndarray:
        return cv2.boxPoints(((self.cx, self.cy), (self.long, self.short), self.angle))


def _value(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (WORK_WIDTH, round(h * WORK_WIDTH / w)), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 2], (5, 5), 0)


def cloth_threshold(frame_bgr: np.ndarray) -> float:
    """Otsu split between cloth and everything else; the floor keeps a cloth-only frame from splitting its own texture."""
    otsu, _ = cv2.threshold(_value(frame_bgr), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(otsu, 90.0)


def segment(frame_bgr: np.ndarray, thresh: float | None = None) -> np.ndarray:
    """Binary mask (WORK_WIDTH wide) of everything brighter than the tablecloth."""
    if thresh is None:
        thresh = cloth_threshold(frame_bgr)
    _, mask = cv2.threshold(_value(frame_bgr), thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def border_mask(mask: np.ndarray) -> np.ndarray:
    """Pixels of the blobs in `mask` that touch the image border."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    sh, sw = mask.shape
    x, y, bw, bh = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
    touching = (x <= 1) | (y <= 1) | (x + bw >= sw - 1) | (y + bh >= sh - 1)
    touching[0] = False  # background label
    return touching[labels]


def detect(frame_bgr: np.ndarray, mask: np.ndarray | None = None, min_area_frac=0.002, max_area_frac=0.25, min_fill=0.7) -> BoxPose | None:
    if mask is None:
        mask = segment(frame_bgr)
    scale = mask.shape[1] / frame_bgr.shape[1]
    sh, sw = mask.shape
    best = None
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, bw, bh = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + bw >= sw - 1 or y + bh >= sh - 1:
            continue
        area = cv2.contourArea(c)
        if not min_area_frac * sw * sh <= area <= max_area_frac * sw * sh:
            continue
        rect = cv2.minAreaRect(c)
        if area < min_fill * rect[1][0] * rect[1][1]:
            continue
        if best is None or area > best[0]:
            best = (area, rect)
    if best is None:
        return None

    area, ((cx, cy), (rw, rh), angle) = best
    if rw < rh:  # make the first side the long one
        rw, rh, angle = rh, rw, angle + 90
    angle = (angle + 90) % 180 - 90
    s = 1 / scale
    return BoxPose(cx * s, cy * s, rw * s, rh * s, angle, area * s * s)


class BoxTracker:
    """Holds the box pose seen with the arm clear, so the gripper covering the box can't corrupt it.

    A new detection replaces the held pose only if its size matches the held one; once the gripper
    occludes or lifts the box, the size changes and the last clean pose is kept.
    """

    def __init__(self, size_tol=0.2):
        self.size_tol = size_tol
        self.pose: BoxPose | None = None

    def reset(self) -> None:
        self.pose = None

    def update(self, frame_bgr: np.ndarray, mask: np.ndarray | None = None) -> BoxPose | None:
        new = detect(frame_bgr, mask)
        if new is not None and (self.pose is None or self._same_size(new)):
            self.pose = new
        return self.pose

    def _same_size(self, new: BoxPose) -> bool:
        tol = self.size_tol
        return abs(new.long / self.pose.long - 1) < tol and abs(new.short / self.pose.short - 1) < tol


class HandDetector:
    """Fraction of the image covered by new bright blobs touching the border (a hand reaching in).

    The first frame of an episode is the baseline, so static border blobs (cardboard, glare) don't count.
    """

    def __init__(self):
        self.baseline: np.ndarray | None = None

    def reset(self) -> None:
        self.baseline = None

    def update(self, mask: np.ndarray) -> float:
        border = border_mask(mask)
        if self.baseline is None:
            self.baseline = cv2.dilate(border.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        return float((border & ~self.baseline).mean())


class TableFrame:
    """Maps image pixels to table centimeters with a homography from 4 marked points.

    table_calib.json: {"image_px": [[u, v] x4], "table_cm": [[x, y] x4]}. Without it, coordinates
    fall back to pixels normalized to [-1, 1] (fine for real-only training, not for sim co-training).
    """

    def __init__(self, image_size: tuple[int, int], calib_path: Path | None = None):
        self.w, self.h = image_size
        self.H = None
        if calib_path and calib_path.exists():
            cal = json.loads(calib_path.read_text())
            self.H = cv2.getPerspectiveTransform(np.float32(cal["image_px"]), np.float32(cal["table_cm"]))

    def to_table(self, pts_px: np.ndarray) -> np.ndarray:
        pts_px = np.asarray(pts_px, np.float32).reshape(-1, 2)
        if self.H is None:
            return pts_px / [self.w / 2, self.h / 2] - 1
        return cv2.perspectiveTransform(pts_px[None], self.H)[0]

    def features(self, pose: BoxPose | None) -> np.ndarray:
        """[x, y, cos 2a, sin 2a, long, short] in table units; 2a because a rectangle is symmetric under 180 deg."""
        if pose is None:
            return np.zeros(len(BOX_FEATURES), np.float32)
        c = self.to_table(pose.corners())  # measure in table space so perspective doesn't skew the angle
        center = c.mean(0)
        e1, e2 = c[1] - c[0], c[2] - c[1]
        if np.linalg.norm(e1) < np.linalg.norm(e2):
            e1, e2 = e2, e1
        a = math.atan2(e1[1], e1[0])
        return np.float32([center[0], center[1], math.cos(2 * a), math.sin(2 * a), np.linalg.norm(e1), np.linalg.norm(e2)])


class Perception:
    """Frame -> env features [box pose (6), hand]; call reset() at the start of every episode."""

    HAND_SCALE = 20.0  # hand fraction is ~0-0.05; scale it to ~0-1 before normalization

    def __init__(self, image_size: tuple[int, int], calib_path: Path | None = None):
        self.table = TableFrame(image_size, calib_path)
        self.tracker, self.hand = BoxTracker(), HandDetector()
        self.thresh: float | None = None

    def reset(self) -> None:
        self.tracker.reset()
        self.hand.reset()
        self.thresh = None

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.thresh is None:  # fixed per episode so the hand signal only reacts to new objects
            self.thresh = cloth_threshold(frame_bgr)
        mask = segment(frame_bgr, self.thresh)
        pose = self.tracker.update(frame_bgr, mask)
        hand = self.hand.update(mask) * self.HAND_SCALE
        return np.append(self.table.features(pose), np.float32(hand)).astype(np.float32)


def annotate(frame_bgr: np.ndarray, pose: BoxPose | None) -> np.ndarray:
    out = frame_bgr.copy()
    if pose is not None:
        cv2.polylines(out, [np.int32(pose.corners())], True, (0, 0, 255), 3)
        cv2.putText(out, f"{pose.angle:.1f} deg", (int(pose.cx), int(pose.cy)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--out")
    args = p.parse_args()
    frame = cv2.imread(args.image)
    pose = detect(frame)
    print(pose)
    if pose is not None:
        print(dict(zip(BOX_FEATURES, TableFrame((frame.shape[1], frame.shape[0])).features(pose).round(3).tolist())))
    if args.out:
        cv2.imwrite(args.out, annotate(frame, pose))
