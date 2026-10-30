"""Pitch homography: fitting, validation, temporal smoothing, camera recovery.

Why this is more than one ``cv2.findHomography`` call:

* **The pose output is noisy frame to frame.** Four points are enough to
  determine a homography exactly, so a single slightly-wrong keypoint shifts
  every projected player. Smoothing the fit over time removes most of the
  jitter that would otherwise show up as players teleporting on the map.
* **A bad fit is not obviously bad.** Every homography "works" -- it just maps
  the pitch somewhere implausible. Validation has to check the *result*
  (reprojection error, where the pitch corners land) rather than the solver's
  return value.
* **The camera is recoverable.** With a known pitch and a homography, the focal
  length follows from the orthogonality of the rotation columns, and the camera
  height follows from the metric scale. That is free calibration at no extra
  cost and is genuinely useful for sanity-checking a frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# A keypoint below this confidence is emitted but not believed. 0.2 rather than
# 0.3: the far keypoints (pitch corners, penalty-area corners) are the ones that
# actually pin down a homography, and they score 0.35-0.58 where the centre
# circle scores 0.97. Discarding them left only a 9 m cluster.
MIN_KEYPOINT_CONF = 0.2
# Within this many pixels of an edge counts as clamped to the frame border.
BORDER_TOLERANCE = 2

# Minimum correspondences. Four is the mathematical minimum; five is treated as
# unreliable because it leaves RANSAC nothing to check.
MIN_POINTS_FOR_FIT = 4
# Correspondences RANSAC must keep before the fit is *believed*.
# This matters more than it looks: with exactly 4 inliers the solve is
# underdetermined, so the reprojection error is zero by construction. Such a fit
# scores perfectly while describing a camera that may be nonsense -- which is
# precisely how a 9 m keypoint cluster produced confidently wrong homographies.
# Redundancy, not reprojection error, is the real measure of trust.
# Measured on the current pose model: no frame in 250 exceeded 5 inliers, so this
# sits at the edge of what the data supports (see FitResult.redundancy).
MIN_TRUSTED_INLIERS = 5
# Spread of the inlier keypoints, in pixels. A cluster smaller than this cannot
# constrain a homography no matter how well it fits: five points inside the
# centre circle determine the transform within 9 m and leave the rest free.
# Reported as a warning rather than rejected, because the current model cannot
# clear it and a pipeline that outputs nothing helps nobody.
MIN_INLIER_SPREAD_PX = 260.0

# Reprojection error above this (metres, on the pitch plane) means the fit is
# describing something other than the pitch.
MAX_REPROJECTION_ERROR_M = 4.0
# Fit tolerance in pixels. Keypoint regression on a 1080p frame is accurate to
# a few pixels, so 5 px discards genuinely good correspondences and starves the
# RANSAC consensus; 12 px keeps them without admitting anything metres out.
RANSAC_PX = 12.0
# Fraction of correspondences RANSAC must keep.
# 0.4 rather than a majority: the pose model emits a handful of reliable points
# plus several mediocre ones, so a frame can have exactly the 4 inliers a
# homography needs and still fail a 0.5 ratio. Rejecting those threw away usable
# fits and fell back to a stale matrix. Downstream validation (reprojection,
# footprint sanity, convexity, jump gate) is what actually protects quality.
MIN_INLIER_RATIO = 0.4
# How far outside the frame corners may land before the fit is discarded.
CORNER_MARGIN = 1.0

# How strongly the previous frame's solution is believed, expressed as 0..1 so it
# maps directly to the --smoothing flag. Default is 0: the fit is taken from the
# data on every frame, because constraining it towards the previous frame also
# damps genuine camera motion. Object positions are smoothed in pitch space
# instead, where a Kalman filter cannot bias the trajectory.
DEFAULT_SMOOTHING = 0.0
# Upper bound on the prior weight (how many times the four previous frame
# corners are added to the correspondences).
MAX_PRIOR_WEIGHT = 3
# Blend factor for the camera estimate, which is noisier than the corners.
CAMERA_SMOOTHING = 0.7
# How far the frame may slide across the pitch between consecutive frames
# before the fit is treated as a different solution rather than camera motion.
# A fast pan moves it a few metres; the ambiguity cases move it by 100+.
MAX_FOOTPRINT_JUMP_M = 25.0
# How many consecutive disagreeing fits before the *previous* footprint is
# treated as the thing that is wrong. Without this the gate can lock onto a
# stale solution and reject every remaining frame of the clip.
JUMP_PATIENCE = 3
# Disagreeing for this many consecutive frames means the previous world frame was
# wrong, not that this frame's fit is: measured on the real clip, the new
# solution explains the keypoints to ~0 m while the old one is 9-93 m out.
RESYNC_AFTER = JUMP_PATIENCE + 1
# A footprint displacement above this is a real camera move, not jitter. The
# temporal prior is switched off for such a frame: dragging the solution back
# towards the previous one is exactly how a pan gets left behind.
PAN_MOTION_M = 1.2
# How many times the previous frame's four frame corners are repeated when
# re-solving a fit. Derived from DEFAULT_SMOOTHING; see HomographyTracker.
PRIOR_WEIGHT = 1


def canonical_keypoints() -> dict[int, tuple[float, float]]:
    """Real-world position in metres of each of the 29 model keypoints.

    Origin is the corner between the top touchline and the left goal line; x
    runs along the 105 m length, y across the 68 m width.
    """
    length, width = PITCH_LENGTH_M, PITCH_WIDTH_M
    cx, cy = length / 2.0, width / 2.0

    pen_depth, pen_half = 16.5, 40.32 / 2.0
    goal_depth, goal_half = 5.5, 18.32 / 2.0
    circle_r, pen_spot = 9.15, 11.0

    top_pen, bottom_pen = cy - pen_half, cy + pen_half
    top_goal, bottom_goal = cy - goal_half, cy + goal_half

    return {
        0: (0.0, 0.0),
        1: (length, 0.0),
        2: (0.0, width),
        3: (length, width),
        4: (0.0, top_pen),
        5: (pen_depth, top_pen),
        6: (0.0, bottom_pen),
        7: (pen_depth, bottom_pen),
        8: (length, top_pen),
        9: (length - pen_depth, top_pen),
        10: (length, bottom_pen),
        11: (length - pen_depth, bottom_pen),
        12: (0.0, top_goal),
        13: (goal_depth, top_goal),
        14: (0.0, bottom_goal),
        15: (goal_depth, bottom_goal),
        16: (length, top_goal),
        17: (length - goal_depth, top_goal),
        18: (length, bottom_goal),
        19: (length - goal_depth, bottom_goal),
        20: (cx, 0.0),
        21: (cx, width),
        22: (cx, cy),
        23: (cx, cy - circle_r),
        24: (cx, cy + circle_r),
        25: (cx - circle_r, cy),
        26: (cx + circle_r, cy),
        27: (pen_spot + circle_r, cy),
        28: (length - pen_spot - circle_r, cy),
    }


# The four pitch corners, used as the smoothing prior and for validation.
CORNER_INDICES = (0, 1, 3, 2)  # tl, tr, br, bl -- in image order


@dataclass
class CameraEstimate:
    """Approximate camera calibration recovered from the homography.

    Assumes square pixels and a principal point at the image centre, which is
    close enough for broadcast footage and avoids needing a calibration target.
    """

    focal_px: float
    height_m: float
    tilt_deg: float
    roll_deg: float
    yaw_deg: float
    hfov_deg: float

    def as_dict(self) -> dict:
        return {
            "focal_px": round(self.focal_px, 1),
            "height_m": round(self.height_m, 2),
            "tilt_deg": round(self.tilt_deg, 1),
            "roll_deg": round(self.roll_deg, 1),
            "yaw_deg": round(self.yaw_deg, 1),
            "hfov_deg": round(self.hfov_deg, 1),
        }

    def summary(self) -> str:
        # ASCII only: this string is also drawn with cv2.putText, which cannot
        # render a degree sign.
        return (f"f={self.focal_px:.0f}px h={self.height_m:.1f}m "
                f"elev={self.tilt_deg:.0f}deg roll={self.roll_deg:.0f}deg "
                f"hfov={self.hfov_deg:.0f}deg")


@dataclass
class FitResult:
    """Outcome of one frame's homography solve."""

    matrix: np.ndarray | None = None
    points_used: int = 0
    inliers: int = 0
    reprojection_error_m: float = float("inf")
    corners: np.ndarray | None = None     # 4x2 in image space
    camera: CameraEstimate | None = None
    accepted: bool = False
    reason: str = ""
    # Inlier correspondences (image pixels, pitch metres), kept so the error
    # can be recomputed after the corners are smoothed.
    inlier_image: np.ndarray | None = None
    inlier_pitch: np.ndarray | None = None
    # Where the frame landed on the pitch using the data alone, before the
    # temporal prior had any say. This is the honest measure of camera motion.
    raw_footprint: np.ndarray | None = None
    inlier_ratio: float = 0.0
    # RMS spread of the inlier pixels. Below MIN_INLIER_SPREAD_PX the fit is an
    # extrapolation and should not be trusted away from the keypoints.
    spread_px: float = 0.0
    # True when the previous frame's solution was used to constrain this fit.
    prior_applied: bool = False
    # Displacement (metres) between the data-only footprint and the prior; above
    # PAN_MOTION_M the prior is deliberately skipped.
    prior_motion_m: float = 0.0

    @property
    def redundancy(self) -> int:
        """Inliers beyond the four a homography needs. 0 means unverified."""
        return max(0, self.inliers - 4)

    @property
    def quality(self) -> float:
        """0..1 heuristic confidence, for reporting and weighting.

        Weighted by redundancy and spread rather than by reprojection error,
        which is worth nothing when the solve is underdetermined.
        """
        if not self.accepted:
            return 0.0
        redundancy = min(1.0, self.redundancy / 4.0)
        spread = min(1.0, self.spread_px / (2 * MIN_INLIER_SPREAD_PX))
        accuracy = max(0.0, 1.0 - self.reprojection_error_m / MAX_REPROJECTION_ERROR_M)
        return round(0.45 * redundancy + 0.35 * spread + 0.20 * accuracy, 3)


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def usable_keypoints(keypoints_xy, keypoint_conf, width: int, height: int):
    """``[(index, (x, y)), ...]`` of keypoints fit to be correspondences.

    Two exclusions matter. Low confidence: the pose head emits a coordinate for
    every keypoint regardless of belief. Border-clamped: Ultralytics clips its
    output to the frame, so a point whose true position is off-screen has been
    snapped onto the edge and no longer describes where the line actually is.
    """
    canon = canonical_keypoints()
    if keypoints_xy is None:
        return []

    out = []
    for index, point in enumerate(keypoints_xy):
        if index not in canon:
            continue
        if keypoint_conf is not None and float(keypoint_conf[index]) < MIN_KEYPOINT_CONF:
            continue
        x, y = float(point[0]), float(point[1])
        if (x <= BORDER_TOLERANCE or x >= width - 1 - BORDER_TOLERANCE
                or y <= BORDER_TOLERANCE or y >= height - 1 - BORDER_TOLERANCE):
            continue
        out.append((index, (x, y)))
    return out


def _inlier_spread(image_points) -> float:
    """RMS distance of the inlier pixels from their centroid.

    A tight cluster cannot constrain a homography however well it fits: the
    transform is determined only where the points are, so everything further
    away is extrapolation. This is the number that says whether a fit means
    anything, and it is invisible in the reprojection error.
    """
    if image_points is None or len(image_points) < 2:
        return 0.0
    points = np.asarray(image_points, dtype=float).reshape(-1, 2)
    centroid = points.mean(axis=0)
    return float(np.sqrt(((points - centroid) ** 2).sum(axis=1).mean()))


def _reprojection_error(matrix: np.ndarray, src, dst) -> float:
    """Mean |pitch - projected| in metres over the given correspondences."""
    if src is None or dst is None or len(src) == 0:
        return float("inf")
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix)
    if projected is None:
        return float("inf")
    return float(np.linalg.norm(projected.reshape(-1, 2) - dst, axis=1).mean())


def _corners_in_image(matrix: np.ndarray):
    """Pitch corners in image space, plus their homogeneous ``w``.

    ``w`` is the projective depth of the point. A corner with ``w <= 0`` lies
    behind the camera: it still has a finite image position, but that position
    is a mathematical artefact on the far side of the horizon and must never be
    treated as a real screen coordinate.
    """
    canon = canonical_keypoints()
    pitch_corners = np.asarray([canon[i] for i in CORNER_INDICES], dtype=np.float64)
    homogeneous = np.hstack([pitch_corners, np.ones((4, 1))])
    projected = homogeneous @ np.linalg.inv(matrix).T
    w = projected[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = projected[:, :2] / w[:, None]
    return xy, w


def fit_homography(keypoints_xy, keypoint_conf, width: int, height: int,
                   ransac_px: float = RANSAC_PX, prior_footprint=None,
                   prior_weight: int = 1,
                   pan_motion_m: float = PAN_MOTION_M) -> FitResult:
    """Solve image -> pitch metres and judge whether the answer is plausible.

    ``prior_footprint`` (four pitch metres for the frame corners) is where the
    previous frame put the frame on the pitch. Supplying it re-solves the
    accepted fit with those four correspondences added, which pins down the
    part of the transform the detected keypoints do not constrain -- see
    :func:`_solve_with_prior`.

    The prior is skipped when the data-alone footprint has moved further than
    ``pan_motion_m``, because there the previous solution is simply out of date
    and constraining towards it would make the estimate lag the camera.
    """
    usable = usable_keypoints(keypoints_xy, keypoint_conf, width, height)
    result = FitResult(points_used=len(usable))

    if len(usable) < MIN_POINTS_FOR_FIT:
        result.reason = f"only {len(usable)} usable keypoints"
        return result

    canon = canonical_keypoints()
    src = np.asarray([p for _, p in usable], dtype=np.float32)
    dst = np.asarray([canon[i] for i, _ in usable], dtype=np.float32)

    # Fit pitch -> image rather than image -> pitch so RANSAC's threshold is in
    # pixels. With the pitch in metres a 5.0 threshold would mean "5 metres",
    # which is far too permissive to identify an outlier.
    inverse, mask = cv2.findHomography(dst, src, cv2.RANSAC, ransac_px)
    if inverse is None:
        result.reason = "solver failed"
        return result
    try:
        matrix = np.linalg.inv(inverse)
    except np.linalg.LinAlgError:
        result.reason = "singular homography"
        return result

    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(usable), bool)
    result.matrix = matrix
    result.inliers = int(inliers.sum())
    result.inlier_image = src[inliers]
    result.inlier_pitch = dst[inliers]
    if result.inliers < MIN_POINTS_FOR_FIT:
        # RANSAC can return a matrix with no consensus at all: the degenerate
        # all-zeros mask. Nothing downstream can use that.
        result.reason = f"{result.inliers} inliers"
        return result
    if result.inliers < MIN_TRUSTED_INLIERS:
        # Reproducing the keypoints proves nothing here: the solve is still
        # close to underdetermined, so a near-zero residual is an artefact.
        result.reason = (f"only {result.inliers} inliers"
                         f" (need {MIN_TRUSTED_INLIERS} to verify)")
        return result
    result.spread_px = _inlier_spread(result.inlier_image)
    result.inlier_ratio = result.inliers / max(1, len(usable))
    # Resdiual is measured only over the correspondences RANSAC kept: a single
    # mislabelled keypoint can be 20 m out and would otherwise dominate.
    result.reprojection_error_m = _reprojection_error(
        matrix, result.inlier_image, result.inlier_pitch
    )
    if result.inlier_ratio < MIN_INLIER_RATIO:
        result.reason = f"inlier ratio {result.inliers}/{len(usable)}"
        return result
    if result.reprojection_error_m > MAX_REPROJECTION_ERROR_M:
        result.reason = f"reprojection {result.reprojection_error_m:.1f}m"
        return result

    # What the keypoints alone say, recorded before the prior gets a say. This
    # is the only honest measure of how far the camera moved this frame.
    result.raw_footprint = _frame_footprint(matrix, width, height)

    if prior_footprint is not None and prior_weight > 0:
        prior = np.asarray(prior_footprint, dtype=np.float32)
        result.prior_motion_m = float(
            np.linalg.norm(result.raw_footprint - prior, axis=1).mean())
        if result.prior_motion_m <= pan_motion_m:
            refined = _solve_with_prior(result.inlier_image, result.inlier_pitch,
                                        prior, width, height, prior_weight)
            if refined is not None:
                error = _reprojection_error(refined, result.inlier_image,
                                            result.inlier_pitch)
                # Only take the constrained solution if it still explains the
                # keypoints; otherwise the prior is stale and the data wins.
                if error <= MAX_REPROJECTION_ERROR_M:
                    matrix = refined
                    result.matrix = refined
                    result.reprojection_error_m = error
                    result.prior_applied = True

    corners, depth = _corners_in_image(matrix)
    if not np.all(np.isfinite(corners)):
        result.reason = "non-finite corners"
        return result

    # Where the frame lands on the pitch is a scale-aware sanity check: a fit
    # that is not describing this pitch maps the frame to somewhere absurd.
    footprint = _frame_footprint(matrix, width, height)
    if not np.all(np.isfinite(footprint)):
        result.reason = "frame maps to infinity"
        return result
    if np.any(np.abs(footprint) > 10 * max(PITCH_LENGTH_M, PITCH_WIDTH_M)):
        result.reason = "frame maps far outside the pitch"
        return result

    # Corners are only usable as a smoothing prior when all four are in front
    # of the camera and in a sane arrangement. A close-up of the centre circle
    # often leaves two corners behind the camera; the fit is still good locally,
    # so this downgrades smoothing rather than rejecting the frame.
    if np.all(depth > 1e-9) and np.all(np.abs(corners) < 50 * max(width, height)):
        if not _is_convex(corners):
            result.reason = "degenerate corner arrangement"
            return result
        result.corners = corners
    else:
        result.corners = None

    result.accepted = True
    result.reason = "ok"
    return result


def _frame_footprint(matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    """The four image corners, in pitch metres."""
    frame = np.asarray([[0, 0], [width, 0], [width, height], [0, height]],
                       dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(frame, matrix).reshape(-1, 2)


def _is_convex(quad: np.ndarray) -> bool:
    """True when the points are in consistent rotational order."""
    edges = np.roll(quad, -1, axis=0) - quad
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    return bool(np.all(cross > 0) or np.all(cross < 0))


def _solve_with_prior(inlier_image, inlier_pitch, prior_footprint, width: int,
                      height: int, weight: int) -> np.ndarray | None:
    """Re-solve image -> pitch with the previous frame's footprint added.

    Keypoints clustered in one part of the pitch determine the transform *there*
    and leave the rest free: several very different homographies reproject them
    to within centimetres, and they disagree by tens of metres at the far
    corners. Adding the four frame corners as correspondences to where the
    previous frame put them pins the free part down.

    ``weight`` repeats those four correspondences; at 1 they are outnumbered by
    the real inliers, so a genuine camera move still wins, while a frame that
    merely wobbles is pulled back onto the previous solution.
    """
    if inlier_image is None or inlier_pitch is None or not len(inlier_image):
        return None
    frame = np.asarray([[0, 0], [width, 0], [width, height], [0, height]],
                       dtype=np.float32)
    # Rows must line up with the direction being solved: pitch metres first,
    # image pixels second, for the real keypoints and the prior alike.
    source = np.vstack([np.asarray(inlier_pitch, np.float32)] +
                       [np.asarray(prior_footprint, np.float32)] * max(1, int(weight)))
    target = np.vstack([np.asarray(inlier_image, np.float32)] +
                       [frame] * max(1, int(weight)))
    inverse = cv2.findHomography(source, target, 0)
    if inverse is None or inverse[0] is None:
        return None
    try:
        matrix = np.linalg.inv(inverse[0])
    except np.linalg.LinAlgError:
        return None
    return matrix if np.all(np.isfinite(matrix)) else None


def matrix_from_frame_corners(footprint: np.ndarray, width: int,
                              height: int) -> np.ndarray | None:
    """Rebuild image -> pitch from where the frame lands on the pitch.

    Four correspondences determine a homography exactly, so this converts a
    temporally smoothed footprint back into a true projection. It is always
    available (unlike the pitch corners, which can fall behind the camera) and
    is the only place a homography may be blended, because averaging matrices
    elementwise does not produce a homography.
    """
    frame = np.asarray([[0, 0], [width, 0], [width, height], [0, height]],
                       dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(frame, np.asarray(footprint,
                                                           dtype=np.float32))
    if matrix is None or not np.all(np.isfinite(matrix)):
        return None
    return matrix


def estimate_camera(matrix: np.ndarray, width: int, height: int) -> CameraEstimate | None:
    """Recover focal length and pose from an image -> pitch homography.

    For ``H = K [r1 r2 t]`` mapping the pitch plane to the image, with
    ``K = diag(f, f, 1)`` (principal point at the frame centre) and orthonormal
    rotation columns, the constraint ``r1 . r2 = 0`` becomes, writing h for the
    image coordinates measured from the centre,

        (h11*h12 + h21*h22) + f^2 * h31*h32 = 0

    so the focal length follows in closed form. Everything else comes from
    decomposing with that focal length.

    Returns None when the numbers are not physically possible, which is a
    genuinely useful signal: a homography fit from a handful of keypoints
    clustered in one small region of the pitch can be locally accurate while
    implying a lens that cannot exist.
    """
    h = np.asarray(matrix, dtype=float)
    if h.shape != (3, 3) or not np.all(np.isfinite(h)):
        return None

    # The decomposition describes pitch -> image; everything else in this
    # module carries image -> pitch, so invert rather than ask callers to.
    try:
        h = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return None

    # Scale is arbitrary; normalise so h33 = 1 for stable arithmetic.
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]

    # The focal-length formula below assumes the principal point is the origin,
    # so shift the image coordinates to put the frame centre at (0, 0).
    shift = np.array([[1, 0, -width / 2.0], [0, 1, -height / 2.0], [0, 0, 1.0]])
    h = shift @ h

    # Two constraints on the same unknown u = 1/f^2, from r1 . r2 = 0 and
    # |r1| = |r2|. Combining them by least squares matters: either one alone
    # degenerates for a camera with no roll (h31 = 0) or no yaw, and real
    # footage hits both cases.
    a1 = h[0, 0] * h[0, 1] + h[1, 0] * h[1, 1]
    b1 = h[2, 0] * h[2, 1]
    a2 = h[0, 0] ** 2 + h[1, 0] ** 2 - h[0, 1] ** 2 - h[1, 1] ** 2
    b2 = h[2, 0] ** 2 - h[2, 1] ** 2

    denominator = a1 * a1 + a2 * a2
    if denominator < 1e-20:
        return None
    u = -(a1 * b1 + a2 * b2) / denominator     # u = 1 / f^2
    if not np.isfinite(u) or u <= 0:
        return None
    focal = float(1.0 / np.sqrt(u))
    # Any real lens on a 1080p frame lands in this range; wildly outside means
    # the fit is not describing a camera.
    if not (200.0 < focal < 20000.0):
        return None

    k_inv = np.diag([1.0 / focal, 1.0 / focal, 1.0])
    b_mat = k_inv @ h
    norm = np.linalg.norm(b_mat[:, 0])
    if norm < 1e-12:
        return None
    b_mat = b_mat / norm

    r1, r2, t = b_mat[:, 0], b_mat[:, 1], b_mat[:, 2]
    rot = np.column_stack([r1, r2, np.cross(r1, r2)])
    u, _, vt = np.linalg.svd(rot)
    rot = u @ vt  # re-orthonormalise against noise
    r1 = rot[:, 0]
    axis = rot[:, 2]                      # optical axis, in pitch coordinates

    # Camera centre in pitch coordinates: C = -R^T t
    origin = -rot.T @ t
    height_m = float(abs(origin[2])) if np.isfinite(origin[2]) else 0.0
    if not (0.5 < height_m < 200.0):
        return None

    # Elevation rather than a signed tilt: a broadcast camera looks down, but
    # the sign of the optical axis in this frame convention is not meaningful
    # on its own, so report the angle off horizontal.
    elevation = float(np.degrees(np.arcsin(np.clip(abs(axis[2]), 0.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(axis[1], axis[0])))
    # Roll is how far the camera's x axis is out of the horizontal plane. The
    # sign is inverted so a clockwise-rolled horizon reads positive.
    roll = float(-np.degrees(np.arcsin(np.clip(r1[2], -1.0, 1.0))))

    hfov = float(np.degrees(2 * np.arctan(width / (2 * focal))))
    if not (5.0 <= hfov <= 170.0):
        return None
    if not np.all(np.isfinite([origin[0], origin[1], elevation, yaw, roll])):
        return None

    return CameraEstimate(focal_px=focal, height_m=height_m, tilt_deg=elevation,
                          roll_deg=roll, yaw_deg=yaw, hfov_deg=hfov)


@dataclass
class FrameFit:
    """One frame's smoothed homography plus what it was derived from."""

    matrix: np.ndarray | None = None
    corners: np.ndarray | None = None
    camera: CameraEstimate | None = None
    raw: FitResult | None = None
    smoothed: bool = False
    held_over: bool = False      # reused a previous frame's matrix
    # The fit settled onto a different global solution, so pitch coordinates
    # are not continuous with the previous frame and anything tracking identity
    # through pitch space has to be restarted.
    resynced: bool = False
    keypoints: list = field(default_factory=list)  # trusted, for the overlay


class HomographyTracker:
    """Fits, validates and temporally smooths the pitch homography.

    Four behaviours worth knowing:

    * **Fits are ambiguous when the keypoints are clustered.** Five points on
      the centre circle constrain the homography well *there* and hardly at all
      elsewhere: two very different global solutions fit the same points to a
      few centimetres. In practice the fit occasionally jumps between them,
      which would teleport every player. So a fit that disagrees wildly with the
      previous frame is treated as an outlier and skipped.
    * **The prior is off by default, and it yields to real camera movement.**
      It measurably reduces frame-to-frame jitter but, being a constraint
      towards the previous frame, it also damps genuine motion: measured, it
      recovers 87% of the camera displacement. Jitter is better removed from the
      object positions (where a Kalman filter is unbiased) than from the
      homography. Set ``--smoothing`` above 0 to enable it.
    * **A failed frame holds over the last good one** for a few frames. The last
      footprint is deliberately *kept* when that runs out, so the jump gate
      still has something to compare against when the next fit arrives.
    """

    def __init__(self, smoothing: float = DEFAULT_SMOOTHING,
                 max_hold: int = 4, max_jump_m: float = MAX_FOOTPRINT_JUMP_M,
                 prior_weight: int | None = None,
                 ransac_px: float = RANSAC_PX,
                 pan_motion_m: float = PAN_MOTION_M,
                 jump_patience: int = JUMP_PATIENCE,
                 resync_after: int = RESYNC_AFTER):
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        # --smoothing maps onto how strongly the previous solution is believed.
        if prior_weight is None:
            prior_weight = int(round(self.smoothing * MAX_PRIOR_WEIGHT / 0.9))
        self.prior_weight = max(0, min(MAX_PRIOR_WEIGHT, int(prior_weight)))
        self.max_hold = max(0, int(max_hold))
        self.max_jump_m = float(max_jump_m)
        self.ransac_px = float(ransac_px)
        self.pan_motion_m = float(pan_motion_m)
        self.jump_patience = max(1, int(jump_patience))
        self.resync_after = max(self.jump_patience + 1, int(resync_after))

        self._footprint: np.ndarray | None = None
        self._camera: CameraEstimate | None = None
        self._matrix: np.ndarray | None = None
        self._hold = 0
        # Consecutive frames whose fit disagreed with the previous footprint.
        self._stuck = 0

        self.accepted = 0
        self.rejected = 0
        self.held = 0
        self.jumps = 0
        self.priors = 0
        self.resyncs = 0
        self.last_reason = ""
        # Per-frame evidence of how much the keypoints really constrain the fit.
        self._inlier_counts: list[int] = []
        self._inlier_spreads: list[float] = []

    # -- per frame ---------------------------------------------------------

    def reset(self) -> None:
        """Forget the previous frame after a cut.

        The counters are deliberately not cleared: they describe the whole clip,
        and a cut is a restart of continuity, not of the run.
        """
        self._footprint = None
        self._matrix = None
        self._camera = None
        self._hold = 0

    def _hold_over(self, raw: FitResult, reason: str) -> FrameFit:
        """Reuse the last good matrix for one more frame, or give up."""
        self.last_reason = reason
        if self._matrix is not None and self._hold < self.max_hold:
            self._hold += 1
            self.held += 1
            return FrameFit(matrix=self._matrix, camera=self._camera, raw=raw,
                            smoothed=True, held_over=True, keypoints=[])
        # Out of runway: stop drawing, but keep the footprint so the next fit
        # still has to justify itself against where the camera last was.
        self._matrix = None
        return FrameFit(raw=raw, keypoints=[])

    def update(self, keypoints_xy, keypoint_conf, width: int, height: int) -> FrameFit:
        # ``fresh`` marks the first frame of a shot. There is deliberately no
        # jump gate then: a cut legitimately moves the camera anywhere, so the
        # previous footprint is not evidence about this frame at all.
        fresh = self._footprint is None

        raw = fit_homography(keypoints_xy, keypoint_conf, width, height,
                             ransac_px=self.ransac_px,
                             prior_footprint=self._footprint,
                             prior_weight=self.prior_weight,
                             pan_motion_m=self.pan_motion_m)
        self.last_reason = raw.reason
        trusted = [p for _, p in usable_keypoints(keypoints_xy, keypoint_conf,
                                                  width, height)]

        if not raw.accepted:
            self.rejected += 1
            return self._hold_over(raw, raw.reason)

        self._inlier_counts.append(raw.inliers)
        self._inlier_spreads.append(raw.spread_px)

        footprint = _frame_footprint(raw.matrix, width, height)
        resynced = False
        if not fresh:
            jump = float(np.linalg.norm(footprint - self._footprint, axis=1).mean())
            if jump > self.max_jump_m:
                self._stuck += 1
                if self._stuck <= self.jump_patience:
                    # A different global solution, not camera motion. A cut
                    # would already have been detected and would not get here.
                    self.jumps += 1
                    self.rejected += 1
                    return self._hold_over(
                        raw, f"implausible jump ({jump:.0f} m)")
                # Persistent disagreement. Measured directly: the new solution
                # explains the keypoints to ~0 m while the old one is 9-93 m out,
                # so this is not jitter or ambiguity -- the data genuinely
                # prefers a different world frame. Forcing continuity here would
                # be knowingly using the worse fit.
                #
                # So take the data's answer, and restart identity: pitch
                # coordinates have just moved tens of metres at once, and a
                # surviving track would appear to teleport across the pitch.
                self.resyncs += 1
                resynced = True

        self._hold = 0
        self._stuck = 0
        self.accepted += 1
        if raw.prior_applied:
            self.priors += 1

        self._footprint = footprint
        self._matrix = raw.matrix

        camera = estimate_camera(raw.matrix, width, height)
        if camera is not None and self._camera is not None:
            blend = CAMERA_SMOOTHING
            camera = CameraEstimate(
                focal_px=blend * self._camera.focal_px + (1 - blend) * camera.focal_px,
                height_m=blend * self._camera.height_m + (1 - blend) * camera.height_m,
                tilt_deg=blend * self._camera.tilt_deg + (1 - blend) * camera.tilt_deg,
                roll_deg=blend * self._camera.roll_deg + (1 - blend) * camera.roll_deg,
                yaw_deg=blend * self._camera.yaw_deg + (1 - blend) * camera.yaw_deg,
                hfov_deg=blend * self._camera.hfov_deg + (1 - blend) * camera.hfov_deg,
            )
        self._camera = camera or self._camera

        return FrameFit(matrix=raw.matrix, camera=self._camera,
                        raw=raw, smoothed=raw.prior_applied,
                        resynced=resynced, keypoints=trusted)

    # -- reporting ---------------------------------------------------------

    def fit_quality(self) -> dict:
        """How well the keypoints actually constrain the homography.

        This is the number that explains most analysis problems, so it is
        reported on every run: a handful of inliers with no redundancy means the
        fit reproduces its own inputs and nothing more.
        """
        inliers = self._inlier_counts
        spreads = self._inlier_spreads
        if not inliers:
            return {"frames": 0}
        return {
            "frames": len(inliers),
            "median_inliers": int(np.median(inliers)),
            "median_redundancy": int(np.median([max(0, i - 4) for i in inliers])),
            "median_spread_px": float(np.median(spreads)),
        }

    def stats(self) -> dict:
        total = self.accepted + self.rejected
        return {
            "frames": total,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "held_over": self.held,
            "jumps": self.jumps,
            "priors_applied": self.priors,
            "resyncs": self.resyncs,
            "solve_rate": round(self.accepted / total, 3) if total else 0.0,
            "last_reason": self.last_reason,
        }
