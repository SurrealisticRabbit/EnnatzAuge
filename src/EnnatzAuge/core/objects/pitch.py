"""Virtual pitch: per-frame objects projected from image space to metres.

Keypoint indices match KEYPOINT_NAMES in convert.py and the trained pose model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .ball import Ball
from .goalkeeper import Goalkeeper
from .player import Player
from .referee import Referee

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
# Drawn around the touchlines so players standing off the pitch stay visible.
PITCH_MARGIN_M = 3.0

# A keypoint below this is emitted by the pose head but not believed.
MIN_KEYPOINT_CONF = 0.3
# Within this many pixels of an edge counts as clamped to the frame border.
BORDER_TOLERANCE = 2
# Boxes below this are ignored entirely.
MIN_PLAYER_CONF = 0.25


def canonical_keypoints() -> dict[int, tuple[float, float]]:
    """Real-world position in metres of each of the 29 keypoints.

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


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _usable_keypoints(keypoints_xy, keypoint_conf, width: int, height: int):
    """Indices and coordinates of keypoints fit to be used as correspondences.

    Low-confidence points are excluded because the pose head emits a
    coordinate for every keypoint regardless of belief, and border-clamped
    points are excluded because Ultralytics clips the output to the frame,
    destroying the true position of anything off-screen.
    """
    canon = canonical_keypoints()
    if keypoints_xy is None:
        return []
    usable = []
    for index, point in enumerate(keypoints_xy):
        if index not in canon:
            continue
        if keypoint_conf is not None and float(keypoint_conf[index]) < MIN_KEYPOINT_CONF:
            continue
        x, y = float(point[0]), float(point[1])
        if (x <= BORDER_TOLERANCE or x >= width - 1 - BORDER_TOLERANCE
                or y <= BORDER_TOLERANCE or y >= height - 1 - BORDER_TOLERANCE):
            continue
        usable.append((index, (x, y)))
    return usable


def solve_homography(keypoints_xy, keypoint_conf, width: int, height: int):
    """Fit image -> pitch metres from one frame's keypoints.

    Returns a 3x3 matrix, or None with fewer than four usable correspondences.
    RANSAC is used rather than a plain least-squares fit because a single wild
    keypoint would otherwise skew the whole projection.
    """
    usable = _usable_keypoints(keypoints_xy, keypoint_conf, width, height)
    if len(usable) < 4:
        return None

    canon = canonical_keypoints()
    src = np.asarray([point for _, point in usable], dtype=np.float32)
    dst = np.asarray([canon[index] for index, _ in usable], dtype=np.float32)

    matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return matrix


@dataclass
class FrameState:
    """What one frame contained, in a form the renderers can draw."""

    players: list = field(default_factory=list)
    referees: list = field(default_factory=list)
    goalkeepers: list = field(default_factory=list)
    balls: list = field(default_factory=list)
    keypoints: list = field(default_factory=list)   # image space, drawable
    homography_ok: bool = False
    projected: int = 0
    rejected: int = 0

    def all_objects(self) -> list:
        return [*self.players, *self.goalkeepers, *self.referees, *self.balls]


class VirtualPitch:
    def __init__(self, scale: float = 10.0):
        self.scale = scale

        # Current frame, kept for the original getters.
        self.players: list[Player] = []
        self.referees: list[Referee] = []
        self.goalkeepers: list[Goalkeeper] = []
        self.balls: list[Ball] = []

        self.homography = None
        self.homography_ok = False
        self.keypoints: list[tuple[float, float]] = []
        self.history: list[FrameState] = []

    # -- ingest ------------------------------------------------------------

    def _ingest_player_model_data(self, player_model_data):
        """Split one frame's detections into typed objects, in image space."""
        players, referees, goalkeepers, balls = [], [], [], []
        if player_model_data is None:
            return players, referees, goalkeepers, balls

        boxes = getattr(player_model_data, "boxes", None)
        if boxes is None:
            return players, referees, goalkeepers, balls

        xyxy = _to_numpy(boxes.xyxy)
        classes = _to_numpy(boxes.cls)
        scores = _to_numpy(boxes.conf)
        names = getattr(player_model_data, "names", {}) or {}

        by_class = {
            "player": players,
            "goalkeeper": goalkeepers,
            "referee": referees,
            "ball": balls,
        }

        for index, box in enumerate(xyxy):
            score = float(scores[index]) if scores is not None else 1.0
            if score < MIN_PLAYER_CONF:
                continue
            class_name = names.get(int(classes[index]), "?") if classes is not None else "?"
            group = by_class.get(class_name)
            if group is None:
                continue  # "other" has no object type in this structure

            x1, y1, x2, y2 = (float(v) for v in box)
            # Foot point: where the player meets the ground, not the box centre.
            foot = ((x1 + x2) / 2.0, y2)

            if class_name == "player":
                obj = Player(index, f"player-{index}", None)
            elif class_name == "goalkeeper":
                obj = Goalkeeper(index, f"goalkeeper-{index}", None)
            elif class_name == "referee":
                obj = Referee(index, f"referee-{index}", None)
            else:
                obj = Ball(index, f"ball-{index}", None)

            obj.class_name = class_name
            obj.confidence = score
            obj.bbox = (x1, y1, x2, y2)
            obj.image_position = foot
            obj.in_frame = True
            group.append(obj)

        return players, referees, goalkeepers, balls

    def _compute_homography(self, pitch_model_data, players, referees,
                            goalkeepers, balls):
        """Solve the homography and project every object onto the pitch."""
        self.homography = None
        self.homography_ok = False
        self.keypoints = []

        if pitch_model_data is None or getattr(pitch_model_data, "keypoints", None) is None:
            return players, referees, goalkeepers, balls

        height, width = pitch_model_data.orig_shape[:2]
        xy = _to_numpy(pitch_model_data.keypoints.xy)
        conf = _to_numpy(pitch_model_data.keypoints.conf)
        if xy is None or not len(xy):
            return players, referees, goalkeepers, balls

        first_xy = xy[0]
        first_conf = conf[0] if conf is not None else None

        self.homography = solve_homography(first_xy, first_conf, width, height)
        if self.homography is None:
            return players, referees, goalkeepers, balls

        self.homography_ok = True
        # Keep the same points the fit used, so the overlay draws what was
        # actually trusted rather than every predicted coordinate.
        self.keypoints = [point for _, point in
                          _usable_keypoints(first_xy, first_conf, width, height)]

        margin = PITCH_MARGIN_M
        for group in (players, referees, goalkeepers, balls):
            for obj in group:
                foot = np.array([obj.image_position], dtype=np.float32).reshape(1, 1, 2)
                point = cv2.perspectiveTransform(foot, self.homography)[0][0]
                px, py = float(point[0]), float(point[1])
                if (-margin <= px <= PITCH_LENGTH_M + margin
                        and -margin <= py <= PITCH_WIDTH_M + margin):
                    obj.current_position = (px, py)
                else:
                    obj.current_position = None  # off the pitch: bad fit
                    obj.in_frame = False

        return players, referees, goalkeepers, balls

    def ingest_frame_data(self, pitch_frame_data, player_frame_data):
        """Consume one aligned pair of model results."""
        players, referees, goalkeepers, balls = self._ingest_player_model_data(
            player_frame_data
        )
        players, referees, goalkeepers, balls = self._compute_homography(
            pitch_frame_data, players, referees, goalkeepers, balls
        )

        self.players = players
        self.referees = referees
        self.goalkeepers = goalkeepers
        self.balls = balls

        every = (players, referees, goalkeepers, balls)
        projected = sum(1 for group in every for obj in group
                        if obj.current_position is not None)
        total = sum(len(group) for group in every)

        self.history.append(FrameState(
            players=players,
            referees=referees,
            goalkeepers=goalkeepers,
            balls=balls,
            keypoints=list(self.keypoints),
            homography_ok=self.homography_ok,
            projected=projected,
            rejected=total - projected,
        ))

    # -- accessors ---------------------------------------------------------

    def get_referees(self):
        return self.referees

    def get_goalkeepers(self):
        return self.goalkeepers

    def get_balls(self):
        return self.balls

    def get_players(self):
        return self.players

    def get_all_objects(self):
        return [*self.players, *self.goalkeepers, *self.referees, *self.balls]
