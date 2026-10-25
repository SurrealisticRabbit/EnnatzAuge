"""Virtual pitch: per-frame objects projected from image space to metres.

Two pieces of machinery sit between the raw model output and the bird's-eye
view, both of which the renderers depend on:

* :class:`~.homography.HomographyTracker` fits and validates the image -> pitch
  transform, smooths it over time and recovers the camera pose. Without the
  validation a bad frame silently teleports every player.
* :class:`~.tracking.Tracker` gives each detection an identity, so the same
  person keeps the same id and their pitch position is filtered rather than
  drawn raw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .ball import Ball
from .goalkeeper import Goalkeeper
from .homography import (
    DEFAULT_SMOOTHING,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    CameraEstimate,
    FitResult,
    FrameFit,
    HomographyTracker,
    canonical_keypoints,
)
from .player import Player
from .referee import Referee
from .tracking import Track, Tracker

__all__ = [
    "PITCH_LENGTH_M",
    "PITCH_WIDTH_M",
    "PITCH_MARGIN_M",
    "FrameState",
    "VirtualPitch",
    "canonical_keypoints",
]

# Drawn around the touchlines so players standing off the pitch stay visible.
PITCH_MARGIN_M = 3.0

# Boxes below this are ignored entirely.
MIN_PLAYER_CONF = 0.25

# Which object class a detection becomes. "other" (category 7 in the dataset)
# has no representation in this structure and is dropped.
OBJECT_TYPES = {
    "player": Player,
    "goalkeeper": Goalkeeper,
    "referee": Referee,
    "ball": Ball,
}


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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

    # Quality of this frame's fit, for the overlay readout.
    camera: CameraEstimate | None = None
    homography_quality: float = 0.0
    homography_smoothed: bool = False
    homography_held: bool = False       # reused the previous frame's fit
    homography_reason: str = ""

    def all_objects(self) -> list:
        return [*self.players, *self.goalkeepers, *self.referees, *self.balls]


class VirtualPitch:
    """Turns paired detection output into a tracked top-down view.

    Per frame the order of operations is forced: the homography can only be fit
    from the pitch keypoints, the detections can only be projected once the
    homography exists, and the tracker can only smooth positions that have been
    projected. So a frame's detections are associated *after* projection, using
    both image-space overlap and pitch-space distance.
    """

    def __init__(self, scale: float = 10.0, smoothing: float = DEFAULT_SMOOTHING,
                 track: bool = True, fps: float = 25.0,
                 compensate: bool = False):
        self.scale = scale
        self.track_objects = track
        self.fps = float(fps) or 25.0
        self.homography_tracker = HomographyTracker(smoothing=smoothing)
        self.tracker = Tracker(compensate=compensate)
        self.cuts = 0
        self.resyncs = 0
        self.last_cut_reason = ""

        # Current frame, kept for the original getters.
        self.players: list[Player] = []
        self.referees: list[Referee] = []
        self.goalkeepers: list[Goalkeeper] = []
        self.balls: list[Ball] = []

        self.homography = None
        self.homography_ok = False
        self.camera: CameraEstimate | None = None
        self.keypoints: list[tuple[float, float]] = []
        self.history: list[FrameState] = []

    # -- ingest ------------------------------------------------------------

    def _detect(self, player_model_data):
        """Flatten one frame's detector output into tracker-ready dicts."""
        detections: list[dict] = []
        if player_model_data is None:
            return detections

        boxes = getattr(player_model_data, "boxes", None)
        if boxes is None:
            return detections

        xyxy = _to_numpy(boxes.xyxy)
        if xyxy is None:
            return detections
        classes = _to_numpy(boxes.cls)
        scores = _to_numpy(boxes.conf)
        names = getattr(player_model_data, "names", {}) or {}

        for index, box in enumerate(xyxy):
            score = float(scores[index]) if scores is not None else 1.0
            if score < MIN_PLAYER_CONF:
                continue
            class_name = names.get(int(classes[index]), "?") if classes is not None else "?"
            if class_name not in OBJECT_TYPES:
                continue

            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append({
                "class_name": class_name,
                "bbox": (x1, y1, x2, y2),
                "confidence": score,
                # Foot point: where the player meets the ground, not the box
                # centre. Projecting the centre lifts everyone up the pitch.
                "foot": ((x1 + x2) / 2.0, y2),
            })
        return detections

    def _fit(self, pitch_model_data, fallback_width: int,
             fallback_height: int) -> FrameFit:
        """Solve and smooth this frame's homography from the pose keypoints."""
        keypoints = getattr(pitch_model_data, "keypoints", None) if pitch_model_data else None
        if keypoints is None:
            self.homography_tracker.last_reason = "no pose output"
            return FrameFit(raw=FitResult(reason="no pose output"))

        shape = getattr(pitch_model_data, "orig_shape", None)
        if shape is not None and len(shape) >= 2:
            height, width = int(shape[0]), int(shape[1])
        else:
            width, height = fallback_width, fallback_height

        xy = _to_numpy(keypoints.xy)
        if xy is None or not len(xy):
            self.homography_tracker.last_reason = "no keypoints"
            return FrameFit(raw=FitResult(reason="no keypoints"))

        conf = _to_numpy(keypoints.conf)
        return self.homography_tracker.update(
            xy[0], conf[0] if conf is not None else None, width, height
        )

    def _project(self, matrix, point):
        """Image pixel -> pitch metres, or None when it lands off the pitch."""
        if matrix is None or point is None:
            return None
        src = np.asarray([point], dtype=np.float32).reshape(1, 1, 2)
        x, y = cv2.perspectiveTransform(src, matrix)[0][0]
        margin = PITCH_MARGIN_M
        if (-margin <= x <= PITCH_LENGTH_M + margin
                and -margin <= y <= PITCH_WIDTH_M + margin):
            return float(x), float(y)
        return None

    def _build(self, track: Track):
        """Materialise the typed object a track describes."""
        kind = OBJECT_TYPES.get(track.class_name)
        if kind is None:
            return None
        obj = kind(track.track_id, f"{track.class_name}-{track.track_id}", track.pitch)
        obj.class_name = track.class_name
        obj.confidence = track.confidence
        obj.bbox = track.bbox
        obj.image_position = track.image_position
        obj.current_position = track.pitch
        obj.in_frame = track.pitch is not None
        obj.track_id = track.track_id
        obj.speed = track.speed
        obj.trail = list(track.trail)
        obj.measured = track.measured
        return obj

    @staticmethod
    def _camera_shift(previous_matrix, matrix, width: int, height: int):
        """Pixels a fixed world point moves across the image between two fits.

        A point seen at the frame centre now corresponds to a world position
        ``H p``; that same world point sat at ``H_prev^-1 (H p)`` in the previous
        frame. The difference is how much of the image movement is the camera's
        rather than the object's, which association can subtract.
        """
        if previous_matrix is None or matrix is None:
            return None
        centre = np.asarray([[[width / 2.0, height / 2.0]]], dtype=np.float32)
        world = cv2.perspectiveTransform(centre, matrix)
        if world is None:
            return None
        try:
            inverse = np.linalg.inv(previous_matrix)
        except np.linalg.LinAlgError:
            return None
        before = cv2.perspectiveTransform(world.astype(np.float32), inverse)
        if before is None or not np.all(np.isfinite(before)):
            return None
        dx = float(centre[0, 0, 0] - before[0, 0, 0])
        dy = float(centre[0, 0, 1] - before[0, 0, 1])
        if not (np.isfinite(dx) and np.isfinite(dy)):
            return None
        # A shift larger than the frame cannot be camera motion between two
        # frames of broadcast footage; treat it as no information.
        if abs(dx) > width or abs(dy) > height:
            return None
        return dx, dy

    def reset(self, reason: str = "cut") -> None:
        """Restart continuity: new shot, so no state carries over.

        ``history`` is kept -- it is what the renderers replay -- but everything
        that describes the *previous* shot stops here.
        """
        self.homography_tracker.reset()
        self.tracker.reset()
        self.homography = None
        self.homography_ok = False
        self.camera = None
        self.keypoints = []
        self.cuts += 1
        self.last_cut_reason = reason

    def ingest_frame_data(self, pitch_frame_data, player_frame_data,
                          fps: float | None = None, cut: bool = False):
        """Consume one aligned pair of model results.

        ``fps`` makes the tracker's time step real, so its velocity comes out in
        m/s instead of metres per frame.
        """
        if fps:
            self.fps = float(fps)
        dt = 1.0 / self.fps if self.fps else 1.0
        detections = self._detect(player_frame_data)

        shape = getattr(player_frame_data, "orig_shape", None)
        height, width = (int(shape[0]), int(shape[1])) if shape is not None else (1080, 1920)

        # A cut means a different camera on different players. Carrying any
        # state across it is what produced teleporting ids and phantom tracks.
        if cut:
            self.reset()

        previous_matrix = self.homography
        fit = self._fit(pitch_frame_data, width, height)

        # A resync moves the whole coordinate frame at once, so existing pitch
        # positions are no longer comparable with the new ones. Only identity is
        # restarted: the homography tracker's own footprint is still the right
        # reference for gating the next few frames.
        if fit.resynced:
            self.tracker.reset()
            self.resyncs += 1
            previous_matrix = None

        self.homography = fit.matrix
        self.homography_ok = fit.matrix is not None
        self.camera = fit.camera
        # Keep only the points the fit actually trusted, so the overlay draws
        # what fed the homography rather than every predicted coordinate.
        self.keypoints = list(fit.keypoints)

        for detection in detections:
            detection["pitch"] = self._project(fit.matrix, detection["foot"])

        # A held-over matrix describes where the camera *was*, so projecting
        # with it puts everyone where they used to be. That is worse than no
        # position at all: the tracker would read the error as a teleport.
        # (A fit that used the temporal prior is stale only by a fraction of a
        # frame, so it counts as trusted.)
        pitch_trusted = fit.matrix is not None and not fit.held_over

        shift = None
        if pitch_trusted and not fit.held_over:
            shift = self._camera_shift(previous_matrix, fit.matrix, width, height)

        if self.track_objects:
            tracks = self.tracker.update(detections, dt=dt,
                                         pitch_trusted=pitch_trusted,
                                         image_shift=shift)
        else:
            tracks = [
                Track(track_id=index + 1,
                      class_name=detection["class_name"],
                      bbox=detection["bbox"],
                      confidence=detection["confidence"],
                      image_position=detection["foot"],
                      pitch=detection["pitch"],
                      hits=1)
                for index, detection in enumerate(detections)
            ]

        players, referees, goalkeepers, balls = [], [], [], []
        by_class = {
            "player": players,
            "goalkeeper": goalkeepers,
            "referee": referees,
            "ball": balls,
        }
        for track in tracks:
            obj = self._build(track)
            if obj is not None:
                by_class[track.class_name].append(obj)

        self.players = players
        self.referees = referees
        self.goalkeepers = goalkeepers
        self.balls = balls

        every = (players, referees, goalkeepers, balls)
        projected = sum(1 for group in every for obj in group
                        if obj.current_position is not None)
        total = sum(len(group) for group in every)

        raw = fit.raw or FitResult()
        self.history.append(FrameState(
            players=players,
            referees=referees,
            goalkeepers=goalkeepers,
            balls=balls,
            keypoints=list(self.keypoints),
            homography_ok=self.homography_ok,
            projected=projected,
            rejected=total - projected,
            camera=fit.camera,
            homography_quality=raw.quality,
            homography_smoothed=fit.smoothed,
            homography_held=fit.held_over,
            homography_reason=self.homography_tracker.last_reason,
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

    def stats(self) -> dict:
        """Whole-clip summary, printed at the end of an analysis run."""
        cameras = [state.camera.focal_px for state in self.history if state.camera]
        projected = sum(state.projected for state in self.history)
        total = sum(state.projected + state.rejected for state in self.history)
        return {
            "frames": len(self.history),
            "cuts": self.cuts,
            "resyncs": self.resyncs,
            "fit_quality": self.homography_tracker.fit_quality(),
            "homography": self.homography_tracker.stats(),
            "projected": projected,
            "unprojected": total - projected,
            "projection_rate": round(projected / total, 3) if total else 0.0,
            "median_focal_px": round(float(np.median(cameras)), 1) if cameras else None,
            "tracks": self.tracker.stats(),
        }
