"""Identity and motion across frames.

The detector reports anonymous boxes. Without linking them frame to frame the
virtual pitch shows a scatter of dots that jump around, and every derived
quantity (speed, distance covered, who is where) is meaningless.

Two problems, handled separately:

* **Association** — deciding that *this* box is the same person as *that* box
  in the previous frame. Greedy IoU matching on image-space boxes, with a
  pitch-space distance fallback when the homography is available, because a
  player standing still against a panning camera has near-zero image motion but
  is obviously the same person.
* **Smoothing** — turning noisy per-frame pitch coordinates into a track. A
  constant-velocity Kalman filter per object; it also gives velocity, which is
  what makes speed and distance-covered possible later.
* **Global motion** — a slightly wrong homography displaces *everything* in the
  frame by the same vector. Per-object filtering cannot see that, so it is
  estimated separately and removed before association.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# IoU above which two boxes are considered the same object.
MIN_IOU = 0.25
# Fallback for a camera that panned: consecutive boxes for the same player can
# overlap by nothing at all, while their centres are still within a box width of
# each other. Expressed in multiples of the mean box size so it scales with
# distance from the camera.
MAX_CENTRE_RATIO = 1.5
# Pitch-space fallback: a player cannot move further than this between frames
# at 25 fps unless tracking has already failed.
MAX_PITCH_JUMP_M = 6.0
# Frames an object may go undetected before its track is dropped. Four frames is
# 0.16 s at 25 fps: long enough to survive a blink, short enough that a lost
# track does not linger as a ghost.
MAX_MISSES = 4
# A track must be seen this many times before it is drawn. A single sighting is
# usually a spurious box, and drawing it produces a dot that never moves.
MIN_HITS_TO_DRAW = 2
# How many consecutive frames a track may be drawn from prediction alone.
MAX_COAST = 2
# Detections below this confidence do not create new tracks.
MIN_CONF = 0.4

# Kalman tuning. Measurement noise in metres; process noise in m/s^2.
# Measurement noise is generous on purpose: after global motion compensation the
# projected position is the noisy quantity, and letting the filter trust it too
# much is what makes reported speeds jump around.
MEASUREMENT_NOISE_M = 2.5
PROCESS_NOISE = 1.2
# A measurement this far from where the motion model says the object should be
# is a projection error, not a movement: a player at 10 m/s covers 0.4 m per
# frame at 25 fps. Coasting through it is much cheaper than letting the filter
# chase it, because a single accepted teleport also corrupts the velocity.
MAX_INNOVATION_M = 4.0

# Global motion compensation.
GMC_WINDOW = 25           # frames of residual history used for the median
GMC_MIN_MATCHES = 4       # fewer people than this and the estimate is noise
GMC_MAX_SHIFT_M = 6.0     # larger than this the *fit* is wrong, not the frame
GMC_MAX_SPREAD_M = 2.5    # residuals must broadly agree to mean anything


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _box_scale(box) -> float:
    """A single length scale for a box: the mean of its width and height."""
    x1, y1, x2, y2 = box
    return max(1.0, (abs(x2 - x1) + abs(y2 - y1)) / 2.0)


def _box_centre(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class Kalman2D:
    """Constant-velocity Kalman filter on a 2D position.

    State is ``[x, y, vx, vy]``. Plain numpy so the package keeps its
    dependency list unchanged; scipy/filterpy would be a heavy import for four
    matrix multiplies.
    """

    def __init__(self, x: float, y: float) -> None:
        self.state = np.array([x, y, 0.0, 0.0], dtype=float)
        self.cov = np.eye(4) * 10.0
        self._initialised = False

    def predict(self, dt: float = 1.0) -> None:
        f = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
                     dtype=float)
        q = PROCESS_NOISE ** 2
        # Simple discrete white-noise acceleration model.
        q_mat = q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                              [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                              [dt ** 3 / 2, 0, dt ** 2, 0],
                              [0, dt ** 3 / 2, 0, dt ** 2]])
        self.state = f @ self.state
        self.cov = f @ self.cov @ f.T + q_mat
        self._initialised = True

    def update(self, x: float, y: float) -> None:
        h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        r = np.eye(2) * MEASUREMENT_NOISE_M ** 2
        z = np.array([x, y], dtype=float)

        innovation = z - h @ self.state
        s = h @ self.cov @ h.T + r
        gain = self.cov @ h.T @ np.linalg.inv(s)
        self.state = self.state + gain @ innovation
        self.cov = (np.eye(4) - gain @ h) @ self.cov
        self._initialised = True

    @property
    def position(self) -> tuple[float, float]:
        return float(self.state[0]), float(self.state[1])

    @property
    def velocity(self) -> tuple[float, float]:
        """m/s, given the filter was stepped with a real time delta."""
        return float(self.state[2]), float(self.state[3])

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return math.hypot(vx, vy)


@dataclass
class Track:
    """One object followed over time."""

    track_id: int
    class_name: str
    kalman: Kalman2D | None = None
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = 0.0
    misses: int = 0
    hits: int = 0
    age: int = 0
    measured: bool = False

    pitch: tuple[float, float] | None = None
    image_position: tuple[float, float] | None = None
    trail: list[tuple[float, float]] = field(default_factory=list)
    # Timestamp of each trail sample, in seconds since the track started. Kept
    # alongside the positions because a track that goes undetected for a while
    # has trail entries that are *not* one frame apart, and treating them as if
    # they were turns a 1 m gap into an apparent 6 m/s sprint.
    trail_time: list[float] = field(default_factory=list)
    clock: float = 0.0

    # Seconds per frame, needed to turn trail distances into m/s.
    dt: float = 1.0
    _speed: float = 0.0

    # Speed is measured over this many seconds of trail rather than from the
    # filter's instantaneous velocity: at 25 fps a single noisy frame produces
    # an apparent 30 m/s sprint that never happened.
    SPEED_WINDOW_S = 0.8
    # Below this many samples the window is too short to average anything out,
    # so a two-frame-old track reports no speed rather than a wrong one.
    MIN_SPEED_SAMPLES = 8
    TRAIL_LENGTH = 50

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def confirmed(self) -> bool:
        """A track worth showing: seen more than once."""
        return self.hits >= 2

    def _update_speed(self) -> None:
        count = len(self.trail)
        if count < 2:
            self._speed = 0.0
            return

        # Oldest sample still inside the window, falling back to the oldest
        # available once enough samples exist.
        target = self.trail_time[-1] - self.SPEED_WINDOW_S
        index = 0
        for candidate in range(count - 1, -1, -1):
            if self.trail_time[candidate] <= target:
                index = candidate
                break
        else:
            if count - 1 < self.MIN_SPEED_SAMPLES:
                self._speed = 0.0
                return

        elapsed = self.trail_time[-1] - self.trail_time[index]
        if elapsed <= 1e-6:
            self._speed = 0.0
            return
        first, last = self.trail[index], self.trail[-1]
        self._speed = math.hypot(last[0] - first[0], last[1] - first[1]) / elapsed

    def smooth(self, position: tuple[float, float] | None, dt: float = 1.0) -> None:
        """Feed a pitch-space measurement through the filter (``dt`` in seconds)."""
        if dt > 0:
            self.dt = dt
        # The clock advances every frame, measured or not, so elapsed time in
        # the trail is real time.
        self.clock += dt if dt > 0 else 0.0

        if position is None:
            if self.kalman is not None:
                self.kalman.predict(dt)
                self.pitch = self.kalman.position
            self.measured = False
            self._update_speed()
            return

        if self.kalman is None:
            self.kalman = Kalman2D(*position)
            self.pitch = position
        else:
            self.kalman.predict(dt)
            predicted = self.kalman.position
            if (self.hits >= 3 and math.hypot(position[0] - predicted[0],
                                              position[1] - predicted[1])
                    > MAX_INNOVATION_M):
                # Disagrees with the motion model: coast rather than absorb it.
                self.pitch = predicted
                self.measured = False
                self._update_speed()
                return
            self.kalman.update(*position)
            self.pitch = self.kalman.position

        self.measured = True
        if self.pitch is not None:
            self.trail.append(self.pitch)
            self.trail_time.append(self.clock)
            if len(self.trail) > self.TRAIL_LENGTH:
                self.trail.pop(0)
                self.trail_time.pop(0)
        self._update_speed()


class GlobalMotion:
    """Estimates the whole-frame component of apparent pitch motion.

    A homography that is even slightly wrong shifts every object in the frame by
    roughly the same vector, which reads as several metres per second of running
    that never happened. Player motion is close to zero-mean across a squad, so
    the *median* pitch displacement between a detection and the track it belongs
    to is a direct estimate of that error, and subtracting it restores the real
    motion.

    The estimate is deliberately conservative: it needs several agreeing
    correspondences, the residuals must be tight, and the shift is capped. When
    any of those fail the previous estimate is reused, so a genuinely bad frame
    cannot poison the correction.
    """

    def __init__(self, window: int = GMC_WINDOW,
                 min_matches: int = GMC_MIN_MATCHES,
                 max_shift: float = GMC_MAX_SHIFT_M,
                 max_spread: float = GMC_MAX_SPREAD_M):
        self.min_matches = max(2, int(min_matches))
        self.max_shift = float(max_shift)
        self.max_spread = float(max_spread)
        self.history: deque = deque(maxlen=max(1, int(window)))
        self.shift = (0.0, 0.0)
        self.accepted = 0
        self.rejected = 0

    def estimate(self, detections: list[dict], tracks) -> tuple[float, float]:
        """Pitch-space correction to subtract from this frame's detections."""
        residuals = []
        for detection in detections:
            here = detection.get("pitch")
            if here is None or detection["class_name"] == "ball":
                continue
            best, best_distance = None, MAX_PITCH_JUMP_M
            for track in tracks:
                if track.class_name != detection["class_name"] or track.pitch is None:
                    continue
                distance = math.hypot(here[0] - track.pitch[0],
                                      here[1] - track.pitch[1])
                if distance < best_distance:
                    best, best_distance = track, distance
            if best is not None:
                residuals.append((here[0] - best.pitch[0], here[1] - best.pitch[1]))

        if len(residuals) < self.min_matches:
            self.rejected += 1
            return self.shift

        values = np.asarray(residuals, dtype=float)
        median = np.median(values, axis=0)
        spread = float(np.median(np.linalg.norm(values - median, axis=1)))
        if spread > self.max_spread or float(np.linalg.norm(median)) > self.max_shift:
            self.rejected += 1
            return self.shift

        self.history.append(median)
        # Median over the window, not an EMA: one bad frame should not move the
        # correction at all, and the median guarantees that for a window of 3+.
        stable = np.median(np.asarray(self.history), axis=0)
        self.shift = (float(stable[0]), float(stable[1]))
        self.accepted += 1
        return self.shift

    def reset(self) -> None:
        """Forget the accumulated shift, e.g. after a cut."""
        self.history.clear()
        self.shift = (0.0, 0.0)

    def stats(self) -> dict:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "shift_m": round(math.hypot(*self.shift), 2),
        }


class Tracker:
    """Assigns detections to tracks, frame by frame.

    Greedy matching rather than Hungarian: with ~25 objects and no scipy, greedy
    over a sorted score list is within a few percent of optimal and far simpler.
    """

    def __init__(self, min_iou: float = MIN_IOU,
                 max_misses: int = MAX_MISSES,
                 max_centre_ratio: float = MAX_CENTRE_RATIO,
                 compensate: bool = False) -> None:
        self.min_iou = min_iou
        self.max_misses = max_misses
        self.max_centre_ratio = float(max_centre_ratio)
        self.tracks: dict[int, Track] = {}
        self.motion = GlobalMotion() if compensate else None
        self._next_id = 1
        self.frames = 0
        self.total_created = 0
        self.total_matches = 0
        self._shift: tuple[float, float] = (0.0, 0.0)
        # Speeds collected every frame rather than read off whatever happens to
        # be alive at the end: the last frame is not representative, and dead
        # tracks are gone by then.
        self._people_speeds: list[float] = []
        self._ball_speeds: list[float] = []
        # Age at which each track died, so mean lifetime describes the whole
        # clip rather than whatever happens to still be alive at the end.
        self._lifetimes: list[int] = []

    def reset(self, keep_ids: bool = True) -> None:
        """Drop all tracks, e.g. after a cut.

        ``keep_ids`` leaves the id counter alone so an id is never reused for a
        different person; downstream reporting, trails and the rendered video all
        rely on that.
        """
        self.tracks.clear()
        if not keep_ids:
            self._next_id = 1
        if self.motion is not None:
            self.motion.reset()
        self._shift = (0.0, 0.0)

    # -- association -------------------------------------------------------

    def _score(self, track: Track, detection: dict) -> float:
        """How likely is this detection to be this track? 0 = no match.

        Previous boxes are shifted by the camera's own movement first, so what
        is compared is object motion rather than object-plus-camera motion. That
        is what keeps ids stable through a pan.

        Tiered rather than a single similarity, so the strongest form of
        evidence always wins: overlap, then proximity, then pitch agreement.
        """
        score = 0.0
        if track.bbox and detection.get("bbox"):
            box = detection["bbox"]
            previous = track.bbox
            if self._shift != (0.0, 0.0):
                dx, dy = self._shift
                previous = (previous[0] + dx, previous[1] + dy,
                            previous[2] + dx, previous[3] + dy)
            iou = _iou(previous, box)
            if iou >= self.min_iou:
                # Definite overlap: 0.5 .. 1.0.
                score = 0.5 + 0.5 * iou
            else:
                # No overlap, but the box may simply have moved. Under a pan the
                # same player's next box can share no pixels at all, and IoU
                # alone would mint a new id every frame.
                ax, ay = _box_centre(previous)
                bx, by = _box_centre(box)
                distance = math.hypot(bx - ax, by - ay)
                reach = self.max_centre_ratio * (_box_scale(previous) + _box_scale(box)) / 2.0
                if distance <= reach:
                    score = 0.5 * (1.0 - distance / reach)

        if track.pitch is not None and detection.get("pitch") is not None:
            dx = detection["pitch"][0] - track.pitch[0]
            dy = detection["pitch"][1] - track.pitch[1]
            distance = math.hypot(dx, dy)
            if distance <= MAX_PITCH_JUMP_M:
                # Pitch agreement is real evidence but weaker than pixels: a
                # mediocre homography turns it into a lie more easily.
                score = max(score, 0.4 * (1.0 - distance / MAX_PITCH_JUMP_M))
        return score

    def update(self, detections: list[dict], dt: float = 1.0,
               pitch_trusted: bool = True,
               image_shift: tuple[float, float] | None = None) -> list[Track]:
        """Match this frame's detections and return the live tracks.

        ``detections`` entries need ``class_name``, ``bbox``, ``confidence`` and
        optionally ``pitch`` (metres, once the homography is known). ``dt`` is
        the frame interval in seconds.

        ``pitch_trusted=False`` means the homography for this frame was reused
        from an earlier one, so the pitch coordinates describe where objects
        *were*, not where they are. They are then ignored entirely: association
        falls back to image overlap and every track coasts on its filter. Using
        them would be worse than having none, because a stale projection is
        several metres out and the innovation gate would read that as the object
        teleporting -- which is how tracks used to die in droves.

        ``image_shift`` is how far a fixed point in the world moved across the
        image between the last frame and this one, in pixels. Subtracting it is
        what lets a box still be recognised after the camera has panned.
        """
        if not pitch_trusted:
            detections = [{**d, "pitch": None} for d in detections]

        self._shift = (float(image_shift[0]), float(image_shift[1])) if image_shift else (0.0, 0.0)
        self.frames += 1

        if pitch_trusted and self.motion is not None and self.tracks:
            dx, dy = self.motion.estimate(detections, self.tracks.values())
            if dx or dy:
                for detection in detections:
                    if detection.get("pitch") is not None:
                        detection["pitch"] = (detection["pitch"][0] - dx,
                                              detection["pitch"][1] - dy)

        # Candidate (score, track_id, detection_index) pairs, best first.
        candidates = []
        for track_id, track in self.tracks.items():
            for index, detection in enumerate(detections):
                if detection["class_name"] != track.class_name:
                    continue
                score = self._score(track, detection)
                if score > 0:
                    candidates.append((score, track_id, index))
        candidates.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        assignments: list[tuple[int, int]] = []
        for score, track_id, index in candidates:
            if track_id in matched_tracks or index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(index)
            assignments.append((track_id, index))

        # Carry matched tracks forward.
        for track_id, index in assignments:
            track = self.tracks[track_id]
            detection = detections[index]
            track.bbox = detection.get("bbox")
            track.confidence = detection.get("confidence", 0.0)
            track.image_position = detection.get("foot")
            track.hits += 1
            track.misses = 0
            track.age += 1
            track.smooth(detection.get("pitch"), dt)
            self.total_matches += 1
            if track.measured:
                if track.class_name == "ball":
                    self._ball_speeds.append(track.speed)
                else:
                    self._people_speeds.append(track.speed)

        # Unmatched tracks coast on the filter, then expire.
        for track_id, track in list(self.tracks.items()):
            if track_id in matched_tracks:
                continue
            track.misses += 1
            track.age += 1
            track.smooth(None, dt)
            if track.misses > self.max_misses:
                self._lifetimes.append(track.age)
                del self.tracks[track_id]

        # Unmatched detections start new tracks, but only with real confidence.
        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            if detection.get("confidence", 0.0) < MIN_CONF:
                continue
            track = Track(track_id=self._next_id,
                          class_name=detection["class_name"],
                          bbox=detection.get("bbox"),
                          confidence=detection.get("confidence", 0.0))
            track.image_position = detection.get("foot")
            track.hits = 1
            track.smooth(detection.get("pitch"), dt)
            self.tracks[self._next_id] = track
            self._next_id += 1
            self.total_created += 1

        return [t for t in self.tracks.values()
                if t.hits >= MIN_HITS_TO_DRAW and t.misses <= MAX_COAST]
    # -- reporting ---------------------------------------------------------

    def stats(self) -> dict:
        tracks = list(self.tracks.values())
        # Speeds are only comparable within a class: a ball at 25 m/s next to
        # players at 2 m/s averages into a meaningless number, so the summary
        # reports people and the ball separately.
        people = sorted(s for s in self._people_speeds if s > 0)
        balls = sorted(s for s in self._ball_speeds if s > 0)
        # Track turnover is the honest measure of whether identity is holding:
        # a well-behaved tracker creates roughly one track per object for the
        # whole clip, so anything near created/frames ≈ 1 per frame means ids
        # are churning every frame and every downstream statistic is noise.
        created = self.total_created or 0
        finished = self._lifetimes + [t.age for t in tracks]
        return {
            "live_tracks": len(tracks),
            "confirmed": sum(1 for t in tracks if t.confirmed),
            "created": created,
            "frames": self.frames,
            "turnover": round(created / self.frames, 3) if self.frames else 0.0,
            "mean_lifetime": round(sum(finished) / len(finished), 1) if finished else 0.0,
            "longest_lifetime": max(finished) if finished else 0,
            "matches": self.total_matches,
            "median_speed_ms": round(people[len(people) // 2], 2) if people else 0.0,
            "max_speed_ms": round(people[-1], 2) if people else 0.0,
            "max_ball_speed_ms": round(balls[-1], 2) if balls else 0.0,
            "gmc": self.motion.stats() if self.motion is not None else None,
        }
