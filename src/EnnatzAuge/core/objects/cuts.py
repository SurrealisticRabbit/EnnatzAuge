"""Hard cuts between shots.

Highlight clips are not continuous footage. ``fb_cut.mp4`` is 11 minutes with
~187 cuts: every cut switches to a different camera, a different part of the
pitch, and different players. Anything that assumes temporal continuity -- a
homography prior, a jump gate, an identity tracker -- is wrong across one.

So a cut is detected and the temporal state is *reset*, rather than the frame
being discarded. Discarding it was the bug: a post-cut frame has a perfectly
good homography, and throwing it away also left the tracker trying to match new
players against old tracks.

Detection is deliberately cheap and physical. A cut changes essentially every
pixel, while even a whip pan keeps most pixels similar, so the mean absolute
difference of a small greyscale thumbnail separates them cleanly. That is ~0.06
MP of work per frame, against 2.5 MP of inference.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

# Thumbnail the comparison runs on; large enough to see structure, small enough
# to be free next to the detector.
THUMB = (320, 180)
# Absolute floor for a cut. Real motion sits around 6 here (measured on the
# source clip), cuts around 25-50.
MIN_DIFF = 18.0
# And it must stand out from recent motion by this factor, so a permanently
# busy sequence does not trigger on its normal level of change.
CONTRAST = 3.5
# Frames of history for the baseline, and how many are needed before the
# relative test is trusted.
BASELINE = 30
WARMUP = 5


class CutDetector:
    """Flags frames that begin a new shot."""

    def __init__(self, min_diff: float = MIN_DIFF, contrast: float = CONTRAST,
                 thumb: tuple[int, int] = THUMB):
        self.min_diff = float(min_diff)
        self.contrast = float(contrast)
        self.thumb = thumb
        self.history: deque[float] = deque(maxlen=BASELINE)
        self._previous: np.ndarray | None = None
        self.cuts = 0
        self.last_delta = 0.0

    def reset(self) -> None:
        """Forget the previous frame, e.g. when analysis restarts mid-clip."""
        self._previous = None
        self.history.clear()

    def update(self, frame) -> bool:
        """True when ``frame`` starts a new shot."""
        if frame is None:
            return False
        small = cv2.resize(frame, self.thumb, interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self._previous is None:
            self._previous = grey
            return False

        delta = float(np.abs(grey - self._previous).mean())
        self._previous = grey
        self.last_delta = delta

        cut = False
        if len(self.history) >= WARMUP:
            baseline = max(float(np.median(self.history)), 1.0)
            cut = delta > max(self.min_diff, self.contrast * baseline)

        # A cut is a one-off event, not the new normal: leaving it in the
        # baseline would make the following frame look quiet and mask a second
        # cut in a fast-cut sequence.
        if not cut:
            self.history.append(delta)
        if cut:
            self.cuts += 1
        return cut

    def stats(self) -> dict:
        return {"cuts": self.cuts, "last_delta": round(self.last_delta, 1)}
