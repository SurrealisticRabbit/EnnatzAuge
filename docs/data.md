
# yolo-players & yolo-pitch
Both come from GSR 2025, which is then split into one dataset containing players (ref, goalie, players & ball), then pitch (pitch keypoints).

- 1: Players & Pitch run
- 2: Homography (output to real coords)
- 3: Player Processing Step
    - Analyzes each player
    - 1: ReID - Player Re-Identification
    - 2: ShirtN - Shirt Number Detection
## Known limit: the pitch keypoints do not constrain a homography

Measured on `fb_cut.mp4` (250 frames) with `pitch-io-1`: every frame yields
**5 RANSAC inliers with 1 redundant point**, spread over only ~200-460 px of a
1920 px frame. A homography needs 4 points, so one spare is not verification.

The cause is what the pose model actually localises. Peak confidence per
keypoint index shows it is confident only on indices 20, 22 and 23-26 --
`center_line_top`, `field_center` and the four centre-circle points. Five of
those six sit within 9.15 m of the centre spot, on a 105 x 68 m pitch. The far
keypoints that would pin the transform down (`sideline_*`, `big_rect_*`,
`small_rect_*`, `semicircle_*`) peak at 0.35-0.58, i.e. the model has barely
learned them.

Consequences, all of them observed rather than predicted:

* a free homography fitted to a 9 m cluster is **underdetermined**, so its
  reprojection error is ~0.02 m by construction and tells you nothing;
* several different world frames fit the same points, so the solution flips.
  That looks like players teleporting across the pitch;
* derived camera values are nonsense (`f=559 px, h=5.4 m, roll=-58deg`).

`analysis summary` now reports `keypoint fit ... N inliers, M redundant` and
warns when redundancy is at or below 1. Treat that warning as the real state of
the pipeline.

**The fix is not in the analysis code.** It is in the pitch dataset or the
training: the model needs to learn the touchlines, penalty areas and
semicircles. Until it does, positions will drift no matter how the homography is
smoothed, gated or filtered.

## Hard cuts

`fb_cut.mp4` is a highlight reel: 20,391 frames with ~187 cuts. Analysis detects
them (`core/objects/cuts.py`) and resets temporal state at each, because a cut is
a different camera on different players. Without that, ids survive a cut and the
tracker matches new players onto old tracks.
