"""Convert SN-GSR-2025 into two YOLO datasets.

Reads directly from the split archives, so the 35 GB of extracted JSON and
images never has to exist on disk at once. Peak usage is one archive plus the
growing output.

Outputs, under the EnnatzAuge datasets directory:

    yolo-players   detect: player, goalkeeper, referee, ball, other
    yolo-pitch     pose:   29 pitch keypoints

The keypoint derivation is the interesting part. The source annotations give
polylines per visible pitch line, and most of those lines are clipped at the
image border -- their first and last points are where the line leaves the
frame, not where it meets another line. Corners are therefore computed as
line-line intersections, and the centre circle is a least-squares fit.
"""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np

OBJECT_CLASS_IDS = {1: 0, 2: 1, 3: 2, 4: 3, 7: 4}
OBJECT_CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball", 4: "other"}

_ASPECT = 1920.0 / 1080.0
MIN_CHORD_LEN = 0.03
MIN_CIRCLE_POINTS = 5
MIN_CIRCLE_ARC_DEG = 20.0
MIN_VISIBLE_KEYPOINTS = 4
BORDER_TOLERANCE = 2

KEYPOINT_NAMES = [
    "sideline_top_left", "sideline_top_right",
    "sideline_bottom_left", "sideline_bottom_right",
    "big_rect_left_top_pt1", "big_rect_left_top_pt2",
    "big_rect_left_bottom_pt1", "big_rect_left_bottom_pt2",
    "big_rect_right_top_pt1", "big_rect_right_top_pt2",
    "big_rect_right_bottom_pt1", "big_rect_right_bottom_pt2",
    "small_rect_left_top_pt1", "small_rect_left_top_pt2",
    "small_rect_left_bottom_pt1", "small_rect_left_bottom_pt2",
    "small_rect_right_top_pt1", "small_rect_right_top_pt2",
    "small_rect_right_bottom_pt1", "small_rect_right_bottom_pt2",
    "center_line_top", "center_line_bottom", "field_center",
    "center_circle_top", "center_circle_bottom",
    "center_circle_left", "center_circle_right",
    "left_semicircle_right", "right_semicircle_left",
]
assert len(KEYPOINT_NAMES) == 29


def _build_flip_index() -> list[int]:
    """Left/right swap pairs, so horizontal-flip augmentation stays valid.

    Without this, Ultralytics' default ``fliplr=0.5`` mirrors the image but
    leaves the keypoints where they were, silently corrupting half the labels.
    """
    index = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
    flip = list(range(len(KEYPOINT_NAMES)))

    def pair(a: str, b: str) -> None:
        flip[index[a]] = index[b]
        flip[index[b]] = index[a]

    pair("sideline_top_left", "sideline_top_right")
    pair("sideline_bottom_left", "sideline_bottom_right")
    for rect in ("big_rect", "small_rect"):
        for band in ("top", "bottom"):
            pair(f"{rect}_left_{band}_pt1", f"{rect}_right_{band}_pt1")
            pair(f"{rect}_left_{band}_pt2", f"{rect}_right_{band}_pt2")
    pair("center_circle_left", "center_circle_right")
    pair("left_semicircle_right", "right_semicircle_left")
    return flip


KEYPOINT_FLIP_IDX = _build_flip_index()
assert len(KEYPOINT_FLIP_IDX) == 29


# ---------------------------------------------------------------------------
# keypoint geometry
# ---------------------------------------------------------------------------


def _dist(p, q):
    return math.hypot((q[0] - p[0]) * _ASPECT, q[1] - p[1])


def _chord(lines, name):
    pts = lines.get(name)
    if not pts or len(pts) < 2:
        return None
    p0 = (pts[0]["x"], pts[0]["y"])
    p1 = (pts[-1]["x"], pts[-1]["y"])
    if _dist(p0, p1) < MIN_CHORD_LEN:
        return None
    return p0, p1


def _line_intersect(p0, p1, q0, q1, min_angle_deg=2.0):
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = q0
    x4, y4 = q1
    d1x, d1y = x2 - x1, y2 - y1
    d2x, d2y = x4 - x3, y4 - y3
    denom = d1x * d2y - d1y * d2x
    len1 = math.hypot(d1x * _ASPECT, d1y)
    len2 = math.hypot(d2x * _ASPECT, d2y)
    if len1 < 1e-9 or len2 < 1e-9:
        return None
    if math.degrees(math.asin(min(1.0, abs(denom) / (len1 * len2)))) < min_angle_deg:
        return None  # near-parallel: the intersection would be unstable
    t = ((x3 - x1) * d2y - (y3 - y1) * d2x) / denom
    return (x1 + t * d1x, y1 + t * d1y)


def _isect(lines, name_a, name_b):
    a = _chord(lines, name_a)
    b = _chord(lines, name_b)
    if a is None or b is None:
        return None
    return _line_intersect(a[0], a[1], b[0], b[1])


def _fit_circle(points):
    pts = np.asarray(points, dtype=float)
    mean = pts.mean(axis=0)
    shifted = pts - mean
    x, y = shifted[:, 0], shifted[:, 1]
    a = np.column_stack([x, y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = math.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0))
    return (cx + mean[0], cy + mean[1]), r


def _arc_span_deg(points, center):
    angles = [math.atan2(p[1] - center[1], (p[0] - center[0]) * _ASPECT) for p in points]
    angles.sort()
    span = angles[-1] - angles[0]
    if len(angles) > 2:
        gaps = [b - a for a, b in zip(angles, angles[1:])]
        span = max(span, 2 * math.pi - max(gaps))
    return math.degrees(span)


def _circle_from(lines, name):
    pts = lines.get(name)
    if not pts or len(pts) < MIN_CIRCLE_POINTS:
        return None
    xy = [(p["x"], p["y"]) for p in pts]
    center, r = _fit_circle(xy)
    if r <= 1e-6 or _arc_span_deg(xy, center) < MIN_CIRCLE_ARC_DEG:
        return None
    return center, r


def _circle_line_intersections(center, r, p0, p1):
    cx, cy = center
    x1, y1 = p0
    x2, y2 = p1
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - cx, y1 - cy
    a = dx * dx + dy * dy
    if a < 1e-12:
        return []
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    sq = math.sqrt(disc)
    return [(x1 + t * dx, y1 + t * dy)
            for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a))]


def derive_keypoints(lines: dict) -> dict[str, tuple[float, float] | None]:
    """``{line_name: [{"x","y"}, ...]}`` -> ``{keypoint_name: (x, y) or None}``.

    Coordinates are left unclipped: the true pitch point is often off-screen,
    and YOLO accepts out-of-range values as long as they are not wild.
    """
    kp: dict[str, tuple[float, float] | None] = {}

    kp["sideline_top_left"] = _isect(lines, "Side line top", "Side line left")
    kp["sideline_top_right"] = _isect(lines, "Side line top", "Side line right")
    kp["sideline_bottom_left"] = _isect(lines, "Side line bottom", "Side line left")
    kp["sideline_bottom_right"] = _isect(lines, "Side line bottom", "Side line right")

    for side in ("left", "right"):
        goal_line = f"Side line {side}"
        big_main = f"Big rect. {side} main"
        small_main = f"Small rect. {side} main"
        for band in ("top", "bottom"):
            box = f"Big rect. {side} {band}"
            kp[f"big_rect_{side}_{band}_pt1"] = _isect(lines, box, goal_line)
            kp[f"big_rect_{side}_{band}_pt2"] = _isect(lines, box, big_main)
            box = f"Small rect. {side} {band}"
            kp[f"small_rect_{side}_{band}_pt1"] = _isect(lines, box, goal_line)
            kp[f"small_rect_{side}_{band}_pt2"] = _isect(lines, box, small_main)

    kp["center_line_top"] = _isect(lines, "Middle line", "Side line top")
    kp["center_line_bottom"] = _isect(lines, "Middle line", "Side line bottom")
    for name in ("field_center", "center_circle_top", "center_circle_bottom",
                 "center_circle_left", "center_circle_right"):
        kp[name] = None

    circle = _circle_from(lines, "Circle central")
    mid = _chord(lines, "Middle line")
    if circle is not None:
        center, r = circle
        kp["field_center"] = center
        if mid is not None:
            along = _circle_line_intersections(center, r, mid[0], mid[1])
            if len(along) == 2:
                ref = kp["center_line_top"] or mid[0]
                along.sort(key=lambda p: _dist(p, ref))
                kp["center_circle_top"], kp["center_circle_bottom"] = along
            dx, dy = mid[1][0] - mid[0][0], mid[1][1] - mid[0][1]
            across = _circle_line_intersections(center, r, center,
                                                (center[0] - dy, center[1] + dx))
            if len(across) == 2:
                across.sort(key=lambda p: p[0])
                kp["center_circle_left"], kp["center_circle_right"] = across

    for name, arc_name, box_name, pick in (
        ("left_semicircle_right", "Circle left", "Big rect. left main", max),
        ("right_semicircle_left", "Circle right", "Big rect. right main", min),
    ):
        kp[name] = None
        arc = _circle_from(lines, arc_name)
        box = _chord(lines, box_name)
        if arc is not None and box is not None:
            sols = _circle_line_intersections(arc[0], arc[1], box[0], box[1])
            if sols:
                kp[name] = pick(sols, key=lambda p: p[0])

    return kp


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------


def players_label(annotations: Iterable[dict], image: dict) -> list[str]:
    """YOLO detection rows for one frame: ``cls cx cy w h``, normalised."""
    width, height = image["width"], image["height"]
    rows = []
    for ann in annotations:
        cls = OBJECT_CLASS_IDS.get(ann.get("category_id"))
        box = ann.get("bbox_image")
        if cls is None or not box:
            continue
        w = box["w"] / width
        h = box["h"] / height
        if w <= 0 or h <= 0:
            continue
        rows.append(f"{cls} {box['x_center'] / width:.6f} "
                    f"{box['y_center'] / height:.6f} {w:.6f} {h:.6f}")
    return rows


def pitch_label(lines: dict) -> str | None:
    """One YOLO pose row: box, then 29 x (x, y, visibility).

    Returns None when too few keypoints are derivable for the frame to teach
    anything.
    """
    kp = derive_keypoints(lines)
    if sum(1 for v in kp.values() if v is not None) < MIN_VISIBLE_KEYPOINTS:
        return None

    raw = [(p["x"], p["y"]) for pts in lines.values() for p in pts]
    if not raw:
        return None
    xs = [p[0] for p in raw]
    ys = [p[1] for p in raw]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    parts = [f"0 {(x_min + x_max) / 2:.6f} {(y_min + y_max) / 2:.6f} "
             f"{max(x_max - x_min, 1e-4):.6f} {max(y_max - y_min, 1e-4):.6f}"]
    for name in KEYPOINT_NAMES:
        value = kp.get(name)
        parts.append("0.000000 0.000000 0"
                     if value is None
                     else f"{value[0]:.6f} {value[1]:.6f} 2")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------


def _ensure_layout(root: Path, splits: Iterable[str], with_images: bool) -> None:
    for split in splits:
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        if with_images:
            (root / "images" / split).mkdir(parents=True, exist_ok=True)


def _write_configs(players_dir: Path, pitch_dir: Path) -> None:
    import yaml

    (players_dir / "data.yaml").write_text(yaml.safe_dump({
        "path": str(players_dir),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": OBJECT_CLASS_NAMES,
    }, sort_keys=False), encoding="utf-8")

    (pitch_dir / "data.yaml").write_text(yaml.safe_dump({
        "path": str(pitch_dir),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {0: "pitch"},
        "kpt_shape": [len(KEYPOINT_NAMES), 3],
        "flip_idx": KEYPOINT_FLIP_IDX,
    }, sort_keys=False), encoding="utf-8")

    (pitch_dir / "keypoints.txt").write_text(
        "# keypoint index -> name, in label order\n"
        + "".join(f"{i}\t{n}\n" for i, n in enumerate(KEYPOINT_NAMES)),
        encoding="utf-8",
    )


def _link_pitch_images(pitch_dir: Path, players_dir: Path, splits: Iterable[str]) -> None:
    """Point pitch images at the players tree instead of copying ~28 GB."""
    for split in splits:
        source = players_dir / "images" / split
        link = pitch_dir / "images" / split
        if not source.exists():
            continue
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.exists():
            shutil.rmtree(link)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source.resolve(), target_is_directory=True)


class Converter:
    """Converts split archives into the two YOLO datasets."""

    def __init__(self, datasets_root: Path, datasets=("players", "pitch"),
                 stride: int = 1, splits: Iterable[str] = ("train", "valid", "test"),
                 pitch_images: str = "link"):
        self.datasets_root = Path(datasets_root)
        self.players_dir = self.datasets_root / "yolo-players"
        self.pitch_dir = self.datasets_root / "yolo-pitch"
        self.want_players = "players" in datasets
        self.want_pitch = "pitch" in datasets
        self.stride = max(1, int(stride))
        self.splits = list(splits)
        self.pitch_images = pitch_images  # "link", "copy" or "none"

    # -- one sequence ------------------------------------------------------

    def _convert_sequence(self, zf, prefix: str, seq_name: str, label_member: str,
                          split: str, counts: dict, stats: dict) -> None:
        data = json.loads(zf.read(label_member))
        stats["sequences"] += 1

        declared = {c["id"]: c.get("name") for c in data.get("categories", [])}
        for cid, yolo_id in OBJECT_CLASS_IDS.items():
            if cid in declared and declared[cid] != OBJECT_CLASS_NAMES[yolo_id]:
                raise ValueError(
                    f"{seq_name}: category {cid} is {declared[cid]!r}, expected "
                    f"{OBJECT_CLASS_NAMES[yolo_id]!r} - class map is out of date"
                )

        objects: dict[str, list] = {}
        pitches: dict[str, dict] = {}
        for ann in data.get("annotations", []):
            if ann.get("supercategory") == "object":
                objects.setdefault(ann["image_id"], []).append(ann)
            elif ann.get("supercategory") == "pitch":
                pitches[ann["image_id"]] = ann.get("lines") or {}

        im_dir = data.get("info", {}).get("im_dir", "img1")

        for index, image in enumerate(data.get("images", [])):
            if index % self.stride or not image.get("is_labeled", False):
                continue

            stem = f"{seq_name}_{Path(image['file_name']).stem}"
            member = f"{prefix}/{im_dir}/{image['file_name']}"

            if self.want_players:
                image_path = self.players_dir / "images" / split / f"{stem}.jpg"
                if not image_path.exists() and not self._extract(zf, member, image_path):
                    stats["missing_images"] += 1
                    continue
                rows = players_label(objects.get(image["image_id"], []), image)
                (self.players_dir / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(rows), encoding="utf-8")
                counts["players"][split] += 1

            if self.want_pitch:
                lines = pitches.get(image["image_id"])
                row = pitch_label(lines) if lines else None
                if row is None:
                    stats["skipped_pitch"] += 1
                    continue
                if self.pitch_images == "copy":
                    self._extract(zf, member,
                                  self.pitch_dir / "images" / split / f"{stem}.jpg")
                (self.pitch_dir / "labels" / split / f"{stem}.txt").write_text(
                    row, encoding="utf-8")
                counts["pitch"][split] += 1

    @staticmethod
    def _extract(zf, member: str, target: Path) -> bool:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, 1024 * 1024)
            return True
        except KeyError:
            return False

    # -- one split ---------------------------------------------------------

    def convert_split(self, zip_path: Path, split: str) -> tuple[dict, dict]:
        counts = {"players": {split: 0}, "pitch": {split: 0}}
        stats = {"sequences": 0, "skipped_pitch": 0, "missing_images": 0}

        if self.want_players:
            _ensure_layout(self.players_dir, [split], with_images=True)
        if self.want_pitch:
            _ensure_layout(self.pitch_dir, [split],
                           with_images=self.pitch_images == "copy")

        with zipfile.ZipFile(zip_path) as zf:
            sequences = {}
            for name in zf.namelist():
                if name.endswith("Labels-GameState.json"):
                    prefix = name.rsplit("/", 1)[0]
                    sequences[prefix.rsplit("/", 1)[-1]] = (prefix, name)

            print(f"  {split}: {len(sequences)} sequences", flush=True)
            ordered = sorted(sequences.items())
            for position, (seq_name, (prefix, member)) in enumerate(ordered, 1):
                self._convert_sequence(zf, prefix, seq_name, member, split,
                                       counts, stats)
                if position % 10 == 0 or position == len(ordered):
                    print(f"    {position}/{len(ordered)} sequences, "
                          f"players={counts['players'][split]}, "
                          f"pitch={counts['pitch'][split]}", flush=True)

        return counts, stats

    # -- driver ------------------------------------------------------------

    def finish(self) -> None:
        # Configs are written for both datasets together, so both roots must
        # exist even when only one was converted (a pitch-only build would
        # otherwise crash writing the players yaml and vice versa).
        if self.want_players or self.want_pitch:
            self.players_dir.mkdir(parents=True, exist_ok=True)
            self.pitch_dir.mkdir(parents=True, exist_ok=True)

        if self.want_players and self.want_pitch and self.pitch_images == "link":
            _link_pitch_images(self.pitch_dir, self.players_dir, self.splits)
        if self.want_players or self.want_pitch:
            _write_configs(self.players_dir, self.pitch_dir)

    def convert_archive(self, zip_path: Path, split: str) -> tuple[dict, dict]:
        counts, stats = self.convert_split(zip_path, split)
        return counts, stats
