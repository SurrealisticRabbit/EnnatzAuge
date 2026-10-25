"""Terminal charts, so training progress is readable at a glance.

Ultralytics prints a table per epoch, which is fine to scroll back through but
useless for spotting a trend while watching. These are small, dependency-free
renderers: a sparkline for the one-liner and a framed chart for the live view.

Everything degrades to plain ASCII when the terminal cannot render box-drawing
or block characters, which is what happens on Windows consoles with a legacy
code page.
"""

from __future__ import annotations

import os
import shutil
import sys

# Blocks, best-first. The final fallback is pure ASCII.
_SPARK_SETS = [
    "▁▂▃▄▅▆▇█",
    ".:-=+*#",
]

_BAR_SETS = [
    ("█", "░", "├", "┤", "─", "│"),
    ("#", ".", "|", "|", "-", "|"),
]


def _pick(charsets):
    """Choose the richest charset the terminal can actually encode."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    for charset in charsets:
        candidate = charset if isinstance(charset, str) else "".join(charset)
        try:
            candidate.encode(encoding)
        except (UnicodeEncodeError, LookupError):
            continue
        return charset
    return charsets[-1]


def sparkline(values, width: int = 40) -> str:
    """Compress a series into one line of block characters."""
    data = [v for v in values if v is not None]
    if not data:
        return ""
    chars = _pick(_SPARK_SETS)
    if len(data) > width:
        # Downsample by averaging buckets rather than truncating: the shape of
        # the trend is the point, not the exact last N values.
        step = len(data) / width
        data = [
            sum(data[int(i * step):max(int((i + 1) * step), int(i * step) + 1)])
            / max(1, len(data[int(i * step):max(int((i + 1) * step), int(i * step) + 1)]))
            for i in range(width)
        ]

    low, high = min(data), max(data)
    if high - low < 1e-12:
        return chars[0] * len(data)
    span = high - low
    return "".join(
        chars[min(len(chars) - 1, int((v - low) / span * (len(chars) - 1)))]
        for v in data
    )


def bar(value: float, maximum: float, width: int = 24) -> str:
    """A single filled bar, for one current value."""
    filled, empty, *_ = _pick(_BAR_SETS)
    if maximum <= 0:
        return empty * width
    ratio = max(0.0, min(1.0, value / maximum))
    n = int(round(ratio * width))
    return filled * n + empty * (width - n)


def chart(series: dict[str, list[float]], height: int = 10,
          width: int | None = None, title: str = "") -> str:
    """Multi-line chart of one or more series, sharing a y axis.

    Series are drawn with different glyphs rather than colours, so it survives
    being piped to a file or a log.
    """
    glyphs = "o*+x#@"
    data = {name: [v for v in values if v is not None]
            for name, values in series.items()}
    data = {name: values for name, values in data.items() if values}
    if not data:
        return ""

    columns = width or min(72, max(30, shutil.get_terminal_size((80, 24)).columns - 12))
    height = max(3, height)

    all_values = [v for values in data.values() for v in values]
    low, high = min(all_values), max(all_values)
    if high - low < 1e-12:
        high = low + 1e-6
    span = high - low

    # Sample each series at the same horizontal positions.
    sampled = {}
    for name, values in data.items():
        if len(values) > columns:
            step = len(values) / columns
            values = [values[min(len(values) - 1, int(i * step))]
                      for i in range(columns)]
        elif len(values) < columns:
            values = values + [values[-1]] * (columns - len(values))
        sampled[name] = values

    grid = [[" "] * columns for _ in range(height)]
    for index, (name, values) in enumerate(sampled.items()):
        glyph = glyphs[index % len(glyphs)]
        row_index = len(data) - 1 - index
        for column, value in enumerate(values):
            # 1-based row indices to leave the axis title row free.
            norm = (value - low) / span
            row = height - 1 - int(round(norm * (height - 1)))
            row = max(0, min(height - 1, row))
            grid[row][column] = glyph if grid[row][column] == " " else "*"

    bar_char, _empty, left, right, horizontal, vertical = _pick(_BAR_SETS)
    lines = []
    if title:
        lines.append(title)
    for row_index, row in enumerate(grid):
        value = high - span * row_index / (height - 1)
        lines.append(f"{value:>8.4f} {vertical}{''.join(row)}{vertical}")
    lines.append(f"{'':>8} {left}{horizontal * columns}{right}")
    if len(data) > 1:
        legend = "  ".join(f"{glyphs[i % len(glyphs)]} {name}"
                           for i, name in enumerate(data))
        lines.append(f"{'':>8} {legend}")
    return "\n".join(lines)


class LiveChart:
    """Redraws a chart in place while training runs.

    The trainer prints its own per-epoch lines *after* this hook fires, so
    simply moving the cursor up would drift by however many lines it printed.
    Instead the cursor position of the block's first line is saved once
    (DECSC) and restored before every redraw (DECRC), and everything below is
    erased. The chart therefore stays put while the trainer's transient output
    is rewritten underneath it.

    Prints nothing at all when stdout is not a tty, so piping to a file or a
    job log still gets the plain per-epoch output rather than escape codes.
    """

    def __init__(self, title: str = "", height: int = 8,
                 fields: tuple[str, ...] = ("fitness", "mAP_50_95")) -> None:
        self.title = title
        self.height = height
        self.fields = fields
        self._rows: list[dict] = []
        self._drawn = 0
        self._saved = False
        self._enabled = (sys.stdout.isatty()
                         and not os.environ.get("ENNATZAUGE_NO_CHART"))

    def add(self, row: dict) -> None:
        self._rows.append(row)

    def render(self) -> str:
        series = {}
        for field_name in self.fields:
            values = [row.get(field_name) for row in self._rows]
            if any(v is not None for v in values):
                series[field_name] = values
        if not series or not self._rows:
            return ""

        last = self._rows[-1]
        fitness = last.get("fitness") or 0.0
        best = max((r.get("fitness") or 0.0) for r in self._rows)
        header = (f"{self.title}  epoch {last.get('epoch', '?')}"
                  f"  fitness {fitness:.4f}"
                  f"  best {best:.4f}"
                  f"  lr {last.get('lr') or 0.0:.1e}")
        block = chart(series, height=self.height, title=header)
        spark = sparkline([r.get("fitness") for r in self._rows], width=48)
        return f"{block}\n{'':>8} {spark}"

    def draw(self) -> None:
        if not self._enabled:
            return
        text = self.render()
        if not text:
            return

        out = ["\0337"] if not self._saved else ["\0338"]
        self._saved = True
        out.append("\033[J")  # erase the previous block and any stale lines
        for line in text.splitlines():
            out.append("\033[2K" + line + "\r\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self._drawn = len(text.splitlines())

    def finish(self) -> None:
        """Leave the final chart on screen."""
        if not self._enabled or not self._drawn:
            return
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._drawn = 0
        self._saved = False


def summary_line(row: dict) -> str:
    """One compact line per epoch, for the scrolling record."""
    epoch = row.get("epoch", "?")
    parts = [f"epoch {epoch:>3}"]
    for key, label in (("mAP_50", "mAP50"), ("mAP_50_95", "mAP50-95"),
                       ("fitness", "fit"), ("lr", "lr")):
        value = row.get(key)
        if value is None:
            continue
        parts.append(f"{label} {value:.1e}" if key == "lr" else f"{label} {value:.4f}")
    if row.get("pose_mAP_50_95") is not None:
        parts.append(f"pose {row['pose_mAP_50_95']:.4f}")
    return "  ".join(parts)
