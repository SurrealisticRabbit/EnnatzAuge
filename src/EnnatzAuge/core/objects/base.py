
class BaseObject:
    """One tracked thing in a frame.

    ``position`` is the pitch-frame coordinate in metres once the homography for
    that frame has been solved, and ``None`` when it could not be projected.
    ``image_position`` keeps the raw foot point in pixels so the overlay can
    draw from the same object without re-running the model.
    """

    def __init__(self, id: int, name: str, position: tuple[float, float] | None):
        self.id = id
        self.name = name
        self.in_frame = False
        self.last_position = None
        self.current_position = position

        # Populated by VirtualPitch when ingesting detection output.
        self.class_name = name
        self.confidence = 0.0
        self.bbox: tuple[float, float, float, float] | None = None
        self.image_position: tuple[float, float] | None = None

    def __repr__(self) -> str:
        x, y = self.current_position if self.current_position else (None, None)
        if x is None:
            return f"<{type(self).__name__} {self.name} unprojected>"
        return f"<{type(self).__name__} {self.name} {x:.1f},{y:.1f}m>"