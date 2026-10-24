from .base import UltralyticsModel


class PitchModel(UltralyticsModel):
    """Pose model: 29 pitch keypoints per frame."""

    task = "pose"