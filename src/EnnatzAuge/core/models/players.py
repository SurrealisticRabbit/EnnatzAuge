from .base import UltralyticsModel


class PlayersModel(UltralyticsModel):
    """Detection model: player, goalkeeper, referee, ball, other boxes."""

    task = "detect"