from .base import BaseObject
from .ball import Ball
from .goalkeeper import Goalkeeper
from .player import Player
from .referee import Referee

__all__ = [
    "BaseObject",
    "VirtualPitch",
    "Player",
    "Referee",
    "Goalkeeper",
    "Ball",
]


def __getattr__(name):
    # VirtualPitch lives in .pitch, which imports the classes above; importing
    # it lazily keeps that dependency one-way and avoids a cycle.
    if name == "VirtualPitch":
        from .pitch import VirtualPitch

        return VirtualPitch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")