from .base import BaseObject

class Player(BaseObject):
    def __init__(self, id: int, name: str, position: tuple[float, float]):
        super().__init__(id, name, position)
        self.team = None
        self.number = None