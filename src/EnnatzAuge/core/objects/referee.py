from .base import BaseObject

class Referee(BaseObject):
    def __init__(self, id: int, name: str, position: tuple[float, float]):
        super().__init__(id, name, position)
        self.head_referee = False
        self.side_referee = False