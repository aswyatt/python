from typing import Tuple
import numpy as np
class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def distance(self, points: Point) -> float:
        dx = self.x - points.x
        dy = self.y - points.y
        dist = np.sqrt((dx**2) + (dy**2))
        return dist

    def __repr__(self) -> str:
        return "(" + str(self.x) + "," + str(self.y) + ")"
