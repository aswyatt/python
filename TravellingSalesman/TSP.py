#   Requires python >=3.9
from typing import Tuple, List
import sys
# import numpy as np
import random
import math
import matplotlib.pyplot as plt


class Point2D:
    """ Defines point in 2D space

    Defines a point (x,y) in 2D space and calculates the distance from this point to another
    """

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def distance_to(self, point) -> float:
        """ Calculates Euclidean distance between self and argument (both Point2D objects) """
        dx = self.x - point.x
        dy = self.y - point.y
        dist = math.sqrt((dx**2) + (dy**2))
        return dist

    def __repr__(self) -> str:
        return "(" + str(round(self.x, 2)) + "," + str(round(self.y, 2)) + ")"


class Route:
    def __init__(self, route: List[Point2D]) -> None:
        """ Initialise a route as a list of points with a total distance to circumnavigate them (back to the first point).

        Distance is the total Euclidean distance between adjacent points along the route, with periodic boundary conditions (i.e. last point is adjacent to first point).
        """
        self.route = route
        self.length = self.calc_length()

    def copy(self):
        return Route(self.route[:])

    def __repr__(self) -> str:
        Str = "[" + str(self.route[0])
        for p in self.route[1:]:
            Str += ", " + str(p)
        Str += "]"
        return Str

    def reverse(self):
        self.route.reverse()

    def calc_length(self) -> float:
        r = self.route
        z = zip(r, r[1:]+[r[0]])
        L = sum([r1.distance_to(r2) for (r1,r2) in z])
        return L

    def sub_length(self, index) -> float:
        """ Calculate length of paths to and from selected point"""
        r = self.route
        l1 = r[index].distance_to(r[index-1])
        l2 = r[index].distance_to(r[(index+1) % len(r)])
        return l1+l2

    def _swap_points(self, index1: int, index2: int) -> None:
        """ Swap two points in list and nothing else """
        r = self.route
        r[index1], r[index2] = r[index2], r[index1]

    def swap_points(self, index1:int, index2:int) -> float:
        """ Swap the position of two points along the route

        The difference in the route distance before and after switching is calculated and this difference is returned by the function
        """
        old_length = self.sub_length(index1) + self.sub_length(index2)
        self._swap_points(index1, index2)
        new_length = self.sub_length(index1) + self.sub_length(index2)
        length_change = new_length - old_length
        self.length += length_change
        return length_change

    def __len__(self) -> int:
        return len(self.route)

    def extract(self) -> Tuple[List[float], List[float]]:
        """ Returns two lists: ([x], [y]]) of points in route """
        return [p.x for p in self.route], [p.y for p in self.route]

    def plot(self) -> None:
        """ Simple plot of route (WIP) """
        (x, y) = self.extract()
        x.append(self.route[0].x)
        y.append(self.route[0].y)
        return plt.plot(x, y, "ko-", alpha=.1)

    def NN_Solution(self, Start:int = 0) -> None:
        pass


def generate_points(N=20, maxPos=Point2D(x=100, y=100)) -> List[Point2D]:
    """ Generate random list of points """
    if N is None or N < 2:
        N = 20
    return [Point2D(x=random.random()*maxPos.x, y=random.random()*maxPos.y) for n in range(N)]


def generate_route(N: int = None) -> Route:
    return Route(generate_points(N))


def main(*args) -> None:
    pass

if __name__ == "__main__":
    main(sys.argv)
