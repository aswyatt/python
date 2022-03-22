import sys
from typing import Tuple, List
import numpy as np
import random

class Point2D:
    """ Defines point in 2D space

    Defines a point (x,y) in 2D space and calculates the distance between this point and a set of points
    """
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def distance(self, points) -> float:
        dx = self.x - points.x
        dy = self.y - points.y
        dist = np.sqrt((dx**2) + (dy**2))
        return dist

    def __repr__(self) -> str:
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Route:
    def __init__(self, route:List[Point2D]) -> None:
        """ Initialise a route as a list of points with a distance and fitness score.

        Distance is the total Euclidean distance between adjacent points along the route, with periodic boundary conditions (i.e. last point is adjacent to first point).

        Fitness score is simple the inverse of the total distance along the route.
        """
        self.route = route
        self.distance = self._calc_distance()
        self.fitness = self._calc_fitness()

    def _calc_distance(self, index=None) -> float:
        """ Calculate distance to traverse (part) of route

        """
        N = len(self.route)
        total = 0
        if index is None:
            for n in range(N-1):
                total += self.route[n].distance(self.route[n+1])
            total = self.route[N-1].distance(self.route[0])
        else:
            index = [
                ((index-1) % N),
                (index % N),
                ((index+1) % N),
                ((index+2) % N)
                ]
            for n in index[:-1]:
                total+=self.route[n].distance(self.route[n+1])
        return total

    def _calc_fitness(self) -> float:
        return 1/self.distance

    def swap_direction(self, index: int) -> None:
        N = len(self.route)
        index1 = (index+1) % N
        old_dist = self._calc_distance(index)
        self.route[index], self.route[index1] = self.route[index1], self.route[index]  # Swap adjacent points on route
        new_dist = self._calc_distance(index)
        self.distance += (new_dist - old_dist)
        self.fitness = self._calc_fitness()


def generate_points(
        N=20,
        maxPos:Point2D=Point2D(x=100,y=100)
    ) -> List[Point2D]:
    """ Generate random list of points
    """
    return [
        Point2D(
            x=random.random()*maxPos.x,
            y=random.random()*maxPos.y)
        for n in range(N)
        ]


def main(*args) -> None:
    print(generate_points())


if __name__ == "__main__":
    main(sys.argv)