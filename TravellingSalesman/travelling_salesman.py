import sys
from typing import Tuple, List
import numpy as np
import random
import math
import matplotlib.pyplot as plt

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
        return "(" + str(round(self.x, 2)) + "," + str(round(self.y, 2)) + ")"


class Route:
    def __init__(self, route:List[Point2D]) -> None:
        """ Initialise a route as a list of points with a distance and fitness score.

        Distance is the total Euclidean distance between adjacent points along the route, with periodic boundary conditions (i.e. last point is adjacent to first point).

        Fitness score is simple the inverse of the total distance along the route.
        """
        self.route = route
        self.distance = self._calc_distance()
        self.fitness = self._calc_fitness()

    def __repr__(self) -> str:
        Str = "["
        for p in self.route:
            Str += str(p)
        Str += "]"
        return Str

    def _calc_distance(self, index=None) -> float:
        """ Calculate distance to traverse (part) of route

        if index is None: calculate distance through whole route

        else if index is scalar: calculate distance from points immediately before and after current point to current point

        else calculate distance between points specified in list
        """
        N = len(self.route)
        total = 0
        if index is None:
            rng = range(N)
            total += self.route[N-1].distance(self.route[0])
        elif not hasattr(index, "__len__"):
            rng = [
                ((index-1) % N),
                (index % N),
                ((index+1) % N)
                ]
            N = len(rng)
        for n in range(N-1):
            total += self.route[rng[n]].distance(self.route[rng[n+1]])
        return total

    def _calc_fitness(self) -> float:
        return 1/self.distance

    def swap_direction(self, index1: int, index2: int=None) -> float:
        N = len(self.route)
        if index2 is None:
            index2 = (index1+1) % N
        old_dist = self._calc_distance(index1) + self._calc_distance(index2)
        self.route[index1], self.route[index2] = self.route[index2], self.route[index1]  # Swap adjacent points on route
        new_dist = self._calc_distance(index1) + self._calc_distance(index2)
        delta = new_dist - old_dist
        self.distance += delta
        self.fitness = self._calc_fitness()
        return delta

    def plot(self) -> None:
        x = []
        y = []
        for p in self.route:
            x.append(p.x)
            y.append(p.y)
        x.append(self.route[0].x)
        y.append(self.route[0].y)
        return plt.plot(x, y, "o-")


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
    plt.ion()
    N = 50
    IterMax = 1000000000
    route = Route(generate_points(N))
    p = route.plot()
    Str = str(route)
    T = .5
    dT = .001
    dE = 0.
    d = []
    for n in range(IterMax):
        indx = random.sample(range(N), k=2)
        d.append(route.distance)
        dE = N*route.swap_direction(indx[0], indx[1])/route.distance
        val = math.exp(-dE/T)
        if val<=random.random():
            route.swap_direction(indx[0], indx[1])
        if not (n%100):
            T = T*(1-dT)
            Str = f"{n}: {route.distance:.2f}, T = {T:.4g}, dE = {dE:.4g}, {val:.3g}"
            print(Str)
            plt.gca().clear()
            # route.plot()
            plt.plot(d)
            plt.title(Str)
            plt.draw()
            plt.pause(1e-3)

    print(Str)
    print(str(route))
    route.plot()
    plt.show()


if __name__ == "__main__":
    main(sys.argv)