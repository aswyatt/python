from typing import Tuple, List
import sys
# import numpy as np
import random
import math
import matplotlib.pyplot as plt

class Point2D:
    """ Defines point in 2D space

    Defines a point (x,y) in 2D space and calculates the distance between this point and another
    """
    def __init__(self, x:float, y:float) -> None:
        self.x = x
        self.y = y

    def distance(self, point) -> float:
        dx = self.x - point.x
        dy = self.y - point.y
        dist = math.sqrt((dx**2) + (dy**2))
        return dist

    def __repr__(self) -> str:
        return "(" + str(round(self.x, 2)) + "," + str(round(self.y, 2)) + ")"


class Route:
    def __init__(self, route:List[Point2D]) -> None:
        """ Initialise a route as a list of points with a total distance to circumnavigate them (back to the first point).

        Distance is the total Euclidean distance between adjacent points along the route, with periodic boundary conditions (i.e. last point is adjacent to first point).
        """
        self.route = route
        self.distance = self._calc_distance()

    def Copy(self):
        return Route(self.route[:])

    def __repr__(self) -> str:
        Str = "["
        for p in self.route:
            Str += str(p)
        Str += "]"
        return Str

    def _calc_distance(self, index=None) -> float:
        """ Calculate distance to traverse (part) of route

        If index is None: calculate distance through whole route back to start

        If index is scalar: calculate sum of distance from points immediately before and after current point to current point

        Otherwise calculate distance between points specified in list (non-periodic)

        The distance of the route is not updated
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

    def _swap_points(self, index1: int, index2:int) -> None:
        """ Swap two points in list and nothing else """
        r = self.route
        r[index1], r[index2] = r[index2], r[index1]

    def swap_points(self, index1: int, index2: int=None) -> float:
        """ Swap the position of two points along the route

        If only one point is selected, it is swapped with the next point along the route.

        The difference in the route distance before and after switching is calculated and this difference is returned by the function
        """
        if index2 is None:
            index2 = (index1+1) % len(self)
        old_dist = self._calc_distance(index1) + self._calc_distance(index2)

        # Swap adjacent points on route and calculate change in distance
        self._swap_points(index1, index2)
        new_dist = self._calc_distance(index1) + self._calc_distance(index2)
        delta = new_dist - old_dist
        self.distance += delta
        return delta

    def __len__(self) -> int:
        return len(self.route)

    def extract(self)->Tuple[List[float],List[float]]:
        """ Returns two lists: x and y co-ordinates of points in route """
        return [p.x for p in self.route], [p.y for p in self.route]

    def plot(self) -> None:
        """ Simple plot of route (WIP) """
        (x,y) = self.extract()
        x.append(self.route[0].x)
        y.append(self.route[0].y)
        return plt.plot(x, y, "o-")


class SimAnneal:
    """ Simulated annealing class for TSP """
    def __init__(self, route:Route, initial_temperature:float=1.0, temperature_decrement:float=.1) -> None:
        self.route = route
        self.temperature = initial_temperature
        self.temperature_decrement = temperature_decrement
        self.iteration = 0

    def _calculate_energy(self, delta):
        return len(self.route) * delta / self.route.distance

    def _select_points(self, k:int=2, counts:List[int]=None) -> List[int]:
        return random.sample(range(len(self.route)), k=k, counts=counts)

    def attempt_swap(self) -> Tuple[float, float]:
        """ Attempt to swap two random points

        Greedy swap if better, otherwise with probability P = exp(-dE/T) where dE = N*dist_incr/total_distance
        """
        indx = self._select_points(2)
        dist = self.route.distance
        delta = self.route.swap_points(indx[0], indx[1])
        dE = self._calculate_energy(delta)
        val = math.exp(-2*dE/self.temperature)
        if val<=random.random():
            #   Return to original value
            self.route._swap_points(indx[0], indx[1])
            self.route.distance = dist
        return (delta, dE, val)

    def anneal(self, iterations:int, schedule:int) -> Tuple[List[int], List[float]]:
        distance = []
        iteration = []
        for ni in range(iterations):
            for ns in range(schedule):
                self.attempt_swap()
                distance.append(self.route.distance)
                iteration.append(self.iteration)
            self.temperature = self.temperature*(1-self.temperature_decrement)
        return (iteration, distance)


class SimAnnealRanked(SimAnneal):
    """Subclass of SimAnneal with weighted point selection

    Instead of selecting two points completely randomly, the probability of selecting any given point increases with the time since it was last selected
    """
    def __init__(self, route:Route, initial_temperature:float=1, temperature_decrement:float=0.1, initial_rank:int=1) -> None:
        super().__init__(route, initial_temperature, temperature_decrement)
        self.initial_rank = initial_rank
        self.rank = [initial_rank]*len(route)

    def _select_points(self, k:int=2) -> List[int]:
        indx=super()._select_points(k=k, counts=self.rank)
        self.rank = [r+1 for r in self.rank]
        for n in indx:
            self.rank[n] = self.initial_rank
        return indx


def generate_points(
        N=20,
        maxPos:Point2D=Point2D(x=100,y=100)
    ) -> List[Point2D]:
    """ Generate random list of points """
    return [
        Point2D(
            x=random.random()*maxPos.x,
            y=random.random()*maxPos.y)
        for n in range(N)
        ]


def generate_route(N:int=None) -> Route:
    if N is None:
        P = generate_points()
    else:
        P = generate_points(N)
    return Route(P)


def main(*args) -> None:
    plt.ion()
    N = 50
    ITER = 10000
    dT = 0.01
    route = generate_route(N)
    TSP = SimAnneal(route.Copy(), temperature_decrement=dT)
    TSPR = SimAnnealRanked(route.Copy(), temperature_decrement=dT, initial_rank=10)
    p = TSPR.route.plot()
    dist = []
    distR = []
    for n in range(1000):
        (_, d) = TSP.anneal(1, ITER)
        dist += d
        (_, d) = TSPR.anneal(1, ITER)
        distR += d
        Str = f"{n}: {TSP.route.distance:.2f}, {TSPR.route.distance:.2f}, T = {TSPR.temperature:.4g}"
        print(Str)
        plt.gca().clear()
        TSP.route.plot()
        TSPR.route.plot()
        # plt.plot(dist)
        plt.title(Str)
        plt.draw()
        plt.pause(1e-3)

if __name__ == "__main__":
    main(sys.argv)