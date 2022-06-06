#   Requires python >=3.9
from typing import Tuple, List
import sys
# import numpy as np
import random
import math
from xmlrpc.client import Boolean
import matplotlib.pyplot as plt


class Point2D:
    """ Defines point in 2D space

    Defines a point (x,y) in 2D space and calculates the distance between this point and another
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
        self.length = self._distance()

    def Copy(self):
        return Route(self.route[:])

    def __repr__(self) -> str:
        Str = "["
        for p in self.route:
            Str += str(p)
        Str += "]"
        return Str

    # def calculate_length(self) -> float:
    #     r = self.route
    #     [r1.distance_to(r2) for r1, r2 in zip(r, )


    def _distance(self, index=None) -> float:
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
            total += self.route[N-1].distance_to(self.route[0])
        elif not hasattr(index, "__len__"):
            rng = [
                ((index-1) % N),
                (index % N),
                ((index+1) % N)
            ]
            N = len(rng)
        for n in range(N-1):
            total += self.route[rng[n]].distance_to(self.route[rng[n+1]])
        return total

    def _swap_points(self, index1: int, index2: int) -> None:
        """ Swap two points in list and nothing else """
        r = self.route
        r[index1], r[index2] = r[index2], r[index1]

    def swap_points(self, index1: int, index2: int = None) -> float:
        """ Swap the position of two points along the route

        If only one point is selected, it is swapped with the next point along the route.

        The difference in the route distance before and after switching is calculated and this difference is returned by the function
        """
        if index2 is None:
            index2 = (index1+1) % len(self)
        old_dist = self._distance(index1) + self._distance(index2)

        # Swap adjacent points on route and calculate change in distance
        self._swap_points(index1, index2)
        new_dist = self._distance(index1) + self._distance(index2)
        delta = new_dist - old_dist
        self.length += delta
        return delta

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


class SimAnneal:
    """ Simulated annealing class for TSP """

    def __init__(self, route: Route, initial_temperature: float = 1.0, temperature_decrement: float = .1, beta: float = 2.0) -> None:
        self.route = route
        self.temperature = initial_temperature
        self.temperature_decrement = temperature_decrement
        self.iteration = 0
        self.beta = beta

    def Update(self, method:str="MovePoint"):
        assert(method in ["MovePoint", "SwapPoints"])
        old_route = self.route.Copy()
        if method=="MovePoint":
            pass
        elif method=="SwapPoints":
            pass
        if not success:
            self.route = old_route



    def _energy(self, delta) -> float:
        return len(self.route) * delta / self.route.length

    def _select_points(self, k: int = 2, counts: List[int] = None) -> List[int]:
        return random.sample(range(len(self.route)), k=k, counts=counts)

    def _attempt_swap(self) -> Tuple[bool, float, float, float]:
        """ Attempt to swap two random points

        Greedy swap if better, otherwise with probability P = exp(-dE/T) where dE = N*dist_incr/total_distance

        Returns (success, delta, dE, val):
            success = true if attempted swap was successful
            delta = Change in distance
            dE = Change in energy of system
            val = exp(-2*dE/T)
        """
        indx = self._select_points(2)
        dist = self.route.length
        delta = self.route.swap_points(indx[0], indx[1])
        dE = self._energy(delta)
        if dE <= 0:
            return (True, delta, dE, dE)
        val = math.exp(-self.beta*dE/self.temperature)
        swapped = True
        if val <= random.random():
            #   Return to original value
            swapped = False
            self.route._swap_points(indx[0], indx[1])
            self.route.length = dist
        return (swapped, delta, dE, val)

    def anneal(self, iterations: int, schedule: int) -> Tuple[List[int], List[float]]:
        """ Runs through the annealing process (WIP)

        1) Run through <schedule> number of iterations
        2) Reduce temperature: T'=T*(1-dt)
        3) Repeate <iterations> times
        """
        distance = []
        iteration = []
        for ni in range(iterations):
            for ns in range(schedule):
                self._attempt_swap()
                distance.append(self.route.length)
                iteration.append(self.iteration)
            self.temperature = self.temperature*(1-self.temperature_decrement)
        return (iteration, distance)


class SimAnnealRanked(SimAnneal):
    """Subclass of SimAnneal with weighted point selection

    Instead of selecting two points completely randomly, the probability of selecting any given point increases with the time since it was last selected
    """

    def __init__(self, route: Route, initial_temperature: float = 1, temperature_decrement: float = 0.1, initial_rank: int = 1) -> None:
        super().__init__(route, initial_temperature, temperature_decrement)
        self.initial_rank = initial_rank
        self.rank = [initial_rank]*len(route)

    def _select_points(self, k: int = 2, counts: List[int] = None) -> List[int]:
        indx = super()._select_points(k=k, counts=self.rank)
        self.rank = [r+1 for r in self.rank]
        for n in indx:
            self.rank[n] = self.initial_rank
        return indx


class STun(SimAnneal):
    def __init__(self, route: Route, initial_temperature: float = 1.0, temperature_decrement: float = 0.1, beta: float = 2.0, gamma: float = 1.0) -> None:
        super().__init__(route, initial_temperature, temperature_decrement, beta)
        self.gamma = gamma
        self.optimum = self.route.length
        self.f_stun = float("inf")

    def _energy(self, delta) -> float:
        # return
        dist = self.route.length
        opt = self.optimum
        if dist < opt:
            self.optimum = dist
            self.f_stun = 0
            return 0
        delta = dist - opt
        # dE = len(self.route) * delta / opt
        dE = len(self.route) * delta / dist
        f_stun = 1.0-math.exp(-self.gamma*dE)
        df = f_stun - self.f_stun
        self.f_stun = f_stun
        return df

    def _attempt_swap(self) -> Tuple[float, float, float]:
        f_stun = self.f_stun
        result = super()._attempt_swap()
        if not result[0]:
            self.f_stun = f_stun
        return result


def generate_points(N=20, maxPos=Point2D(x=100, y=100)) -> List[Point2D]:
    """ Generate random list of points """
    if N is None or N < 2:
        N = 20
    return [Point2D(x=random.random()*maxPos.x, y=random.random()*maxPos.y) for n in range(N)]


def generate_route(N: int = None) -> Route:
    return Route(generate_points(N))


# def main(*args) -> None:
#     """
#         main function for testing and debugging purposes
#     """
#     plt.ion()
#     NPoints = 50
#     ITER_SCHED = 1000
#     ITER_TOTAL = 10000
#     dT = 0.001
#     route = generate_route(NPoints)
#     TSP1 = SimAnneal(route.Copy(), temperature_decrement=dT)
#     TSP2 = SimAnneal(route.Copy(), temperature_decrement=dT/10)
#     TSP3 = SimAnneal(route.Copy(), temperature_decrement=dT/2, initial_temperature=.5)
#     # TSP1 = STun(route.Copy(), temperature_decrement=dT, gamma=10.0)
#     # TSP2 = STun(route.Copy(), temperature_decrement=dT, initial_temperature=.1)
#     # TSP3 = STun(route.Copy(), temperature_decrement=dT, initial_temperature=.1)
#     p = TSP2.route.plot()
#     dist1 = []
#     dist2 = []
#     dist3 = []
#     for n in range(ITER_TOTAL):
#         (_, d) = TSP1.anneal(1, ITER_SCHED)
#         dist1 += d
#         (_, d) = TSP2.anneal(1, ITER_SCHED)
#         dist2 += d
#         (_, d) = TSP3.anneal(1, ITER_SCHED)
#         dist3 += d
#         Str = f"{n}: D = [{TSP1.route.distance:.2f}, {TSP2.route.distance:.2f}, {TSP3.route.distance:.2f}], T = [{TSP1.temperature:.4g}, {TSP2.temperature:.4g}, {TSP3.temperature:.4g}]"
#         print(Str)
#         plt.gca().clear()
#         TSP1.route.plot()
#         TSP2.route.plot()
#         TSP3.route.plot()
#         # plt.plot(dist)
#         plt.title(Str)
#         plt.draw()
#         plt.pause(1e-3)
#     plt.show(block=True)

def main(*args) -> None:
    """
        main function for testing and debugging purposes
    """
    plt.ion()
    NPoints = 10
    N_TSP = 10
    ITER_SCHED = N_TSP*10
    ITER_PLOT = 1
    ITER_TOTAL = 1000000
    route = generate_route(NPoints)
    TSP = [SimAnneal(route.Copy(), temperature_decrement=.01,
                     initial_temperature=.5, beta=3) for n in range(N_TSP)]
    plt.gca().clear()
    for tsp in TSP:
        tsp.route.plot()
    plt.draw()
    plt.pause(1)
    for ni in range(ITER_TOTAL):
        for tsp in TSP:
            if tsp.temperature < 10*tsp.temperature_decrement:
                tsp.temperature_decrement /= 10
            tsp.anneal(ITER_PLOT, ITER_SCHED)
        plt.gca().clear()
        dist = []
        for tsp in TSP:
            tsp.route.plot()
            dist.append(tsp.route.length)
        # dist = [tsp.route.distance for tsp in TSP]
        # mn = min([tsp.route.distance for tsp in TSP])
        plt.title(
            f"{ni}: D_min={min(dist):.4g}, D_max={max(dist):.4g}, D_av={sum(dist)/len(dist):.4g}, T={TSP[0].temperature:.4g}, dT={TSP[0].temperature_decrement}")
        plt.draw()
        plt.pause(1e-3)
    plt.show(block=True)


if __name__ == "__main__":
    main(sys.argv)
