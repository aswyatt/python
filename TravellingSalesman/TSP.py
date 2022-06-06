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
        self.length = self.total_length()

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

    def total_length(self) -> float:
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


class SimAnneal:
    """ Simulated annealing class for TSP """

    def __init__(self, route: Route, initial_temperature: float = 1.0, temperature_decrement: float = .1, beta: float = 2.0) -> None:
        self.route = route
        self.temperature = initial_temperature
        self.temperature_decrement = temperature_decrement
        self.iteration = 0
        self.beta = beta

    def _update_success(self, energy_change:float) -> Tuple[bool, float]:
        success = True
        if energy_change>0:
            prob = math.exp(-self.beta*energy_change/self.temperature)
            if prob <= random.random():
                success = False
        else:
            prob = 1.0
        return (success, prob)

    def _move_point(self) -> float:
        r = self.route
        l0 = r.length
        p = r.route.pop(self._select_points(1)[0])
        dl = l0 - r.length
        N = len(r)
        max_prob = 0
        min_length_change = 0
        min_energy_change = 0
        for n in random.sample(range(N), N):
            l1 = p.distance_to(r.route[n-1])
            l2 = p.distance_to(r.route[n])
            length_change = l1+l2-dl
            energy_change = self._energy_change(length_change)
            (success, prob) = self._update_success(energy_change)
            if prob>max_prob:
                max_prob = prob
                min_length_change = length_change
                min_energy_change = energy_change
            if success:
                r.route.insert(n, p)
                return (success, prob, energy_change, length_change)
        return (False, max_prob, min_energy_change, min_length_change)



    def _swap_points(self) -> Tuple[bool, float, float, float]:
        """ Swap the locations of two random (unique) points along the route """
        ind = self._select_points(2)
        length_change = self.route.swap_points(ind[0], ind[1])
        energy_change = self._energy_change(length_change)
        (success, prob) = self._update_success(energy_change)
        return (success, prob, energy_change, length_change)

    def _iterate_method(self) -> Tuple[bool, float, float, float]:
        return self._move_point()

    def iterate(self) -> Tuple[bool, float, float, float]:
        """ Attempt an iteration improvement. Keep the modification with probability P=min(1, exp(-beta*dE/T)), otherwise return to original.
        return (success, probability, energy_change, length_change)
        """
        self.iteration += 1
        old_route = self.route.copy()
        (success, success_prob, energy_change, length_change) = self._iterate_method()
        if not success:
            self.route = old_route
        return (success, success_prob, energy_change, length_change)

    def _energy_change(self, length_change) -> float:
        return len(self.route) * length_change / self.route.length

    def _select_points(self, num_points: int = 2, counts: List[int] = None) -> List[int]:
        return random.sample(range(len(self.route)), k=num_points, counts=counts)

    def anneal(self, iterations: int, schedule: int, logging:bool=False) -> Tuple[List[int], List[float]]:
        """ Runs through the annealing process (WIP)
        1) Run through <schedule> number of iterations
        2) Reduce temperature: T'=T*(1-dt)
        3) Repeate <iterations> times
        """
        if logging:
            length = []
            iteration = []
            success_prob = []
            energy_change = []
            success = []
            temperature = []
        for ni in range(iterations):
            for ns in range(schedule):
                (s, p, d, _) = self.iterate()
                if logging:
                    length.append(self.route.length)
                    iteration.append(self.iteration)
                    success_prob.append(p)
                    energy_change.append(d)
                    success.append(s)
                    temperature.append(self.temperature)
            self.temperature = self.temperature*(1-self.temperature_decrement)
        if logging:
            return (iteration, length, temperature, success_prob, energy_change, success)
        return tuple([None]*6)


class SimAnnealRanked(SimAnneal):
    """Subclass of SimAnneal with weighted point selection

    Instead of selecting two points completely randomly, the probability of selecting any given point increases with the time since it was last selected
    """

    def __init__(self, route: Route, initial_temperature: float = 1, temperature_decrement: float = 0.1, initial_rank: int = 1) -> None:
        super().__init__(route, initial_temperature, temperature_decrement)
        self.initial_rank = initial_rank
        self.rank = [initial_rank]*len(route)

    def _select_points(self, k: int = 2, counts: List[int] = None) -> List[int]:
        indx = super()._select_points(num_points=k, counts=self.rank)
        self.rank = [r+1 for r in self.rank]
        for n in indx:
            self.rank[n] = self.initial_rank
        return indx

#   Currently broken!!
class STun(SimAnneal):
    def __init__(self, route: Route, initial_temperature: float = 1.0, temperature_decrement: float = 0.1, beta: float = 2.0, gamma: float = 1.0) -> None:
        super().__init__(route, initial_temperature, temperature_decrement, beta)
        self.gamma = gamma
        self.optimum = self.route.length
        self.f_stun = float("inf")

    def _energy_change(self, delta) -> float:
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
    NPoints = 20
    N_TSP = 20
    ITER_SCHED = N_TSP*10
    ITER_PLOT = 10
    ITER_TOTAL = 100000
    dT = .01
    T0 = 5
    # route = generate_route(NPoints)
    # TSP = [SimAnneal(route.copy(), temperature_decrement=.01,
    #                  initial_temperature=.5, beta=3) for n in range(N_TSP)]
    points = generate_points(NPoints)
    TSP = [SimAnneal(
        Route(random.sample(points, NPoints)),
        temperature_decrement=dT,
        initial_temperature=T0,
        beta=3) for n in range(N_TSP)]
    plt.gca().clear()
    for tsp in TSP:
        tsp.route.plot()
    plt.draw()
    plt.pause(1)
    for ni in range(ITER_TOTAL):
        for tsp in TSP:
            # if tsp.temperature < 10*tsp.temperature_decrement:
                # tsp.temperature_decrement /= 2
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
