from TSP import *

#   Simulation parameters
IDENTICAL_START = False
NPoints = 100
N_TSP = 10
ITER_SCHED = N_TSP**2
ITER_PLOT = 1
ITER_TOTAL = 1000000
dT = .01
T0 = 1
BETA = 3

class SimAnneal:
    """ Simulated annealing class for TSP """

    def __init__(self, route: Route, initial_temperature: float = 1.0, temperature_decrement: float = .1, beta: float = 2.0) -> None:
        self.route = route
        self.temperature = initial_temperature
        self.temperature_decrement = temperature_decrement
        self.iteration = 0
        self.beta = beta
        self.history = 1000
        self.success_rate = [1]*6
        # self.attempts = [1]*6

    def _update_success(self, energy_change:float) -> Tuple[bool, float]:
        success = True
        if energy_change>0:
            prob = math.exp(-self.beta*energy_change/self.temperature)
            if prob <= random.random():
                success = False
        else:
            prob = 1.0
        return (success, prob)

    def _extract_route(self, ind:List[int]) -> List:
        r = self.route.route
        (ind1, ind2) = (min(ind), max(ind))
        ind1 = max(ind1, 1)
        r1 = r[ind1:ind2]
        self.route = Route(r[:ind1] + r[ind2:])
        return r1

    def _insert_route(self, route:List, indx:int) -> None:
        r = self.route.route
        self.route = Route(r[:indx] + route + r[indx:])

    def _modify_section(self, reverse:bool=False) -> Tuple[bool, float, float, float]:
        l0 = self.route.length
        ind = self._select_points(2)
        r = self._extract_route(ind)
        if reverse:
            r.reverse()
            self._insert_route(r, ind[0])
        else:
            self._insert_route(r, self._select_points(1)[0])
        length_change = self.route.length-l0
        energy_change = self._energy_change(length_change)
        (success, prob) = self._update_success(energy_change)
        return (success, prob, energy_change, length_change)

    def _move_point(self) -> Tuple[bool, float, float, float]:
        r = self.route.copy()
        indx = self._select_points(1)[0] - 1
        p = r.route.pop(indx)
        pm = r.route[indx-1]
        pp = r.route[indx]
        dl = pm.distance_to(pp) - \
            (p.distance_to(pm) + p.distance_to(pp))
        N = len(r)
        max_prob = 0
        min_length_change = 0
        min_energy_change = 0
        for n in random.sample(range(N), N):
            pm = r.route[n-1]
            pp = r.route[n]
            length_change = dl + \
                p.distance_to(pm) + p.distance_to(pp) - pm.distance_to(pp)
            energy_change = self._energy_change(length_change)
            (success, prob) = self._update_success(energy_change)
            if success:
                r.route.insert(n, p)
                r.length += length_change
                self.route = r
                return (success, prob, energy_change, length_change)
            elif prob>max_prob:
                max_prob = prob
                min_length_change = length_change
                min_energy_change = energy_change
        return (False, max_prob, min_energy_change, min_length_change)

    def _switch_direction(self) -> Tuple[bool, float, float, float]:
        indx = self._select_points(1)[0]-1
        r = self.route.route
        dl1 = r[indx-2].distance_to(r[indx-1]) + r[indx].distance_to(r[indx+1])
        r.insert(indx-1, r.pop(indx))
        dl2 = r[indx-2].distance_to(r[indx-1]) + r[indx].distance_to(r[indx+1])
        length_change = dl2-dl1
        self.route.length += length_change
        energy_change = self._energy_change(length_change)
        (success, prob) = self._update_success(energy_change)
        return (success, prob, energy_change, length_change)

    def _swap_points(self) -> Tuple[bool, float, float, float]:
        """ Swap the locations of two random (unique) points along the route """
        ind = self._select_points(2)
        length_change = self.route.swap_points(ind[0], ind[1])
        energy_change = self._energy_change(length_change)
        (success, prob) = self._update_success(energy_change)
        return (success, prob, energy_change, length_change)

    def _move_closest(self) -> Tuple[bool, float, float, float]:
        r = self.route.route
        ind0 = self._select_points(1)[0]-1
        p0 = r.pop(ind0)
        dist = float("inf")
        ind1 = None
        for (p, n) in zip(r, range(len(r))):
            d = p0.distance_to(p)
            if d<dist:
                ind1 = n
                dist = d
        ind2 = (ind1+1) % len(r)
        dl1 = p0.distance_to(r[ind0-1]) + p0.distance_to(r[ind0])
        dm = p0.distance_to(r[ind1-1])
        dp = p0.distance_to(r[ind2])
        dl2 = dist
        if dp<dm:
            dl2 += dp
            r.insert(ind2, p0)
        else:
            dl2 += dm
            r.insert(ind1, p0)
        length_change = dl2-dl1
        self.route.length += length_change
        energy_change = self._energy_change(length_change)
        (success, prob) = self._update_success(energy_change)
        return (success, prob, energy_change, length_change)

    def _iterate_method(self) -> Tuple[bool, float, float, float]:
        return self._move_point()
        choice = random.choices(range(6), self.success_rate)[0]
        if choice==0:
            ret = self._modify_section(False)
        elif choice==1:
            ret = self._modify_section(True)
        elif choice==2:
            ret = self._switch_direction()
        elif choice==3:
            ret = self._swap_points()
        elif choice==4:
            ret = self._move_point()
        else:
            ret = self._move_closest()

        for n in range(6):
            self.success_rate[choice] = (n==choice)*(ret[0]/self.history + (1-1/self.history)*self.success_rate[choice]) + (n!=choice)*(self.success_rate[choice] + .01/self.history)

        return ret

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

def main(*args) -> None:
    points = generate_points(NPoints)
    if IDENTICAL_START:
        TSP = [SimAnneal(
            Route(points),
            temperature_decrement=dT,
            initial_temperature=T0,
            beta=BETA) for n in range(N_TSP)]
    else:
        TSP = [SimAnneal(
            Route(random.sample(points, NPoints)),
            temperature_decrement=dT,
            initial_temperature=T0,
            beta=3) for n in range(N_TSP)]
    # plt.ion()   #   enable interactive mode
    plt.gca().clear()
    plt.get_current_fig_manager().full_screen_toggle()
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