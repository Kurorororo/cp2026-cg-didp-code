import numpy as np


def vrp_round(x, n_decimal_places):
    return (np.floor(x * (10**n_decimal_places) + 0.5)).astype(np.int64)


class VRPTW:
    def __init__(
        self, n_vehicles, capacity, demand, earliest, latest, service_time, travel_time
    ):
        self.n_nodes = len(demand)
        self.n_vehicles = n_vehicles
        self.capacity = capacity
        self.demand = demand
        self.earliest = earliest
        self.latest = latest
        self.service_time = service_time
        self.travel_time = travel_time

    @classmethod
    def from_solomon(cls, vrptw, n_decimal_places=None):
        earliest = vrptw["time_window"][:, 0]
        latest = vrptw["time_window"][:, 1]
        service_time = vrptw["service_time"]
        travel_time = vrptw["edge_weight"]

        if n_decimal_places is not None:
            earliest = vrp_round(earliest, n_decimal_places)
            latest = vrp_round(latest, n_decimal_places)
            service_time = vrp_round(service_time, n_decimal_places)
            travel_time = vrp_round(travel_time, n_decimal_places)

        return cls(
            vrptw["vehicles"],
            vrptw["capacity"],
            vrptw["demand"],
            earliest,
            latest,
            service_time,
            travel_time,
        )

    def minify(self, n_vehicles, n_nodes):
        return VRPTW(
            n_vehicles,
            self.capacity,
            self.demand[:n_nodes],
            self.earliest[:n_nodes],
            self.latest[:n_nodes],
            self.service_time[:n_nodes],
            self.travel_time[:n_nodes, :n_nodes],
        )

    def validate(self, routes, cost):
        visited = set()
        recomputed_cost = 0

        for i, r in enumerate(routes):
            r_cost = self.validate_route(visited, r)

            if r_cost is None:
                print("The {}-th route {} is invalid".format(i, r))

                return False

            recomputed_cost += r_cost

        if cost is not None and recomputed_cost != cost:
            print(
                "Recomputed cost {} != reported cost {}".format(recomputed_cost, cost)
            )

            return False

        return True

    def validate_route(self, visited, route):
        current = 0
        load = 0
        time = 0
        cost = 0

        for next in route:
            cost += self.travel_time[current, next]
            load += self.demand[next]
            time = max(
                time + self.service_time[current] + self.travel_time[current, next],
                self.earliest[next],
            )

            if next in visited:
                print("Customer {} is already visited".format(next))

                return None

            if load > self.capacity:
                print(
                    "Load {} at node {} > capacity {}".format(load, next, self.capacity)
                )

                return None

            if time > self.latest[next]:
                print(
                    "Arrival time {} at node {} > latest time {}".format(
                        time, self.latest[next], next
                    )
                )

                return None

            current = next
            visited.add(next)

        cost += self.travel_time[current, 0]

        return cost

    def compute_route_cost(self, route):
        current = 0
        load = 0
        cost = 0

        for next in route:
            cost += self.travel_time[current, next]
            load += self.demand[next]
            current = next

        cost += self.travel_time[current, 0]

        return cost

    def get_edges(self):
        n = len(self.demand)
        edges = set()

        for i in range(n):
            for j in range(n):
                if (
                    i == 0
                    or j == 0
                    or self.earliest[i] + self.service_time[i] + self.travel_time[i, j]
                    <= self.latest[j]
                ) and i != j:
                    edges.add((i, j))

        return edges

    def compute_shortest_travel_time(self, edges):
        n = len(self.demand)
        ub = max(self.service_time.sum() + self.travel_time.sum(), self.latest[0])
        dist = np.full((n, n), ub)

        for i, j in edges:
            dist[i, j] = self.travel_time[i, j] + self.service_time[i]

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, j] > dist[i, k] + dist[k, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        return dist
