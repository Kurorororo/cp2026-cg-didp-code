import numpy as np


def round_to(x, n_decimal_places):
    return np.floor(x * (10**n_decimal_places) + 0.5).astype(np.int32)


class PDPTW:
    def __init__(
        self,
        n_vehicles,
        capacity,
        demand,
        earliest,
        latest,
        service_time,
        travel_time,
        pickup_locations,
        delivery_locations,
    ):
        self.n_nodes = len(demand)
        self.n_vehicles = n_vehicles
        self.capacity = capacity
        self.demand = demand
        self.earliest = earliest
        self.latest = latest
        self.service_time = service_time
        self.travel_time = travel_time
        self.pickup_locations = pickup_locations
        self.delivery_locations = delivery_locations
        self.pickup_to_delivery = {
            p: d for p, d in zip(pickup_locations, delivery_locations)
        }
        self.delivery_to_pickup = {
            d: p for p, d in zip(pickup_locations, delivery_locations)
        }

    @classmethod
    def load_from_ropke_file(cls, filename, n_decimal_places=None):
        xs = []
        ys = []
        demand = []
        earliest = []
        latest = []
        service_time = []
        pickup_locations = []
        delivery_locations = []

        with open(filename, "r") as f:
            n_vehicles, n_tasks, _, capacity, _ = map(int, f.readline().split())

            for i in range(2 * n_tasks + 2):
                line = f.readline().rstrip()
                _, x, y, s, d, a, b = map(float, line.split())
                xs.append(x)
                ys.append(y)
                demand.append(d)
                earliest.append(a)
                latest.append(b)
                service_time.append(s)

                if 0 < i <= n_tasks:
                    pickup_locations.append(i)
                    delivery_locations.append(i + n_tasks)

        demand = np.array(demand, dtype=np.int32)
        earliest = np.array(earliest, dtype=np.int32)
        latest = np.array(latest, dtype=np.int32)
        service_time = np.array(service_time, dtype=np.int32)
        travel_time = np.zeros((len(xs), len(xs)), dtype=np.float64)

        for i in range(len(xs)):
            for j in range(len(xs)):
                travel_time[i, j] = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)

        if n_decimal_places is not None:
            earliest = np.array(round_to(earliest, n_decimal_places), dtype=np.int32)
            latest = np.array(round_to(latest, n_decimal_places), dtype=np.int32)
            service_time = np.array(
                round_to(service_time, n_decimal_places), dtype=np.int32
            )
            travel_time = np.array(
                round_to(travel_time, n_decimal_places), dtype=np.int32
            )

        return PDPTW(
            n_vehicles,
            capacity,
            demand,
            earliest,
            latest,
            service_time,
            travel_time,
            pickup_locations,
            delivery_locations,
        )

    @classmethod
    def load_from_li_lim_file(cls, filename, n_decimal_places=None):
        xs = []
        ys = []
        demand = []
        earliest = []
        latest = []
        service_time = []
        pickup_locations = []
        delivery_locations = []

        with open(filename, "r") as f:
            n_vehicles, capacity, _ = map(int, f.readline().split())

            line = f.readline().rstrip()

            while len(line) > 0:
                id, x, y, d, a, b, s, p, _ = map(int, line.split())
                xs.append(x)
                ys.append(y)
                demand.append(d)
                earliest.append(a)
                latest.append(b)
                service_time.append(s)

                if p != 0:
                    pickup_locations.append(p)
                    delivery_locations.append(id)

                line = f.readline().rstrip()

        xs.append(xs[0])
        ys.append(ys[0])
        demand.append(0)
        earliest.append(earliest[0])
        latest.append(latest[0])
        service_time.append(0)

        demand = np.array(demand, dtype=np.int32)
        earliest = np.array(earliest, dtype=np.int32)
        latest = np.array(latest, dtype=np.int32)
        service_time = np.array(service_time, dtype=np.int32)
        travel_time = np.zeros((len(xs), len(xs)), dtype=np.float64)

        for i in range(len(xs)):
            for j in range(len(xs)):
                travel_time[i, j] = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)

        if n_decimal_places is not None:
            earliest = np.array(round_to(earliest, n_decimal_places), dtype=np.int32)
            latest = np.array(round_to(latest, n_decimal_places), dtype=np.int32)
            service_time = np.array(
                round_to(service_time, n_decimal_places), dtype=np.int32
            )
            travel_time = np.array(
                round_to(travel_time, n_decimal_places), dtype=np.int32
            )

        return PDPTW(
            n_vehicles,
            capacity,
            demand,
            earliest,
            latest,
            service_time,
            travel_time,
            pickup_locations,
            delivery_locations,
        )

    def __repr__(self):
        return f"""
PDPTWInstance(
    n_vehicles={self.n_vehicles},
    capacity={self.capacity},
    demand={self.demand},
    earliest={self.earliest},
    latest={self.latest},
    service_time={self.service_time},
    travel_time={self.travel_time},
    pickup_locations={self.pickup_locations},
    delivery_locations={self.delivery_locations}
)
"""

    def compute_route_cost(self, route):
        cost = 0
        location = 0
        time = 0
        load = 0

        for i in route:
            cost += self.travel_time[location, i]
            time = (
                max(time + self.travel_time[location, i], self.earliest[i])
                + self.service_time[i]
            )
            load += self.demand[i]
            location = i

        cost += self.travel_time[location, self.n_nodes - 1]

        return cost

    def compute_solution_cost(self, routes, vehicle_penalty=0):
        return sum(self.compute_route_cost(r) for r in routes) + vehicle_penalty * len(
            routes
        )

    def validate_route(self, route, visited):
        location = 0
        time = 0
        load = 0
        visited_by_route = set()

        for i in route:
            if i in visited:
                print(f"Location {i} visited is visited by another route.")
                return None

            if i in visited_by_route:
                print(f"Location {i} is visited twice.")
                return None

            time = max(time + self.travel_time[location, i], self.earliest[i])

            if time > self.latest[i]:
                print(
                    f"Location {i} is visited at {time} after the deadline {self.latest[i]}."
                )
                return None

            if i in self.delivery_to_pickup:
                p = self.delivery_to_pickup[i]

                if p not in visited_by_route:
                    print(
                        f"Delivery location {i} is visited before pickup location {p}."
                    )
                    return None

            time += self.service_time[i]
            load += self.demand[i]
            location = i

            if load > self.capacity:
                print(f"The vehicle exceeds the capacity at {i}.")
                return None

            visited_by_route.add(i)

        time += self.travel_time[location, self.n_nodes - 1]

        if time > self.latest[0]:
            print("The vehicle returns to the depot after the due time.")
            return False

        for p, d in self.pickup_to_delivery.items():
            if p in visited_by_route and d not in visited_by_route:
                print(
                    f"The vehicle visits pickup location {p} but not delivery location {d}."
                )
                return False

        return True

    def validate_feasibility(self, routes):
        visited = set()

        for v, r in enumerate(routes):
            if not self.validate_route(r, visited):
                print(f"Vehicle {v} fails.")
                return False

            visited |= set(r)

        for i in self.pickup_locations:
            if i not in visited:
                print(f"Pickup location {i} is not visited.")
                return False

        return True

    def delete_edges_by_capacity(self, edges):
        deleted_edges = set()

        for i, (p1, d1) in enumerate(
            zip(self.pickup_locations, self.delivery_locations)
        ):
            for p2, d2 in zip(
                self.pickup_locations[i + 1 :], self.delivery_locations[i + 1 :]
            ):
                if self.demand[p1] + self.demand[p2] > self.capacity:
                    deleted_edges.discard((p1, p2))
                    deleted_edges.discard((p2, p1))
                    deleted_edges.discard((p1, d2))
                    deleted_edges.discard((p2, d1))
                    deleted_edges.discard((d1, d2))
                    deleted_edges.discard((d2, d1))

    def extract_shortest_travel_time(self, edges):
        n = self.n_nodes
        ub = self.latest.max() - self.earliest.min() + 1
        dist = np.full((n, n), ub)

        for i, j in edges:
            dist[i, j] = self.travel_time[i, j] + self.service_time[i]

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, j] > dist[i, k] + dist[k, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        return dist

    def reduce_time_windows(self, shortest_travel_time):
        for p, d in zip(self.pickup_locations, self.delivery_locations):
            self.latest[d] = min(
                self.latest[d],
                self.latest[self.n_nodes - 1]
                - shortest_travel_time[d, self.n_nodes - 1],
            )
            self.latest[p] = min(
                self.latest[p],
                self.latest[d] - shortest_travel_time[p, d],
            )
            self.earliest[p] = max(
                self.earliest[p],
                self.earliest[0] + shortest_travel_time[0, p],
            )
            self.earliest[d] = max(
                self.earliest[d],
                self.earliest[p] + shortest_travel_time[p, d],
            )

    def delete_edges_by_time_windows(self, edges):
        n = self.travel_time.shape[0]

        for i in range(n):
            for j in range(n):
                if (
                    i != j
                    and (i, j) in edges
                    and self.earliest[i] + self.travel_time[i, j] + self.service_time[i]
                    > self.latest[j]
                ):
                    edges.remove((i, j))

    def is_infeasible(self, shortest_travel_time, i, j):
        return self.earliest[i] + shortest_travel_time[i, j] > self.latest[j]

    def delete_edges_by_pairwise_reasoning(self, edges, shortest_travel_time):
        for p1, d1 in zip(self.pickup_locations, self.delivery_locations):
            for p2, d2 in zip(self.pickup_locations, self.delivery_locations):
                if p1 != p2 and d1 != d2:
                    if self.is_infeasible(
                        shortest_travel_time, p2, p1
                    ) or self.is_infeasible(shortest_travel_time, d2, d1):
                        edges.discard((p1, d2))

                    if (
                        self.is_infeasible(shortest_travel_time, p2, d1)
                        or self.is_infeasible(shortest_travel_time, d1, d2)
                    ) and self.is_infeasible(shortest_travel_time, d2, d1):
                        edges.discard((p1, p2))

                    if (
                        self.is_infeasible(shortest_travel_time, p1, p2)
                        or self.is_infeasible(shortest_travel_time, p2, d1)
                    ) and self.is_infeasible(shortest_travel_time, p2, p1):
                        edges.discard((d1, d2))

    def preprocess(self):
        edges = set()

        for p in self.pickup_locations:
            edges.add((0, p))

        for d in self.delivery_locations:
            edges.add((d, self.n_nodes - 1))

        for i in range(1, self.n_nodes - 1):
            for j in range(1, self.n_nodes - 1):
                if i != j and (
                    i not in self.delivery_to_pickup or self.delivery_to_pickup[i] != j
                ):
                    edges.add((i, j))

        self.delete_edges_by_capacity(edges)
        shortest_travel_time = None
        changed = True

        while changed:
            n_edges = len(edges)
            shortest_travel_time = self.extract_shortest_travel_time(edges)
            self.reduce_time_windows(shortest_travel_time)
            self.delete_edges_by_time_windows(edges)
            self.delete_edges_by_pairwise_reasoning(edges, shortest_travel_time)
            changed = len(edges) < n_edges

        return edges, shortest_travel_time
