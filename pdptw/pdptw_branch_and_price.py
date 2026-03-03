#!/usr/bin/env python

import argparse
import time

import didppy as dp
from vrpbap import RoutePricer, VRPBaP
import gurobipy as gp
import ortools.sat.python.cp_model as cp

from pdptw import PDPTW


def create_pricing_dp_model(
    instance,
    edges,
    node_cost,
    shortest_travel_time,
    farkas=False,
    bound="knapsack",
    disable_set_resource=False,
    disable_filter=False,
):
    n = instance.n_nodes
    n_tasks = len(instance.pickup_locations)

    model = dp.Model(maximize=False, float_cost=True)

    node = model.add_object_type(number=n)
    task = model.add_object_type(number=n_tasks)

    current = model.add_element_var(node, target=0)
    t = model.add_int_resource_var(target=0, less_is_better=True)
    load = model.add_int_resource_var(target=0, less_is_better=True)

    if disable_set_resource:
        reachable = model.add_set_var(task, target=list(range(n_tasks)))
    else:
        reachable = model.add_set_resource_var(
            task, target=list(range(n_tasks)), less_is_better=False
        )

    open = model.add_set_var(task, target=[])
    x = model.add_local_var()

    travel_time = model.add_int_table(instance.travel_time)
    service_and_travel_time = model.add_int_table(
        (instance.travel_time + instance.service_time[:, None])
    )
    shortest_pickup_time = model.add_int_table(
        [
            [shortest_travel_time[i, p] for p in instance.pickup_locations]
            for i in range(n)
        ]
    )
    shortest_delivery_time = model.add_int_table(
        [
            [shortest_travel_time[i, d] for d in instance.delivery_locations]
            for i in range(n)
        ]
    )
    pickup_latest = model.add_int_table(
        [instance.latest[p] for p in instance.pickup_locations]
    )
    delivery_latest = model.add_int_table(
        [instance.latest[d] for d in instance.delivery_locations]
    )

    connected = model.add_bool_table(
        [[(i, j) in edges for j in range(n)] for i in range(n)]
    )

    for i, next in enumerate(instance.pickup_locations):
        arrival_time = t + service_and_travel_time[current, next]
        start_time = dp.max(arrival_time, instance.earliest[next])
        new_load = load + instance.demand[next]

        if farkas:
            cost = node_cost[next] + dp.FloatExpr.state_cost()
        else:
            cost = (
                node_cost[next] + travel_time[current, next] + dp.FloatExpr.state_cost()
            )

        new_reachable = reachable.remove(i)

        if not disable_filter:
            new_reachable = new_reachable.filter(
                x,
                (start_time + shortest_pickup_time[next, x] <= pickup_latest[x]),
            )

        pickup = dp.Transition(
            name=str(next),
            cost=cost,
            effects=[
                (current, next),
                (t, start_time),
                (load, new_load),
                (open, open.add(i)),
                (
                    reachable,
                    new_reachable,
                ),
            ],
            preconditions=[
                connected[current, next],
                arrival_time <= instance.latest[next],
                new_load <= instance.capacity,
                reachable.contains(i),
            ],
        )
        model.add_transition(pickup)

    for i, next in enumerate(instance.delivery_locations):
        arrival_time = t + service_and_travel_time[current, next]
        start_time = dp.max(arrival_time, instance.earliest[next])

        if farkas:
            cost = dp.FloatExpr.state_cost()
        else:
            cost = travel_time[current, next] + dp.FloatExpr.state_cost()

        effects = [
            (current, next),
            (t, start_time),
            (load, load + instance.demand[next]),
            (open, open.remove(i)),
        ]

        if not disable_filter:
            effects.append(
                (
                    reachable,
                    reachable.filter(
                        x,
                        (
                            start_time + shortest_pickup_time[next, x]
                            <= pickup_latest[x]
                        ),
                    ),
                )
            )

        delivery = dp.Transition(
            name=str(next),
            cost=cost,
            effects=effects,
            preconditions=[
                connected[current, next],
                arrival_time <= instance.latest[next],
                open.contains(i),
            ],
        )
        model.add_transition(delivery)

    arrival_time = t + service_and_travel_time[current, n - 1]

    if farkas:
        cost = dp.FloatExpr.state_cost()
    else:
        cost = travel_time[current, n - 1] + dp.FloatExpr.state_cost()

    finish = dp.Transition(
        name=str(n - 1),
        cost=cost,
        effects=[
            (current, n - 1),
            (t, arrival_time),
        ],
        preconditions=[
            connected[current, n - 1],
            current != 0,
            arrival_time <= instance.latest[n - 1],
            load == 0,
        ],
    )
    model.add_transition(finish)

    model.add_base_case([current == n - 1], cost=0)

    for j in range(n_tasks):
        model.add_state_constr(
            ~open.contains(j)
            | (t + shortest_delivery_time[current, j] <= delivery_latest[j])
        )

    if bound is not None:
        min_to = [
            (
                0
                if j == 0
                else min(
                    instance.travel_time[i, j] for i in range(n) if (i, j) in edges
                )
            )
            for j in range(n)
        ]
        min_to_weight = [
            min_to[p] + instance.service_time[p] + min_to[d] + instance.service_time[d]
            for p, d in zip(instance.pickup_locations, instance.delivery_locations)
        ]
        min_to_delivery_time_table = model.add_int_table(
            [min_to[d] + instance.service_time[d] for d in instance.delivery_locations]
        )
        min_to_delivery_cost_table = model.add_int_table(
            [min_to[d] for d in instance.delivery_locations]
        )

        min_from = [
            (
                0
                if i == n - 1
                else min(
                    instance.travel_time[i, j] for j in range(n) if (i, j) in edges
                )
            )
            for i in range(n)
        ]
        min_from_weight = [
            instance.service_time[p]
            + min_from[p]
            + instance.service_time[d]
            + min_from[d]
            for p, d in zip(instance.pickup_locations, instance.delivery_locations)
        ]
        min_from_table = model.add_int_table(min_from)
        min_from_time_table = model.add_int_table(
            [instance.service_time[i] + min_from[i] for i in range(n)]
        )
        min_from_delivery_time_table = model.add_int_table(
            [
                instance.service_time[d] + min_from[d]
                for d in instance.delivery_locations
            ]
        )
        min_from_delivery_cost_table = model.add_int_table(
            [min_from[d] for d in instance.delivery_locations]
        )

        if farkas:
            node_value = [-min(-node_cost[p], 0) for p in instance.pickup_locations]

            if bound == "knapsack":
                model.add_dual_bound(
                    (current == n - 1).if_then_else(
                        0,
                        -dp.fractional_knapsack(
                            reachable,
                            instance.latest[n - 1]
                            - t
                            - min_to_delivery_time_table[open]
                            - min_to[n - 1],
                            node_value,
                            min_to_weight,
                        ),
                    )
                )
                model.add_dual_bound(
                    (current == n - 1).if_then_else(
                        0,
                        -dp.fractional_knapsack(
                            reachable,
                            instance.latest[n - 1]
                            - t
                            - min_from_delivery_time_table[open]
                            - min_from_time_table[current],
                            node_value,
                            min_from_weight,
                        ),
                    )
                )
            elif bound == "profit":
                node_value_table = model.add_float_table(node_value)
                model.add_dual_bound(
                    (current == n - 1).if_then_else(
                        0,
                        -node_value_table[reachable],
                    )
                )
        else:
            min_to_value = [
                -min(min_to[p] + node_cost[p] + min_to[d], 0)
                for p, d in zip(instance.pickup_locations, instance.delivery_locations)
            ]
            min_from_value = [
                -min(node_cost[p] + min_from[p] + min_from[d], 0)
                for p, d in zip(instance.pickup_locations, instance.delivery_locations)
            ]

            if bound == "knapsack":
                model.add_dual_bound(
                    (current == n - 1).if_then_else(
                        0,
                        -dp.fractional_knapsack(
                            reachable,
                            instance.latest[n - 1]
                            - t
                            - min_to_delivery_time_table[open]
                            - min_to[n - 1],
                            min_to_value,
                            min_to_weight,
                        )
                        + min_to_delivery_cost_table[open]
                        + min_to[n - 1],
                    )
                )
                model.add_dual_bound(
                    (current == n - 1).if_then_else(
                        0,
                        -dp.fractional_knapsack(
                            reachable,
                            instance.latest[n - 1]
                            - t
                            - min_from_delivery_time_table[open]
                            - min_from_time_table[current],
                            min_from_value,
                            min_from_weight,
                        )
                        + min_from_delivery_cost_table[open]
                        + min_from_table[current],
                    )
                )
            elif bound == "profit":
                min_to_value_table = model.add_float_table(min_to_value)
                model.add_dual_bound(
                    (current == n - 1).if_then_else(
                        0,
                        -min_to_value_table[reachable]
                        + min_to_delivery_cost_table[open]
                        + min_to[n - 1],
                    )
                )
                min_from_value_table = model.add_float_table(min_from_value)
                model.add_dual_bound(
                    (current == n - 1).if_then_else(
                        0,
                        -min_from_value_table[reachable]
                        + min_from_delivery_cost_table[open]
                        + min_from_table[current],
                    )
                )

    return model


def create_relaxed_pricing_dp_model(
    instance,
    edges,
    node_cost,
    shortest_travel_time,
    farkas=False,
    disable_bound=False,
):
    n = instance.n_nodes
    n_tasks = len(instance.pickup_locations)

    model = dp.Model(maximize=False, float_cost=True)

    node = model.add_object_type(number=n)
    task = model.add_object_type(number=n_tasks)

    current = model.add_element_var(node, target=0)
    t = model.add_int_resource_var(target=0, less_is_better=True)
    load = model.add_int_resource_var(target=0, less_is_better=True)
    open = model.add_set_var(task, target=[])

    travel_time = model.add_int_table(instance.travel_time)
    service_and_travel_time = model.add_int_table(
        (instance.travel_time + instance.service_time[:, None])
    )
    shortest_pickup_time = model.add_int_table(
        [
            [shortest_travel_time[i, p] for p in instance.pickup_locations]
            for i in range(n)
        ]
    )
    shortest_delivery_time = model.add_int_table(
        [
            [shortest_travel_time[i, d] for d in instance.delivery_locations]
            for i in range(n)
        ]
    )

    connected = model.add_bool_table(
        [[(i, j) in edges for j in range(n)] for i in range(n)]
    )

    for i, next in enumerate(instance.pickup_locations):
        arrival_time = t + service_and_travel_time[current, next]
        start_time = dp.max(arrival_time, instance.earliest[next])
        new_load = load + instance.demand[next]

        if farkas:
            cost = node_cost[next] + dp.FloatExpr.state_cost()
        else:
            cost = (
                node_cost[next] + travel_time[current, next] + dp.FloatExpr.state_cost()
            )

        pickup = dp.Transition(
            name=str(next),
            cost=cost,
            effects=[
                (current, next),
                (t, start_time),
                (load, new_load),
                (open, open.add(i)),
            ],
            preconditions=[
                connected[current, next],
                arrival_time <= instance.latest[next],
                new_load <= instance.capacity,
                ~open.contains(i),
            ],
        )
        model.add_transition(pickup)

    for i, next in enumerate(instance.delivery_locations):
        arrival_time = t + service_and_travel_time[current, next]

        if farkas:
            cost = dp.FloatExpr.state_cost()
        else:
            cost = travel_time[current, next] + dp.FloatExpr.state_cost()

        delivery = dp.Transition(
            name=str(next),
            cost=cost,
            effects=[
                (current, next),
                (t, dp.max(arrival_time, instance.earliest[next])),
                (load, load + instance.demand[next]),
                (open, open.remove(i)),
            ],
            preconditions=[
                connected[current, next],
                arrival_time <= instance.latest[next],
                open.contains(i),
            ],
        )
        model.add_transition(delivery)

    arrival_time = t + service_and_travel_time[current, n - 1]

    if farkas:
        cost = dp.FloatExpr.state_cost()
    else:
        cost = travel_time[current, n - 1] + dp.FloatExpr.state_cost()

    finish = dp.Transition(
        name=str(n - 1),
        cost=cost,
        effects=[
            (current, n - 1),
            (t, arrival_time),
        ],
        preconditions=[
            connected[current, n - 1],
            current != 0,
            arrival_time <= instance.latest[n - 1],
            load == 0,
        ],
    )
    model.add_transition(finish)

    model.add_base_case([current == n - 1], cost=0)

    pickup_latest = model.add_int_table(
        [instance.latest[p] for p in instance.pickup_locations]
    )
    delivery_latest = model.add_int_table(
        [instance.latest[d] for d in instance.delivery_locations]
    )

    for j in range(n_tasks):
        model.add_state_constr(
            ~open.contains(j)
            | (t + shortest_delivery_time[current, j] <= delivery_latest[j])
        )

    if not disable_bound:
        min_to = [
            (
                0
                if j == 0
                else min(
                    instance.travel_time[i, j] for i in range(n) if (i, j) in edges
                )
            )
            for j in range(n)
        ]
        min_to_delivery_time_table = model.add_int_table(
            [min_to[d] + instance.service_time[d] for d in instance.delivery_locations]
        )
        min_to_delivery_cost_table = model.add_int_table(
            [min_to[d] for d in instance.delivery_locations]
        )

        min_from = [
            (
                0
                if i == n - 1
                else min(
                    instance.travel_time[i, j] for j in range(n) if (i, j) in edges
                )
            )
            for i in range(n)
        ]
        min_from_table = model.add_int_table(min_from)
        min_from_time_table = model.add_int_table(
            [instance.service_time[i] + min_from[i] for i in range(n)]
        )
        min_from_delivery_time_table = model.add_int_table(
            [
                instance.service_time[d] + min_from[d]
                for d in instance.delivery_locations
            ]
        )
        min_from_delivery_cost_table = model.add_int_table(
            [min_from[d] for d in instance.delivery_locations]
        )

        min_task_time = [
            min(
                instance.service_time[i] + instance.travel_time[i, p]
                for i in range(n)
                if (i, p) in edges
            )
            + shortest_travel_time[p, d]
            + instance.service_time[d]
            + min_from[d]
            for (p, d) in zip(instance.pickup_locations, instance.delivery_locations)
        ]
        after_delivery_deadline = [
            instance.latest[d] + instance.service_time[d] + min_from[d]
            for d in instance.delivery_locations
        ]
        n_task_completions = [
            (after_delivery_deadline[i] - t) // min_task_time[i] for i in range(n_tasks)
        ]

        min_to_weight = [
            (
                min_to[p]
                + instance.service_time[p]
                + min_to[d]
                + instance.service_time[d]
            )
            * n_task_completions[i]
            for i, (p, d) in enumerate(
                zip(instance.pickup_locations, instance.delivery_locations)
            )
        ]
        min_from_weight = [
            (
                instance.service_time[p]
                + min_from[p]
                + instance.service_time[d]
                + min_from[d]
            )
            * n_task_completions[i]
            for i, (p, d) in enumerate(
                zip(instance.pickup_locations, instance.delivery_locations)
            )
        ]

        if farkas:
            candidates = model.create_set_const(
                task,
                [
                    i
                    for i, p in enumerate(instance.pickup_locations)
                    if node_cost[p] < 0
                ],
            )
            x = model.add_local_var()
            node_value = [
                (
                    -node_cost[p] * n_task_completions[i]
                    if p in node_cost and node_cost[p] < 0
                    else 0
                )
                for i, p in enumerate(instance.pickup_locations)
            ]
            reachable = model.add_set_state_fun(
                candidates.filter(
                    x, (t + shortest_pickup_time[current, x] <= pickup_latest[x])
                )
            )

            model.add_dual_bound(
                (current == n - 1).if_then_else(
                    0,
                    -dp.fractional_knapsack(
                        reachable,
                        instance.latest[n - 1]
                        - t
                        - min_to_delivery_time_table[open]
                        - min_to[n - 1],
                        node_value,
                        min_to_weight,
                    ),
                )
            )
            model.add_dual_bound(
                (current == n - 1).if_then_else(
                    0,
                    -dp.fractional_knapsack(
                        reachable,
                        instance.latest[n - 1]
                        - t
                        - min_from_delivery_time_table[open]
                        - min_from_time_table[current],
                        node_value,
                        min_from_weight,
                    ),
                )
            )
        else:
            candidates = model.create_set_const(
                task,
                [
                    i
                    for i, (p, d) in enumerate(
                        zip(instance.pickup_locations, instance.delivery_locations)
                    )
                    if min_to[p] + node_cost[p] + min_to[d] < 0
                    or node_cost[p] + min_from[p] + min_from[d] < 0
                ],
            )
            x = model.add_local_var()
            reachable = model.add_set_state_fun(
                candidates.filter(
                    x, t + shortest_pickup_time[current, x] <= pickup_latest[x]
                )
            )

            min_to_value = [
                (
                    -(min_to[p] + node_cost[p] + min_to[d]) * n_task_completions[i]
                    if p in node_cost and min_to[p] + node_cost[p] + min_to[d] < 0
                    else 0
                )
                for i, (p, d) in enumerate(
                    zip(instance.pickup_locations, instance.delivery_locations)
                )
            ]
            model.add_dual_bound(
                (current == n - 1).if_then_else(
                    0,
                    -dp.fractional_knapsack(
                        reachable,
                        instance.latest[n - 1]
                        - t
                        - min_to_delivery_time_table[open]
                        - min_to[n - 1],
                        min_to_value,
                        min_to_weight,
                    )
                    + min_to_delivery_cost_table[open]
                    + min_to[n - 1],
                )
            )

            min_from_value = [
                (
                    -(node_cost[p] + min_from[p] + min_from[d]) * n_task_completions[i]
                    if p in node_cost and min_from[p] + node_cost[p] + min_from[d] < 0
                    else 0
                )
                for i, (p, d) in enumerate(
                    zip(instance.pickup_locations, instance.delivery_locations)
                )
            ]
            model.add_dual_bound(
                (current == n - 1).if_then_else(
                    0,
                    -dp.fractional_knapsack(
                        reachable,
                        instance.latest[n - 1]
                        - t
                        - min_from_delivery_time_table[open]
                        - min_from_time_table[current],
                        min_from_value,
                        min_from_weight,
                    )
                    + min_from_delivery_cost_table[open]
                    + min_from_table[current],
                )
            )

    return model


class DIDPPricer(RoutePricer):
    def __init__(
        self,
        instance,
        shortest_travel_time,
        vehicle_penalty,
        elementary=False,
        bound="knapsack",
        disable_set_resource=False,
        disable_filter=False,
        f_first=False,
        get_all_solutions=False,
        one_column_farkas=False,
        epsilon=1e-6,
        solver="labeling",
    ):
        super().__init__()
        self.instance = instance
        self.shortest_travel_time = shortest_travel_time
        self.vehicle_penalty = vehicle_penalty
        self.elementary = elementary
        self.bound = bound
        self.disable_set_resource = disable_set_resource
        self.disable_filter = disable_filter
        self.f_first = f_first
        self.get_all_solutions = get_all_solutions
        self.one_column_farkas = one_column_farkas
        self.epsilon = epsilon
        self.pricing_stats = {"time": 0.0, "calls": 0, "columns": 0, "expanded": 0}

        # Store the solver type
        assert solver in [
            "cabs",
            "caasdy",
            "labeling",
        ], f'Invalid DIDP sovler "{solver}"'
        if solver == "cabs":
            self.solver_type = dp.CABS
            self.get_all_solutions = False
            self.f_first = False
        elif solver == "caasdy":
            self.solver_type = dp.CAASDy
            self.f_first = False
        elif solver == "labeling":
            self.solver_type = dp.Labeling

    def solve(self, model, constant_cost, farkas, time_limit):
        if farkas:
            primal_bound = -self.epsilon - constant_cost
        else:
            primal_bound = -self.epsilon - self.vehicle_penalty - constant_cost

        verbosity = self.model.data["verbosity"]

        if self.get_all_solutions and self.f_first:
            solver = self.solver_type(
                model,
                primal_bound=primal_bound,
                quiet=verbosity <= 2,
                get_all_solutions=True,
                node_first=True,
                time_limit=time_limit,
            )
        elif self.get_all_solutions:
            solver = self.solver_type(
                model,
                primal_bound=primal_bound,
                quiet=verbosity <= 2,
                get_all_solutions=True,
                time_limit=time_limit,
            )
        elif self.f_first:
            solver = self.solver_type(
                model,
                primal_bound=primal_bound,
                quiet=verbosity <= 2,
                node_first=True,
                time_limit=time_limit,
            )
        else:
            solver = self.solver_type(
                model,
                primal_bound=primal_bound,
                quiet=verbosity <= 2,
                time_limit=time_limit,
            )

        is_terminated = False
        best_cost = None

        while not is_terminated:
            solution, is_terminated = solver.search_next()

            if is_terminated:
                self.pricing_stats["expanded"] += solution.expanded

            if solution.is_optimal and solution.cost == best_cost:
                break

            if solution.cost is not None:
                if best_cost is None or solution.cost < best_cost:
                    best_cost = solution.cost

                route = (0,) + tuple(int(t.name) for t in solution.transitions)

                coeffs = {}

                for i in route:
                    if i in self.instance.pickup_to_delivery:
                        if i not in coeffs:
                            coeffs[i] = 0

                        coeffs[i] += 1

                cost = (
                    self.instance.compute_route_cost(route[1:-1]) + self.vehicle_penalty
                )
                redcost = solution.cost + constant_cost

                if not farkas:
                    redcost += self.vehicle_penalty

                yield route, coeffs, cost, redcost, solution.time_out

                if self.one_column_farkas and farkas:
                    break

            elif solution.time_out:
                yield None, None, None, None, True

    def empty(self):
        yield from ()

    def pricer_column(self, edges, node_costs, constant_cost, farkas, time_limit):
        for i in range(self.instance.n_nodes - 1):
            if not any((i, j) in edges for j in range(1, self.instance.n_nodes)):
                return self.empty()

        for i in range(1, self.instance.n_nodes):
            if not any((j, i) in edges for j in range(self.instance.n_nodes - 1)):
                return self.empty()

        if self.elementary:
            model = create_pricing_dp_model(
                self.instance,
                edges,
                node_costs,
                self.shortest_travel_time,
                farkas=farkas,
                bound=self.bound,
                disable_set_resource=self.disable_set_resource,
                disable_filter=self.disable_filter,
            )
        else:
            model = create_relaxed_pricing_dp_model(
                self.instance,
                edges,
                node_costs,
                self.shortest_travel_time,
                farkas=farkas,
                disable_bound=(self.bound == "none"),
            )

        return self.solve(model, constant_cost, farkas, time_limit)

    def pricerredcost_column(self, edges, node_costs, constant_cost, time_limit=None):
        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        try:
            for column in self.pricer_column(
                edges, node_costs, constant_cost, False, time_limit
            ):
                if column[0] is not None:
                    self.pricing_stats["columns"] += 1

                yield column

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start

    def pricerfarkas_column(self, edges, node_costs, constant_cost, time_limit=None):
        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        try:
            for column in self.pricer_column(
                edges, node_costs, constant_cost, True, time_limit
            ):
                if column[0] is not None:
                    self.pricing_stats["columns"] += 1

                yield column

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start


class GurobiPricer(RoutePricer):
    def __init__(
        self, instance, shortest_travel_time, vehicle_penalty, epsilon=1e-6, threads=1
    ):
        super().__init__()
        self.instance = instance
        self.shortest_travel_time = shortest_travel_time
        self.vehicle_penalty = vehicle_penalty
        self.epsilon = epsilon
        self.threads = threads
        self.pricing_model = None
        self.pricing_stats = {"time": 0.0, "calls": 0, "columns": 0}

    def create_model(self, edges, node_cost, farkas=False):
        n = self.instance.n_nodes
        nodes = list(range(n))

        if self.pricing_model is None:
            model = gp.Model()

            x = model.addVars(edges, vtype=gp.GRB.BINARY, name="x")
            t = model.addVars(
                nodes,
                vtype=gp.GRB.CONTINUOUS,
                lb=self.instance.earliest,
                ub=self.instance.latest,
            )
            q_lb = [max(0, self.instance.demand[i]) for i in nodes]
            q_ub = [
                min(
                    self.instance.capacity,
                    self.instance.capacity + self.instance.demand[i],
                )
                for i in nodes
            ]
            q = model.addVars(
                nodes,
                vtype=gp.GRB.CONTINUOUS,
                lb=q_lb,
                ub=q_ub,
            )

            objective = gp.quicksum(
                node_cost[i] * x[i, j] for i, j in edges if i in node_cost
            )

            if not farkas:
                objective += gp.quicksum(
                    self.instance.travel_time[i, j] * x[i, j] for i, j in edges
                )

            model.setObjective(objective, gp.GRB.MINIMIZE)

            model.addConstr(gp.quicksum(x[0, j] for j in nodes if (0, j) in edges) == 1)
            model.addConstrs(
                gp.quicksum(x[i, j] for i in nodes if (i, j) in edges)
                == gp.quicksum(x[j, i] for i in nodes if (j, i) in edges)
                for j in nodes[1:-1]
            )
            model.addConstr(
                gp.quicksum(x[i, n - 1] for i in nodes[1:] if (i, n - 1) in edges) == 1
            )
            model.addConstrs(
                gp.quicksum(x[p, j] for j in nodes if (p, j) in edges)
                == gp.quicksum(x[d, j] for j in nodes if (d, j) in edges)
                for p, d in zip(
                    self.instance.pickup_locations, self.instance.delivery_locations
                )
            )

            t_big_m = {
                (i, j): self.instance.latest[i]
                + self.instance.service_time[i]
                + self.instance.travel_time[i, j]
                - self.instance.earliest[j]
                for i, j in edges
            }
            model.addConstrs(
                t[i]
                + self.instance.service_time[i]
                + self.instance.travel_time[i, j]
                - t[j]
                <= t_big_m[i, j] * (1 - x[i, j])
                for i, j in edges
            )
            model.addConstrs(
                t[d] >= t[p] + self.shortest_travel_time[p, d]
                for p, d in zip(
                    self.instance.pickup_locations, self.instance.delivery_locations
                )
            )

            q_big_m = {
                (i, j): q_ub[i] + self.instance.demand[j] - q_lb[j] for i, j in edges
            }
            model.addConstrs(
                q[i] + self.instance.demand[j] - q[j] <= q_big_m[i, j] * (1 - x[i, j])
                for i, j in edges
            )

            self.pricing_model = model

            return model, x
        else:
            x = {}

            for i in nodes:
                for j in nodes:
                    x_ij = self.pricing_model.getVarByName(f"x[{i},{j}]")

                    if x_ij is not None:
                        if (i, j) in edges:
                            x_ij.UB = 1
                            x_ij.Obj = (
                                0 if farkas else self.instance.travel_time[i, j]
                            ) + (node_cost[i] if i in node_cost else 0.0)
                            x[i, j] = x_ij
                        else:
                            x_ij.UB = 0

            return self.pricing_model, x

    def solve(self, model, x, constant_cost, farkas, time_limit):
        model.params.cutoff = -self.epsilon - constant_cost
        model.params.outputFlag = 1 if self.model.data["verbosity"] >= 3 else 0
        model.params.threads = self.threads

        if time_limit is not None:
            model.params.timeLimit = time_limit

        model.optimize()

        for i in range(model.SolCount):
            model.params.SolutionNumber = i
            objval = model.poolObjVal

            route = [0]
            current = 0

            while True:
                for next in range(self.instance.n_nodes):
                    if (current, next) in x and x[current, next].Xn > 0.5:
                        current = next
                        break

                route.append(current)

                if next == self.instance.n_nodes - 1:
                    break

            route = tuple(route)

            coeffs = {}

            for i in route[1:-1]:
                if i in self.instance.pickup_to_delivery:
                    if i not in coeffs:
                        coeffs[i] = 0

                    coeffs[i] += 1

            cost = self.instance.compute_route_cost(route[1:-1]) + self.vehicle_penalty
            redcost = objval + constant_cost

            if not farkas:
                redcost += self.vehicle_penalty

            if redcost < -self.epsilon:
                yield route, coeffs, cost, redcost, False

        if model.status == gp.GRB.TIME_LIMIT:
            yield None, None, None, None, True

    def pricerredcost_column(self, edges, node_costs, constant_cost, time_limit=None):
        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        try:
            model, x = self.create_model(edges, node_costs, farkas=False)
            for column in self.solve(
                model, x, constant_cost, farkas=False, time_limit=time_limit
            ):
                if column[0] is not None:
                    self.pricing_stats["columns"] += 1

                yield column

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start

    def pricerfarkas_column(self, edges, node_costs, constant_cost, time_limit=None):
        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        try:
            model, x = self.create_model(edges, node_costs, farkas=True)
            for column in self.solve(
                model, x, constant_cost, farkas=True, time_limit=time_limit
            ):
                if column[0] is not None:
                    self.pricing_stats["columns"] += 1

                yield column

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start


class CPPricer(RoutePricer):
    def __init__(
        self, instance, shortest_travel_time, vehicle_penalty, epsilon=1e-6, threads=1
    ):
        super().__init__()
        self.instance = instance
        self.shortest_travel_time = shortest_travel_time
        self.vehicle_penalty = vehicle_penalty
        self.epsilon = epsilon
        self.threads = threads
        self.pricing_stats = {"time": 0.0, "calls": 0, "columns": 0}

    def create_model(self, edges, node_cost, farkas=False):
        n = self.instance.n_nodes
        nodes = list(range(n))

        model = cp.CpModel()

        x = {(i, j): model.new_bool_var(f"x_{i}_{j}") for i, j in edges if j != 0}
        y = {p: model.new_bool_var(f"y_{p}") for p in self.instance.pickup_locations}
        t = [
            model.new_int_var(
                self.instance.earliest[i], self.instance.latest[i], f"t_{i}"
            )
            for i in nodes
        ] + [
            model.new_int_var(
                self.instance.earliest[0], self.instance.latest[0], f"t_{n}"
            )
        ]
        q = [
            model.new_int_var(
                max(0, self.instance.demand[i]),
                min(
                    self.instance.capacity,
                    self.instance.capacity + self.instance.demand[i],
                ),
                f"q_{i}",
            )
            for i in nodes
        ]

        model.add_circuit(
            [(i, j, x_ij) for (i, j), x_ij in x.items()]
            + [(p, p, ~y[p]) for p in self.instance.pickup_locations]
            + [
                (d, d, ~y[p])
                for p, d in zip(
                    self.instance.pickup_locations, self.instance.delivery_locations
                )
            ]
            + [(n - 1, 0, True)]
        )

        for p, d in zip(
            self.instance.pickup_locations, self.instance.delivery_locations
        ):
            model.add(t[d] >= t[p] + self.shortest_travel_time[p, d])

        for (i, j), x_ij in x.items():
            d_ij = (
                self.instance.travel_time[i, 0]
                if j == n
                else self.instance.travel_time[i, j]
            )
            model.add(
                t[j] >= t[i] + self.instance.service_time[i] + d_ij
            ).only_enforce_if(x_ij)
            model.add(q[j] == q[i] + self.instance.demand[j]).only_enforce_if(x_ij)

        model.minimize(
            sum(
                (0 if farkas else self.instance.travel_time[i, j] * x_ij)
                for (i, j), x_ij in x.items()
            )
            + sum(node_cost[i] * y[i] for i in nodes[1:-1] if i in node_cost)
        )

        return model, x

    def solve(self, model, x, node_cost, constant_cost, farkas, time_limit):
        class SolutionCallback(cp.CpSolverSolutionCallback):
            def __init__(self, nodes, x):
                super().__init__()
                self.nodes = nodes
                self.x = x
                self.routes = []

            def on_solution_callback(self):
                route = [0]
                current = 0

                while current != len(self.nodes) - 1:
                    for next in self.nodes:
                        if (current, next) in self.x and self.Value(
                            self.x[current, next]
                        ) > 0.5:
                            route.append(next)
                            current = next
                            break

                self.routes.append(route)

        solver = cp.CpSolver()
        solver.parameters.num_search_workers = self.threads

        if self.model.data["verbosity"] >= 3:
            solver.parameters.log_search_progress = True

        if time_limit is not None:
            solver.parameters.max_time_in_seconds = time_limit

        callback = SolutionCallback(list(range(self.instance.n_nodes)), x)
        status = solver.solve(model, callback)

        for route in callback.routes:
            route = tuple(route)

            coeffs = {}

            for i in route[1:-1]:
                if i in self.instance.pickup_to_delivery:
                    if i not in coeffs:
                        coeffs[i] = 0

                    coeffs[i] += 1

            cost = self.instance.compute_route_cost(route[1:-1]) + self.vehicle_penalty
            redcost = (
                sum(node_cost[i] for i in route[1:-1] if i in node_cost) + constant_cost
            ) + (0 if farkas else cost)

            if redcost < -self.epsilon:
                yield route, coeffs, cost, redcost, False

        if status != cp.OPTIMAL and status != cp.INFEASIBLE:
            yield None, None, None, None, True

    def pricerredcost_column(self, edges, node_costs, constant_cost, time_limit=None):
        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        try:
            model, x = self.create_model(edges, node_costs, farkas=False)
            for column in self.solve(
                model, x, node_costs, constant_cost, False, time_limit
            ):
                if column[0] is not None:
                    self.pricing_stats["columns"] += 1

                yield column

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start

    def pricerfarkas_column(self, edges, node_costs, constant_cost, time_limit=None):
        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        try:
            model, x = self.create_model(edges, node_costs, farkas=True)
            for column in self.solve(
                model, x, node_costs, constant_cost, True, time_limit
            ):
                if column[0] is not None:
                    self.pricing_stats["columns"] += 1

                yield column

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start


if __name__ == "__main__":
    start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--n-decimal-places", type=int, default=3)
    parser.add_argument(
        "--pricer",
        type=str,
        choices=["relaxed-dp", "dp", "mip", "cp"],
        default="dp",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["cabs", "caasdy", "labeling"],
        default="labeling",
    )
    parser.add_argument("--vehicle-penalty", type=float, default=1e4)
    parser.add_argument(
        "--bound", type=str, choices=["knapsack", "profit", "none"], default="knapsack"
    )
    parser.add_argument("--disable-set-resource", action="store_true")
    parser.add_argument("--disable-filter", action="store_true")
    parser.add_argument("--get-all-solutions", action="store_true")
    parser.add_argument("--one-column-farkas", action="store_true")
    parser.add_argument(
        "--file-format", type=str, choices=["LiLim", "Ropke"], default="LiLim"
    )
    parser.add_argument("--f-first", action="store_true")
    args = parser.parse_args()

    vehicle_penalty = args.vehicle_penalty * (10**args.n_decimal_places)

    if args.file_format == "LiLim":
        instance = PDPTW.load_from_li_lim_file(
            args.input, n_decimal_places=args.n_decimal_places
        )
    else:
        instance = PDPTW.load_from_ropke_file(
            args.input, n_decimal_places=args.n_decimal_places
        )

    edges, shortest_travel_time = instance.preprocess()

    if args.pricer == "mip":
        pricer = GurobiPricer(
            instance,
            shortest_travel_time,
            vehicle_penalty,
            epsilon=args.epsilon,
            threads=args.threads,
        )
    elif args.pricer == "cp":
        pricer = CPPricer(
            instance,
            shortest_travel_time,
            vehicle_penalty,
            epsilon=args.epsilon,
            threads=args.threads,
        )
    else:
        pricer = DIDPPricer(
            instance,
            shortest_travel_time,
            vehicle_penalty,
            elementary=args.pricer == "dp",
            epsilon=args.epsilon,
            bound=args.bound,
            disable_set_resource=args.disable_set_resource,
            disable_filter=args.disable_filter,
            f_first=args.f_first,
            get_all_solutions=args.get_all_solutions,
            one_column_farkas=args.one_column_farkas,
            solver=args.solver,
        )

    routes = [
        (0, p, d, instance.n_nodes - 1)
        for p, d in zip(instance.pickup_locations, instance.delivery_locations)
    ]
    route_costs = [
        instance.compute_route_cost(r[1:-1]) + vehicle_penalty for r in routes
    ]
    route_coeffs = [{p: 1} for p in instance.pickup_locations]

    model = VRPBaP(
        instance.pickup_locations,
        edges,
        [0],
        [instance.n_nodes - 1],
        pricer,
        routes=routes,
        route_coeffs=route_coeffs,
        route_costs=route_costs,
        verbosity=args.verbosity,
        time_limit=args.timeout,
        objective_integral=True,
    )

    solution = model.optimize()

    print("Solving time: {}s".format(time.perf_counter() - start))
    print("Nodes:", solution["nodes"])
    print("Pricing time: {}s".format(pricer.pricing_stats["time"]))
    print("Pricing calls:", pricer.pricing_stats["calls"])
    print("Pricing columns:", pricer.pricing_stats["columns"])

    if args.pricer in ["relaxed-dp", "dp"]:
        print("Pricing expanded::", pricer.pricing_stats["expanded"])

    if solution["infeasible"]:
        print("The problem is infeasible")
    else:
        print("best bound:", solution["best_bound"])

        if solution["solution_found"]:
            cost = int(round(solution["cost"]))
            print("cost:", cost)

            if solution["optimal"]:
                print("optimal cost:", cost)

            routes = [r[1:-1] for r in solution["routes"]]
            print(routes)

            is_feasibility_validated = instance.validate_feasibility(routes)

            if is_feasibility_validated:
                print("The solution is valid")

            recomputed_cost = instance.compute_solution_cost(
                routes,
                vehicle_penalty=vehicle_penalty,
            )

            if cost == recomputed_cost:
                print("The cost is correct")
            else:
                print(
                    "Cost {} does not match recmoputed cost {}".format(
                        cost, recomputed_cost
                    )
                )
        else:
            print("No solution found")
