#!/usr/bin/env python


import argparse
import time

import didppy as dp
import gurobipy as gp
import ortools.sat.python.cp_model as cp
from vrpbap import RoutePricer, VRPBaP
import vrplib

from vrptw import VRPTW


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

    model = dp.Model(maximize=False, float_cost=True)

    node = model.add_object_type(number=n)

    current = model.add_element_var(node, target=0)
    t = model.add_int_resource_var(target=0, less_is_better=True)
    load = model.add_int_resource_var(target=0, less_is_better=True)

    if disable_set_resource:
        reachable = model.add_set_var(node, target=list(range(1, n)))
    else:
        reachable = model.add_set_resource_var(
            node, target=list(range(1, n)), less_is_better=False
        )

    x = model.add_local_var()

    travel_time = model.add_int_table(instance.travel_time)
    service_and_travel_time = model.add_int_table(
        (instance.travel_time + instance.service_time[:, None])
    )
    shortest_travel_time = model.add_int_table(shortest_travel_time)
    connected = model.add_bool_table(
        [[(i, j) in edges for j in range(n)] for i in range(n)]
    )
    latest = model.add_int_table(instance.latest)
    demand = model.add_int_table(instance.demand)

    for next in range(1, n):
        arrival_time = t + service_and_travel_time[current, next]
        start_time = dp.max(arrival_time, instance.earliest[next])
        new_load = load + instance.demand[next]

        if farkas:
            cost = node_cost[next] + dp.FloatExpr.state_cost()
        else:
            cost = (
                node_cost[next] + travel_time[current, next] + dp.FloatExpr.state_cost()
            )

        new_reachable = reachable.remove(next)

        if not disable_filter:
            new_reachable = new_reachable.filter(
                x,
                (start_time + shortest_travel_time[next, x] <= latest[x])
                & (new_load + demand[x] <= instance.capacity),
            )

        visit = dp.Transition(
            name=str(next),
            cost=cost,
            effects=[
                (current, next),
                (t, start_time),
                (load, new_load),
                (reachable, new_reachable),
            ],
            preconditions=[
                connected[current, next],
                arrival_time <= instance.latest[next],
                reachable.contains(next),
            ],
        )
        model.add_transition(visit)

    arrival_time = t + service_and_travel_time[current, 0]

    if farkas:
        cost = dp.FloatExpr.state_cost()
    else:
        cost = travel_time[current, 0] + dp.FloatExpr.state_cost()

    finish = dp.Transition(
        name="0",
        cost=cost,
        effects=[(current, n), (t, arrival_time)],
        preconditions=[
            connected[current, 0],
            current != 0,
            arrival_time <= instance.latest[0],
        ],
    )
    model.add_transition(finish)

    model.add_base_case([current == n], cost=0)

    if bound is not None:
        min_to = [
            min(instance.travel_time[i, j] for i in range(n) if (i, j) in edges)
            for j in range(n)
        ]
        min_to_weight = [min_to[i] + instance.service_time[i] for i in range(n)]
        min_from = [
            min(instance.travel_time[i, j] for j in range(n) if (i, j) in edges)
            for i in range(n)
        ]
        min_from_table = model.add_int_table(min_from)
        min_from_weight = [instance.service_time[i] + min_from[i] for i in range(n)]
        min_from_weight_table = model.add_int_table(min_from_weight)

        if farkas:
            node_profit = [
                max(-node_cost[i], 0) if i in node_cost else 0 for i in range(n)
            ]

            if bound == "knapsack":
                model.add_dual_bound(
                    (current == n).if_then_else(
                        0,
                        -dp.fractional_knapsack(
                            reachable,
                            instance.capacity - load,
                            node_profit,
                            instance.demand,
                        ),
                    )
                )
                model.add_dual_bound(
                    (current == n).if_then_else(
                        0,
                        -dp.fractional_knapsack(
                            reachable,
                            instance.latest[0] - t - min_to[0],
                            node_profit,
                            min_to_weight,
                        ),
                    )
                )
                model.add_dual_bound(
                    (current == n).if_then_else(
                        0,
                        -dp.fractional_knapsack(
                            reachable,
                            instance.latest[0] - t - min_from_weight_table[current],
                            node_profit,
                            min_from_weight,
                        ),
                    )
                )
            elif bound == "profit":
                node_profit_table = model.add_float_table(node_profit)
                model.add_dual_bound(
                    (current == n).if_then_else(0, -node_profit_table[reachable])
                )
        else:
            min_to_value = [
                (
                    -(min_to[i] + node_cost[i])
                    if i in node_cost and min_to[i] + node_cost[i] < 0
                    else 0
                )
                for i in range(n)
            ]
            min_from_value = [
                (
                    -(node_cost[i] + min_from[i])
                    if i in node_cost and node_cost[i] + min_from[i] < 0
                    else 0
                )
                for i in range(n)
            ]

            if bound == "knapsack":
                model.add_dual_bound(
                    (current == n).if_then_else(
                        0,
                        -dp.fractional_knapsack(
                            reachable,
                            instance.capacity - load,
                            min_to_value,
                            instance.demand,
                        )
                        + min_to[0],
                    )
                )
                model.add_dual_bound(
                    (current == n).if_then_else(
                        0,
                        -dp.fractional_knapsack(
                            reachable,
                            instance.latest[0] - t - min_to[0],
                            min_to_value,
                            min_to_weight,
                        )
                        + min_to[0],
                    )
                )
                model.add_dual_bound(
                    (current == n).if_then_else(
                        0,
                        -dp.fractional_knapsack(
                            reachable,
                            instance.capacity - load,
                            min_from_value,
                            instance.demand,
                        )
                        + min_from_table[current],
                    )
                )
                model.add_dual_bound(
                    (current == n).if_then_else(
                        0,
                        -dp.fractional_knapsack(
                            reachable,
                            instance.latest[0] - t - min_from_weight_table[current],
                            min_from_value,
                            min_from_weight,
                        )
                        + min_from_table[current],
                    )
                )
            elif bound == "profit":
                min_to_value_table = model.add_float_table(min_to_value)
                model.add_dual_bound(
                    (current == n).if_then_else(
                        0, -min_to_value_table[reachable] + min_to[0]
                    )
                )
                min_from_value_table = model.add_float_table(min_from_value)
                model.add_dual_bound(
                    (current == n).if_then_else(
                        0, -min_from_value_table[reachable] + min_from_table[current]
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

    model = dp.Model(maximize=False, float_cost=True)

    node = model.add_object_type(number=n)

    current = model.add_element_var(node, target=0)
    previous = model.add_element_var(node, target=0)
    t = model.add_int_resource_var(target=0, less_is_better=True)
    load = model.add_int_resource_var(target=0, less_is_better=True)

    travel_time = model.add_int_table(instance.travel_time)
    service_and_travel_time = model.add_int_table(
        (instance.travel_time + instance.service_time[:, None])
    )
    connected = model.add_bool_table(
        [[(i, j) in edges for j in range(n)] for i in range(n)]
    )

    for next in range(1, n):
        arrival_time = t + service_and_travel_time[current, next]
        start_time = dp.max(arrival_time, instance.earliest[next])
        new_load = load + instance.demand[next]

        if farkas:
            cost = node_cost[next] + dp.FloatExpr.state_cost()
        else:
            cost = (
                node_cost[next] + travel_time[current, next] + dp.FloatExpr.state_cost()
            )

        visit = dp.Transition(
            name=str(next),
            cost=cost,
            effects=[
                (current, next),
                (previous, current),
                (t, start_time),
                (load, new_load),
            ],
            preconditions=[
                connected[current, next],
                arrival_time <= instance.latest[next],
                new_load <= instance.capacity,
                next != current,
                next != previous,
            ],
        )
        model.add_transition(visit)

    arrival_time = t + service_and_travel_time[current, 0]

    if farkas:
        cost = dp.FloatExpr.state_cost()
    else:
        cost = travel_time[current, 0] + dp.FloatExpr.state_cost()

    finish = dp.Transition(
        name="0",
        cost=cost,
        effects=[
            (current, n),
            (previous, current),
            (t, arrival_time),
        ],
        preconditions=[
            connected[current, 0],
            arrival_time <= instance.latest[0],
            current != 0,
        ],
    )
    model.add_transition(finish)

    model.add_base_case([current == n], cost=0)

    if not disable_bound:
        latest = model.add_int_table([instance.latest[i] for i in range(n)])
        demand = model.add_int_table([instance.demand[i] for i in range(n)])
        shortest_travel_time = model.add_int_table(shortest_travel_time)

        min_from = [
            min(instance.travel_time[i, j] for j in range(n) if (i, j) in edges)
            for i in range(n)
        ]
        min_visit_time = [
            min(
                instance.service_time[i] + instance.travel_time[i, j]
                for i in range(n)
                if (i, j) in edges
            )
            + instance.service_time[j]
            + min_from[j]
            for j in range(n)
        ]
        after_visit_deadline = [
            instance.latest[j] + instance.service_time[j] + min_from[j]
            for j in range(n)
        ]
        n_visits = [
            (after_visit_deadline[i] - t) // min_visit_time[i] for i in range(n)
        ]
        demand_weight = [instance.demand[i] * n_visits[i] for i in range(n)]
        min_to = [
            min(instance.travel_time[i, j] for i in range(n) if (i, j) in edges)
            for j in range(n)
        ]
        min_to_weight = [
            (min_to[i] + instance.service_time[i]) * n_visits[i] for i in range(n)
        ]
        min_from = [
            min(instance.travel_time[i, j] for j in range(n) if (i, j) in edges)
            for i in range(n)
        ]
        min_from_time_table = model.add_int_table(
            [instance.service_time[i] + min_from[i] for i in range(n)] + [0]
        )
        min_from_weight = [
            (instance.service_time[i] + min_from[i]) * n_visits[i] for i in range(n)
        ]
        min_from_table = model.add_int_table(min_from + [0])

        if farkas:
            candidates = model.create_set_const(
                node,
                [i for i in range(1, n) if node_cost[i] < 0],
            )
            x = model.add_local_var()
            node_value = [
                (
                    -node_cost[i] * n_visits[i]
                    if i in node_cost and node_cost[i] < 0
                    else 0
                )
                for i in range(n)
            ]
            reachable = model.add_set_state_fun(
                candidates.filter(
                    x,
                    (t + shortest_travel_time[current, x] <= latest[x])
                    & (load + demand[x] <= instance.capacity),
                )
            )

            model.add_dual_bound(
                (current == n).if_then_else(
                    0,
                    -dp.fractional_knapsack(
                        reachable,
                        instance.capacity - load,
                        node_value,
                        demand_weight,
                    ),
                )
            )
            model.add_dual_bound(
                (current == n).if_then_else(
                    0,
                    -dp.fractional_knapsack(
                        reachable,
                        instance.latest[0] - t - min_to[0],
                        node_value,
                        min_to_weight,
                    ),
                )
            )
            model.add_dual_bound(
                (current == n).if_then_else(
                    0,
                    -dp.fractional_knapsack(
                        reachable,
                        instance.latest[0] - t - min_from_time_table[current],
                        node_value,
                        min_from_weight,
                    ),
                )
            )
        else:
            candidates = model.create_set_const(
                node,
                [
                    i
                    for i in range(1, n)
                    if min_to[i] + node_cost[i] < 0 or node_cost[i] + min_from[i] < 0
                ],
            )
            x = model.add_local_var()
            reachable = model.add_set_state_fun(
                candidates.filter(
                    x,
                    (t + shortest_travel_time[current, x] <= latest[x])
                    & (load + demand[x] <= instance.capacity),
                ),
            )

            min_to_value = [
                (
                    -(node_cost[i] + min_to[i]) * n_visits[i]
                    if i in node_cost and node_cost[i] + min_to[i] < 0
                    else 0
                )
                for i in range(n)
            ]
            model.add_dual_bound(
                (current == n).if_then_else(
                    0,
                    -dp.fractional_knapsack(
                        reachable, instance.capacity - load, min_to_value, demand_weight
                    )
                    + min_to[0],
                )
            )
            model.add_dual_bound(
                (current == n).if_then_else(
                    0,
                    -dp.fractional_knapsack(
                        reachable,
                        instance.latest[0] - t - min_to[0],
                        min_to_value,
                        min_to_weight,
                    )
                    + min_to[0],
                )
            )

            min_from_value = [
                (
                    -(node_cost[i] + min_from[i]) * n_visits[i]
                    if i in node_cost and node_cost[i] + min_from[i] < 0
                    else 0
                )
                for i in range(n)
            ]
            model.add_dual_bound(
                (current == n).if_then_else(
                    0,
                    -dp.fractional_knapsack(
                        reachable,
                        instance.capacity - load,
                        min_from_value,
                        demand_weight,
                    )
                    + min_from_table[current],
                )
            )
            model.add_dual_bound(
                (current == n).if_then_else(
                    0,
                    -dp.fractional_knapsack(
                        reachable,
                        instance.latest[0] - t - min_from_time_table[current],
                        min_from_value,
                        min_from_weight,
                    )
                    + min_from_table[current],
                )
            )

    return model


class DIDPPricer(RoutePricer):
    def __init__(
        self,
        instance,
        shortest_travel_time,
        elementary=False,
        bound="knapsack",
        disable_set_resource=False,
        disable_filter=False,
        get_all_solutions=False,
        one_column_farkas=False,
        epsilon=1e-6,
        solver="labeling",
        f_first=False,
    ):
        super().__init__()
        self.instance = instance
        self.shortest_travel_time = shortest_travel_time
        self.elementary = elementary
        self.bound = bound
        self.disable_set_resource = disable_set_resource
        self.disable_filter = disable_filter
        self.get_all_solutions = get_all_solutions
        self.one_column_farkas = one_column_farkas
        self.epsilon = epsilon
        self.f_first = f_first
        self.pricing_stats = {"time": 0.0, "calls": 0, "columns": 0, "expanded": 0}

        # Store the solver type.
        assert solver in [
            "cabs",
            "caasdy",
            "labeling",
        ], f'Invalid DIDP solver "{solver}"'
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
        primal_bound = -self.epsilon - constant_cost

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

                for i in route[1:-1]:
                    if i not in coeffs:
                        coeffs[i] = 0

                    coeffs[i] += 1

                cost = self.instance.compute_route_cost(route[1:-1])
                redcost = solution.cost + constant_cost

                yield route, coeffs, cost, redcost, solution.time_out

                if self.one_column_farkas and farkas:
                    break

            elif solution.time_out:
                yield None, None, None, None, True

    def empty(self):
        yield from ()

    def pricer_column(self, edges, node_costs, constant_cost, farkas, time_limit):
        for i in range(self.instance.n_nodes):
            if not any(
                (i, j) in edges for j in range(self.instance.n_nodes)
            ) or not any((j, i) in edges for j in range(self.instance.n_nodes)):
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
                disable_bound=(self.bound == "none" or self.bound is None),
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
        self,
        instance,
        epsilon=1e-6,
        threads=1,
    ):
        super().__init__()
        self.instance = instance
        self.epsilon = epsilon
        self.threads = threads
        self.nodes = list(range(instance.n_nodes))
        self.pricing_model = None
        self.pricing_stats = {"time": 0.0, "calls": 0, "columns": 0}

    def create_model(self, edges, node_cost, farkas=False):
        if self.pricing_model is None:
            model = gp.Model()

            x = model.addVars(edges, vtype=gp.GRB.BINARY, name="x")
            t = model.addVars(
                self.nodes,
                vtype=gp.GRB.CONTINUOUS,
                lb=self.instance.earliest,
                ub=self.instance.latest,
            )

            objective = gp.quicksum(
                node_cost[i] * x[i, j] for i, j in edges if i in node_cost
            )

            if not farkas:
                objective += gp.quicksum(
                    self.instance.travel_time[i, j] * x[i, j] for i, j in edges
                )

            model.setObjective(objective, gp.GRB.MINIMIZE)

            model.addConstr(
                gp.quicksum(x[0, j] for j in self.nodes[1:] if (0, j) in edges) == 1
            )
            model.addConstrs(
                gp.quicksum(x[i, j] for j in self.nodes if (i, j) in edges)
                == gp.quicksum(x[j, i] for j in self.nodes if (j, i) in edges)
                for i in self.nodes[1:]
            )
            model.addConstr(
                gp.quicksum(x[i, 0] for i in self.nodes[1:] if (i, 0) in edges) == 1
            )

            model.addConstrs(
                t[j]
                >= (self.instance.service_time[0] + self.instance.travel_time[0, j])
                * x[0, j]
                for j in self.nodes
                if (0, j) in edges
            )
            big_m = {
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
                <= big_m[i, j] * (1 - x[i, j])
                for i, j in edges
                if i != 0
            )

            model.addConstr(
                gp.quicksum(self.instance.demand[i] * x[i, j] for (i, j) in edges)
                <= self.instance.capacity
            )

            self.pricing_model = model

            return model, x
        else:
            x = {}

            for i in self.nodes:
                for j in self.nodes:
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

    def solve(self, model, x, constant_cost, time_limit):
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
                for next in self.nodes:
                    if (current, next) in x and x[current, next].Xn > 0.5:
                        current = next
                        break

                route.append(current)

                if next == 0:
                    break

            route = tuple(route)

            coeffs = {}

            for i in route[1:-1]:
                if i not in coeffs:
                    coeffs[i] = 0

                coeffs[i] += 1

            cost = self.instance.compute_route_cost(route[1:-1])
            redcost = objval + constant_cost

            if redcost < -self.epsilon:
                yield route, coeffs, cost, redcost, False

        if model.status == gp.GRB.TIME_LIMIT:
            yield None, None, None, None, True

    def pricerredcost_column(self, edges, node_costs, constant_cost, time_limit=None):
        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        try:
            model, x = self.create_model(edges, node_costs, farkas=False)
            for column in self.solve(model, x, constant_cost, time_limit):
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
            for column in self.solve(model, x, constant_cost, time_limit):
                if column[0] is not None:
                    self.pricing_stats["columns"] += 1

                yield column

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start


class CPPricer(RoutePricer):
    def __init__(
        self,
        instance,
        epsilon=1e-6,
        threads=1,
    ):
        super().__init__()
        self.instance = instance
        self.epsilon = epsilon
        self.threads = threads
        self.nodes = list(range(instance.n_nodes))
        self.pricing_stats = {"time": 0.0, "calls": 0, "columns": 0}

    def create_model(self, edges, node_cost, farkas=False):
        n = self.instance.n_nodes

        model = cp.CpModel()

        x = {(i, j): model.new_bool_var(f"x_{i}_{j}") for i, j in edges if j != 0}

        for i in self.nodes[1:]:
            if (i, 0) in edges:
                x[i, n] = model.new_bool_var(f"x_{i}_{n}")

        y = {i: model.new_bool_var(f"y_{i}") for i in self.nodes[1:]}

        t = [
            model.new_int_var(
                self.instance.earliest[i], self.instance.latest[i], f"t_{i}"
            )
            for i in self.nodes
        ] + [
            model.new_int_var(
                self.instance.earliest[0], self.instance.latest[0], f"t_{n}"
            )
        ]

        model.add_circuit(
            [(i, j, x_ij) for (i, j), x_ij in x.items()]
            + [(i, i, ~y[i]) for i in self.nodes[1:]]
            + [(n, 0, True)]
        )

        for (i, j), x_ij in x.items():
            d_ij = (
                self.instance.travel_time[i, 0]
                if j == n
                else self.instance.travel_time[i, j]
            )
            model.add(
                t[j] >= t[i] + self.instance.service_time[i] + d_ij
            ).only_enforce_if(x_ij)

        model.add(
            sum(self.instance.demand[i] * y[i] for i in self.nodes[1:])
            <= self.instance.capacity
        )

        model.minimize(
            sum(
                (
                    0
                    if farkas
                    else (
                        self.instance.travel_time[i, 0]
                        if j == n
                        else self.instance.travel_time[i, j]
                    )
                    * x_ij
                )
                for (i, j), x_ij in x.items()
            )
            + sum(node_cost[i] * y[i] for i in self.nodes[1:] if i in node_cost)
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

                while current != len(self.nodes):
                    for next in self.nodes + [len(self.nodes)]:
                        if (current, next) in self.x and self.Value(
                            self.x[current, next]
                        ) > 0.5:
                            if next == len(self.nodes):
                                route.append(0)
                            else:
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

        callback = SolutionCallback(self.nodes, x)
        status = solver.solve(model, callback)

        for route in callback.routes:
            route = tuple(route)

            coeffs = {}

            for i in route[1:-1]:
                if i not in coeffs:
                    coeffs[i] = 0

                coeffs[i] += 1

            cost = self.instance.compute_route_cost(route[1:-1])
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
    parser.add_argument("--n-decimal-places", type=int, default=2)
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
    parser.add_argument(
        "--bound",
        type=str,
        choices=["knapsack", "profit", "none"],
        default="knapsack",
    )
    parser.add_argument(
        "--disable-set-resource",
        action="store_true",
    )
    parser.add_argument(
        "--disable-filter",
        action="store_true",
    )
    parser.add_argument("--get-all-solutions", action="store_true")
    parser.add_argument("--one-column-farkas", action="store_true")
    parser.add_argument("--f-first", action="store_true")
    args = parser.parse_args()

    vrptw_instance = vrplib.read_instance(args.input, instance_format="solomon")
    instance = VRPTW.from_solomon(
        vrptw_instance, n_decimal_places=args.n_decimal_places
    )
    edges = instance.get_edges()

    if args.pricer == "mip":
        pricer = GurobiPricer(
            instance,
            epsilon=args.epsilon,
            threads=args.threads,
        )
    elif args.pricer == "cp":
        pricer = CPPricer(
            instance,
            epsilon=args.epsilon,
            threads=args.threads,
        )
    else:
        shortest_travel_time = instance.compute_shortest_travel_time(edges)
        pricer = DIDPPricer(
            instance,
            shortest_travel_time,
            elementary=args.pricer == "dp",
            epsilon=args.epsilon,
            bound=args.bound,
            disable_set_resource=args.disable_set_resource,
            disable_filter=args.disable_filter,
            get_all_solutions=args.get_all_solutions,
            one_column_farkas=args.one_column_farkas,
            solver=args.solver,
            f_first=args.f_first,
        )

    nodes = list(range(1, instance.n_nodes))
    routes = [(0, i, 0) for i in nodes]
    route_costs = [instance.compute_route_cost(r[1:-1]) for r in routes]
    route_coeffs = [{i: 1} for i in nodes]

    model = VRPBaP(
        nodes,
        edges,
        [0],
        [0],
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
        print("Pricing expanded:", pricer.pricing_stats["expanded"])

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

            if instance.validate(routes, cost):
                print("The solution is valid")
            else:
                print("The solution is invalid")
        else:
            print("no solution found")
