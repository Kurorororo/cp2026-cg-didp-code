#!/usr/bin/env python3

import argparse
import time

import pygcgopt as gcg
import numpy as np

from pdptw import PDPTW


def create_three_index_model(instance, edges, shortest_travel_time, vehicle_penalty):
    vehicles = list(range(len(instance.pickup_locations)))
    nodes = list(range(instance.n_nodes))
    edges.add((0, instance.n_nodes - 1))

    model = gcg.Model()

    x = {
        (i, j, k): model.addVar(
            f"x_{i}_{j}_{k}",
            vtype="B",
            obj=(
                instance.travel_time[i, j] + vehicle_penalty
                if i == 0 and j != instance.n_nodes - 1
                else instance.travel_time[i, j]
            ),
        )
        for i, j in edges
        for k in vehicles
    }
    t = {
        (i, k): model.addVar(
            f"t_{i}_{k}",
            vtype="C",
            lb=instance.earliest[i],
            ub=instance.latest[i],
        )
        for i in nodes
        for k in vehicles
    }
    q = {
        (i, k): model.addVar(
            f"q_{i}_{k}",
            vtype="C",
            lb=max(0, instance.demand[i]),
            ub=min(instance.capacity, instance.capacity + instance.demand[i]),
        )
        for i in nodes
        for k in vehicles
    }

    # Multi-commodity flow on the network
    model.addConss(
        gcg.quicksum(x[0, j, k] for j in nodes[1:] if (0, j) in edges) == 1
        for k in vehicles
    )
    model.addConss(
        gcg.quicksum(x[i, j, k] for j in nodes if (i, j) in edges)
        == gcg.quicksum(x[j, i, k] for j in nodes if (j, i) in edges)
        for k in vehicles
        for i in nodes[1:-1]
    )
    model.addConss(
        gcg.quicksum(
            x[i, instance.n_nodes - 1, k]
            for i in nodes[:-1]
            if (i, instance.n_nodes - 1) in edges
        )
        == 1
        for k in vehicles
    )

    # Every pickup is served
    model.addConss(
        gcg.quicksum(x[i, j, k] for k in vehicles for j in nodes if (i, j) in edges)
        == 1
        for i in instance.pickup_locations
    )

    # Delivery on the same vehicle as the pickup
    model.addConss(
        gcg.quicksum(x[p, j, k] for j in nodes if (p, j) in edges)
        == gcg.quicksum(x[d, j, k] for j in nodes if (d, j) in edges)
        for p, d in zip(instance.pickup_locations, instance.delivery_locations)
        for k in vehicles
    )

    # Travel time
    big_m = {
        (i, j): instance.latest[i]
        + instance.service_time[i]
        + instance.travel_time[i, j]
        - instance.earliest[j]
        for i, j in edges
    }
    model.addConss(
        t[j, k]
        >= t[i, k]
        + instance.service_time[i]
        + instance.travel_time[i, j]
        - big_m[i, j] * (1 - x[i, j, k])
        for i, j in edges
        for k in vehicles
    )

    # Delivery time
    model.addConss(
        t[d, k] >= t[p, k] + shortest_travel_time[p, d]
        for p, d in zip(instance.pickup_locations, instance.delivery_locations)
        for k in vehicles
    )

    # Load
    model.addConss(
        q[j, k]
        >= q[i, k]
        + instance.demand[j]
        - (instance.capacity + instance.demand[j]) * (1 - x[i, j, k])
        for i, j in edges
        for k in vehicles
    )

    return model, x


def extract_three_index_routes(instance, model, x):
    routes = []

    for k in range(len(instance.pickup_locations)):
        r = []
        current = 0

        while current != instance.n_nodes - 1:
            if current != 0:
                r.append(current)

            for next in range(instance.n_nodes):
                if (current, next, k) in x and model.getVal(x[current, next, k]) > 0.5:
                    current = next
                    break

        if len(r) > 0:
            routes.append(r)

    return routes


def create_two_index_model(instance, edges, shortest_travel_time, vehicle_penalty):
    nodes = list(range(instance.n_nodes))

    model = gcg.Model()

    x = {
        (i, j): model.addVar(
            f"x_{i}_{j}",
            vtype="B",
            obj=(
                instance.travel_time[i, j] + vehicle_penalty
                if i == 0
                else instance.travel_time[i, j]
            ),
        )
        for i, j in edges
    }
    t = {
        i: model.addVar(
            f"t_{i}",
            vtype="C",
            lb=instance.earliest[i],
            ub=instance.latest[i],
        )
        for i in nodes
    }
    q = {
        i: model.addVar(
            f"q_{i}",
            vtype="C",
            lb=max(0, instance.demand[i]),
            ub=min(instance.capacity, instance.capacity + instance.demand[i]),
        )
        for i in nodes
    }
    v = {i: model.addVar(f"v_{i}", vtype="C") for i in nodes[1:-1]}

    # Every request is served
    model.addConss(
        gcg.quicksum(x[i, j] for i in nodes if (i, j) in edges) == 1
        for j in nodes[1:-1]
    )
    model.addConss(
        gcg.quicksum(x[i, j] for j in nodes if (i, j) in edges) == 1
        for i in nodes[1:-1]
    )

    # Travel time
    big_m = {
        (i, j): instance.latest[i]
        + instance.service_time[i]
        + instance.travel_time[i, j]
        - instance.earliest[j]
        for i, j in edges
    }
    model.addConss(
        t[j]
        >= t[i]
        + instance.service_time[i]
        + instance.travel_time[i, j]
        - big_m[i, j] * (1 - x[i, j])
        for i, j in edges
    )

    # Precedence
    model.addConss(
        t[d] >= t[p] + shortest_travel_time[p, d]
        for p, d in zip(instance.pickup_locations, instance.delivery_locations)
    )

    # Load
    model.addConss(
        q[j]
        >= q[i]
        + instance.demand[j]
        - (instance.capacity + instance.demand[j]) * (1 - x[i, j])
        for i, j in edges
    )

    # Same vehicle for delivery
    n_tasks = len(instance.pickup_locations)
    location_to_index = {
        p: i + 1
        for i, p in enumerate(instance.pickup_locations + instance.delivery_locations)
    }
    model.addConss(
        v[d] == v[p]
        for p, d in zip(instance.pickup_locations, instance.delivery_locations)
    )
    model.addConss(
        v[j] >= location_to_index[j] * x[0, j] for j in nodes[1:-1] if (0, j) in edges
    )
    model.addConss(
        v[j] <= location_to_index[j] * x[0, j] - n_tasks * (x[0, j] - 1)
        for j in nodes[1:-1]
        if (0, j) in edges
    )
    model.addConss(
        v[j] >= v[i] + n_tasks * (x[i, j] - 1)
        for i in nodes[1:-1]
        for j in nodes[1:-1]
        if (i, j) in edges
    )
    model.addConss(
        v[j] <= v[i] + n_tasks * (1 - x[i, j])
        for i in nodes[1:-1]
        for j in nodes[1:-1]
        if (i, j) in edges
    )

    return model, x


def extract_two_index_routes(instance, model, x):
    routes = []

    for current in range(1, instance.n_nodes - 1):
        if (0, current) in x and model.getVal(x[0, current]) > 0.5:
            r = []

            while current != instance.n_nodes - 1:
                r.append(current)

                for next in range(instance.n_nodes):
                    if (current, next) in x and model.getVal(x[current, next]) > 0.5:
                        current = next
                        break

            routes.append(r)

    return routes


if __name__ == "__main__":
    start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n-decimal-places", type=int, default=3)
    parser.add_argument("--vehicle-penalty", type=float, default=1e4)
    parser.add_argument("--three-index", action="store_true")
    parser.add_argument(
        "--file-format", type=str, choices=["LiLim", "Ropke"], default="LiLim"
    )
    args = parser.parse_args()

    vehicle_penalty = args.vehicle_penalty * (10**args.n_decimal_places)

    if args.file_format == "LiLim":
        pdptw_instance = PDPTW.load_from_li_lim_file(
            args.input, n_decimal_places=args.n_decimal_places
        )
    else:
        pdptw_instance = PDPTW.load_from_ropke_file(
            args.input, n_decimal_places=args.n_decimal_places
        )

    edges, shortest_travel_time = pdptw_instance.preprocess()

    if args.three_index:
        model, x = create_three_index_model(
            pdptw_instance, edges, shortest_travel_time, vehicle_penalty
        )
    else:
        model, x = create_two_index_model(
            pdptw_instance, edges, shortest_travel_time, vehicle_penalty
        )

    if not args.verbose:
        model.hideOutput()

    model.setParam("display/freq", 1)
    model.setParam("display/headerfreq", 1)

    model.setParam("lp/threads", 1)
    model.setParam("parallel/maxnthreads", 1)

    if args.timeout is not None:
        model.setParam("limits/time", args.timeout)

    model.optimize()

    print("Solving time: {}s".format(time.perf_counter() - start))
    print("Nodes:", int(model.getNTotalNodes()))

    status = model.getStatus()
    if status == "infeasible":
        print("The problem is infeasible")
    else:
        print("best bound:", model.getDualbound())

        if model.getNSols() > 0:
            cost = int(round(model.getPrimalbound()))
            print("cost:", cost)

            if status == "optimal":
                print("optimal cost:", cost)

            if args.three_index:
                routes = extract_three_index_routes(pdptw_instance, model, x)
            else:
                routes = extract_two_index_routes(pdptw_instance, model, x)

            print(routes)

            is_feasibility_validated = pdptw_instance.validate_feasibility(routes)

            if is_feasibility_validated:
                print("The solution is valid")

            recomputed_cost = pdptw_instance.compute_solution_cost(
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
