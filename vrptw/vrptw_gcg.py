#!/usr/bin/env python3

import argparse
import time

import pygcgopt as gcg
import numpy as np
import vrplib

from vrptw import VRPTW


def create_three_index_model(instance):
    n_vehicles = instance.n_nodes - 1
    vehicles = list(range(n_vehicles))
    nodes = list(range(instance.n_nodes + 1))
    edges = instance.get_edges()
    edges = set(
        [(i, j) for i, j in edges if j != 0]
        + [(i, instance.n_nodes) for i in nodes if (i, 0) in edges]
        + [(0, instance.n_nodes)]
    )
    demand = np.append(instance.demand, instance.demand[0])
    capacity = instance.capacity
    earliest = np.append(instance.earliest, instance.earliest[0])
    latest = np.append(instance.latest, instance.latest[0])
    service_time = np.append(instance.service_time, instance.service_time[0])
    travel_time = np.vstack([instance.travel_time, instance.travel_time[0]])
    travel_time = np.column_stack((travel_time, travel_time[:, 0]))

    model = gcg.Model()

    x = {
        (i, j, k): model.addVar(f"x_{i}_{j}_{k}", vtype="B", obj=travel_time[i, j])
        for i, j in edges
        for k in vehicles
    }
    t = {
        (i, k): model.addVar(
            f"t_{i}_{k}",
            vtype="C",
            lb=earliest[i],
            ub=latest[i],
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
            x[i, instance.n_nodes, k]
            for i in nodes[:-1]
            if (i, instance.n_nodes) in edges
        )
        == 1
        for k in vehicles
    )

    # Every request is served
    model.addConss(
        gcg.quicksum(x[i, j, k] for k in vehicles for j in nodes if (i, j) in edges)
        == 1
        for i in nodes[1:-1]
    )

    # Travel time
    big_m = {
        (i, j): latest[i]
        + service_time[i]
        + travel_time[i, j]
        - earliest[j]
        for i, j in edges
    }
    model.addConss(
        t[j, k] >= t[i, k] + service_time[i] + travel_time[i, j] - big_m[i, j] * (1 - x[i, j, k])
        for i, j in edges
        for k in vehicles
    )

    # Load
    model.addConss(
        gcg.quicksum(demand[j] * x[i, j, k] for (i, j) in edges) <= capacity
        for k in vehicles
    )

    return model, x


def extract_three_index_routes(instance, model, x):
    routes = []

    n_vehicles = instance.n_nodes - 1
    for k in range(n_vehicles):
        r = []
        current = 0

        first = None
        for next in range(1, instance.n_nodes + 1):
            if (current, next, k) in x and model.getVal(x[current, next, k]) > 0.5:
                first = next
                break
        if first is None:
            continue
        current = first

        while current != instance.n_nodes:
            r.append(current)

            found = False
            for next in range(instance.n_nodes + 1):
                if (current, next, k) in x and model.getVal(x[current, next, k]) > 0.5:
                    current = next
                    found = True
                    break

            if not found:
                break

        if len(r) > 0:
            routes.append(r)

    return routes


def create_two_index_model(instance):
    nodes = list(range(instance.n_nodes + 1))
    edges = instance.get_edges()
    edges = set(
        [(i, j) for i, j in edges if j != 0]
        + [(i, instance.n_nodes) for i in nodes if (i, 0) in edges and i != 0]
    )
    demand = np.append(instance.demand, instance.demand[0])
    capacity = instance.capacity
    earliest = np.append(instance.earliest, instance.earliest[0])
    latest = np.append(instance.latest, instance.latest[0])
    service_time = np.append(instance.service_time, instance.service_time[0])
    travel_time = np.vstack([instance.travel_time, instance.travel_time[0]])
    travel_time = np.column_stack((travel_time, travel_time[:, 0]))

    model = gcg.Model()

    x = {
        (i, j): model.addVar(f"x_{i}_{j}", vtype="B", obj=travel_time[i, j])
        for i, j in edges
    }
    t = {
        i: model.addVar(
            f"t_{i}",
            vtype="C",
            lb=earliest[i],
            ub=latest[i],
        )
        for i in nodes
    }
    q = {
        i: model.addVar(
            f"q_{i}",
            vtype="C",
            lb=demand[i],
            ub=capacity,
        )
        for i in nodes
    }

    # Every request is served
    model.addConss(
        gcg.quicksum(x[i, j] for i in nodes if (i, j) in edges) == 1 for j in nodes[1:-1]
    )
    model.addConss(
        gcg.quicksum(x[i, j] for j in nodes if (i, j) in edges) == 1 for i in nodes[1:-1]
    )

    # Travel time
    big_m = {
        (i, j): latest[i]
        + service_time[i]
        + travel_time[i, j]
        - earliest[j]
        for i, j in edges
    }
    model.addConss(
        t[j]
        >= t[i]
        + service_time[i]
        + travel_time[i, j]
        - big_m[i, j] * (1 - x[i, j])
        for i, j in edges
    )

    # Load
    model.addConss(
        q[j]
        >= q[i]
        + demand[j]
        - (capacity + demand[j]) * (1 - x[i, j])
        for i, j in edges
    )

    return model, x


def extract_two_index_routes(instance, model, x):
    routes = []

    for current in range(1, instance.n_nodes):
        if (0, current) in x and model.getVal(x[0, current]) > 0.5:
            r = []

            while current != instance.n_nodes:
                r.append(current)

                for next in range(instance.n_nodes + 1):
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
    parser.add_argument("--n-decimal-places", type=int, default=2)
    parser.add_argument("--three-index", action="store_true")
    args = parser.parse_args()

    vrptw_instance = vrplib.read_instance(args.input, instance_format="solomon")
    instance = VRPTW.from_solomon(
        vrptw_instance, n_decimal_places=args.n_decimal_places
    )

    if args.three_index:
        model, x = create_three_index_model(instance)
    else:
        model, x = create_two_index_model(instance)

    if not args.verbose:
        model.hideOutput()

    model.setParam("display/freq", 1)
    model.setParam("display/headerfreq", 1)

    model.setParam("lp/threads", 1)
    model.setParam("parallel/maxnthreads", 1)

    if args.timeout is not None:
        model.setParam("limits/time", args.timeout)

    model.setObjIntegral()

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
                routes = extract_three_index_routes(instance, model, x)
            else:
                routes = extract_two_index_routes(instance, model, x)

            print(routes)

            if instance.validate(routes, cost):
                print("The solution is valid")
            else:
                print("The solution is invalid")
        else:
            print("no solution found")
