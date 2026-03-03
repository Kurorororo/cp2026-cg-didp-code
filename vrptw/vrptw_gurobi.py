#!/usr/bin/env python3

import argparse
import time

import gurobipy as gp
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
    )
    demand = np.append(instance.demand, instance.demand[0])
    capacity = instance.capacity
    earliest = np.append(instance.earliest, instance.earliest[0])
    latest = np.append(instance.latest, instance.latest[0])
    service_time = np.append(instance.service_time, instance.service_time[0])
    travel_time = np.vstack([instance.travel_time, instance.travel_time[0]])
    travel_time = np.column_stack((travel_time, travel_time[:, 0]))

    model = gp.Model()

    x = model.addVars(
        edges,
        vehicles,
        vtype="B",
        obj={(i, j, k): travel_time[i, j] for i, j in edges for k in vehicles}
    )
    t = model.addVars(
        nodes,
        vehicles,
        vtype="C",
        lb={(i, k): earliest[i] for i in nodes for k in vehicles},
        ub={(i, k): latest[i] for i in nodes for k in vehicles}
    )

    # Multi-commodity flow on the network
    model.addConstrs(
        gp.quicksum(x[0, j, k] for j in nodes[1:] if (0, j) in edges) == 1
        for k in vehicles
    )
    model.addConstrs(
        gp.quicksum(x[i, j, k] for j in nodes if (i, j) in edges)
        == gp.quicksum(x[j, i, k] for j in nodes if (j, i) in edges)
        for k in vehicles
        for i in nodes[1:-1]
    )
    model.addConstrs(
        gp.quicksum(
            x[i, instance.n_nodes, k]
            for i in nodes[:-1]
            if (i, instance.n_nodes) in edges
        )
        == 1
        for k in vehicles
    )

    # Every request is served
    model.addConstrs(
        gp.quicksum(x[i, j, k] for k in vehicles for j in nodes if (i, j) in edges)
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
    model.addConstrs(
        t[j, k] >= t[i, k] + service_time[i] + travel_time[i, j] - big_m[i, j] * (1 - x[i, j, k])
        for i, j in edges
        for k in vehicles
    )

    # Load
    model.addConstrs(
        gp.quicksum(demand[j] * x[i, j, k] for (i, j) in edges) <= capacity
        for k in vehicles
    )

    return model, x


def extract_three_index_routes(instance, model, x):
    routes = []

    n_vehicles = instance.n_nodes - 1
    for k in range(n_vehicles):
        r = []
        current = 0

        while current != instance.n_nodes:
            if current != 0:
                r.append(current)

            for next in range(instance.n_nodes + 1):
                if (current, next, k) in x and x[current, next, k].x > 0.5:
                    current = next
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

    model = gp.Model()

    x = model.addVars(
        edges,
        vtype=gp.GRB.BINARY,
        obj={(i, j): travel_time[i, j] for i, j in edges}
    )
    t = model.addVars(
        nodes,
        vtype=gp.GRB.CONTINUOUS,
        lb=earliest,
        ub=latest,
    )
    q = model.addVars(
        nodes,
        vtype=gp.GRB.CONTINUOUS,
        lb=demand,
        ub=capacity
    )

    # Every request is served
    model.addConstrs(
        gp.quicksum(x[i, j] for i in nodes if (i, j) in edges) == 1 for j in nodes[1:-1]
    )
    model.addConstrs(
        gp.quicksum(x[i, j] for j in nodes if (i, j) in edges) == 1 for i in nodes[1:-1]
    )

    # Travel time
    big_m = {
        (i, j): latest[i]
        + service_time[i]
        + travel_time[i, j]
        - earliest[j]
        for i, j in edges
    }
    model.addConstrs(
        t[j]
        >= t[i]
        + service_time[i]
        + travel_time[i, j]
        - big_m[i, j] * (1 - x[i, j])
        for i, j in edges
    )

    # Load
    model.addConstrs(
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
        if (0, current) in x and x[0, current].x > 0.5:
            r = []

            while current != instance.n_nodes:
                r.append(current)

                for next in range(instance.n_nodes + 1):
                    if (current, next) in x and x[current, next].x > 0.5:
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
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-decimal-places", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-6)
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
        model.params.outputFlag = 0

    model.params.threads = args.threads

    if args.timeout is not None:
        model.params.timeLimit = args.timeout

    model.params.MIPGap = args.tolerance

    model.optimize()

    print("Solving time: {}s".format(time.perf_counter() - start))
    print("Nodes:", int(model.NodeCount))

    if model.status == gp.GRB.INFEASIBLE:
        print("The problem is infeasible")
    else:
        print("best bound:", model.ObjBound)

        if model.sol_count > 0:
            cost = int(round(model.ObjVal))
            print("cost:", cost)

            if model.status == gp.GRB.OPTIMAL:
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
