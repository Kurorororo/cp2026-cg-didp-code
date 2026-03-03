import argparse
import time

import gurobipy as gp

import pmwc


def create_assignment_model(instance):
    model = gp.Model()

    n = instance["n"]
    m = instance["m"]
    p = instance["p"]
    w = instance["w"]
    r = instance["r"]
    d = instance["d"]

    # Job j is assigned to machine i
    x = model.addVars(n, m, vtype=gp.GRB.BINARY)
    # Completion time of job j
    lb = [r[j] + p[j] for j in range(n)]
    c = model.addVars(n, vtype=gp.GRB.CONTINUOUS, lb=lb, ub=d, obj=w)

    model.addConstrs(gp.quicksum(x[j, i] for i in range(m)) == 1 for j in range(n))
    model.addConstrs(
        c[j]
        >= gp.quicksum(p[k] * x[k, i] for k in range(j + 1))
        - sum(p[:j]) * (1 - x[j, i])
        for j in range(n)
        for i in range(m)
    )

    return model, x


def extract_assignment_solution(instance, x):
    n = instance["n"]
    m = instance["m"]
    schedule = [[] for _ in range(m)]

    for j in range(n):
        for i in range(m):
            if x[j, i].X > 0.5:
                schedule[i].append(j)

    return schedule


def create_precedence_model(instance):
    model = gp.Model()

    n = instance["n"]
    m = instance["m"]
    p = instance["p"]
    w = instance["w"]
    r = instance["r"]
    d = instance["d"]

    # Job k is right after job j, -1 for dummy start job and n for dummy end job
    pairs = [
        (k, j) for k in range(-1, n) for j in range(k + 1, n + 1) if k != -1 or j != n
    ]
    x = model.addVars(pairs, vtype=gp.GRB.BINARY)
    lb = [r[j] + p[j] for j in range(n)]
    c = model.addVars(n, vtype=gp.GRB.CONTINUOUS, lb=lb, ub=d, obj=w)

    model.addConstrs(gp.quicksum(x[k, j] for k in range(-1, j)) == 1 for j in range(n))
    model.addConstr(gp.quicksum(x[-1, j] for j in range(n)) <= m)
    model.addConstrs(
        gp.quicksum(x[k, j] for k in range(-1, j))
        == gp.quicksum(x[j, k] for k in range(j + 1, n + 1))
        for j in range(n)
    )

    model.addConstrs(
        c[j] >= c[k] + p[j] - (d[k] + p[j]) * (1 - x[k, j])
        for k in range(n - 1)
        for j in range(k + 1, n)
    )

    return model, x


def extract_precedence_solution(instance, x):
    n = instance["n"]
    schedule = []

    for j in range(n):
        if x[-1, j].X > 0.5:
            schedule.append([])
            current = j

            while current != n:
                schedule[-1].append(current)

                for k in range(current + 1, n + 1):
                    if x[current, k].X > 0.5:
                        current = k
                        break

    return schedule


if __name__ == "__main__":
    start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--model", type=str, choices=["assignment", "precedence"], default="assignment"
    )
    args = parser.parse_args()

    instance = pmwc.read_instance(args.input)
    pmwc.compute_bound(instance)

    if args.model == "assignment":
        model, x = create_assignment_model(instance)
    else:
        model, x = create_precedence_model(instance)

    if args.timeout is not None:
        model.params.timeLimit = args.timeout

    model.params.threads = args.threads

    if not args.verbose:
        model.params.OutputFlag = 0

    model.optimize()

    print("Solving time: {}s".format(time.perf_counter() - start))
    print("Nodes:", int(model.NodeCount))

    if model.status == gp.GRB.INFEASIBLE:
        print("The problem is infeasible")
    else:
        print("best bound:", model.ObjBound)

        if model.sol_count > 0:
            if args.model == "assignment":
                schedule = extract_assignment_solution(instance, x)
            else:
                schedule = extract_precedence_solution(instance, x)

            cost = int(round(model.ObjVal))

            print("cost:", cost)

            if model.status == gp.GRB.OPTIMAL:
                print("optimal cost:", cost)

            print(schedule)

            recomputed_cost = pmwc.validate(instance, schedule)

            if recomputed_cost is None:
                print("The solution is invalid")
            else:
                print("The solution is valid")

                if cost == recomputed_cost:
                    print("The cost is correct")
                else:
                    print(
                        "Cost {} does not match recomputed cost {}".format(
                            cost, recomputed_cost
                        )
                    )

        else:
            print("No solution found")
