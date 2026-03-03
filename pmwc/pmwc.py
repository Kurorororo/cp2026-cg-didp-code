import math


def read_instance(filename):
    with open(filename) as f:
        n, m = map(int, f.readline().strip().split())
        instance = {"n": n, "m": m, "p": [], "w": []}

        for line in f:
            p, w = map(int, line.strip().split())
            instance["p"].append(p)
            instance["w"].append(w)

        return instance


def compute_bound(instance):
    n = instance["n"]
    m = instance["m"]
    p = instance["p"]
    w = instance["w"]

    p_max = max(p)
    p_sum = sum(p)
    h_min = math.ceil(p_sum / m - (m - 1) * p_max / m)
    h_max = math.floor(p_sum / m + (m - 1) * p_max / m)

    predecessors = [
        [k for k in range(j) if w[k] >= w[j] and p[k] <= p[j]] for j in range(n)
    ]
    followers = [
        [k for k in range(j + 1, n) if w[k] <= w[j] and p[k] >= p[j]] for j in range(n)
    ]

    rho = [sum(p[k] for k in predecessors[j][: -m + 1]) for j in range(n)]
    r = [math.ceil(rho[j] / m) for j in range(n)]

    d = [math.floor(p_sum / m + (m - 1) * p[j] / m) for j in range(n)]
    delta = [
        h_max - math.ceil((sum(p[k] for k in followers[j]) + p[j]) / m)
        for j in range(n)
    ]
    d = [min(delta[j] + p[j], d[j]) for j in range(n)]

    instance["r"] = r
    instance["d"] = d
    instance["h_min"] = h_min
    instance["h_max"] = h_max


def validate(instance, schedule):
    if len(schedule) > instance["m"]:
        print(f"{len(schedule)} > {instance["m"]} machines are used")
        return None

    scheduled_jobs = set()
    cost = 0

    for s in schedule:
        time = 0

        for j in s:
            if j < 0 or j >= instance["n"]:
                print(f"Job {j} is out of range.")
                return None

            if j in scheduled_jobs:
                print(f"Job {j} is scheduled more than once.")
                return None

            scheduled_jobs.add(j)
            time += instance["p"][j]
            cost += instance["w"][j] * time

    if len(scheduled_jobs) != instance["n"]:
        print("Not all jobs are scheduled.")
        return None

    return cost
