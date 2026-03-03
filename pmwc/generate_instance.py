import argparse

import numpy as np

parser = argparse.ArgumentParser(description="Generate random instance")
parser.add_argument(
    "--ns",
    type=int,
    nargs="+",
    default=[20, 30, 40, 20, 30, 40, 50],
    help="Number of jobs",
)
parser.add_argument(
    "--ms", type=int, nargs="+", default=[3, 4, 5], help="Number of machines"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--n-instances",
    type=int,
    default=5,
    help="Number of instances for each configuration",
)
parser.add_argument(
    "--dir-path", type=str, default="instances", help="Path to save instances"
)
args = parser.parse_args()

seed = args.seed

job_configurations = [((1, 10), (10, 100)), ((1, 100), (1, 100)), ((10, 20), (10, 20))]

np.random.seed(args.seed)

for n in args.ns:
    for m in args.ms:
        for (p_lb, p_ub), (w_lb, w_ub) in job_configurations:
            for i in range(args.n_instances):
                jobs = [
                    (
                        np.random.randint(p_lb, p_ub + 1),
                        np.random.randint(w_lb, w_ub + 1),
                    )
                    for _ in range(n)
                ]

                jobs.sort(key=lambda x: -x[1] / x[0])  # Sort by w/p in descending order

                filename = (
                    f"n{n}_m{m}_p{p_lb}_{p_ub}_w{w_lb}_{w_ub}_seed{seed}_id{i}.txt"
                )

                with open(f"{args.dir_path}/{filename}", "w") as f:
                    f.write(f"{n} {m}\n")

                    for p, w in jobs:
                        f.write(f"{p} {w}\n")
