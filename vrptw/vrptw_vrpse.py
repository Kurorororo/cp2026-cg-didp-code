import argparse
import time

import VRPSolverEasy as vrpse
import vrplib

from vrptw import VRPTW


def create_model(instance):
    edges = instance.get_edges()

    model = vrpse.Model()

    model.add_vehicle_type(
        id=1,
        start_point_id=0,
        end_point_id=0,
        capacity=instance.capacity,
        max_number=instance.n_nodes,
        tw_end=int(instance.latest[0]),
        var_cost_dist=1,
    )
    model.add_depot(
        id=0, tw_begin=int(instance.earliest[0]), tw_end=int(instance.latest[0])
    )

    for i in range(1, instance.n_nodes):
        model.add_customer(
            id=i,
            demand=int(instance.demand[i]),
            tw_begin=int(instance.earliest[i]),
            tw_end=int(instance.latest[i] + instance.service_time[i]),
            service_time=int(instance.service_time[i]),
        )

    for i, j in edges:
        model.add_link(
            start_point_id=i,
            end_point_id=j,
            distance=int(instance.travel_time[i, j]),
            time=int(instance.travel_time[i, j]),
        )

    return model


if __name__ == "__main__":
    start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    # parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--n-decimal-places", type=int, default=2)
    # parser.add_argument("--verbose", action="store_true")
    # parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    vrptw_instance = vrplib.read_instance(args.input, instance_format="solomon")
    instance = VRPTW.from_solomon(
        vrptw_instance, n_decimal_places=args.n_decimal_places
    )

    model = create_model(instance)

    model.parameters.time_limit = args.timeout

    model.solve()
    model.export()

    if model.solution.is_defined():
        print(model.solution)
