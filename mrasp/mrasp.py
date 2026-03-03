import didppy as dp
import pyscipopt as scip
import numpy as np
import math
import re
import time

# iter = 0

# Print numpy arrays without line wrap
np.set_printoptions(linewidth=np.inf)

# Functions for approximate comparison of floating point numbers
EPS = 1e-6


def is_eq(x, y):
    return abs(x - y) <= EPS


def is_lt(x, y):
    return x - y < -EPS


def is_le(x, y):
    return x - y <= EPS


def is_gt(x, y):
    return x - y > EPS


def is_ge(x, y):
    return x - y >= -EPS


def eps_floor(x):
    return math.floor(x + EPS)


def eps_ceil(x):
    return math.ceil(x - EPS)


def eps_round(x):
    return math.ceil(x - 0.5 + EPS)


def eps_frac(x):
    return x - eps_floor(x)


def is_integral(x):
    return eps_frac(x) <= EPS


class Master:
    """
    Master problem
    """

    def __init__(self, data, time_limit=None):
        """
        Create the master problem
        """

        from pyscipopt import quicksum

        # Store the instance data.
        self.data = data
        num_runways = self.data["num_runways"]
        num_aircraft = self.data["num_aircraft"]

        # Create model.
        self.model = scip.Model()

        # Set number of threads.
        self.model.setParam("lp/threads", 1)
        self.model.setParam("parallel/maxnthreads", 1)

        # Set time limit.
        if time_limit is not None:
            self.model.setRealParam("limits/time", time_limit)

        # Turn off presolving and cutting planes.
        self.model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
        self.model.setSeparating(scip.SCIP_PARAMSETTING.OFF)
        self.model.setIntParam("propagating/rootredcost/freq", -1)
        self.model.setIntParam("display/freq", 1)

        # Indicate that the objective function is integer.
        self.model.setObjIntegral()

        # Limit the number of runways.
        self.runway_constraint = self.model.addCons(
            quicksum([]) <= num_runways, separate=False, modifiable=True, name=f"runway"
        )

        # Every aircraft must appear in at least one schedule.
        self.aircraft_constraints = [
            self.model.addCons(
                quicksum([]) >= 1.0,
                separate=False,
                modifiable=True,
                name=f"aircraft[{a}]",
            )
            for a in range(num_aircraft)
        ]

        # Prepare somewhere to store the schedules.
        self.schedules = {}

    def transform_constraints(self):
        """
        Get the transformed constraints in the master problem
        """

        # Get the constraints in the transformed problem.
        self.runway_constraint = self.model.getTransformedCons(self.runway_constraint)
        self.aircraft_constraints = [
            self.model.getTransformedCons(cons) for cons in self.aircraft_constraints
        ]

    def check_schedule(self, cost, aircraft_time, aircraft_order):
        """
        Verify that a schedule is valid
        """

        # Get the instance data.
        aircraft_release = self.data["release"]
        aircraft_due = self.data["due"]
        aircraft_separation_time = self.data["aircraft_separation_time"]
        aircraft_cost = self.data["weight"]

        # Check the solution.
        assert len(aircraft_time) == len(aircraft_order)
        for i in range(len(aircraft_order) - 1):
            current = aircraft_order[i]
            next = aircraft_order[i + 1]
            current_time = aircraft_time[i]
            next_time = aircraft_time[i + 1]
            assert is_ge(
                next_time, current_time + aircraft_separation_time[current, next]
            )

        # Check the time and cost.
        check_cost = 0.0
        for t, a in zip(aircraft_time, aircraft_order):
            check_cost += t * aircraft_cost[a]
            assert is_le(aircraft_release[a], t)
            assert is_le(t, aircraft_due[a])
        assert is_eq(cost, check_cost)

    def add_var(self, cost, aircraft_time, aircraft_order):
        """
        Add a priced variable
        """

        # Check solution for feasibility.
        self.check_schedule(cost, aircraft_time, aircraft_order)

        # Create the variable.
        name = f'schedule[{",".join(map(str, aircraft_order))}]'
        # assert name not in self.schedules
        if name in self.schedules:
            existing_time, existing_order, existing_var = self.schedules[name]
            assert existing_time == aircraft_time
            assert existing_order == aircraft_order
            print(f"Duplicate column {name}, original in LP: {existing_var.isInLP()}")
        else:
            var = self.model.addVar(
                vtype="I", lb=0, ub=None, obj=cost, name=name, pricedVar=True
            )

            # Attach coefficients to the constraints.
            self.model.addConsCoeff(self.runway_constraint, var, 1.0)
            for a in aircraft_order:
                self.model.addConsCoeff(self.aircraft_constraints[a], var, 1.0)

            # Store the schedule.
            self.schedules[name] = (aircraft_time, aircraft_order, var)

    def time_remaining(self):
        """
        Get the time remaining in seconds
        """

        time_limit = self.model.getParam("limits/time")
        elapsed_time = self.model.getSolvingTime()
        time_remaining = time_limit - elapsed_time
        return time_remaining

    def write(self):
        """
        Write the model to file
        """

        self.model.writeProblem("master.lp", trans=True)


class DIDPPricer(scip.Pricer):
    """
    Pricing problem
    """

    def __init__(
        self,
        data,
        solver,
        get_all_solutions,
        one_column_farkas,
        bound,
        master,
        branching_conshdlr,
        disable_set_resource,
        disable_filter,
        f_first=False,
    ):
        """
        Create the pricing problem
        """

        # Store the instance data.
        self.data = data

        # Store the solver type.
        assert solver in [
            "cabs",
            "caasdy",
            "labeling",
        ], f'Invalid DIDP solver "{solver}"'
        self.get_all_solutions = get_all_solutions
        self.one_column_farkas = one_column_farkas
        self.f_first = f_first
        if solver == "cabs":
            self.solver_type = dp.CABS
            self.get_all_solutions = False
            self.f_first = False
        elif solver == "caasdy":
            self.solver_type = dp.CAASDy
            self.f_first = False
        elif solver == "labeling":
            self.solver_type = dp.Labeling

        # Store the other parts of the solver.
        self.master = master
        self.branching_conshdlr = branching_conshdlr
        self.disable_set_resource = disable_set_resource
        self.disable_filter = disable_filter
        self.bound = bound
        self.pricing_stats = {"time": 0.0, "calls": 0, "columns": 0, "expanded": 0}

        # Attach the pricer.
        self.master.model.includePricer(self, "didp", "Pricer using DIDP", delay=True)

    def pricerinit(self):

        # Get the master problem constraints. TODO: This should be called somewhere else.
        self.master.transform_constraints()

    def price(self, master_is_feasible, runway_dual_sol, aircraft_dual_sol):
        """
        Run the pricing algorithm
        """

        # global iter
        # iter += 1
        # print(f'ITER {iter}')

        # Solve using DIDP.
        reduced_cost, is_optimal = self.solve_didp(
            master_is_feasible, runway_dual_sol, aircraft_dual_sol
        )
        return reduced_cost, is_optimal

    def solve_didp(
        self, master_is_feasible, runway_dual_sol, aircraft_dual_sol, add_var=True
    ):

        verbose = False

        # Get the instance data.
        num_aircraft = self.data["num_aircraft"]
        aircraft_reduced_cost = [-x for x in aircraft_dual_sol]
        aircraft_type = self.data["aircraft_type"]
        aircraft_operation = self.data["operation"]
        aircraft_release = self.data["release"]
        aircraft_due = self.data["due"]
        aircraft_separation_time = self.data["aircraft_separation_time"]
        type_separation_time = self.data["separation_times"]
        aircraft_cost = self.data["weight"]

        # Create the arcs.
        release_matrix = np.expand_dims(np.array(aircraft_release), 1).repeat(
            num_aircraft, axis=1
        )
        due_matrix = np.expand_dims(np.array(aircraft_due), 0).repeat(
            num_aircraft, axis=0
        )
        arcs = ~np.eye(num_aircraft + 1, dtype=np.bool_)
        arcs[:num_aircraft, :num_aircraft] &= (
            release_matrix + aircraft_separation_time[:num_aircraft] <= due_matrix
        )
        no_cons = True

        for cons in self.branching_conshdlr.conss:
            if cons.isActive():
                (i, j, direction) = cons.data
                if direction:
                    original_val = arcs[i, j]
                    arcs[:, j] = False
                    arcs[i, :] = False
                    arcs[i, j] = original_val
                else:
                    arcs[i, j] = False

                no_cons = False

        # Create model.
        model = dp.Model(maximize=False, float_cost=True)
        future = model.add_local_var()

        # Create input data.
        aircraft = model.add_object_type(number=num_aircraft)
        aircraft_reduced_cost_table = model.add_float_table(aircraft_reduced_cost)
        aircraft_cost_table = model.add_int_table(aircraft_cost)
        aircraft_release_table = model.add_int_table(aircraft_release)
        aircraft_due_table = model.add_int_table(
            [
                (
                    min(
                        aircraft_due[a],
                        int(eps_floor(-aircraft_reduced_cost[a] / aircraft_cost[a])),
                    )
                    if master_is_feasible and no_cons
                    else aircraft_due[a]
                )
                for a in range(num_aircraft)
            ]
        )
        aircraft_separation_time_table = model.add_int_table(
            aircraft_separation_time.tolist()
        )
        arcs = model.add_bool_table(arcs.tolist())
        is_aircraft_small_arrival_table = model.add_bool_table(
            [
                aircraft_operation[a] == "Arrival" and aircraft_type[a] == "Small"
                for a in range(num_aircraft)
            ]
            + [False]
        )
        is_aircraft_large_arrival_table = model.add_bool_table(
            [
                aircraft_operation[a] == "Arrival" and aircraft_type[a] == "Large"
                for a in range(num_aircraft)
            ]
            + [False]
        )

        # Create resource variables.
        time = model.add_int_resource_var(target=0, less_is_better=True)

        initial_aircraft = [
            a
            for a in range(num_aircraft)
            if not no_cons
            or aircraft_reduced_cost[a]
            + (aircraft_cost[a] * aircraft_release[a] if master_is_feasible else 0.0)
            < 0.0
        ]

        if self.disable_set_resource:
            reachable = model.add_set_var(object_type=aircraft, target=initial_aircraft)
        else:
            reachable = model.add_set_resource_var(
                object_type=aircraft,
                target=initial_aircraft,
                less_is_better=False,
            )

        earliest_small_arrival = model.add_int_resource_var(
            target=0, less_is_better=True
        )
        earliest_large_arrival = model.add_int_resource_var(
            target=0, less_is_better=True
        )

        # Create a transition for each aircraft.
        current = model.add_element_var(object_type=aircraft, target=num_aircraft)
        all_transitions = []

        for next in range(num_aircraft):
            next_time_lb = time + aircraft_separation_time_table[current, next]
            if aircraft_operation[next] == "Arrival" and aircraft_type[next] == "Small":
                next_time_lb = dp.max(next_time_lb, earliest_small_arrival)
            elif (
                aircraft_operation[next] == "Arrival" and aircraft_type[next] == "Large"
            ):
                next_time_lb = dp.max(next_time_lb, earliest_large_arrival)

            next_time = dp.max(next_time_lb, aircraft_release_table[next])

            transition_reduced_cost = aircraft_reduced_cost_table[next]
            if master_is_feasible:
                transition_cost = next_time * aircraft_cost_table[next]
                transition_reduced_cost += transition_cost

            next_earliest_small_arrival = dp.max(
                earliest_small_arrival,
                next_time
                + type_separation_time[
                    aircraft_operation[next], aircraft_type[next], "Arrival", "Small"
                ],
            )
            next_earliest_large_arrival = dp.max(
                earliest_large_arrival,
                next_time
                + type_separation_time[
                    aircraft_operation[next], aircraft_type[next], "Arrival", "Large"
                ],
            )

            next_reachable = reachable.remove(next)

            preconditions = [
                arcs[current, next],
                reachable.contains(next),
            ]

            if self.disable_filter:
                preconditions.append(next_time_lb <= aircraft_due_table[next])
            else:
                filter_expression = (
                    dp.max(
                        next_time + aircraft_separation_time_table[next, future],
                        is_aircraft_small_arrival_table[future].if_then_else(
                            next_earliest_small_arrival,
                            is_aircraft_large_arrival_table[future].if_then_else(
                                next_earliest_large_arrival,
                                0,
                            ),
                        ),
                    )
                    <= aircraft_due_table[future]
                )
                next_reachable = next_reachable.filter(future, filter_expression)

            visit = dp.Transition(
                name=f"visit_{next}",
                cost=dp.FloatExpr.state_cost() + transition_reduced_cost,
                preconditions=preconditions,
                effects=[
                    (current, next),
                    (time, next_time),
                    (earliest_small_arrival, next_earliest_small_arrival),
                    (earliest_large_arrival, next_earliest_large_arrival),
                    (reachable, next_reachable),
                ],
            )
            model.add_transition(visit)
            all_transitions.append(visit)

        # Create a transition to indicate termination.
        done = dp.Transition(
            name="done",
            cost=dp.FloatExpr.state_cost(),
            preconditions=[arcs[current, num_aircraft]],
            effects=[
                (current, num_aircraft + 1),
            ],
        )
        model.add_transition(done)
        all_transitions.append(done)

        # Create the finishing state.
        model.add_base_case([current == num_aircraft + 1])

        # Add a dual bound comprising the sum of reachable nodes.
        if self.bound is not None:
            min_cost = [
                min(
                    aircraft_reduced_cost[next]
                    + (
                        aircraft_cost[next] * aircraft_release[next]
                        if master_is_feasible
                        else 0.0
                    ),
                    0,
                )
                for next in range(num_aircraft)
            ]

            if self.bound == "profit" or (
                self.bound == "reduce" and not master_is_feasible
            ):
                min_cost_table = model.add_float_table(min_cost)
                model.add_dual_bound(
                    (current == num_aircraft + 1).if_then_else(
                        0.0, min_cost_table[reachable]
                    )
                )
            elif self.bound == "knapsack":
                values = [-c for c in min_cost]
                sep = aircraft_separation_time[:num_aircraft, :num_aircraft].copy()
                int_max = np.iinfo(sep.dtype).max
                np.fill_diagonal(sep, int_max)
                assert (sep.min(axis=0) < int_max).all()
                weights = [int(x) for x in sep.min(axis=0).tolist()]
                max_due = max(aircraft_due)
                capacity = max_due - time
                model.add_dual_bound(
                    -dp.fractional_knapsack(reachable, capacity, values, weights)
                )
            elif self.bound == "reduce":
                aircraft_reduced_cost_table = model.add_float_table(
                    aircraft_reduced_cost
                )

                if no_cons and not self.disable_filter:
                    model.add_dual_bound(
                        (current == num_aircraft + 1).if_then_else(
                            0.0,
                            reachable.sum(
                                future,
                                aircraft_reduced_cost_table[future]
                                + aircraft_cost_table[future]
                                * dp.max(
                                    time
                                    + aircraft_separation_time_table[current, future],
                                    is_aircraft_small_arrival_table[
                                        future
                                    ].if_then_else(
                                        earliest_small_arrival,
                                        is_aircraft_large_arrival_table[
                                            future
                                        ].if_then_else(
                                            earliest_large_arrival,
                                            aircraft_release_table[future],
                                        ),
                                    ),
                                ),
                            ),
                        )
                    )
                else:
                    model.add_dual_bound(
                        (current == num_aircraft + 1).if_then_else(
                            0.0,
                            reachable.sum(
                                future,
                                dp.min(
                                    0.0,
                                    aircraft_reduced_cost_table[future]
                                    + aircraft_cost_table[future]
                                    * dp.max(
                                        time
                                        + aircraft_separation_time_table[
                                            current, future
                                        ],
                                        is_aircraft_small_arrival_table[
                                            future
                                        ].if_then_else(
                                            earliest_small_arrival,
                                            is_aircraft_large_arrival_table[
                                                future
                                            ].if_then_else(
                                                earliest_large_arrival,
                                                aircraft_release_table[future],
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        )
                    )

        # Create the solver.
        time_limit = max(self.master.time_remaining(), 0.0)
        # print(f'Time limit: {time_limit:.1f} seconds')
        if self.get_all_solutions and self.f_first:
            solver = self.solver_type(
                model=model,
                quiet=(not verbose),
                get_all_solutions=True,
                node_first=True,
                time_limit=time_limit,
                primal_bound=-EPS + runway_dual_sol,
            )
        elif self.get_all_solutions:
            solver = self.solver_type(
                model=model,
                quiet=(not verbose),
                get_all_solutions=True,
                time_limit=time_limit,
                primal_bound=-EPS + runway_dual_sol,
            )
        elif self.f_first:
            solver = self.solver_type(
                model=model,
                quiet=(not verbose),
                node_first=True,
                time_limit=time_limit,
                primal_bound=-EPS + runway_dual_sol,
            )
        else:
            solver = self.solver_type(
                model=model,
                quiet=(not verbose),
                time_limit=time_limit,
                primal_bound=-EPS + runway_dual_sol,
            )

        # Add columns to the master problem.
        is_terminated = False
        is_optimal = True
        ub = 0.0
        while not is_terminated:
            sol, is_terminated = solver.search_next()

            if is_terminated:
                self.pricing_stats["expanded"] += sol.expanded

            if sol.is_optimal and is_eq(sol.cost - runway_dual_sol, ub):
                break

            if sol.cost is not None:
                sol_reduced_cost = sol.cost - runway_dual_sol
                assert self.get_all_solutions or sol_reduced_cost < -EPS
                if (self.get_all_solutions and sol_reduced_cost < -EPS) or is_lt(
                    sol_reduced_cost, ub
                ):

                    # Get the solution.
                    aircraft_order = []
                    for transition in sol.transitions:
                        if transition.name == "done":
                            break
                        else:
                            next = int(re.match(r"visit_(\d+)", transition.name)[1])
                            aircraft_order.append(next)

                    # Create a schedule.
                    cost = 0
                    time = 0
                    current = num_aircraft
                    earliest_small_arrival = 0
                    earliest_large_arrival = 0
                    aircraft_time = []
                    for next in aircraft_order:
                        next_time = max(
                            time + aircraft_separation_time[current, next],
                            aircraft_release[next],
                        )
                        if (
                            aircraft_operation[next] == "Arrival"
                            and aircraft_type[next] == "Small"
                        ):
                            next_time = max(next_time, earliest_small_arrival)
                        elif (
                            aircraft_operation[next] == "Arrival"
                            and aircraft_type[next] == "Large"
                        ):
                            next_time = max(next_time, earliest_large_arrival)

                        transition_cost = next_time * aircraft_cost[next]
                        cost += transition_cost

                        transition_reduced_cost = aircraft_reduced_cost[next]
                        if master_is_feasible:
                            transition_reduced_cost += transition_cost

                        current = next
                        time = next_time
                        earliest_small_arrival = max(
                            earliest_small_arrival,
                            next_time
                            + type_separation_time[
                                aircraft_operation[next],
                                aircraft_type[next],
                                "Arrival",
                                "Small",
                            ],
                        )
                        earliest_large_arrival = max(
                            earliest_large_arrival,
                            next_time
                            + type_separation_time[
                                aircraft_operation[next],
                                aircraft_type[next],
                                "Arrival",
                                "Large",
                            ],
                        )

                        aircraft_time.append(next_time)

                    # Add a column.
                    # if verbose: print(solution.cost, solution_reduced_cost, included_items)
                    if add_var:
                        self.master.add_var(cost, aircraft_time, aircraft_order)

                    # Require future solutions to be improving.
                    ub = min(ub, sol_reduced_cost)

                    # Print.
                    if verbose:
                        print(f"    sol.is_optimal: {sol.is_optimal}")
                        print(f"    sol.expanded: {sol.expanded}")

                    # If the master problem is infeasible, only one column is sufficient.
                    if self.one_column_farkas and not master_is_feasible:
                        break

            elif sol.time_out:
                is_optimal = False

        return (ub, is_optimal)

    def pricerredcost(self):
        """
        Run the pricing algorithm on a feasible master problem
        """

        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        num_columns = len(self.master.schedules)
        try:
            # Run the pricer.
            aircraft_dual_sol = [
                self.master.model.getDualsolLinear(cons)
                for cons in self.master.aircraft_constraints
            ]
            runway_dual_sol = self.master.model.getDualsolLinear(
                self.master.runway_constraint
            )
            reduced_cost, is_optimal = self.price(
                True, runway_dual_sol, aircraft_dual_sol
            )

            # Determine if the pricer ran.
            result = scip.SCIP_RESULT.DIDNOTRUN
            if is_optimal or reduced_cost < 0.0:
                result = scip.SCIP_RESULT.SUCCESS

            # Compute a lower bound.
            master_lb = float("-inf")
            if is_optimal:
                num_runways = self.data["num_runways"]
                master_lb = self.master.model.getLPObjVal() + num_runways * reduced_cost

            return {"result": result, "lowerbound": master_lb}

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start
            self.pricing_stats["columns"] += len(self.master.schedules) - num_columns

    def pricerfarkas(self):
        """
        Run the pricing algorithm on an infeasible master problem
        """

        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        num_columns = len(self.master.schedules)
        try:
            # Run the pricer.
            aircraft_dual_sol = [
                self.master.model.getDualfarkasLinear(cons)
                for cons in self.master.aircraft_constraints
            ]
            runway_dual_sol = self.master.model.getDualfarkasLinear(
                self.master.runway_constraint
            )
            reduced_cost, is_optimal = self.price(
                False, runway_dual_sol, aircraft_dual_sol
            )

            # Determine if the pricer ran.
            result = scip.SCIP_RESULT.DIDNOTRUN
            if is_optimal or reduced_cost < 0.0:
                result = scip.SCIP_RESULT.SUCCESS

            return {"result": result}

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start
            self.pricing_stats["columns"] += len(self.master.schedules) - num_columns


class GurobiPricer(scip.Pricer):
    def __init__(
        self,
        data,
        solver,
        get_all_solutions,
        one_column_farkas,
        master,
        branching_conshdlr,
    ):
        """
        Create the pricing problem
        """

        # Store the instance data.
        self.data = data

        self.solver = solver

        # Store the solver type.
        self.get_all_solutions = get_all_solutions
        self.one_column_farkas = one_column_farkas

        # Store the other parts of the solver.
        self.master = master
        self.branching_conshdlr = branching_conshdlr

        # Attach the pricer.
        self.master.model.includePricer(
            self, "gurobi", "Pricer using Gurobi", delay=True
        )

        self.pricing_model = None
        self.pricing_stats = {"time": 0.0, "calls": 0, "columns": 0}

    def pricerinit(self):

        # Get the master problem constraints. TODO: This should be called somewhere else.
        self.master.transform_constraints()

    def time_remaining(self):
        if self.model.data["has_time_limit"]:
            time_limit = self.model.getParam("limits/time")
            elapsed_time = self.model.getSolvingTime()

            return time_limit - elapsed_time

        return None

    def price(self, master_is_feasible, runway_dual_sol, aircraft_dual_sol):
        """
        Run the pricing algorithm
        """

        # global iter
        # iter += 1
        # print(f'ITER {iter}')

        # Solve using DIDP.
        if self.solver == "mip":
            reduced_cost, is_optimal = self.solve_gurobi(
                master_is_feasible, runway_dual_sol, aircraft_dual_sol
            )
        else:
            reduced_cost, is_optimal = self.solve_gurobi_assignment(
                master_is_feasible, runway_dual_sol, aircraft_dual_sol
            )

        return reduced_cost, is_optimal

    def solve_gurobi(
        self, master_is_feasible, runway_dual_sol, aircraft_dual_sol, add_var=True
    ):
        import gurobipy as gp

        verbose = False

        # Get the instance data.
        num_aircraft = self.data["num_aircraft"]
        aircraft_type = self.data["aircraft_type"]
        aircraft_operation = self.data["operation"]
        aircraft_release = self.data["release"]
        aircraft_due = self.data["due"]
        aircraft_separation_time = self.data["aircraft_separation_time"]
        type_separation_time = self.data["separation_times"]
        aircraft_cost = self.data["weight"]

        # Build arc feasibility matrix (including dummy start node).
        arcs = np.ones((num_aircraft + 1, num_aircraft), dtype=np.bool_)
        for i in range(num_aircraft):
            for j in range(num_aircraft):
                if i == j:
                    arcs[i, j] = False
                else:
                    arcs[i, j] = (
                        aircraft_release[i] + aircraft_separation_time[i, j]
                        <= aircraft_due[j]
                    )
        arcs[num_aircraft, :] = True
        no_cons = True

        # Apply branching constraints.
        for cons in self.branching_conshdlr.conss:
            if cons.isActive():
                (i, j, direction) = cons.data
                if direction:
                    original_val = arcs[i, j]
                    arcs[:, j] = False
                    arcs[i, :] = False
                    arcs[i, j] = original_val
                else:
                    arcs[i, j] = False

                no_cons = False

        # Precompute separation times to future arrivals.
        sep_to_small = [
            type_separation_time[
                aircraft_operation[a], aircraft_type[a], "Arrival", "Small"
            ]
            for a in range(num_aircraft)
        ]
        sep_to_large = [
            type_separation_time[
                aircraft_operation[a], aircraft_type[a], "Arrival", "Large"
            ]
            for a in range(num_aircraft)
        ]

        max_due = max(aircraft_due) if num_aircraft > 0 else 0
        max_sep = 0
        if num_aircraft > 0:
            max_sep = max(
                int(np.max(aircraft_separation_time[:num_aircraft, :num_aircraft])),
                max(sep_to_small),
                max(sep_to_large),
            )
        big_m_time = max_due + max_sep

        # Create model.
        if self.pricing_model is None:
            model = gp.Model()
            model.Params.OutputFlag = 1 if verbose else 0
            model.Params.Threads = 1

            # Create variables.
            y = model.addVars(num_aircraft, vtype=gp.GRB.BINARY, name="y")
            t = model.addVars(num_aircraft, vtype=gp.GRB.INTEGER, lb=0, name="t")
            u = model.addVars(
                num_aircraft, vtype=gp.GRB.INTEGER, lb=0, ub=num_aircraft, name="u"
            )
            es = model.addVars(
                num_aircraft, vtype=gp.GRB.INTEGER, lb=0, ub=big_m_time, name="es"
            )
            el = model.addVars(
                num_aircraft, vtype=gp.GRB.INTEGER, lb=0, ub=big_m_time, name="el"
            )

            # Create arc variables only for feasible arcs.
            x = {}
            for i in range(num_aircraft + 1):
                for j in range(num_aircraft):
                    if arcs[i, j]:
                        x[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f"x[{i},{j}]")

            # Each scheduled aircraft has exactly one predecessor (possibly the dummy start).
            for j in range(num_aircraft):
                incoming = [x[i, j] for i in range(num_aircraft + 1) if (i, j) in x]
                if incoming:
                    model.addConstr(gp.quicksum(incoming) == y[j], name=f"pred[{j}]")
                else:
                    model.addConstr(y[j] == 0, name=f"pred[{j}]")

            # Each scheduled aircraft has at most one successor.
            for i in range(num_aircraft):
                outgoing = [x[i, j] for j in range(num_aircraft) if (i, j) in x]
                if outgoing:
                    model.addConstr(gp.quicksum(outgoing) <= y[i], name=f"succ[{i}]")

            # At most one aircraft can start the sequence.
            start_outgoing = [
                x[num_aircraft, j]
                for j in range(num_aircraft)
                if (num_aircraft, j) in x
            ]
            if start_outgoing:
                model.addConstr(gp.quicksum(start_outgoing) <= 1, name="start_outgoing")

            # MTZ constraints to prevent subtours.
            for j in range(num_aircraft):
                model.addConstr(u[j] <= num_aircraft * y[j], name=f"u_ub[{j}]")
            for j in range(num_aircraft):
                if (num_aircraft, j) in x:
                    model.addConstr(
                        u[j] >= 1 - num_aircraft * (1 - x[num_aircraft, j]),
                        name=f"u_start[{j}]",
                    )
            for i in range(num_aircraft):
                for j in range(num_aircraft):
                    if i != j and (i, j) in x:
                        model.addConstr(
                            u[j] >= u[i] + 1 - num_aircraft * (1 - x[i, j]),
                            name=f"u_mtz[{i},{j}]",
                        )

            # Time window constraints.
            for j in range(num_aircraft):
                model.addConstr(
                    t[j] >= aircraft_release[j] * y[j], name=f"release[{j}]"
                )
                model.addConstr(t[j] <= aircraft_due[j] * y[j], name=f"due[{j}]")

            # Separation constraints for consecutive aircraft.
            for i in range(num_aircraft):
                for j in range(num_aircraft):
                    if i != j and (i, j) in x:
                        model.addConstr(
                            t[j]
                            >= t[i]
                            + aircraft_separation_time[i, j]
                            - big_m_time * (1 - x[i, j]),
                            name=f"sep[{i},{j}]",
                        )

            # Arrival-specific constraints and resource updates.
            for i in range(num_aircraft + 1):
                for j in range(num_aircraft):
                    if (i, j) not in x:
                        continue
                    # Arrival lower bounds (dummy start has es/el = 0).
                    if aircraft_operation[j] == "Arrival":
                        if aircraft_type[j] == "Small":
                            if i == num_aircraft:
                                model.addConstr(
                                    t[j] >= -big_m_time * (1 - x[i, j]),
                                    name=f"arr_small_start[{j}]",
                                )
                            else:
                                model.addConstr(
                                    t[j] >= es[i] - big_m_time * (1 - x[i, j]),
                                    name=f"arr_small[{i},{j}]",
                                )
                        elif aircraft_type[j] == "Large":
                            if i == num_aircraft:
                                model.addConstr(
                                    t[j] >= -big_m_time * (1 - x[i, j]),
                                    name=f"arr_large_start[{j}]",
                                )
                            else:
                                model.addConstr(
                                    t[j] >= el[i] - big_m_time * (1 - x[i, j]),
                                    name=f"arr_large[{i},{j}]",
                                )

                    # Update earliest arrival resources after scheduling j.
                    if i == num_aircraft:
                        model.addConstr(
                            es[j]
                            >= t[j] + sep_to_small[j] - big_m_time * (1 - x[i, j]),
                            name=f"es_start[{j}]",
                        )
                        model.addConstr(
                            el[j]
                            >= t[j] + sep_to_large[j] - big_m_time * (1 - x[i, j]),
                            name=f"el_start[{j}]",
                        )
                    else:
                        model.addConstr(
                            es[j] >= es[i] - big_m_time * (1 - x[i, j]),
                            name=f"es_prop[{i},{j}]",
                        )
                        model.addConstr(
                            es[j]
                            >= t[j] + sep_to_small[j] - big_m_time * (1 - x[i, j]),
                            name=f"es_update[{i},{j}]",
                        )
                        model.addConstr(
                            el[j] >= el[i] - big_m_time * (1 - x[i, j]),
                            name=f"el_prop[{i},{j}]",
                        )
                        model.addConstr(
                            el[j]
                            >= t[j] + sep_to_large[j] - big_m_time * (1 - x[i, j]),
                            name=f"el_update[{i},{j}]",
                        )

            # Force resources to 0 when aircraft is not scheduled (helps numerics).
            for j in range(num_aircraft):
                model.addConstr(es[j] <= big_m_time * y[j], name=f"es_ub[{j}]")
                model.addConstr(el[j] <= big_m_time * y[j], name=f"el_ub[{j}]")

            # Objective: reduced cost (including runway dual as a constant shift).
            if master_is_feasible:
                obj = gp.quicksum(
                    aircraft_cost[j] * t[j] - aircraft_dual_sol[j] * y[j]
                    for j in range(num_aircraft)
                )
            else:
                obj = gp.quicksum(
                    -aircraft_dual_sol[j] * y[j] for j in range(num_aircraft)
                )
            model.setObjective(obj, gp.GRB.MINIMIZE)
        else:
            model = self.pricing_model
            y = []

            for j in range(num_aircraft):
                y_j = model.getVarByName(f"y[{j}]")
                y_j.Obj = -aircraft_dual_sol[j]
                y.append(y_j)

                t_j = model.getVarByName(f"t[{j}]")

                if master_is_feasible:
                    t_j.Obj = aircraft_cost[j]
                else:
                    t_j.Obj = 0.0

            x = {}

            for i in range(num_aircraft + 1):
                for j in range(num_aircraft):
                    x_ij = model.getVarByName(f"x[{i},{j}]")

                    if x_ij is not None:
                        if arcs[i, j]:
                            x_ij.UB = 1
                            x[i, j] = x_ij
                        else:
                            x_ij.UB = 0

        if no_cons:
            for j in range(num_aircraft):
                if (
                    -aircraft_dual_sol[j]
                    + (
                        aircraft_cost[j] * aircraft_release[j]
                        if master_is_feasible
                        else 0.0
                    )
                    >= 0.0
                ):
                    y[j].UB = 0

                    for i in range(num_aircraft + 1):
                        if arcs[i, j]:
                            x[i, j].UB = 0

                    for i in range(num_aircraft):
                        if arcs[j, i]:
                            x[j, i].UB = 0

        # Set time limit.
        time_limit = max(self.master.time_remaining(), 0.01)
        if time_limit is not None:
            model.Params.TimeLimit = time_limit

        # Solve.
        model.Params.Cutoff = -EPS + runway_dual_sol
        model.optimize()
        status = model.Status

        # Exit if no solution found.
        is_optimal = (
            status == gp.GRB.OPTIMAL
            or status == gp.GRB.INFEASIBLE
            or status == gp.GRB.CUTOFF
        )
        ub = 0.0
        if model.SolCount == 0:
            return (ub, is_optimal)

        # Get a solution.
        for i in range(model.SolCount):
            model.params.SolutionNumber = i
            order = self.extract_order(x, num_aircraft)
            cost, aircraft_time = self.build_schedule(
                order,
                num_aircraft,
                aircraft_separation_time,
                aircraft_release,
                aircraft_operation,
                aircraft_type,
                aircraft_cost,
                type_separation_time,
            )
            sol_reduced_cost = model.PoolObjVal - runway_dual_sol
            ub = min(sol_reduced_cost, ub)
            if add_var:
                self.master.add_var(cost, aircraft_time, order)

        return (ub, is_optimal)

    def extract_order(self, x, num_aircraft):
        succ = {}
        for (i, j), var in x.items():
            if var.Xn > 0.5:
                succ[i] = j
        if num_aircraft not in succ:
            return []
        order = []
        current = num_aircraft
        visited = set()
        while current in succ:
            next = succ[current]
            if next in visited:
                break
            visited.add(next)
            order.append(next)
            current = next
        return order

    def build_schedule(
        self,
        order,
        num_aircraft,
        aircraft_separation_time,
        aircraft_release,
        aircraft_operation,
        aircraft_type,
        aircraft_cost,
        type_separation_time,
    ):
        cost = 0.0
        time_now = 0
        current = num_aircraft
        earliest_small_arrival = 0
        earliest_large_arrival = 0
        aircraft_time = []
        for next in order:
            next_time = max(
                time_now + aircraft_separation_time[current, next],
                aircraft_release[next],
            )
            if aircraft_operation[next] == "Arrival" and aircraft_type[next] == "Small":
                next_time = max(next_time, earliest_small_arrival)
            elif (
                aircraft_operation[next] == "Arrival" and aircraft_type[next] == "Large"
            ):
                next_time = max(next_time, earliest_large_arrival)

            transition_cost = next_time * aircraft_cost[next]
            cost += transition_cost

            earliest_small_arrival = max(
                earliest_small_arrival,
                next_time
                + type_separation_time[
                    aircraft_operation[next],
                    aircraft_type[next],
                    "Arrival",
                    "Small",
                ],
            )
            earliest_large_arrival = max(
                earliest_large_arrival,
                next_time
                + type_separation_time[
                    aircraft_operation[next],
                    aircraft_type[next],
                    "Arrival",
                    "Large",
                ],
            )

            time_now = next_time
            current = next
            aircraft_time.append(next_time)

        return cost, aircraft_time

    def solve_gurobi_assignment(
        self, master_is_feasible, runway_dual_sol, aircraft_dual_sol, add_var=True
    ):
        import gurobipy as gp

        verbose = False

        # Get the instance data.
        num_aircraft = self.data["num_aircraft"]
        aircraft_type = self.data["aircraft_type"]
        aircraft_operation = self.data["operation"]
        aircraft_release = self.data["release"]
        aircraft_due = self.data["due"]
        type_separation_time = self.data["separation_times"]
        aircraft_separation_time = self.data["aircraft_separation_time"]
        aircraft_cost = self.data["weight"]

        # Apply branching constraints.
        enforced_arcs = set()
        forbidden_arcs = set()
        no_cons = True

        # Apply branching constraints.
        for cons in self.branching_conshdlr.conss:
            if cons.isActive():
                (i, j, direction) = cons.data
                if direction:
                    enforced_arcs.add((i, j))
                else:
                    forbidden_arcs.add((i, j))

                no_cons = False

        if self.pricing_model is None:
            model = gp.Model()
            model.Params.OutputFlag = 1 if verbose else 0
            model.Params.Threads = 1

            y = model.addVars(
                [
                    (i, j)
                    for i in range(num_aircraft)
                    for j in range(num_aircraft)
                    if i != j
                ],
                vtype=gp.GRB.BINARY,
                name="y",
            )
            z = model.addVars(
                num_aircraft,
                vtype=gp.GRB.BINARY,
                obj=[-pi for pi in aircraft_dual_sol],
                name="z",
            )
            t = model.addVars(
                num_aircraft,
                vtype=gp.GRB.INTEGER,
                lb=0,
                obj=[c if master_is_feasible else 0 for c in aircraft_cost],
                name="t",
            )

            # Relase time
            model.addConstrs(
                (t[j] >= aircraft_release[j] * z[j] for j in range(num_aircraft)),
                name="release",
            )
            # Due time
            model.addConstrs(
                (t[j] <= aircraft_due[j] * z[j] for j in range(num_aircraft)),
                name="due",
            )

            # Separation time
            max_due = max(aircraft_due)
            model.addConstrs(
                (
                    t[j]
                    >= t[i]
                    + aircraft_separation_time[i, j] * y[i, j]
                    - (1 - y[i, j]) * max_due
                    for i in range(num_aircraft)
                    for j in range(num_aircraft)
                    if i != j
                ),
                name="separation",
            )

            # Precedence
            model.addConstrs(
                (
                    y[i, j] + y[j, i] >= z[i] + z[j] - 1
                    for i in range(num_aircraft)
                    for j in range(num_aircraft)
                    if i != j
                ),
                name="precedence_lb",
            )
            model.addConstrs(
                (
                    y[i, j] + y[j, i] <= z[i]
                    for i in range(num_aircraft)
                    for j in range(num_aircraft)
                    if i != j
                ),
                name="precedence_ub",
            )

            for i in range(num_aircraft):
                for j in range(num_aircraft):
                    if (
                        i != j
                        and aircraft_release[i] + aircraft_separation_time[i, j]
                        > aircraft_due[j]
                    ):
                        y[i, j].UB = 0

                    if (
                        i != j
                        and aircraft_release[i] < aircraft_release[j]
                        and aircraft_due[i] <= aircraft_due[j]
                        and aircraft_operation[i] == aircraft_operation[j]
                        and aircraft_type[i] == aircraft_type[j]
                    ):
                        y[j, i].UB = 0

            self.pricing_model = model
        else:
            model = self.pricing_model
            z = []
            t = []

            for j in range(num_aircraft):
                z_j = model.getVarByName(f"z[{j}]")
                z_j.Obj = -aircraft_dual_sol[j]
                z.append(z_j)

                t_j = model.getVarByName(f"t[{j}]")
                if master_is_feasible:
                    t_j.Obj = aircraft_cost[j]
                else:
                    t_j.Obj = 0.0
                t.append(t_j)

            y = {}

            for i in range(num_aircraft):
                for j in range(num_aircraft):
                    y_ij = model.getVarByName(f"y[{i},{j}]")

                    if y_ij is not None:
                        y[i, j] = y_ij

        for j in range(num_aircraft):
            if no_cons and (
                -aircraft_dual_sol[j]
                + (
                    aircraft_cost[j] * aircraft_release[j]
                    if master_is_feasible
                    else 0.0
                )
                >= 0.0
            ):
                z[j].UB = 0
            else:
                z[j].UB = 1

        # Apply branching constraints.
        if not no_cons:
            for i in range(num_aircraft):
                forbid_start_cons = model.getConstrByName(f"forbid_start[{i}]")

                if (
                    num_aircraft,
                    i,
                ) in forbidden_arcs and forbid_start_cons is None:
                    model.addConstr(
                        gp.quicksum(y[j, i] for j in range(num_aircraft) if j != i)
                        >= z[i],
                        name=f"forbid_start[{i}]",
                    )

                if (
                    num_aircraft,
                    i,
                ) not in forbidden_arcs and forbid_start_cons is not None:
                    model.remove(forbid_start_cons)

                forbit_end_cons = model.getConstrByName(f"forbid_end[{i}]")

                if (
                    i,
                    num_aircraft,
                ) in forbidden_arcs and forbit_end_cons is None:
                    model.addConstr(
                        gp.quicksum(y[i, j] for j in range(num_aircraft) if j != i)
                        >= z[i],
                        name=f"forbid_end[{i}]",
                    )

                if (
                    i,
                    num_aircraft,
                ) not in forbidden_arcs and forbit_end_cons is not None:
                    model.remove(forbit_end_cons)

                for j in range(num_aircraft):
                    if (num_aircraft, i) in enforced_arcs:
                        y[j, i].UB = 0
                    else:
                        y[j, i].UB = 1

                    if (i, num_aircraft) in enforced_arcs:
                        y[i, j].UB = 0
                    else:
                        y[i, j].UB = 1

                    enforce_cons = model.getConstrByName(f"enforce_1[{i},{j}]")

                    if (i, j) in enforced_arcs and enforce_cons is None:
                        y[j, i].UB = 0
                        model.addConstr(y[i, j] >= z[i], name=f"enforce_1[{i},{j}]")
                        model.addConstr(y[i, j] >= z[j], name=f"enforce_2[{i},{j}]")
                        model.addConstr(
                            gp.quicksum(y[k, i] for k in range(num_aircraft) if k != i)
                            + 1
                            >= gp.quicksum(
                                y[k, j] for k in range(num_aircraft) if k != j
                            )
                            - (sum(1 for k in range(num_aircraft) if k != i) - 1)
                            * (1 - y[i, j]),
                            name=f"enforce_3[{i},{j}]",
                        )

                    if (i, j) not in enforced_arcs and enforce_cons is not None:
                        y[j, i].UB = 1
                        model.remove(enforce_cons)
                        model.remove(model.getConstrByName(f"enforce_2[{i},{j}]"))
                        model.remove(model.getConstrByName(f"enforce_3[{i},{j}]"))

                    forbid_cons = model.getConstrByName(f"forbid[{i},{j}]")

                    if (i, j) in forbidden_arcs and forbid_cons is None:
                        model.addConstr(
                            gp.quicksum(y[k, i] for k in range(num_aircraft) if k != i)
                            + 2
                            <= gp.quicksum(
                                y[k, j] for k in range(num_aircraft) if k != j
                            )
                            + (sum(1 for k in range(num_aircraft) if k != i) + 2)
                            * (1 - y[i, j]),
                            name=f"forbid[{i},{j}]",
                        )

                    if (i, j) not in forbidden_arcs and forbid_cons is not None:
                        model.remove(forbid_cons)

        # Set time limit.
        time_limit = max(self.master.time_remaining(), 0.01)
        if time_limit is not None:
            model.Params.TimeLimit = time_limit

        # Solve.
        model.Params.Cutoff = -EPS + runway_dual_sol
        model.optimize()
        status = model.Status

        # Exit if no solution found.
        is_optimal = (
            status == gp.GRB.OPTIMAL
            or status == gp.GRB.INFEASIBLE
            or status == gp.GRB.CUTOFF
        )
        ub = 0.0
        if model.SolCount == 0:
            return (ub, is_optimal)

        # Get a solution.
        for i in range(model.SolCount):
            model.params.SolutionNumber = i

            # Extract order from solution.
            order = []

            for i in range(num_aircraft):
                if z[i].Xn >= 0.5:
                    order.append(i)

            order.sort(key=lambda a: t[a].Xn)

            cost, aircraft_time = self.build_schedule(
                order,
                num_aircraft,
                aircraft_separation_time,
                aircraft_release,
                aircraft_operation,
                aircraft_type,
                aircraft_cost,
                type_separation_time,
            )
            sol_reduced_cost = model.PoolObjVal - runway_dual_sol
            ub = min(sol_reduced_cost, ub)
            if add_var:
                self.master.add_var(cost, aircraft_time, order)

        return (ub, is_optimal)

    def pricerredcost(self):
        """
        Run the pricing algorithm on a feasible master problem
        """

        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        num_columns = len(self.master.schedules)
        try:
            # Run the pricer.
            aircraft_dual_sol = [
                self.master.model.getDualsolLinear(cons)
                for cons in self.master.aircraft_constraints
            ]
            runway_dual_sol = self.master.model.getDualsolLinear(
                self.master.runway_constraint
            )
            reduced_cost, is_optimal = self.price(
                True, runway_dual_sol, aircraft_dual_sol
            )

            # Determine if the pricer ran.
            result = scip.SCIP_RESULT.DIDNOTRUN
            if is_optimal or reduced_cost < 0.0:
                result = scip.SCIP_RESULT.SUCCESS

            # Compute a lower bound.
            master_lb = float("-inf")
            if is_optimal:
                num_runways = self.data["num_runways"]
                master_lb = self.master.model.getLPObjVal() + num_runways * reduced_cost

            return {"result": result, "lowerbound": master_lb}

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start
            self.pricing_stats["columns"] += len(self.master.schedules) - num_columns

    def pricerfarkas(self):
        """
        Run the pricing algorithm on an infeasible master problem
        """

        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        num_columns = len(self.master.schedules)
        try:
            # Run the pricer.
            aircraft_dual_sol = [
                self.master.model.getDualfarkasLinear(cons)
                for cons in self.master.aircraft_constraints
            ]
            runway_dual_sol = self.master.model.getDualfarkasLinear(
                self.master.runway_constraint
            )
            reduced_cost, is_optimal = self.price(
                False, runway_dual_sol, aircraft_dual_sol
            )

            # Determine if the pricer ran.
            result = scip.SCIP_RESULT.DIDNOTRUN
            if is_optimal or reduced_cost < 0.0:
                result = scip.SCIP_RESULT.SUCCESS

            return {"result": result}

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start
            self.pricing_stats["columns"] += len(self.master.schedules) - num_columns


class CpPricer(scip.Pricer):
    def __init__(
        self,
        data,
        solver,
        get_all_solutions,
        one_column_farkas,
        master,
        branching_conshdlr,
    ):
        """
        Create the pricing problem
        """

        # Store the instance data.
        self.data = data

        self.solver = solver

        # Store the solver type.
        self.get_all_solutions = get_all_solutions
        self.one_column_farkas = one_column_farkas

        # Store the other parts of the solver.
        self.master = master
        self.branching_conshdlr = branching_conshdlr

        # Attach the pricer.
        self.master.model.includePricer(self, "cp", "Pricer using CP", delay=True)

        self.pricing_model = None
        self.pricing_stats = {"time": 0.0, "calls": 0, "columns": 0}

    def pricerinit(self):

        # Get the master problem constraints. TODO: This should be called somewhere else.
        self.master.transform_constraints()

    def time_remaining(self):
        if self.model.data["has_time_limit"]:
            time_limit = self.model.getParam("limits/time")
            elapsed_time = self.model.getSolvingTime()

            return time_limit - elapsed_time

        return None

    def price(self, master_is_feasible, runway_dual_sol, aircraft_dual_sol):
        """
        Run the pricing algorithm
        """

        # global iter
        # iter += 1
        # print(f'ITER {iter}')

        reduced_cost, is_optimal = self.solve_cp(
            master_is_feasible, runway_dual_sol, aircraft_dual_sol
        )

        return reduced_cost, is_optimal

    def solve_cp(
        self, master_is_feasible, runway_dual_sol, aircraft_dual_sol, add_var=True
    ):
        import ortools.sat.python.cp_model as cp

        class SolutionCallback(cp.CpSolverSolutionCallback):
            def __init__(self, z, t):
                super().__init__()
                self.z = z
                self.t = t
                self.orders = []

            def on_solution_callback(self):
                order = [j for j, z_j in enumerate(self.z) if self.Value(z_j)]
                order.sort(key=lambda a: self.Value(self.t[a]))
                self.orders.append(order)

        verbose = False

        # Get the instance data.
        num_aircraft = self.data["num_aircraft"]
        aircraft_type = self.data["aircraft_type"]
        aircraft_operation = self.data["operation"]
        aircraft_release = self.data["release"]
        aircraft_due = self.data["due"]
        type_separation_time = self.data["separation_times"]
        aircraft_separation_time = self.data["aircraft_separation_time"]
        aircraft_cost = self.data["weight"]

        # Apply branching constraints.
        enforced_arcs = set()
        forbidden_arcs = set()
        no_cons = True

        # Apply branching constraints.
        for cons in self.branching_conshdlr.conss:
            if cons.isActive():
                (i, j, direction) = cons.data
                if direction:
                    enforced_arcs.add((i, j))
                else:
                    forbidden_arcs.add((i, j))

                no_cons = False

        model = cp.CpModel()

        y = {
            (i, j): model.new_bool_var(f"y[{i},{j}]")
            for i in range(num_aircraft)
            for j in range(num_aircraft)
            if i != j
        }
        z = [model.new_bool_var(f"z[{j}]") for j in range(num_aircraft)]
        t = [
            model.new_int_var(0, aircraft_due[j], f"t[{j}]")
            for j in range(num_aircraft)
        ]

        model.minimize(
            sum(
                (aircraft_cost[j] * t[j] if master_is_feasible else 0.0)
                - aircraft_dual_sol[j] * z[j]
                for j in range(num_aircraft)
            )
        )

        for j in range(num_aircraft):
            model.add(t[j] >= aircraft_release[j]).only_enforce_if(z[j])

        for i in range(num_aircraft):
            for j in range(num_aircraft):
                if i != j:
                    model.add(
                        t[j] >= t[i] + aircraft_separation_time[i, j]
                    ).only_enforce_if(y[i, j])
                    model.add_at_most_one(y[i, j], y[j, i])
                    model.add_at_least_one(y[i, j], y[j, i]).only_enforce_if(
                        [z[i], z[j]]
                    )
                    model.add_implication(y[i, j], z[i])

        for i in range(num_aircraft):
            for j in range(num_aircraft):
                if (
                    i != j
                    and aircraft_release[i] + aircraft_separation_time[i, j]
                    > aircraft_due[j]
                ):
                    model.add(y[i, j] == 0)

                if (
                    i != j
                    and aircraft_release[i] < aircraft_release[j]
                    and aircraft_due[i] <= aircraft_due[j]
                    and aircraft_operation[i] == aircraft_operation[j]
                    and aircraft_type[i] == aircraft_type[j]
                ):
                    model.add(y[j, i] == 0)

        for j in range(num_aircraft):
            if no_cons and (
                -aircraft_dual_sol[j]
                + (
                    aircraft_cost[j] * aircraft_release[j]
                    if master_is_feasible
                    else 0.0
                )
                >= 0.0
            ):
                model.add(z[j] == 0)

        # Apply branching constraints.
        if not no_cons:
            for i in range(num_aircraft):
                if (num_aircraft, i) in forbidden_arcs:
                    model.add_at_least_one(
                        y[j, i] for j in range(num_aircraft) if j != i
                    ).only_enforce_if(z[i])

                if (i, num_aircraft) in forbidden_arcs:
                    model.add_at_least_one(
                        y[i, j] for j in range(num_aircraft) if j != i
                    ).only_enforce_if(z[i])

                for j in range(num_aircraft):
                    if (num_aircraft, i) in enforced_arcs:
                        model.add(y[j, i] == 0)

                    if (i, num_aircraft) in enforced_arcs:
                        model.add(y[i, j] == 0)

                    if (i, j) in enforced_arcs:
                        model.add(y[j, i] == 0)
                        model.add_implication(z[i], y[i, j])
                        model.add_implication(z[j], y[i, j])
                        model.add(
                            sum(y[k, i] for k in range(num_aircraft) if k != i) + 1
                            == sum(y[k, j] for k in range(num_aircraft) if k != j)
                        ).only_enforce_if(y[i, j])

                    if (i, j) in forbidden_arcs:
                        model.add(
                            sum(y[k, i] for k in range(num_aircraft) if k != i) + 2
                            <= sum(y[k, j] for k in range(num_aircraft) if k != j)
                        ).only_enforce_if(y[i, j])

        solver = cp.CpSolver()
        solver.parameters.num_search_workers = 1

        if verbose:
            solver.parameters.log_search_progress = True

        time_limit = max(self.master.time_remaining(), 0.01)
        if time_limit is not None:
            solver.parameters.max_time_in_seconds = time_limit

        callback = SolutionCallback(z, t)
        status = solver.solve(model, callback)

        # Exit if no solution found.
        is_optimal = status == cp.OPTIMAL or status == cp.INFEASIBLE
        ub = 0.0
        if len(callback.orders) == 0:
            return (ub, is_optimal)

        # Get a solution.
        for order in callback.orders:
            cost, aircraft_time = self.build_schedule(
                order,
                num_aircraft,
                aircraft_separation_time,
                aircraft_release,
                aircraft_operation,
                aircraft_type,
                aircraft_cost,
                type_separation_time,
            )
            sol_reduced_cost = (
                -sum(aircraft_dual_sol[j] for j in order) - runway_dual_sol
            )

            if master_is_feasible:
                sol_reduced_cost += cost

            ub = min(sol_reduced_cost, ub)
            if add_var and sol_reduced_cost < -EPS:
                self.master.add_var(cost, aircraft_time, order)

        return (ub, is_optimal)

    def build_schedule(
        self,
        order,
        num_aircraft,
        aircraft_separation_time,
        aircraft_release,
        aircraft_operation,
        aircraft_type,
        aircraft_cost,
        type_separation_time,
    ):
        cost = 0.0
        time_now = 0
        current = num_aircraft
        earliest_small_arrival = 0
        earliest_large_arrival = 0
        aircraft_time = []
        for next in order:
            next_time = max(
                time_now + aircraft_separation_time[current, next],
                aircraft_release[next],
            )
            if aircraft_operation[next] == "Arrival" and aircraft_type[next] == "Small":
                next_time = max(next_time, earliest_small_arrival)
            elif (
                aircraft_operation[next] == "Arrival" and aircraft_type[next] == "Large"
            ):
                next_time = max(next_time, earliest_large_arrival)

            transition_cost = next_time * aircraft_cost[next]
            cost += transition_cost

            earliest_small_arrival = max(
                earliest_small_arrival,
                next_time
                + type_separation_time[
                    aircraft_operation[next],
                    aircraft_type[next],
                    "Arrival",
                    "Small",
                ],
            )
            earliest_large_arrival = max(
                earliest_large_arrival,
                next_time
                + type_separation_time[
                    aircraft_operation[next],
                    aircraft_type[next],
                    "Arrival",
                    "Large",
                ],
            )

            time_now = next_time
            current = next
            aircraft_time.append(next_time)

        return cost, aircraft_time

    def pricerredcost(self):
        """
        Run the pricing algorithm on a feasible master problem
        """

        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        num_columns = len(self.master.schedules)
        try:
            # Run the pricer.
            aircraft_dual_sol = [
                self.master.model.getDualsolLinear(cons)
                for cons in self.master.aircraft_constraints
            ]
            runway_dual_sol = self.master.model.getDualsolLinear(
                self.master.runway_constraint
            )
            reduced_cost, is_optimal = self.price(
                True, runway_dual_sol, aircraft_dual_sol
            )

            # Determine if the pricer ran.
            result = scip.SCIP_RESULT.DIDNOTRUN
            if is_optimal or reduced_cost < 0.0:
                result = scip.SCIP_RESULT.SUCCESS

            # Compute a lower bound.
            master_lb = float("-inf")
            if is_optimal:
                num_runways = self.data["num_runways"]
                master_lb = self.master.model.getLPObjVal() + num_runways * reduced_cost

            return {"result": result, "lowerbound": master_lb}

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start
            self.pricing_stats["columns"] += len(self.master.schedules) - num_columns

    def pricerfarkas(self):
        """
        Run the pricing algorithm on an infeasible master problem
        """

        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        num_columns = len(self.master.schedules)
        try:
            # Run the pricer.
            aircraft_dual_sol = [
                self.master.model.getDualfarkasLinear(cons)
                for cons in self.master.aircraft_constraints
            ]
            runway_dual_sol = self.master.model.getDualfarkasLinear(
                self.master.runway_constraint
            )
            reduced_cost, is_optimal = self.price(
                False, runway_dual_sol, aircraft_dual_sol
            )

            # Determine if the pricer ran.
            result = scip.SCIP_RESULT.DIDNOTRUN
            if is_optimal or reduced_cost < 0.0:
                result = scip.SCIP_RESULT.SUCCESS

            return {"result": result}

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start
            self.pricing_stats["columns"] += len(self.master.schedules) - num_columns


class ArcBranchingRule(scip.Branchrule):
    """
    Arc branching that branches on an immediate successor
    """

    def __init__(self, data, master, conshdlr):
        """
        Create the branching rule
        """

        # Store the instance data.
        self.data = data

        # Store the other parts of the solver.
        self.master = master
        self.conshdlr = conshdlr

        # Attach the branching rule.
        self.master.model.includeBranchrule(
            self, "arc", "Arc branching", priority=1000000, maxdepth=-1, maxbounddist=1
        )

    def branchexeclp(self, _allowaddcons):
        """
        Run the branching rule on a feasible LP relaxation
        """

        # Get the instance data.
        num_aircraft = self.data["num_aircraft"]

        # Get the columns with fractional value.
        branch_vars, *_ = self.master.model.getLPBranchCands()

        # Compute the value of each arc.
        arc_vals = np.zeros((num_aircraft + 1, num_aircraft + 1))
        for var in branch_vars:
            var_val = self.master.model.getVal(var)
            aircraft_time, aircraft_order, _ = self.master.schedules[var.name]
            arc_vals[num_aircraft, aircraft_order[0]] += var_val

            for i, j in zip(aircraft_order[:-1], aircraft_order[1:]):
                arc_vals[i, j] += var_val

            arc_vals[aircraft_order[-1], num_aircraft] += var_val

        # Choose an arc to branch on.
        best_val = 1.0
        best_i = None
        best_j = None
        for i in range(num_aircraft + 1):
            for j in range(num_aircraft + 1):
                val = abs(arc_vals[i, j] - 0.5)
                if val < best_val:
                    best_val = val
                    best_i = i
                    best_j = j
        assert best_i is not None

        # Create the left child with a constraint requiring the pairing.
        left_child = self.master.model.createChild(
            0, self.master.model.getLocalEstimate()
        )
        left_cons = self.conshdlr.create_constraint(best_i, best_j, True)
        self.master.model.addConsNode(left_child, left_cons)

        # Create the right child with a constraint preventing the pairing.
        right_child = self.master.model.createChild(
            0, self.master.model.getLocalEstimate()
        )
        right_cons = self.conshdlr.create_constraint(best_i, best_j, False)
        self.master.model.addConsNode(right_child, right_cons)

        return {"result": scip.SCIP_RESULT.BRANCHED}

    def branchexecps(self, _allowaddcons):
        """
        Run the branching rule when a feasible LP relaxation is not available
        """

        return {"result": scip.SCIP_RESULT.DIDNOTRUN}


class ArcBranchingConshdlr(scip.Conshdlr):
    """
    Constraint handler for implementing the arc branching decisions
    """

    def __init__(self, master):
        """
        Create the constraint handler
        """

        # Store the other parts of the solver.
        self.master = master

        # Attach the constraint handler.
        self.master.model.includeConshdlr(
            self,
            "arc",
            "Constraint handler implementing arc branching decisions",
            enfopriority=0,
            chckpriority=9999999,
            propfreq=1,
            eagerfreq=1,
        )

        # Prepare somewhere to store the constraints.
        self.conss = []

    def create_constraint(self, i, j, direction):
        """
        Create one constraint implementing a decision
        """

        # Create a constraint with the associated data.
        name = f"arc[{i},{j},{direction}]"
        cons = self.master.model.createCons(
            conshdlr=self,
            name=name,
            initial=False,
            separate=False,
            enforce=False,
            check=False,
            propagate=True,
            local=True,
            modifiable=False,
            dynamic=False,
            removable=False,
            stickingatnode=True,
        )
        cons.data = (i, j, direction)
        self.conss.append(cons)

        return cons

    def consprop(self, conss, _nusefulconss, _namrkedconss, _proptiming):
        """
        Remove variables in the master problem that violate a decision
        """

        # Enforce the branching decision of every constraint.
        result = scip.SCIP_RESULT.DIDNOTFIND
        for cons in conss:
            assert cons.isActive()

            # Get the decision.
            i, j, direction = cons.data

            # Disable variables incompatible with the branching decision.
            num_aircraft = self.master.data["num_aircraft"]

            for _, aircraft_order, var in self.master.schedules.values():
                for a1, a2 in zip(
                    [num_aircraft] + aircraft_order,
                    aircraft_order + [num_aircraft],
                ):
                    if (a1 == i and (a2 == j) != direction) or (
                        a2 == j and (a1 == i) != direction
                    ):
                        self.model.fixVar(var, 0)
                        result = scip.SCIP_RESULT.REDUCEDDOM
                        break

        return {"result": result}


class BranchAndPriceSolver:
    """
    Branch-and-price solver
    """

    def __init__(
        self,
        data,
        solver,
        get_all_solutions=False,
        one_column_farkas=False,
        bound="knapsack",
        time_limit=None,
        disable_set_resource=False,
        disable_filter=False,
        f_first=False,
    ):
        """
        Create the components of the branch-and-price solver
        """

        self.master = Master(data, time_limit)
        self.branching_conshdlr = ArcBranchingConshdlr(self.master)
        if solver == "mip" or solver == "mip-assignment":
            self.pricer = GurobiPricer(
                data,
                solver,
                get_all_solutions,
                one_column_farkas,
                self.master,
                self.branching_conshdlr,
            )
        elif solver == "cp":
            self.pricer = CpPricer(
                data,
                solver,
                get_all_solutions,
                one_column_farkas,
                self.master,
                self.branching_conshdlr,
            )
        else:
            self.pricer = DIDPPricer(
                data,
                solver,
                get_all_solutions,
                one_column_farkas,
                bound,
                self.master,
                self.branching_conshdlr,
                disable_set_resource,
                disable_filter,
                f_first,
            )
        self.branching_rule = ArcBranchingRule(
            data, self.master, self.branching_conshdlr
        )

    def solve(self):
        """
        Solve
        """

        # Solve.
        self.master.model.optimize()

        # Get the solution.
        if self.master.model.getNSols() > 0:
            print("")
            cost = self.master.model.getObjVal()
            print(f"Cost {cost}")
            runway = 0
            for aircraft_time, aircraft_order, var in self.master.schedules.values():
                if is_gt(self.master.model.getVal(var), 0.0):
                    print(f"Runway {runway}:")
                    for t, a in zip(aircraft_time, aircraft_order):
                        print(f"    Time {t}, aircraft {a}")

                    runway += 1
            print("")

        if hasattr(self.pricer, "pricing_stats"):
            pricing_stats = self.pricer.pricing_stats
            print(f"Pricing time {pricing_stats['time']}s")
            print(f"Pricing calls {pricing_stats['calls']}")
            print(f"Pricing columns {pricing_stats['columns']}")

            if "expanded" in pricing_stats:
                print(f"Pricing expanded {pricing_stats['expanded']}")

            print("")


def solve_branch_and_price(
    data,
    solver,
    get_all_solutions=False,
    one_column_farkas=False,
    bound="knapsack",
    time_limit=None,
    disable_set_resource=False,
    disable_filter=False,
    f_first=False,
):

    solver = BranchAndPriceSolver(
        data,
        solver,
        get_all_solutions,
        one_column_farkas,
        bound,
        time_limit,
        disable_set_resource,
        disable_filter,
        f_first,
    )
    solver.solve()


def solve_scip(data, solver, time_limit=None):

    verbose = False

    assert solver in ["scip", "gcg"], f'Invalid solver "{solver}"'
    if solver == "scip":
        from pyscipopt import quicksum

        solver = scip
    elif solver == "gcg":
        import pygcgopt as gcg
        from pygcgopt import quicksum

        solver = gcg

    # Get the instance data.
    num_runways = data["num_runways"]
    num_aircraft = data["num_aircraft"]
    aircraft_release = data["release"]
    aircraft_due = data["due"]
    aircraft_separation_time = data["aircraft_separation_time"]
    aircraft_cost = data["weight"]

    # Create model.
    model = solver.Model()
    model.hideOutput(not verbose)

    # Set number of threads.
    model.setParam("lp/threads", 1)
    model.setParam("parallel/maxnthreads", 1)

    # Set time limit.
    if time_limit is not None:
        model.setRealParam("limits/time", time_limit)

    # Create variables.
    assign = {
        (r, a): model.addVar(vtype="B", name=f"assign[{r},{a}]")
        for r in range(num_runways)
        for a in range(num_aircraft)
    }
    successor = {
        (a1, a2): model.addVar(vtype="B", name=f"successor[{a1},{a2}]")
        for a1 in range(num_aircraft)
        for a2 in range(num_aircraft)
        if a1 != a2
    }
    start_time = {
        a: model.addVar(
            vtype="C", obj=cost, lb=release, ub=due, name=f"start_time[{a}]"
        )
        for a, (cost, release, due) in enumerate(
            zip(aircraft_cost, aircraft_release, aircraft_due)
        )
    }

    # Assign every aircraft to one runway.
    assignment_constraints = [
        model.addCons(
            quicksum(assign[r, a] for r in range(num_runways)) == 1,
            name=f"assignment[{a}]",
        )
        for a in range(num_aircraft)
    ]

    # Create constraints enforcing separation time.
    separation_time_constraints = [
        model.addCons(
            start_time[a2]
            >= start_time[a1]
            + aircraft_separation_time[a1, a2]
            - (1 - successor[a1, a2])
            * (
                aircraft_due[a1]
                - aircraft_release[a2]
                + aircraft_separation_time[a1, a2]
            ),
            name=f"separation_time[{a1},{a2}]",
        )
        for a1 in range(num_aircraft)
        for a2 in range(num_aircraft)
        if a1 != a2
    ]

    # Create constraints computing successors.
    successor_constraints = [
        model.addCons(
            successor[a1, a2] + successor[a2, a1] >= assign[r, a1] + assign[r, a2] - 1,
            name=f"successor[{r},{a1},{a2}]",
        )
        for r in range(num_runways)
        for a1 in range(num_aircraft)
        for a2 in range(num_aircraft)
        if a1 < a2
    ]

    # Solve.
    # model.writeProblem('monolithic.lp', trans=False)
    model.optimize()

    # Get the solution.
    if model.getNSols() > 0:
        print("")
        cost = model.getObjVal()
        print(f"Cost {cost}")

        schedules = [[] for _ in range(num_runways)]
        for (r, a), var in assign.items():
            if is_gt(model.getVal(var), 0.0):
                t = eps_round(model.getVal(start_time[a]))
                schedules[r].append((t, a))
        for r, schedule in enumerate(schedules):
            schedule = sorted(schedule)
            print(f"Runway {r}:")
            for t, a in schedule:
                print(f"    Time {t}, aircraft {a}")
            # for i, (t1, a1) in enumerate(schedule):
            #     for j, (t2, a2) in enumerate(schedule):
            #         if i < j:
            #             type1 = aircraft_type[a1]
            #             type2 = aircraft_type[a2]
            #             op1 = aircraft_operation[a1]
            #             op2 = aircraft_operation[a2]
            #             assert aircraft_separation_time[a1, a2] == type_separation_time[op1, type1, op2, type2]
            #             assert aircraft_separation_time[a1, a2] <= t2 - t1
        print("")


def solve_gurobi(data, time_limit=None):
    import gurobipy as gp
    from gurobipy import quicksum

    # Get the instance data.
    num_runways = data["num_runways"]
    num_aircraft = data["num_aircraft"]
    aircraft_release = data["release"]
    aircraft_due = data["due"]
    aircraft_separation_time = data["aircraft_separation_time"]
    aircraft_cost = data["weight"]

    # Create model.
    model = gp.Model()

    # Set number of threads.
    model.params.threads = 1

    # Set time limit.
    if time_limit is not None:
        model.params.timeLimit = time_limit

    # Create variables.
    assign = {
        (r, a): model.addVar(vtype="B", name=f"assign[{r},{a}]")
        for r in range(num_runways)
        for a in range(num_aircraft)
    }
    successor = {
        (a1, a2): model.addVar(vtype="B", name=f"successor[{a1},{a2}]")
        for a1 in range(num_aircraft)
        for a2 in range(num_aircraft)
        if a1 != a2
    }
    start_time = {
        a: model.addVar(
            vtype="C", obj=cost, lb=release, ub=due, name=f"start_time[{a}]"
        )
        for a, (cost, release, due) in enumerate(
            zip(aircraft_cost, aircraft_release, aircraft_due)
        )
    }

    # Assign every aircraft to one runway.
    assignment_constraints = [
        model.addConstr(
            quicksum(assign[r, a] for r in range(num_runways)) == 1,
            name=f"assignment[{a}]",
        )
        for a in range(num_aircraft)
    ]

    # Create constraints enforcing separation time.
    separation_time_constraints = [
        model.addConstr(
            start_time[a2]
            >= start_time[a1]
            + aircraft_separation_time[a1, a2]
            - (1 - successor[a1, a2])
            * (
                aircraft_due[a1]
                - aircraft_release[a2]
                + aircraft_separation_time[a1, a2]
            ),
            name=f"separation_time[{a1},{a2}]",
        )
        for a1 in range(num_aircraft)
        for a2 in range(num_aircraft)
        if a1 != a2
    ]

    # Create constraints computing successors.
    successor_constraints = [
        model.addConstr(
            successor[a1, a2] + successor[a2, a1] >= assign[r, a1] + assign[r, a2] - 1,
            name=f"successor[{r},{a1},{a2}]",
        )
        for r in range(num_runways)
        for a1 in range(num_aircraft)
        for a2 in range(num_aircraft)
        if a1 < a2
    ]

    # Solve.
    # model.writeProblem('monolithic.lp', trans=False)
    model.optimize()

    # Get the solution.
    if model.sol_count > 0:
        print("")
        cost = model.objVal
        print(f"Cost {cost}")

        schedules = [[] for _ in range(num_runways)]
        for (r, a), var in assign.items():
            if is_gt(var.x, 0.0):
                t = eps_round(start_time[a].x)
                schedules[r].append((t, a))
        for r, schedule in enumerate(schedules):
            schedule = sorted(schedule)
            print(f"Runway {r}:")
            for t, a in schedule:
                print(f"    Time {t}, aircraft {a}")
            # for i, (t1, a1) in enumerate(schedule):
            #     for j, (t2, a2) in enumerate(schedule):
            #         if i < j:
            #             type1 = aircraft_type[a1]
            #             type2 = aircraft_type[a2]
            #             op1 = aircraft_operation[a1]
            #             op2 = aircraft_operation[a2]
            #             assert aircraft_separation_time[a1, a2] == type_separation_time[op1, type1, op2, type2]
            #             assert aircraft_separation_time[a1, a2] <= t2 - t1
        print("")


# def solve_monolithic_gurobi(data, time_limit=None):
#
#     from gurobipy import Model, quicksum
#
#     # Get the instance data.
#     num_runways = data['num_runways']
#     num_aircraft = data['num_aircraft']
#     release = data['release']
#     due = data['due']
#     aircraft_type = data['aircraft_type']
#     operation = data['operation']
#     cost = data['weight']
#     separation_times = data['separation_times']
#
#     # Compute separation_times.
#     aircraft_separation_time = {
#         (a1, a2): separation_times[op1, type1, op2, type2]
#         for a1, (type1, op1) in enumerate(zip(aircraft_type, operation))
#         for a2, (type2, op2) in enumerate(zip(aircraft_type, operation))
#         if a1 != a2
#     }
#
#     # Create model.
#     model = Model()
#
#     # Set number of threads.
#     model.Params.Threads = 1
#
#     # Set time limit.
#     if time_limit is not None:
#         model.Params.TimeLimit = time_limit
#
#     # Create variables.
#     assign = {(r, a): model.addVar(vtype="B", name=f'assign[{r},{a}]')
#               for r in range(num_runways) for a in range(num_aircraft)}
#     successor = {(a1, a2): model.addVar(vtype="B", name=f'successor[{a1},{a2}]')
#                  for a1 in range(num_aircraft) for a2 in range(num_aircraft) if a1 != a2}
#     start_time = {a: model.addVar(vtype="C", obj=cost, lb=release, ub=due, name=f'start_time[{a}]')
#                   for a, (cost, release, due) in enumerate(zip(cost, release, due))}
#
#     # Create assignment constraints. Every aircraft is assigned to a runway.
#     assignment_constraints = [
#         model.addConstr(
#             quicksum(assign[r, a] for r in range(num_runways)) == 1.0,
#             name=f'assignment[{a}]'
#         )
#         for a in range(num_aircraft)
#     ]
#
#     # Create constraints enforcing separation time.
#     time_constraints = [
#         model.addConstr(
#             start_time[a2] >= start_time[a1] + aircraft_separation_time[a1, a2] - \
#                 (1 - successor[a1, a2]) * (due[a1] - release[a2] + aircraft_separation_time[a1, a2]),
#             name=f'separation_time[{a1},{a2}]'
#         )
#         for a1 in range(num_aircraft) for a2 in range(num_aircraft) if a1 != a2
#     ]
#
#     # Create constraints computing successors.
#     successor_constraints = [
#         model.addConstr(
#             successor[a1, a2] + successor[a2, a1] >= assign[r, a1] + assign[r, a2] - 1,
#             name=f'successor[{r},{a1},{a2}]')
#         for r in range(num_runways)
#         for a1 in range(num_aircraft) for a2 in range(num_aircraft) if a1 < a2
#     ]
#
#     # Solve.
#     # model.writeProblem('monolithic.lp', trans=False)
#     model.optimize()
#
#     # Get the solution.
#     if model.SolCount > 0:
#         print('')
#         cost = model.ObjVal
#         print(f'Cost {cost}')
#         schedules = [[] for _ in range(num_runways)]
#         for (r, a), var in assign.items():
#             if is_gt(var.X, 0.0):
#                 t = eps_round(start_time[a].X)
#                 schedules[r].append((t, a))
#         for r, schedule in enumerate(schedules):
#             schedule = sorted(schedule)
#             print(f'Runway {r}:')
#             for t, a in schedule:
#                 print(f'    Time {t}, aircraft {a}')
#             for i, (t1, a1) in enumerate(schedule):
#                 for j, (t2, a2) in enumerate(schedule):
#                     if i < j:
#                         type1 = aircraft_type[a1]
#                         type2 = aircraft_type[a2]
#                         op1 = operation[a1]
#                         op2 = operation[a2]
#                         assert aircraft_separation_time[a1, a2] == separation_times[op1, type1, op2, type2]
#                         assert aircraft_separation_time[a1, a2] <= t2 - t1
#         print('')


def read_instance(instance_path):
    import json

    # Read the instance.
    with open(instance_path, "r") as f:
        data = json.load(f)
    data["separation_times"] = {
        tuple(key.split("-")): value for key, value in data["separation_times"].items()
    }

    # Check separation times.
    for x in data["separation_times"].values():
        assert is_integral(x), f"Separation time {x} must be an integer value"

    # Calculate the matrix of separation times.
    num_aircraft = data["num_aircraft"]
    aircraft_type = data["aircraft_type"]
    aircraft_operation = data["operation"]
    type_separation_times = data["separation_times"]
    data["aircraft_separation_time"] = np.array(
        [
            [
                type_separation_times[op1, type1, op2, type2] if a1 != a2 else 0
                for a2, (type2, op2) in enumerate(
                    zip(aircraft_type, aircraft_operation)
                )
            ]
            for a1, (type1, op1) in enumerate(zip(aircraft_type, aircraft_operation))
        ]
        + [[0 for _ in range(num_aircraft)]],
        dtype=np.int32,
    )

    return data


if __name__ == "__main__":

    import argparse

    # Prepare parser for input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", help="Instance file")
    parser.add_argument(
        "--solver", help="Solver (gurobi, scip, gcg, cabs, caasdy, labeling)"
    )
    parser.add_argument("--timeout", help="Time limit in seconds")
    parser.add_argument("--disable-set-resource", action="store_true")
    parser.add_argument("--disable-filter", action="store_true")
    parser.add_argument("--get-all-solutions", action="store_true")
    parser.add_argument("--one-column-farkas", action="store_true")
    parser.add_argument(
        "--bound",
        type=str,
        choices=["knapsack", "profit", "reduce", "none"],
        default="knapsack",
        help="Dual bound in DIDP pricing (knapsack, profit, reduce, none)",
    )
    parser.add_argument("--f-first", action="store_true")
    args = parser.parse_args()

    # Get arguments.
    time_limit = float(args.timeout) if args.timeout else None
    instance_path = args.instance

    # Read instance data.
    data = read_instance(instance_path)

    # Solve.
    solver = args.solver
    get_all_solutions = args.get_all_solutions
    one_column_farkas = args.one_column_farkas
    if solver == "gurobi":
        print(f"Solving monolithic MIP model using {solver}")
        print("", flush=True)
        solve_gurobi(data, time_limit)
    elif solver in ["scip", "gcg"]:
        print(f"Solving monolithic MIP model using {solver}")
        print("", flush=True)
        solve_scip(data, solver, time_limit)
    else:
        print(f"Solving branch-and-price using DIDP with {solver} pricer")
        print("", flush=True)
        bound = None if args.bound == "none" else args.bound
        solve_branch_and_price(
            data,
            solver,
            get_all_solutions,
            one_column_farkas,
            bound,
            time_limit,
            args.disable_set_resource,
            args.disable_filter,
            args.f_first,
        )
