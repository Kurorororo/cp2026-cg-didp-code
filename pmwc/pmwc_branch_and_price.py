import argparse
import copy
import time

import didppy as dp
import pyscipopt as scip
import gurobipy as gp
import ortools.sat.python.cp_model as cp

import pmwc


def create_master_model(instance):
    model = scip.Model()

    machine_cons = model.addCons(
        scip.quicksum([]) <= instance["m"], separate=False, modifiable=True
    )
    job_conss = {
        j: model.addCons(scip.quicksum([]) == 1, separate=False, modifiable=True)
        for j in range(instance["n"])
    }

    model.data = {
        "instance": instance,
        "y": [],
        "schedules": [],
        "machine_cons": machine_cons,
        "job_conss": job_conss,
    }

    return model


def create_pricing_dp_model(instance, dual, farkas=False, bound="knapsack"):
    model = dp.Model(float_cost=True)

    n = instance["n"]
    h_min = instance["h_min"]
    h_max = instance["h_max"]

    job = model.add_object_type(number=n)
    p = model.add_int_table(instance["p"])
    w = model.add_int_table(instance["w"])
    r = model.add_int_table(instance["r"])
    d = model.add_int_table(instance["d"])
    dual_table = model.add_float_table(dual)

    j = model.add_element_var(object_type=job, target=0)
    t = model.add_int_var(target=0)

    cost_weight = -dual_table[j] if farkas else w[j] * (t + p[j]) - dual_table[j]
    include = dp.Transition(
        name="include",
        cost=cost_weight + dp.FloatExpr.state_cost(),
        effects=[(j, j + 1), (t, t + p[j])],
        preconditions=[j < n, r[j] <= t, t + p[j] <= d[j]],
    )
    model.add_transition(include)

    exclude = dp.Transition(
        name="exclude",
        cost=dp.FloatExpr.state_cost(),
        effects=[(j, j + 1)],
        preconditions=[j < n],
    )
    model.add_transition(exclude)

    model.add_base_case([j == n, t >= h_min])

    candidates = model.add_set_table(
        [list(range(k, n)) for k in range(n + 1)], object_type=job
    )
    values = [
        max(
            (
                dual[k]
                if farkas
                else dual[k] - instance["w"][k] * (instance["r"][k] + instance["p"][k])
            ),
            0,
        )
        for k in range(n)
    ]

    if bound == "knapsack":
        model.add_dual_bound(
            -dp.fractional_knapsack(candidates[j], h_max - t, values, instance["p"])
        )
    elif bound == "profit":
        value_table = model.add_float_table(values)
        model.add_dual_bound(-value_table[candidates[j]])

    return model


def var_to_index(var):
    if var.name.startswith("t_"):
        return int(var.name[2:])
    else:
        return int(var.name)


class SimpleRoundingHeuristic(scip.Heur):
    def __init__(self):
        self.machine_cons = None
        self.job_conss = None

    def heurinit(self):
        self.machine_cons = self.model.getTransformedCons(
            self.model.data["machine_cons"]
        )
        self.job_conss = {
            i: self.model.getTransformedCons(c)
            for i, c in self.model.data["job_conss"].items()
        }

    def add_column(self, schedule):
        c = []
        t = 0
        cost = 0

        for j in schedule:
            t += self.model.data["instance"]["p"][j]
            c.append(t)
            cost += self.model.data["instance"]["w"][j] * t

        index = len(self.model.data["schedules"])
        v = self.model.addVar(
            vtype="I", lb=0, obj=cost, name=str(index), pricedVar=False
        )

        self.model.addConsCoeff(self.machine_cons, v, 1)

        for j in schedule:
            self.model.addConsCoeff(self.job_conss[j], v, 1)

        self.model.data["y"].append(v)
        self.model.data["schedules"].append(
            {"schedule": schedule, "completion_time": c}
        )

        for j, c_j in zip(schedule, c):
            self.model.data["job_to_schedule_and_completion_time"][j].append(
                (index, c_j)
            )

        return v

    def heurexec(self, _heurtiming, nodeinfeasible_):
        result = scip.SCIP_RESULT.DIDNOTRUN

        # This heuristic does not run if the LP status is not optimal
        lpsolstat = self.model.getLPSolstat()

        if lpsolstat != scip.SCIP_LPSOLSTAT.OPTIMAL:
            return {"result": result}

        # Get the current branching candidate, i.e., the current fractional variables with integer requirements
        (
            branch_vars,
            _,
            _,
            ncands,
            *_,
        ) = self.model.getLPBranchCands()

        # Ignore if there are no branching candidates
        if ncands == 0:
            return {"result": result}

        # Create a solution that is initialised to the LP values
        sol = self.model.createSol(self, initlp=True)

        n = self.model.data["instance"]["n"]
        c = [None] * n

        for var in branch_vars:
            s = self.model.data["schedules"][var_to_index(var)]

            for j, c_j in zip(s["schedule"], s["completion_time"]):
                if c[j] is None:
                    c[j] = c_j
                elif c[j] != c_j:
                    return {"result": result}

            self.model.setSolVal(sol, var, 0)

        done = [False] * n
        n_unfixed_jobs = sum(1 for c_j in c if c_j is not None)
        n_done = 0

        while n_done < n_unfixed_jobs:
            schedule = []
            t = 0

            for j in range(n):
                p_j = self.model.data["instance"]["p"][j]

                if c[j] is not None and not done[j] and c[j] - p_j == t:
                    schedule.append(j)
                    t = c[j]
                    done[j] = True
                    n_done += 1

            if len(schedule) > 0:
                v = self.add_column(schedule)
                self.model.setSolVal(sol, v, 1)

        # Now try the solution. Note: This will free the solution afterwards by default.
        stored = self.model.trySol(sol)

        if stored:
            return {"result": scip.SCIP_RESULT.FOUNDSOL}
        else:
            return {"result": scip.SCIP_RESULT.DIDNOTFIND}


class TimeWindowConshdlr(scip.Conshdlr):
    def __init__(self):
        self.conss = []

    def consdata_create(self, name, j, r_j, d_j, node):
        cons = self.model.createCons(
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
        cons.data = {
            "job": j,
            "release_date": r_j,
            "deadline": d_j,
            "node": node,
            "propagated": False,
            "npropagated": 0,
        }
        self.conss.append(cons)

        return cons

    def consactive(self, constraint):
        if constraint.data["npropagated"] != len(self.model.data["y"]):
            constraint.data["propagated"] = False
            self.model.repropagateNode(constraint.data["node"])

    def consdeactivate(self, constraint):
        constraint.data["npropagated"] = len(self.model.data["y"])

    def consprop(self, constraints, _nusefulconss, _nmarkedconss, _proptiming):
        result = scip.SCIP_RESULT.DIDNOTFIND

        p = self.model.data["instance"]["p"]

        for c in constraints:
            if c.data["propagated"]:
                continue

            j = c.data["job"]
            r_j = c.data["release_date"]
            d_j = c.data["deadline"]

            if self.model.data["verbosity"] >= 3:
                node_id = self.model.getCurrentNode().getNumber()

                print(f"Job {j} deadline {d_j} at node {node_id}")

            for i, c_j in self.model.data["job_to_schedule_and_completion_time"][j]:
                if r_j > c_j - p[j] or d_j < c_j:
                    self.model.fixVar(
                        self.model.getTransformedVar(self.model.data["y"][i]), 0
                    )

                    if result == scip.SCIP_RESULT.DIDNOTFIND:
                        result = scip.SCIP_RESULT.REDUCEDDOM

            c.data["propagated"] = True
            c.data["npropagated"] = len(self.model.data["y"])

        return {"result": result}


class TimeWindowBranchrule(scip.Branchrule):
    def branch(self, j, d_j):
        if self.model.data["verbosity"] >= 2:
            node_id = self.model.getCurrentNode().getNumber()
            print(
                f"At node {node_id}, branching on job {j} with deadline {d_j}",
            )

        left_child = self.model.createChild(0, self.model.getLocalEstimate())
        left_id = left_child.getNumber()
        left_r_j = self.model.data["instance"]["r"][j]
        name = str((left_id, j, left_r_j, d_j))
        left_cons = self.model.data["conshdlr"].consdata_create(
            name,
            j,
            left_r_j,
            d_j,
            left_child,
        )
        self.model.addConsNode(left_child, left_cons)

        right_child = self.model.createChild(0, self.model.getLocalEstimate())
        right_id = right_child.getNumber()
        right_r_j = d_j + 1 - self.model.data["instance"]["p"][j]
        right_d_j = self.model.data["instance"]["d"][j]
        name = str((right_id, j, right_r_j, right_d_j))
        right_cons = self.model.data["conshdlr"].consdata_create(
            name,
            j,
            right_r_j,
            right_d_j,
            right_child,
        )
        self.model.addConsNode(right_child, right_cons)

        return {"result": scip.SCIP_RESULT.BRANCHED}

    def branchexeclp(self, _allowaddcons):
        branch_vars, sol_vals, *_ = self.model.getLPBranchCands()

        n = self.model.data["instance"]["n"]
        c_average = [0] * n
        c_min = [None] * n

        for var, val in zip(branch_vars, sol_vals):
            s = self.model.data["schedules"][var_to_index(var)]

            for j, c_j in zip(s["schedule"], s["completion_time"]):
                c_average[j] += val * c_j

                if c_min[j] is None or c_j < c_min[j]:
                    c_min[j] = c_j

        branch_j = next(
            filter(
                lambda j: c_min[j] is not None
                and c_average[j] > c_min[j] + self.model.data["tolerance"],
                range(n),
            )
        )

        return self.branch(branch_j, c_min[branch_j])

    def branchexecps(self, _allowaddcons):
        branch_vars, *_ = self.model.getPseudoBranchCands()
        c_min = [None] * (self.model.data["instance"]["n"])

        for var in branch_vars:
            s = self.model.data["schedules"][var_to_index(var)]

            for j, c_j in zip(s["schedule"], s["completion_time"]):
                if c_min[j] is None or c_j < c_min[j]:
                    c_min[j] = c_j

        branch_j = next(filter(lambda j: c_min[j] is not None, range(len(c_min))))

        print("WARNING: branching on pseudosolution")

        return self.branch(branch_j, c_min[branch_j])


class DIDPPricer(scip.Pricer):
    def __init__(
        self,
        get_all_solutions=False,
        solver="labeling",
        bound="knapsack",
        f_first=False,
    ):
        self.machine_cons = None
        self.job_conss = None
        self.get_all_solutions = get_all_solutions
        self.bound = bound
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

    def pricerinit(self):
        self.machine_cons = self.model.getTransformedCons(
            self.model.data["machine_cons"]
        )
        self.job_conss = {
            i: self.model.getTransformedCons(c)
            for i, c in self.model.data["job_conss"].items()
        }
        self.model.data["job_to_schedule_and_completion_time"] = [
            [] for _ in range(self.model.data["instance"]["n"])
        ]

    def time_remaining(self):
        if self.model.data["has_time_limit"]:
            time_limit = self.model.getParam("limits/time")
            elapsed_time = self.model.getSolvingTime()

            return time_limit - elapsed_time

        return None

    def add_column(self, schedule):
        c = []
        t = 0
        cost = 0

        for j in schedule:
            t += self.model.data["instance"]["p"][j]
            c.append(t)
            cost += self.model.data["instance"]["w"][j] * t

        index = len(self.model.data["schedules"])
        v = self.model.addVar(
            vtype="I", lb=0, obj=cost, name=str(index), pricedVar=True
        )

        self.model.addConsCoeff(self.machine_cons, v, 1)

        for j in schedule:
            self.model.addConsCoeff(self.job_conss[j], v, 1)

        self.model.data["y"].append(v)
        self.model.data["schedules"].append(
            {"schedule": schedule, "completion_time": c}
        )

        for j, c_j in zip(schedule, c):
            self.model.data["job_to_schedule_and_completion_time"][j].append(
                (index, c_j)
            )

    def price(self, farkas):
        if self.model.data["verbosity"] >= 2:
            current_id = self.model.getCurrentNode().getNumber()
            print(f"At node {current_id}, LP obj: {self.model.getLPObjVal()}")

        time_limit = self.time_remaining()

        if time_limit is not None and time_limit <= 0.0:
            return {"result": scip.SCIP_RESULT.DIDNOTRUN}

        timed_out = False
        n_generated = 0

        instance = copy.deepcopy(self.model.data["instance"])
        n = instance["n"]
        p = instance["p"]
        w = instance["w"]

        for cons in self.model.data["conshdlr"].conss:
            if cons.isActive():
                j = cons.data["job"]
                r_j = cons.data["release_date"]
                d_j = cons.data["deadline"]

                instance["r"][j] = max(instance["r"][j], r_j)
                instance["d"][j] = min(instance["d"][j], d_j)

                if r_j > instance["r"][j]:
                    for k in range(j + 1, n):
                        if w[k] <= w[j] and p[k] >= p[j]:
                            instance["r"][k] = max(instance["r"][k], r_j)

                if d_j < instance["d"][j]:
                    for k in range(j):
                        if w[k] >= w[j] and p[k] <= p[j]:
                            instance["d"][k] = min(instance["d"][k], d_j - p[j] + p[k])

        if farkas:
            dual = [self.model.getDualfarkasLinear(self.job_conss[j]) for j in range(n)]
            constant_cost = -self.model.getDualfarkasLinear(self.machine_cons)
        else:
            dual = [self.model.getDualsolLinear(self.job_conss[j]) for j in range(n)]
            constant_cost = -self.model.getDualsolLinear(self.machine_cons)

        model = create_pricing_dp_model(instance, dual, farkas=farkas, bound=self.bound)
        primal_bound = -self.model.data["tolerance"] - constant_cost

        if self.get_all_solutions and self.f_first:
            solver = self.solver_type(
                model,
                primal_bound=primal_bound,
                quiet=self.model.data["verbosity"] <= 2,
                get_all_solutions=True,
                node_first=True,
                time_limit=time_limit,
            )
        elif self.get_all_solutions:
            solver = self.solver_type(
                model,
                primal_bound=primal_bound,
                quiet=self.model.data["verbosity"] <= 2,
                get_all_solutions=True,
                time_limit=time_limit,
            )
        elif self.f_first:
            solver = self.solver_type(
                model,
                primal_bound=primal_bound,
                quiet=self.model.data["verbosity"] <= 2,
                node_first=True,
                time_limit=time_limit,
            )
        else:
            solver = self.solver_type(
                model,
                primal_bound=primal_bound,
                quiet=self.model.data["verbosity"] <= 2,
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

                schedule = [
                    j for j, t in enumerate(solution.transitions) if t.name == "include"
                ]
                self.add_column(schedule)
                n_generated += 1

            timed_out |= solution.time_out

        lowerbound = self.model.getLPObjVal()

        if best_cost is not None:
            lowerbound += (best_cost + constant_cost) * instance["m"]

        if self.model.data["verbosity"] >= 2:
            print(f"Generated {n_generated} columns")

        if timed_out and n_generated == 0:
            return {"result": scip.SCIP_RESULT.DIDNOTRUN}
        elif timed_out or farkas:
            return {"result": scip.SCIP_RESULT.SUCCESS}
        else:
            return {"result": scip.SCIP_RESULT.SUCCESS, "lowerbound": lowerbound}

    def pricerredcost(self):
        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        num_columns = len(self.model.data["schedules"])
        try:
            return self.price(farkas=False)

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start
            self.pricing_stats["columns"] += (
                len(self.model.data["schedules"]) - num_columns
            )

    def pricerfarkas(self):
        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        num_columns = len(self.model.data["schedules"])
        try:
            return self.price(farkas=True)

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start
            self.pricing_stats["columns"] += (
                len(self.model.data["schedules"]) - num_columns
            )


class GurobiPricer(scip.Pricer):
    def __init__(self, threads=1):
        self.machine_cons = None
        self.job_conss = None
        self.threads = threads
        self.pricing_model = None
        self.pricing_stats = {"time": 0.0, "calls": 0, "columns": 0}

    def pricerfree(self):
        if self.pricing_model is not None:
            self.pricing_model = self.pricing_model.dispose()

    def pricerinit(self):
        self.machine_cons = self.model.getTransformedCons(
            self.model.data["machine_cons"]
        )
        self.job_conss = {
            i: self.model.getTransformedCons(c)
            for i, c in self.model.data["job_conss"].items()
        }
        self.model.data["job_to_schedule_and_completion_time"] = [
            [] for _ in range(self.model.data["instance"]["n"])
        ]

    def time_remaining(self):
        if self.model.data["has_time_limit"]:
            time_limit = self.model.getParam("limits/time")
            elapsed_time = self.model.getSolvingTime()

            return time_limit - elapsed_time

        return None

    def add_column(self, schedule):
        c = []
        t = 0
        cost = 0

        for j in schedule:
            t += self.model.data["instance"]["p"][j]
            c.append(t)
            cost += self.model.data["instance"]["w"][j] * t

        index = len(self.model.data["schedules"])
        v = self.model.addVar(
            vtype="I", lb=0, obj=cost, name=str(index), pricedVar=True
        )

        self.model.addConsCoeff(self.machine_cons, v, 1)

        for j in schedule:
            self.model.addConsCoeff(self.job_conss[j], v, 1)

        self.model.data["y"].append(v)
        self.model.data["schedules"].append(
            {"schedule": schedule, "completion_time": c}
        )

        for j, c_j in zip(schedule, c):
            self.model.data["job_to_schedule_and_completion_time"][j].append(
                (index, c_j)
            )

    def create_model(self, instance, dual, constant_cost, farkas=False):
        n = instance["n"]
        p = instance["p"]
        w = [0 if farkas else w_i for w_i in instance["w"]]
        r = instance["r"]
        d = instance["d"]

        if self.pricing_model is None:
            model = gp.Model()

            # Job j is included in the schedule
            negative_dual = [-v for v in dual]
            x = model.addVars(n, vtype=gp.GRB.BINARY, obj=negative_dual, name="x")
            # Completion time of job j
            c = model.addVars(n, vtype=gp.GRB.CONTINUOUS, lb=0, ub=d, obj=w, name="c")

            model.addConstrs(
                (c[j] >= (r[j] + p[j]) * x[j] for j in range(n)), name="c_first"
            )
            model.addConstrs(
                c[j]
                >= gp.quicksum(p[k] * x[k] for k in range(j + 1))
                - sum(p[:j]) * (1 - x[j])
                for j in range(n)
            )

            self.pricing_model = model

            return model, x
        else:
            x = self.pricing_model.getVars()[:n]
            c = self.pricing_model.getVars()[n:]

            for j in range(n):
                x[j].Obj = -dual[j]
                c[j].UB = d[j]
                c[j].Obj = 0.0 if farkas else w[j]
                self.pricing_model.chgCoeff(
                    self.pricing_model.getConstrByName("c_first[%d]" % j),
                    x[j],
                    -(r[j] + p[j]),
                )

            return self.pricing_model, x

    def price(self, farkas):
        if self.model.data["verbosity"] >= 2:
            current_id = self.model.getCurrentNode().getNumber()
            print(f"At node {current_id}, LP obj: {self.model.getLPObjVal()}")

        time_limit = self.time_remaining()

        if time_limit is not None and time_limit <= 0.0:
            return {"result": scip.SCIP_RESULT.DIDNOTRUN}

        timed_out = False
        n_generated = 0

        instance = copy.deepcopy(self.model.data["instance"])
        n = instance["n"]
        p = instance["p"]
        w = instance["w"]

        for cons in self.model.data["conshdlr"].conss:
            if cons.isActive():
                j = cons.data["job"]
                r_j = cons.data["release_date"]
                d_j = cons.data["deadline"]

                instance["r"][j] = r_j
                instance["d"][j] = d_j

                if r_j > instance["r"][j]:
                    for k in range(j + 1, n):
                        if w[k] <= w[j] and p[k] >= p[j]:
                            instance["r"][k] = max(instance["r"][k], r_j)

                if d_j < instance["d"][j]:
                    for k in range(j):
                        if w[k] >= w[j] and p[k] <= p[j]:
                            instance["d"][k] = min(instance["d"][k], d_j - p[j])

        if farkas:
            dual = [self.model.getDualfarkasLinear(self.job_conss[j]) for j in range(n)]
            constant_cost = -self.model.getDualfarkasLinear(self.machine_cons)
        else:
            dual = [self.model.getDualsolLinear(self.job_conss[j]) for j in range(n)]
            constant_cost = -self.model.getDualsolLinear(self.machine_cons)

        model, x = self.create_model(instance, dual, constant_cost, farkas=farkas)

        model.params.cutoff = -constant_cost
        model.params.outputFlag = 1 if self.model.data["verbosity"] >= 3 else 0
        model.params.threads = self.threads

        if time_limit is not None:
            model.params.timeLimit = time_limit

        model.optimize()
        best_cost = None

        for i in range(model.SolCount):
            model.params.SolutionNumber = i

            if best_cost is None or model.ObjVal < best_cost:
                best_cost = model.ObjVal

            if model.ObjVal + constant_cost < -self.model.data["tolerance"]:
                schedule = [j for j in range(instance["n"]) if x[j].Xn > 0.5]
                self.add_column(schedule)
                n_generated += 1

        if model.status == gp.GRB.TIME_LIMIT:
            timed_out = True

        lowerbound = self.model.getLPObjVal()

        if best_cost is not None:
            lowerbound += (best_cost + constant_cost) * instance["m"]

        if self.model.data["verbosity"] >= 2:
            print(f"Generated {n_generated} columns")

        if timed_out and n_generated == 0:
            return {"result": scip.SCIP_RESULT.DIDNOTRUN}
        elif timed_out or farkas:
            return {"result": scip.SCIP_RESULT.SUCCESS}
        else:
            return {"result": scip.SCIP_RESULT.SUCCESS, "lowerbound": lowerbound}

    def pricerredcost(self):
        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        num_columns = len(self.model.data["schedules"])
        try:
            return self.price(farkas=False)

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start
            self.pricing_stats["columns"] += (
                len(self.model.data["schedules"]) - num_columns
            )

    def pricerfarkas(self):
        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        num_columns = len(self.model.data["schedules"])
        try:
            return self.price(farkas=True)

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start
            self.pricing_stats["columns"] += (
                len(self.model.data["schedules"]) - num_columns
            )


class CPPricer(scip.Pricer):
    def __init__(self, threads=1):
        self.machine_cons = None
        self.job_conss = None
        self.threads = threads
        self.pricing_stats = {"time": 0.0, "calls": 0, "columns": 0}

    def pricerfree(self):
        if self.pricing_model is not None:
            self.pricing_model = self.pricing_model.dispose()

    def pricerinit(self):
        self.machine_cons = self.model.getTransformedCons(
            self.model.data["machine_cons"]
        )
        self.job_conss = {
            i: self.model.getTransformedCons(c)
            for i, c in self.model.data["job_conss"].items()
        }
        self.model.data["job_to_schedule_and_completion_time"] = [
            [] for _ in range(self.model.data["instance"]["n"])
        ]

    def time_remaining(self):
        if self.model.data["has_time_limit"]:
            time_limit = self.model.getParam("limits/time")
            elapsed_time = self.model.getSolvingTime()

            return time_limit - elapsed_time

        return None

    def add_column(self, schedule):
        c = []
        t = 0
        cost = 0

        for j in schedule:
            t += self.model.data["instance"]["p"][j]
            c.append(t)
            cost += self.model.data["instance"]["w"][j] * t

        index = len(self.model.data["schedules"])
        v = self.model.addVar(
            vtype="I", lb=0, obj=cost, name=str(index), pricedVar=True
        )

        self.model.addConsCoeff(self.machine_cons, v, 1)

        for j in schedule:
            self.model.addConsCoeff(self.job_conss[j], v, 1)

        self.model.data["y"].append(v)
        self.model.data["schedules"].append(
            {"schedule": schedule, "completion_time": c}
        )

        for j, c_j in zip(schedule, c):
            self.model.data["job_to_schedule_and_completion_time"][j].append(
                (index, c_j)
            )

    def create_model(self, instance, dual, constant_cost, farkas=False):
        n = instance["n"]
        p = instance["p"]
        w = [0 if farkas else w_i for w_i in instance["w"]]
        r = instance["r"]
        d = instance["d"]

        model = cp.CpModel()

        x = [model.new_bool_var(f"x_{j}") for j in range(n)]
        c = [model.new_int_var(0, d[j], f"t_{j}") for j in range(n)]

        for j in range(n):
            model.add(c[j] >= r[j] + p[j]).only_enforce_if(x[j])

        for j in range(1, n):
            model.add(
                c[j] == p[j] + sum(p[k] * x[k] for k in range(j))
            ).only_enforce_if(x[j])

        model.minimize(
            sum((0 if farkas else w[j] * c[j]) - dual[j] * x[j] for j in range(n))
        )

        return model, x

    def price(self, farkas):
        class SolutionCallback(cp.CpSolverSolutionCallback):
            def __init__(self, x):
                super().__init__()
                self.x = x
                self.schedules = []

            def on_solution_callback(self):
                self.schedules.append(
                    [j for j, x_j in enumerate(self.x) if self.Value(x_j)]
                )

        if self.model.data["verbosity"] >= 2:
            current_id = self.model.getCurrentNode().getNumber()
            print(f"At node {current_id}, LP obj: {self.model.getLPObjVal()}")

        time_limit = self.time_remaining()

        if time_limit is not None and time_limit <= 0.0:
            return {"result": scip.SCIP_RESULT.DIDNOTRUN}

        timed_out = False
        n_generated = 0

        instance = copy.deepcopy(self.model.data["instance"])
        n = instance["n"]
        p = instance["p"]
        w = instance["w"]

        for cons in self.model.data["conshdlr"].conss:
            if cons.isActive():
                j = cons.data["job"]
                r_j = cons.data["release_date"]
                d_j = cons.data["deadline"]

                instance["r"][j] = r_j
                instance["d"][j] = d_j

                if r_j > instance["r"][j]:
                    for k in range(j + 1, n):
                        if w[k] <= w[j] and p[k] >= p[j]:
                            instance["r"][k] = max(instance["r"][k], r_j)

                if d_j < instance["d"][j]:
                    for k in range(j):
                        if w[k] >= w[j] and p[k] <= p[j]:
                            instance["d"][k] = min(instance["d"][k], d_j - p[j])

        if farkas:
            dual = [self.model.getDualfarkasLinear(self.job_conss[j]) for j in range(n)]
            constant_cost = -self.model.getDualfarkasLinear(self.machine_cons)
        else:
            dual = [self.model.getDualsolLinear(self.job_conss[j]) for j in range(n)]
            constant_cost = -self.model.getDualsolLinear(self.machine_cons)

        model, x = self.create_model(instance, dual, constant_cost, farkas=farkas)

        solver = cp.CpSolver()
        solver.parameters.num_search_workers = self.threads

        if self.model.data["verbosity"] >= 3:
            solver.parameters.log_search_progress = True

        if time_limit is not None:
            solver.parameters.max_time_in_seconds = time_limit

        callback = SolutionCallback(x)
        status = solver.solve(model, callback)
        best_cost = None

        for schedule in callback.schedules:
            obj_value = 0
            t = 0

            for j in schedule:
                t += instance["p"][j]
                obj_value += -dual[j]

                if not farkas:
                    obj_value += instance["w"][j] * t

            if obj_value + constant_cost < -self.model.data["tolerance"]:
                self.add_column(schedule)
                n_generated += 1

            if best_cost is None or obj_value < best_cost:
                best_cost = obj_value

        lowerbound = self.model.getLPObjVal()

        if best_cost is not None:
            lowerbound += (best_cost + constant_cost) * instance["m"]

        if status != cp.OPTIMAL and status != cp.INFEASIBLE:
            timed_out = True

        if self.model.data["verbosity"] >= 2:
            print(f"Generated {n_generated} columns")

        if timed_out and n_generated == 0:
            return {"result": scip.SCIP_RESULT.DIDNOTRUN}
        elif timed_out or farkas:
            return {"result": scip.SCIP_RESULT.SUCCESS}
        else:
            return {"result": scip.SCIP_RESULT.SUCCESS, "lowerbound": lowerbound}

    def pricerredcost(self):
        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        num_columns = len(self.model.data["schedules"])
        try:
            return self.price(farkas=False)

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start
            self.pricing_stats["columns"] += (
                len(self.model.data["schedules"]) - num_columns
            )

    def pricerfarkas(self):
        self.pricing_stats["calls"] += 1
        start = time.perf_counter()
        num_columns = len(self.model.data["schedules"])
        try:
            return self.price(farkas=True)

        finally:
            self.pricing_stats["time"] += time.perf_counter() - start
            self.pricing_stats["columns"] += (
                len(self.model.data["schedules"]) - num_columns
            )


if __name__ == "__main__":
    start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verbosity", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--pricer", type=str, choices=["dp", "mip", "cp"], default="dp")
    parser.add_argument(
        "--solver", type=str, choices=["cabs", "caasdy", "labeling"], default="labeling"
    )
    parser.add_argument(
        "--bound", type=str, choices=["knapsack", "profit", "none"], default="knapsack"
    )
    parser.add_argument("--get-all-solutions", action="store_true")
    parser.add_argument("--f-first", action="store_true")
    args = parser.parse_args()

    instance = pmwc.read_instance(args.input)
    pmwc.compute_bound(instance)

    model = create_master_model(instance)

    if args.pricer == "mip":
        pricer = GurobiPricer(threads=args.threads)
        model.includePricer(
            pricer, "Gurobi Pricer", "Gurobi pricing problem", delay=True
        )
    elif args.pricer == "cp":
        pricer = CPPricer(threads=args.threads)
        model.includePricer(pricer, "CP Pricer", "CP pricing problem", delay=True)
    else:
        pricer = DIDPPricer(
            solver=args.solver,
            get_all_solutions=args.get_all_solutions,
            bound=args.bound,
            f_first=args.f_first,
        )
        model.includePricer(pricer, "DIDP Pricer", "DIDP pricing problem", delay=True)

    heuristic = SimpleRoundingHeuristic()
    model.includeHeur(
        heuristic,
        "HandlingSpecialFractionalSolution",
        "custom heuristic implemented in python",
        "Y",
        timingmask=scip.SCIP_HEURTIMING.DURINGLPLOOP,
    )

    conshdlr = TimeWindowConshdlr()
    model.includeConshdlr(
        conshdlr,
        "TimeWindowConshdlr",
        "Conshdlr imposing a time window constraint for a job",
        chckpriority=1000000,
        propfreq=1,
    )
    model.data["conshdlr"] = conshdlr

    branchrule = TimeWindowBranchrule()
    model.includeBranchrule(
        branchrule,
        "TimeWindowBranchrule",
        "Branchrule for enforcing time window constraints",
        priority=1000000,
        maxdepth=-1,
        maxbounddist=1,
    )

    model.data["verbosity"] = args.verbosity

    if args.verbosity == 0:
        model.hideOutput()

    model.setParam("display/freq", 1)
    model.setParam("display/headerfreq", 1)

    if args.timeout is None:
        model.data["has_time_limit"] = False
    else:
        model.setParam("limits/time", args.timeout)
        model.data["has_time_limit"] = True

    model.setParam("limits/gap", args.tolerance)
    model.data["tolerance"] = args.tolerance
    model.setObjIntegral()

    model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
    model.setSeparating(scip.SCIP_PARAMSETTING.OFF)
    model.setIntParam("propagating/rootredcost/freq", -1)

    model.optimize()

    print("Solving time: {}s".format(time.perf_counter() - start))
    print("Nodes:", model.getNTotalNodes())
    print("Pricing time: {}s".format(pricer.pricing_stats["time"]))
    print("Pricing calls:", pricer.pricing_stats["calls"])
    print("Pricing columns:", pricer.pricing_stats["columns"])

    if args.pricer == "dp":
        print("Pricing expanded:", pricer.pricing_stats["expanded"])

    status = model.getStatus()

    if status == "infeasible":
        print("The problem is infeasible")
    else:
        print("best bound:", model.getDualbound())

        n_sols = model.getNSols()

        if n_sols == 0:
            print("No solution found")
        else:
            cost = int(round(model.getObjVal()))
            print("cost:", cost)

            if status == "optimal":
                print("optimal cost:", cost)

            solution = []

            for i, v in enumerate(model.data["y"]):
                if model.getVal(v) > 0.5:
                    solution.append(model.data["schedules"][i]["schedule"])

            print(solution)

            recomputed_cost = pmwc.validate(instance, solution)

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
