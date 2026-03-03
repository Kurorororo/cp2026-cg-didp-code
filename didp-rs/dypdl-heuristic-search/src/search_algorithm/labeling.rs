use super::data_structure::{
    exceed_bound, BfsNode, ParentAndChildStateFunctionCache, ResourceLexicographicNode,
    StateInRegistry, StateInformation, StateRegistry, SuccessorGenerator,
};
use super::rollout::get_solution_cost_and_suffix;
use super::search::{Parameters, Search, SearchInput, Solution};
use super::util::{update_bound_if_better, update_solution, TimeKeeper};
use dypdl::prelude::*;
use dypdl::variable_type::Numeric;
use std::collections::BinaryHeap;
use std::error::Error;
use std::fmt::Display;
use std::marker::PhantomData;
use std::rc::Rc;

/// Parameters for the labeling algorithm.
#[derive(Default)]
pub struct LabelingParameters {
    /// Lazily insert states into the registry.
    pub lazy_insertion: bool,
    /// Use lexicographic ordering for dominance detection.
    pub lexicographic_storing: bool,
    /// Use the node evaluation value as the primary key and the resource variables as the secondary key for lexicographic ordering.
    pub node_first: bool,
}

/// Labeling solver.
pub struct Labeling<'a, T, N, G, B, const NODE_FIRST: bool, V = Transition>
where
    T: Numeric + Ord + Display,
    N: BfsNode<T, V>,
    G: FnMut(
        StateInRegistry,
        &mut StateFunctionCache,
        T,
        &N,
        &Rc<V>,
        Option<T>,
        Option<&N>,
    ) -> Option<N>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    generator: SuccessorGenerator<V>,
    suffix: &'a [V],
    node_generator: G,
    base_cost_evaluator: B,
    primal_bound: Option<T>,
    original_primal_bound: Option<T>,
    get_all_solutions: bool,
    quiet: bool,
    labeling_parameters: LabelingParameters,
    open: BinaryHeap<Rc<ResourceLexicographicNode<T, N, NODE_FIRST, V>>>,
    registry: StateRegistry<T, ResourceLexicographicNode<T, N, NODE_FIRST, V>>,
    function_cache: ParentAndChildStateFunctionCache,
    applicable_transitions: Vec<Rc<V>>,
    time_keeper: TimeKeeper,
    solution: Solution<T>,
    phantom: PhantomData<N>,
}

impl<'a, T, N, G, B, const NODE_FIRST: bool, V> Labeling<'a, T, N, G, B, NODE_FIRST, V>
where
    T: Numeric + Ord + Display,
    N: BfsNode<T, V>,
    G: FnMut(
        StateInRegistry,
        &mut StateFunctionCache,
        T,
        &N,
        &Rc<V>,
        Option<T>,
        Option<&N>,
    ) -> Option<N>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    /// Creates a new labeling solver.
    pub fn new(
        input: SearchInput<'a, ResourceLexicographicNode<T, N, NODE_FIRST, V>, V>,
        node_generator: G,
        base_cost_evaluator: B,
        parameters: Parameters<T>,
        labeling_parameters: LabelingParameters,
    ) -> Self {
        let mut time_keeper = parameters
            .time_limit
            .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
        let primal_bound = parameters.primal_bound;
        let get_all_solutions = parameters.get_all_solutions;
        let quiet = parameters.quiet;

        let mut open = BinaryHeap::new();
        let mut registry = StateRegistry::new(input.generator.model.clone());

        if let Some(capacity) = parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let mut solution = Solution::default();
        if let Some(node) = input.node {
            if labeling_parameters.lazy_insertion {
                open.push(Rc::new(node));
            } else {
                let result = registry.insert(node);
                open.push(result.information.unwrap());
            }
            solution.generated += 1;
        } else {
            solution.is_infeasible = true;
        }

        time_keeper.stop();

        let function_cache =
            ParentAndChildStateFunctionCache::new(&input.generator.model.state_functions);

        Self {
            generator: input.generator,
            suffix: input.solution_suffix,
            node_generator,
            base_cost_evaluator,
            primal_bound,
            original_primal_bound: primal_bound,
            get_all_solutions,
            quiet,
            labeling_parameters,
            open,
            registry,
            function_cache,
            applicable_transitions: Vec::new(),
            time_keeper,
            solution,
            phantom: PhantomData,
        }
    }
}

impl<T, N, G, B, const NODE_FIRST: bool, V> Search<T> for Labeling<'_, T, N, G, B, NODE_FIRST, V>
where
    T: Numeric + Ord + Display,
    N: BfsNode<T, V>,
    G: FnMut(
        StateInRegistry,
        &mut StateFunctionCache,
        T,
        &N,
        &Rc<V>,
        Option<T>,
        Option<&N>,
    ) -> Option<N>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        self.time_keeper.start();
        let model = &self.generator.model;

        while let Some(node) = self.open.pop() {
            let node = if self.labeling_parameters.lazy_insertion {
                let result = if self.labeling_parameters.lexicographic_storing {
                    self.registry.insert_rc_lexicographic(node.clone())
                } else {
                    self.registry.insert_rc(node.clone())
                };

                if let Some(node) = result.information {
                    node
                } else {
                    continue;
                }
            } else {
                if node.is_closed() {
                    continue;
                }

                node
            };

            node.close();

            if let Some(dual_bound) = node.bound(model) {
                if exceed_bound(model, dual_bound, self.primal_bound) {
                    if ResourceLexicographicNode::<T, N, NODE_FIRST, V>::ordered_by_bound() {
                        self.open.clear();
                    }
                    continue;
                }

                if ResourceLexicographicNode::<T, N, NODE_FIRST, V>::ordered_by_bound()
                    || self.open.is_empty()
                {
                    self.solution.time = self.time_keeper.elapsed_time();
                    update_bound_if_better(&mut self.solution, dual_bound, model, self.quiet);
                }
            }

            self.function_cache.parent.clear();

            if let Some((cost, suffix)) = get_solution_cost_and_suffix(
                model,
                &*node,
                self.suffix,
                &mut self.base_cost_evaluator,
                &mut self.function_cache,
            ) {
                if !exceed_bound(model, cost, self.primal_bound) {
                    self.primal_bound = Some(cost);
                    let time = self.time_keeper.elapsed_time();
                    update_solution(&mut self.solution, &*node, cost, suffix, time, self.quiet);
                    self.time_keeper.stop();

                    return Ok((self.solution.clone(), self.solution.is_optimal));
                } else if self.get_all_solutions
                    && !exceed_bound(model, cost, self.original_primal_bound)
                {
                    let mut solution = self.solution.clone();
                    let time = self.time_keeper.elapsed_time();
                    update_solution(&mut solution, &*node, cost, suffix, time, true);
                    self.time_keeper.stop();

                    return Ok((solution, false));
                }
                continue;
            }

            if self.time_keeper.check_time_limit(self.quiet) {
                self.solution.time_out = true;
                self.solution.time = self.time_keeper.elapsed_time();
                self.time_keeper.stop();

                return Ok((self.solution.clone(), true));
            }

            self.solution.expanded += 1;
            self.generator.generate_applicable_transitions(
                node.state(),
                &mut self.function_cache.parent,
                &mut self.applicable_transitions,
            );

            for transition in self.applicable_transitions.drain(..) {
                self.function_cache.child.clear();

                if let Some((successor_state, successor_cost)) = model.generate_successor_state(
                    node.state(),
                    &mut self.function_cache.child,
                    node.cost(model),
                    &*transition,
                    None,
                ) {
                    if self.labeling_parameters.lazy_insertion {
                        if self.labeling_parameters.lexicographic_storing {
                            if self
                                .registry
                                .get_lexicographic(&successor_state, successor_cost)
                                .is_some()
                            {
                                continue;
                            }
                        } else if self
                            .registry
                            .get(&successor_state, successor_cost)
                            .is_some()
                        {
                            continue;
                        }

                        if let Some(successor_node) = (self.node_generator)(
                            successor_state,
                            &mut self.function_cache.child,
                            successor_cost,
                            &node.node,
                            &transition,
                            self.primal_bound,
                            None,
                        ) {
                            self.solution.generated += 1;
                            let successor_node =
                                ResourceLexicographicNode::new(successor_node, model.clone());
                            self.open.push(Rc::new(successor_node));
                        }
                    } else {
                        let constructor = |state,
                                           cost,
                                           other: Option<
                            &ResourceLexicographicNode<T, N, NODE_FIRST, V>,
                        >| {
                            (self.node_generator)(
                                state,
                                &mut self.function_cache.child,
                                cost,
                                &node.node,
                                &transition,
                                self.primal_bound,
                                other.map(|node| &node.node),
                            )
                            .map(|node| ResourceLexicographicNode::new(node, model.clone()))
                        };

                        let result = if self.labeling_parameters.lexicographic_storing {
                            self.registry.insert_with_lexicographic(
                                successor_state,
                                successor_cost,
                                constructor,
                            )
                        } else {
                            self.registry
                                .insert_with(successor_state, successor_cost, constructor)
                        };

                        if let Some(successor_node) = result.information {
                            if result.dominated.is_empty() {
                                self.solution.generated += 1;
                            }

                            self.open.push(successor_node);
                        }

                        for dominated in result.dominated {
                            if !dominated.is_closed() {
                                dominated.close();
                            }
                        }
                    }
                }
            }
        }

        self.solution.is_infeasible = self.solution.cost.is_none();
        self.solution.is_optimal = self.solution.cost.is_some();
        self.solution.best_bound = self.solution.cost;
        self.solution.time = self.time_keeper.elapsed_time();
        self.time_keeper.stop();
        Ok((self.solution.clone(), true))
    }
}
