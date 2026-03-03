use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::{
    CostNode, FNode, Labeling, LabelingParameters, Parameters, Search, SearchInput,
    SuccessorGenerator,
};
use crate::caasdy;
use crate::search_algorithm::data_structure::{
    CreateTransitionChain, RcChain, ResourceLexicographicNode,
};
use crate::search_algorithm::FNodeEvaluators;
use dypdl::{variable_type, StateFunctionCache, Transition};
use std::fmt;
use std::rc::Rc;
use std::str;

/// Creates a labeling solver.
pub fn create_labeling<T>(
    model: Rc<dypdl::Model>,
    parameters: Parameters<T>,
    f_evaluator_type: FEvaluatorType,
    labeling_parameters: LabelingParameters,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let metadata = &model.state_metadata;

    if metadata.set_resource_variable_names.is_empty()
        && metadata.element_resource_variable_names.is_empty()
        && metadata.integer_resource_variable_names.is_empty()
        && metadata.continuous_resource_variable_names.is_empty()
    {
        return caasdy::create_caasdy(model, parameters, f_evaluator_type);
    }

    let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
    let base_cost_evaluator = move |cost, base_cost| f_evaluator_type.eval(cost, base_cost);
    let cost = match f_evaluator_type {
        FEvaluatorType::Plus => T::zero(),
        FEvaluatorType::Product => T::one(),
        FEvaluatorType::Max => T::min_value(),
        FEvaluatorType::Min => T::max_value(),
        FEvaluatorType::Overwrite => T::zero(),
    };

    let state = model.target.clone();
    let mut cache = StateFunctionCache::new(&model.state_functions);

    if model.has_dual_bounds() {
        let h_model = model.clone();
        let h_evaluator = move |state: &_, cache: &mut _| h_model.eval_dual_bound(state, cache);
        let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
        let node = FNode::generate_root_node(
            state,
            &mut cache,
            cost,
            &generator.model,
            &h_evaluator,
            &f_evaluator,
            parameters.primal_bound,
        );
        let generator_model = model.clone();
        let node_generator = move |state,
                                   cache: &mut _,
                                   cost,
                                   parent: &FNode<_>,
                                   transition: &Rc<Transition>,
                                   primal_bound,
                                   other: Option<&_>| {
            let evaluators = FNodeEvaluators {
                h: &h_evaluator,
                f: &f_evaluator,
            };

            if let Some((h, f)) = FNode::evaluate_state(
                &state,
                cache,
                cost,
                &generator_model,
                evaluators,
                primal_bound,
                other,
            ) {
                let transition_chain = Rc::new(RcChain::new(
                    parent.transition_chain().clone(),
                    transition.clone(),
                ));
                let node = CostNode::new(state, cost, &generator_model, Some(transition_chain));
                Some(FNode::with_node_and_h_and_f(node, h, f))
            } else {
                None
            }
        };

        if labeling_parameters.node_first {
            let node = node.map(|node| ResourceLexicographicNode::<_, _, true>::new(node, model));
            let input = SearchInput {
                node,
                generator,
                solution_suffix: &[],
            };

            Box::new(Labeling::new(
                input,
                node_generator,
                base_cost_evaluator,
                parameters,
                labeling_parameters,
            ))
        } else {
            let node = node.map(|node| ResourceLexicographicNode::<_, _, false>::new(node, model));
            let input = SearchInput {
                node,
                generator,
                solution_suffix: &[],
            };

            Box::new(Labeling::new(
                input,
                node_generator,
                base_cost_evaluator,
                parameters,
                labeling_parameters,
            ))
        }
    } else {
        let node = CostNode::generate_root_node(state, cost, &generator.model);
        let generator_model = model.clone();
        let node_generator = move |state,
                                   _: &mut _,
                                   cost,
                                   parent: &CostNode<_>,
                                   transition: &Rc<Transition>,
                                   _,
                                   _: Option<&_>| {
            let transition_chain = Rc::new(RcChain::new(
                parent.transition_chain().clone(),
                transition.clone(),
            ));
            Some(CostNode::new(
                state,
                cost,
                &generator_model,
                Some(transition_chain),
            ))
        };

        if labeling_parameters.node_first {
            let node = ResourceLexicographicNode::<_, _, true>::new(node, model);
            let input = SearchInput {
                node: Some(node),
                generator,
                solution_suffix: &[],
            };
            Box::new(Labeling::new(
                input,
                node_generator,
                base_cost_evaluator,
                parameters,
                labeling_parameters,
            ))
        } else {
            let node = ResourceLexicographicNode::<_, _, false>::new(node, model);
            let input = SearchInput {
                node: Some(node),
                generator,
                solution_suffix: &[],
            };
            Box::new(Labeling::new(
                input,
                node_generator,
                base_cost_evaluator,
                parameters,
                labeling_parameters,
            ))
        }
    }
}
