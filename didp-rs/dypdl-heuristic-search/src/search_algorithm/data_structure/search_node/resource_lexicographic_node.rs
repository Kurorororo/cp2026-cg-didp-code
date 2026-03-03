use crate::search_algorithm::data_structure::HashableSignatureVariables;

use super::super::state_registry::{StateInRegistry, StateInformation};
use super::super::transition_chain::GetTransitions;
use super::super::util;
use super::BfsNode;
use delegate::delegate;
use dypdl::variable_type::Numeric;
use dypdl::{Model, State, Transition, TransitionInterface};
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct ResourceLexicographicNode<
    T,
    N,
    const NODE_FIRST: bool = false,
    V = Transition,
    R = Rc<V>,
    K = Rc<HashableSignatureVariables>,
    M = Rc<Model>,
> {
    pub node: N,
    model: M,
    _phantom: PhantomData<(T, V, R, K)>,
}

impl<T, N, const NODE_FIRST: bool, V, R, K, M>
    ResourceLexicographicNode<T, N, NODE_FIRST, V, R, K, M>
{
    pub fn new(node: N, model: M) -> Self {
        Self {
            node,
            model,
            _phantom: PhantomData,
        }
    }
}

impl<T, N, const NODE_FIRST: bool, V, R, K, M> PartialEq
    for ResourceLexicographicNode<T, N, NODE_FIRST, V, R, K, M>
where
    N: StateInformation<T, K> + PartialEq,
    T: Numeric,
    K: Hash + Eq + Clone + Debug,
    M: Deref<Target = Model>,
{
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
            && self.node.state().resource_variables == other.node.state().resource_variables
    }
}

impl<T, N, const NODE_FIRST: bool, V, R, K, M> Eq
    for ResourceLexicographicNode<T, N, NODE_FIRST, V, R, K, M>
where
    N: StateInformation<T, K> + Eq,
    T: Numeric,
    K: Hash + Eq + Clone + Debug,
    M: Deref<Target = Model>,
{
}

impl<T, N, const NODE_FIRST: bool, V, R, K, M> Ord
    for ResourceLexicographicNode<T, N, NODE_FIRST, V, R, K, M>
where
    N: StateInformation<T, K> + Ord,
    T: Numeric + PartialEq,
    K: Hash + Eq + Clone + Debug + Deref<Target = HashableSignatureVariables>,
    StateInRegistry<K>: From<State>,
    M: Deref<Target = Model>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        if NODE_FIRST {
            let result = self.node.cmp(&other.node);

            if result != Ordering::Equal {
                return result;
            }

            util::lexicographic_resource_variables_cmp(
                &self.model.state_metadata,
                self.node.state(),
                other.node.state(),
            )
        } else {
            let result = util::lexicographic_resource_variables_cmp(
                &self.model.state_metadata,
                self.node.state(),
                other.node.state(),
            );

            if result != Ordering::Equal {
                return result;
            }

            self.node.cmp(&other.node)
        }
    }
}

impl<T, N, const NODE_FIRST: bool, V, R, K, M> PartialOrd
    for ResourceLexicographicNode<T, N, NODE_FIRST, V, R, K, M>
where
    N: StateInformation<T, K> + Ord,
    T: Numeric,
    K: Hash + Eq + Clone + Debug + Deref<Target = HashableSignatureVariables>,
    StateInRegistry<K>: From<State>,
    M: Deref<Target = Model>,
    ResourceLexicographicNode<T, N, NODE_FIRST, V, R, K, M>: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, N, const NODE_FIRST: bool, V, R, K, M> StateInformation<T, K>
    for ResourceLexicographicNode<T, N, NODE_FIRST, V, R, K, M>
where
    N: StateInformation<T, K> + Eq,
    T: Numeric,
    K: Hash + Eq + Clone + Debug,
{
    delegate! {
        to self.node {
            fn state(&self) -> &StateInRegistry<K>;
            fn state_mut(&mut self) -> &mut StateInRegistry<K>;
            fn cost(&self, model: &Model) -> T;
            fn bound(&self, model: &Model) -> Option<T>;
            fn is_closed(&self) -> bool;
            fn close(&self);
        }
    }
}

impl<T, N, const NODE_FIRST: bool, V, R, K, M> GetTransitions<V>
    for ResourceLexicographicNode<T, N, NODE_FIRST, V, R, K, M>
where
    N: GetTransitions<V>,
{
    delegate! {
        to self.node {
            fn transitions(&self) -> Vec<V>;
            fn last(&self) -> Option<&V>;
        }
    }
}

impl<T, N, const NODE_FIRST: bool, V, R, K, M> BfsNode<T, V, K>
    for ResourceLexicographicNode<T, N, NODE_FIRST, V, R, K, M>
where
    N: BfsNode<T, V, K>,
    T: Numeric + Display,
    V: TransitionInterface + Clone,
    K: Hash + Eq + Clone + Debug + Deref<Target = HashableSignatureVariables>,
    StateInRegistry<K>: From<State>,
    M: Deref<Target = Model>,
    ResourceLexicographicNode<T, N, NODE_FIRST, V, R, K, M>: Ord,
{
    #[inline]
    fn ordered_by_bound() -> bool {
        NODE_FIRST && N::ordered_by_bound()
    }
}

#[cfg(test)]
mod tests {
    use crate::search_algorithm::FNode;

    use super::super::CostNode;
    use dypdl::ResourceVariables;

    use super::*;

    #[test]
    fn test_state_information() {
        let model = Model::default();
        let model = Rc::new(model);
        let node = ResourceLexicographicNode::<_, _>::new(
            CostNode::<_>::new(StateInRegistry::default(), 0, &model, None),
            model.clone(),
        );
        assert_eq!(node.state(), &StateInRegistry::default());
        assert_eq!(node.cost(&model), 0);
        assert_eq!(node.bound(&model), None);
        assert!(!node.is_closed());
        node.close();
        assert!(node.is_closed());
    }

    #[test]
    fn test_get_transitions() {
        let model = Model::default();
        let model = Rc::new(model);
        let node = ResourceLexicographicNode::<i32, CostNode<_>>::new(
            CostNode::<_>::new(StateInRegistry::default(), 0, &model, None),
            model,
        );
        assert_eq!(node.transitions(), vec![]);
        assert_eq!(node.last(), None);
    }

    #[test]
    fn test_first_ordering() {
        let mut model = Model::default();

        let iv1 = model.add_integer_resource_variable("iv1", false, 0);
        assert!(iv1.is_ok());

        let iv2 = model.add_integer_resource_variable("iv2", true, 0);
        assert!(iv2.is_ok());

        let model = Rc::new(model);

        let state1 = StateInRegistry {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, -1],
                ..Default::default()
            },
            ..Default::default()
        };
        let node1 = ResourceLexicographicNode::<i32, FNode<_>>::new(
            FNode::with_node_and_h_and_f(CostNode::<_>::new(state1, 2, &model, None), 2, 4),
            model.clone(),
        );

        let state2 = StateInRegistry {
            resource_variables: ResourceVariables {
                integer_variables: vec![0, -2],
                ..Default::default()
            },
            ..Default::default()
        };
        let node2 = ResourceLexicographicNode::<i32, FNode<_>>::new(
            FNode::with_node_and_h_and_f(CostNode::<_>::new(state2, 0, &model, None), 0, 0),
            model.clone(),
        );

        let state3 = StateInRegistry {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, -1],
                ..Default::default()
            },
            ..Default::default()
        };
        let node3 = ResourceLexicographicNode::<i32, FNode<_>>::new(
            FNode::with_node_and_h_and_f(CostNode::<_>::new(state3, 3, &model, None), 0, 3),
            model.clone(),
        );

        let state4 = StateInRegistry {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, -1],
                ..Default::default()
            },
            ..Default::default()
        };
        let node4 = ResourceLexicographicNode::<i32, FNode<_>>::new(
            FNode::with_node_and_h_and_f(CostNode::<_>::new(state4, 2, &model, None), 1, 3),
            model.clone(),
        );

        assert!(node1 == node1);
        assert!(node1 >= node2);
        assert!(node1 >= node3);
        assert!(node1 >= node4);
    }

    #[test]
    fn test_ordered_by_bound() {
        assert!(!ResourceLexicographicNode::<i32, CostNode<i32>>::ordered_by_bound(),);
    }
}
