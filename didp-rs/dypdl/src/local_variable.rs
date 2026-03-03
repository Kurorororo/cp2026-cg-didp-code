use crate::util;
use crate::ModelErr;
use rustc_hash::FxHashMap;

/// Local variable bound to an element in a set.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct LocalVariable(usize);

impl LocalVariable {
    /// Returns the index.
    #[inline]
    pub fn id(&self) -> usize {
        self.0
    }
}

/// Data for local variables.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct LocalVariableData {
    /// Names of local variables.
    pub names: Vec<String>,
    /// Mapping from a local variable name to its ID.
    pub name_to_id: FxHashMap<String, usize>,
}

impl LocalVariableData {
    /// Adds a local variable;
    pub fn add<T>(&mut self, name: T) -> Result<LocalVariable, ModelErr>
    where
        String: From<T>,
    {
        let id = util::add_name(name, &mut self.names, &mut self.name_to_id)?;

        Ok(LocalVariable(id))
    }

    /// Returns a local variable given a name.
    pub fn get(&self, name: &str) -> Result<LocalVariable, ModelErr> {
        let id = util::get_id(name, &self.name_to_id)?;

        Ok(LocalVariable(id))
    }

    /// Returns the name of local variables .
    pub fn number_of_variables(&self) -> usize {
        self.names.len()
    }
}
