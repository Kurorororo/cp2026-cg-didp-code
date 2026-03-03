/// A local environment mapping a local element variable index to a value.
#[derive(Debug, Default, Clone)]
pub struct LocalEnvironment {
    id_to_value: Vec<Option<usize>>,
}

impl LocalEnvironment {
    /// Sets the value of a local element variable.
    pub fn set(&mut self, id: usize, value: usize) -> Option<usize> {
        if id >= self.id_to_value.len() {
            self.id_to_value.resize(id + 1, None);
        }

        let before = self.id_to_value[id];
        self.id_to_value[id] = Some(value);

        before
    }

    /// Unsets the value of a local element variable.
    pub fn unset(&mut self, id: usize) -> Option<usize> {
        if id >= self.id_to_value.len() {
            None
        } else {
            let before = self.id_to_value[id];
            self.id_to_value[id] = None;

            before
        }
    }

    /// Gets the value of a local element variable.
    pub fn get(&self, id: usize) -> Option<usize> {
        if id >= self.id_to_value.len() {
            None
        } else {
            self.id_to_value[id]
        }
    }
}
