use anyhow::{Context, Result};
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis, RemoveAxis};
use std::collections::HashMap;
use std::hash::Hash;

/// switch the key and value
#[extend::ext(name = HashMapExt)]
pub impl<K, V> HashMap<K, V> {
    fn invert(&self) -> HashMap<V, K>
    where
        K: Clone,
        V: Clone + Hash + Eq,
    {
        self.iter().map(|(k, v)| (v.clone(), k.clone())).collect()
    }
}
