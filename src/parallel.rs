#[cfg(not(feature = "rayon"))]
use std::cmp::Ordering;
#[cfg(not(feature = "rayon"))]
use std::slice::{Iter, IterMut};

#[cfg(feature = "rayon")]
pub use rayon::prelude::*;
#[cfg(not(feature = "rayon"))]
use slotmap::SlotMap;

#[cfg(not(feature = "rayon"))]
pub trait OptionalParIter: IntoIterator {
    fn into_par_iter(self) -> Self::IntoIter
    where
        Self: Sized,
    {
        self.into_iter()
    }

    fn par_bridge(self) -> Self
    where
        Self: Sized,
    {
        self
    }

    fn collect_into_vec(self, vec: &mut Vec<Self::Item>)
    where
        Self: Sized,
    {
        vec.extend(self);
    }
}

#[cfg(not(feature = "rayon"))]
pub trait OptionalMutParIter<'a, T> {
    fn par_iter_mut(&'a mut self) -> IterMut<'a, T>
    where
        Self: 'a;

    fn par_sort_by<F>(&'a mut self, compare: F)
    where
        Self: 'a,
        F: FnMut(&T, &T) -> Ordering;

    fn par_sort(&'a mut self)
    where
        Self: 'a,
        T: std::cmp::Ord;
}

#[cfg(not(feature = "rayon"))]
impl<'a, T> OptionalMutParIter<'a, T> for Vec<T> {
    fn par_iter_mut(&'a mut self) -> IterMut<'a, T>
    where
        Self: 'a,
    {
        self.iter_mut()
    }

    fn par_sort_by<F>(&'a mut self, compare: F)
    where
        Self: 'a,
        F: FnMut(&T, &T) -> Ordering,
    {
        self.sort_by(compare)
    }

    fn par_sort(&'a mut self)
    where
        Self: 'a,
        T: std::cmp::Ord,
    {
        self.sort();
    }
}

#[cfg(not(feature = "rayon"))]
pub trait ParIter<'a, T> {
    fn par_iter(&'a self) -> Iter<'a, T>
    where
        Self: 'a;
}

#[cfg(not(feature = "rayon"))]
impl<'a, T> ParIter<'a, T> for Vec<T> {
    fn par_iter(&'a self) -> Iter<'a, T>
    where
        Self: 'a,
    {
        self.iter()
    }
}

#[cfg(not(feature = "rayon"))]
pub trait SlotMapMutParIter<'a, T, K: slotmap::Key> {
    fn par_iter_mut(&'a mut self) -> slotmap::basic::IterMut<'a, K, T>
    where
        Self: 'a;
}

#[cfg(not(feature = "rayon"))]
impl<'a, T, K: slotmap::Key> SlotMapMutParIter<'a, T, K> for SlotMap<K, T> {
    fn par_iter_mut(&'a mut self) -> slotmap::basic::IterMut<'a, K, T>
    where
        Self: 'a,
    {
        self.iter_mut()
    }
}

#[cfg(not(feature = "rayon"))]
impl<I: IntoIterator> OptionalParIter for I {}
