//! Implementation of heapque for float values.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::traits::FloatISeq;

/// Orderable float value.
#[derive(Clone, Copy)]
pub struct OrdFloat<T: FloatISeq> {
    pub value: T,
}

impl<T: FloatISeq> OrdFloat<T> {
    /// Make a float value orderable.
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: FloatISeq> Ord for OrdFloat<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<T: FloatISeq> PartialOrd for OrdFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: FloatISeq> PartialEq for OrdFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
}

impl<T: FloatISeq> Eq for OrdFloat<T> {}

/// Heapque with a fixed capacity.
pub struct HeapCap<T> {
    heap: BinaryHeap<T>,
    capacity: usize,
    full: bool,
}

impl<T: Ord + Copy> HeapCap<T> {
    /// Create HeapCap with a `capacity`.
    pub fn new(capacity: usize) -> Option<Self> {
        if capacity == 0 {
            return None;
        }

        let heap = BinaryHeap::with_capacity(capacity);
        let full = false;

        Some(Self { heap, capacity, full })
    }

    /// Add new elements.
    /// 
    /// When the number of elements are exceeds the capacity, delete the largest element.
    pub fn push(&mut self, value: T) {
        if self.full {
            let greatest = match self.heap.peek() {
                Some(&g) => g,
                _ => panic!(),
            };
            if greatest > value {
                let _ = self.heap.pop();
                self.heap.push(value);
            }
        } else {
            self.heap.push(value);
            if self.heap.len() >= self.capacity {
                self.full = true;
            }
        }
    }

    /// Show the laegest value.
    pub fn peek(&self) -> Option<&T> {
        self.heap.peek()
    }
}