use crate::num::randf;
use elara_log::prelude::*;
use num_traits::{Float, NumAssignOps};
use std::iter::{Product, Sum};
use std::ops::{AddAssign, SubAssign};
use std::{
    fmt::Debug,
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
};
use libblas;

pub mod utils;
use utils::{One, Zero};

pub mod blas;

/// Macro for quickly creating 1D or 2D arrays
#[macro_export]
macro_rules! arr {
    [$([$($x:expr),* $(,)*]),+ $(,)*] => {{
        $crate::NdArray::from_vec2(vec![$([$($x,)*],)*])
    }};
    [$($x:expr),* $(,)*] => {{
        $crate::NdArray::from_vec1(vec![$($x,)*])
    }};
}

/// A general NdArray (multi-dimensional
/// array type)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd)]
pub struct NdArray<T: Clone, const N: usize> {
    pub shape: [usize; N],
    pub data: Vec<T>,
}

impl<T: Clone, const N: usize> NdArray<T, N> {
    /// Creates a new NdArray from an array of
    /// values with a given shape
    pub fn new(array: &[T], shape: [usize; N]) -> Self {
        NdArray {
            shape,
            data: array.to_vec(),
        }
    }

    /// Creates a new NdArray with a `Vec` of
    /// values with a given shape
    pub fn from(array: Vec<T>, shape: [usize; N]) -> Self {
        NdArray { shape, data: array }
    }

    /// Creates a new NdArray filled with
    /// only zeroes
    pub fn zeros(shape: [usize; N]) -> Self
    where
        T: Clone + Zero,
    {
        NdArray {
            shape,
            data: vec![T::zero(); shape.iter().product()],
        }
    }

    /// Creates a new NdArray filled with
    /// only ones
    pub fn ones(shape: [usize; N]) -> Self
    where
        T: Clone + One,
    {
        NdArray {
            shape,
            data: vec![T::one(); shape.iter().product()],
        }
    }

    /// Creates a new NdArray of a shape without
    /// specifying values
    pub fn empty(shape: [usize; N]) -> Self {
        NdArray {
            shape,
            data: Vec::new(),
        }
    }

    pub fn from_vec2(array: Vec<[T; N]>) -> NdArray<T, 2>
    where
        T: Debug,
    {
        let mut shape: [usize; 2] = [0; 2];
        shape[0] = array.len();
        shape[1] = array[0].len();
        let flattened_arr: Vec<T> = array.into_iter().flatten().collect();
        NdArray {
            shape,
            data: flattened_arr,
        }
    }

    /// Finds the number of elements present
    /// in a NdArray
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Allows for iterating through elements
    /// of a NdArray
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn first(&self) -> Option<&T> {
        self.data.first()
    }

    pub fn mapv<B, F>(&self, f: F) -> NdArray<B, N>
    where
        T: Clone,
        F: FnMut(T) -> B,
        B: Clone,
    {
        let data = self.data.clone();
        NdArray {
            data: data.into_iter().map(f).collect(),
            shape: self.shape,
        }
    }

    pub fn mmapv<F>(&mut self, f: F) 
    where
        T: Clone,
        F: FnMut(T) -> T,
    {
        let data = self.data.clone().into_iter().map(f).collect();
        self.data = data;
    }

    pub fn fill(&mut self, val: T) {
        self.data = vec![val; self.shape.iter().product()]
    }

    // Referenced https://codereview.stackexchange.com/questions/256345/n-dimensional-array-in-rust
    fn get_index(&self, idx: &[usize; N]) -> usize {
        let mut i = 0;
        for j in 0..self.shape.len() {
            if idx[j] >= self.shape[j] {
                error!(
                    "[elara-math] Index {} is out of bounds for dimension {} with size {}",
                    idx[j], j, self.shape[j]
                )
            }
            i = i * self.shape[j] + idx[j];
        }
        i
    }

    /// Change the shape of a NdArray
    pub fn reshape(self, shape: [usize; N]) -> NdArray<T, N> {
        if self.len() != shape.iter().product() {
            error!(
                "[elara-math] Cannot reshape into provided shape {:?}",
                shape
            );
        }
        NdArray::from(self.data, shape)
    }

    /// Convert a higher-dimensional NdArray into
    /// a 1D NdArray
    pub fn flatten(self) -> NdArray<T, 1> {
        NdArray {
            data: self.data,
            shape: [self.shape.iter().product(); 1],
        }
    }

    /// Find the dot product of a NdArray with
    /// another NdArray
    pub fn dot(&self, other: &NdArray<T, N>) -> T
    where
        T: Float + NumAssignOps
    {
        if self.len() != other.len() {
            error!("[elara-math] Dot product cannot be found between NdArrays of shape {:?} and {:?}, consider using matmul()",
        self.shape, other.shape)
        }
        libblas::level1::dot(self.len(), &self.data, 1, &other.data, 1)
    }

    /// Create a NdArray from a range of values
    pub fn arange<I: Iterator<Item = T>>(range: I) -> NdArray<T, N> {
        let vec: Vec<T> = range.collect();
        let len = vec.len();
        NdArray::from(vec, [len; N])
    }

    pub fn max(&self) -> T
    where
        T: Ord,
    {
        self.data.iter().max().unwrap().clone()
    }

    pub fn min(&self) -> T
    where
        T: Ord,
    {
        self.data.iter().min().unwrap().clone()
    }

    pub fn sum(&self) -> T
    where
        T: Clone + Sum,
    {
        self.data.iter().cloned().sum()
    }

    pub fn product(&self) -> T
    where
        T: Clone + Product,
    {
        self.data.iter().cloned().product()
    }

    pub fn mean(&self) -> T
    where
        T: Clone + Sum + Div<usize, Output = T>,
    {
        self.sum() / self.len()
    }

    pub fn scaled_add(&mut self, alpha: T, rhs: &NdArray<T, N>) 
    where T: Float + NumAssignOps
    {
        if self.shape != rhs.shape {
            error!("[elara-array] Attempting to add two ndarrays of differing shapes {:?} and {:?}", self.shape, rhs.shape);
        }
        libblas::level1::axpy(self.len(), alpha, &rhs.data, 1, &mut self.data, 1);
    }
}

impl<const N: usize> NdArray<f64, N> {
    /// Creates a new NdArray filled with
    /// random values
    pub fn random(shape: [usize; N]) -> Self {
        // There's GOT to be a more efficient way to do this
        let empty_vec = vec![0.0; shape.iter().product()];
        let data = empty_vec.iter().map(|_| randf()).collect();
        NdArray { shape, data }
    }
}

impl NdArray<f64, 1> {
    /// Creates a equally linearly-spaced vector
    pub fn linspace(x_start: f64, x_end: f64, n_samples: i32) -> NdArray<f64, 1> {
        let dx = (x_end - x_start) / ((n_samples - 1) as f64);
        let vec: Vec<f64> = (0..n_samples).map(|i| x_start + (i as f64) * dx).collect();
        let len = vec.len();
        NdArray::from(vec, [len])
    }

}

impl<T: Clone> NdArray<T, 1> {
    pub fn from_vec1(array: Vec<T>) -> NdArray<T, 1> {
        let shape = [array.len()];
        NdArray { shape, data: array }
    }
}

impl NdArray<f64, 2> {
    /// Finds the matrix product of 2 matrices
    pub fn matmul(&self, b: &NdArray<f64, 2>) -> NdArray<f64, 2> {
        assert_eq!(self.shape[1], b.shape[0]);
        let matmul_vec = blas::dgemm(self.shape[0], self.shape[1], b.shape[1], &self.data, &b.data);
        let shape = [self.shape[0], b.shape[1]];
        NdArray::from(matmul_vec, shape)
    }

    pub fn transpose(&self) -> NdArray<f64, 2> {
        let transpose_vec = blas::transpose(&self.data, self.shape[0], self.shape[1]);
        let mut shape = self.shape;
        shape.reverse();
        NdArray::from(transpose_vec, shape)
    }
}

// Referenced https://codereview.stackexchange.com/questions/256345/n-dimensional-array-in-rust
impl<T: Clone, const N: usize> Index<&[usize; N]> for NdArray<T, N> {
    type Output = T;

    fn index(&self, idx: &[usize; N]) -> &T {
        let i = self.get_index(idx);
        &self.data[i]
    }
}

impl<T: Clone, const N: usize> IndexMut<&[usize; N]> for NdArray<T, N> {
    fn index_mut(&mut self, idx: &[usize; N]) -> &mut T {
        let i = self.get_index(idx);
        &mut self.data[i]
    }
}

macro_rules! impl_binary_ops {
    [$trait:ident, $op_name:ident, $op:tt] => {
        // Elementwise op by reference
        impl<T: Clone + $trait<Output = T>, const N: usize> $trait<&NdArray<T, N>> for &NdArray<T, N> 
        {
            type Output = NdArray<T, N>;

            fn $op_name(self, rhs: &NdArray<T, N>) -> Self::Output {
                if self.shape != rhs.shape {
                    error!("[elara-array] Cannot {} two NdArrays of differing shapes {:?}, {:?}", stringify!($op_name), self.shape, rhs.shape);
                }

                let output_vec = self
                    .data
                    .iter()
                    .zip(&rhs.data)
                    .map(|(a, b)| a.clone() $op b.clone())
                    .collect();

                NdArray::from(output_vec, self.shape)
            }
        }

        // Elementwise ops without reference
        impl<T: Clone + $trait<Output = T>, const N: usize> $trait<NdArray<T, N>> for NdArray<T, N> 
        {

            type Output = NdArray<T, N>;

            fn $op_name(self, rhs: NdArray<T, N>) -> Self::Output {
                &self $op &rhs
            }
        }

        // Scalar ops by reference
        impl<T: Clone + $trait<Output = T>, const N: usize> $trait<T> for &NdArray<T, N> 
        {
            type Output = NdArray<T, N>;
        
            fn $op_name(self, rhs: T) -> Self::Output {
                self.mapv(|a| a $op rhs.clone())
            }
        }

        // Scalar ops without reference
        impl<T: Clone + $trait<Output = T>, const N: usize> $trait<T> for NdArray<T, N> 
        {
            type Output = NdArray<T, N>;
        
            fn $op_name(self, val: T) -> Self::Output {
                &self $op val
            }
        }
    };
}

impl_binary_ops![Add, add, +];
impl_binary_ops![Sub, sub, -];
impl_binary_ops![Mul, mul, *];
impl_binary_ops![Div, div, /];

// Scalar addassign
impl<T: Clone + Add<Output = T>, const N: usize> AddAssign<T> for NdArray<T, N> {
    fn add_assign(&mut self, rhs: T) {
        self.mmapv(|a| a + rhs.clone())
        // let sum_vec = self.data.iter().map(|a| a.clone() + rhs.clone()).collect();
        // self.data = sum_vec;
    }
}


// Elementwise addasign by reference
impl<T: Clone + Add<Output = T>, const N: usize> AddAssign<&NdArray<T, N>> for &mut NdArray<T, N> {
    fn add_assign(&mut self, rhs: &NdArray<T, N>) {
        let sum_vec = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        self.data = sum_vec;
    }
}

// Elementwise addassign without reference
impl<T: Clone + Add<Output = T>, const N: usize> AddAssign<NdArray<T, N>> for NdArray<T, N> {
    fn add_assign(&mut self, rhs: NdArray<T, N>) {
        let sum_vec: Vec<T> = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        self.data = sum_vec;
    }
}

// Elementwise subassign
// recommended not to use subassign, rather use scaled_add(-1.0, rhs) instead
impl<T: Clone + Sub<Output = T>, const N: usize> SubAssign<&NdArray<T, N>> for &mut NdArray<T, N> {
    fn sub_assign(&mut self, rhs: &NdArray<T, N>) {
        let sub_vec = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a.clone() - b.clone())
            .collect();
        self.data = sub_vec;
    }
}

impl<T: Clone + Sub<Output = T>, const N: usize> SubAssign<&NdArray<T, N>> for NdArray<T, N> {
    fn sub_assign(&mut self, rhs: &NdArray<T, N>) {
        let sub_vec = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a.clone() - b.clone())
            .collect();
        self.data = sub_vec;
    }
}

impl<T: Clone + Sub<Output = T>, const N: usize> SubAssign<NdArray<T, N>> for NdArray<T, N> {
    fn sub_assign(&mut self, rhs: NdArray<T, N>) {
        let sub_vec: Vec<T> = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a.clone() - b.clone())
            .collect();
        self.data = sub_vec;
    }
}