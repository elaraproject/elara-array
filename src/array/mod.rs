use crate::num::randf;
use elara_log::prelude::*;
use num_traits::{Float, NumAssignOps};
use std::iter::{Product, Sum};
use std::ops::{AddAssign, SubAssign};
use std::{
    fmt,
    fmt::Debug,
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
};
// use crate::num::{min, max};
use std::cmp::{min, max};
use libblas;

mod utils;
use utils::{One, Zero};

mod blas;

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
#[derive(Clone, PartialEq, Eq, PartialOrd)]
pub struct NdArray<T: Clone, const N: usize> {
    shape: [usize; N],
    data: Vec<T>,
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

    /// Create a new NdArray from a nested vector.
    /// This is usually not used directly; instead,
    /// use the macro [`arr!`] instead.
    /// 
    /// [`arr`]: #macro.arr
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
    /// of an NdArray
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Returns the first element of an NdArray
    pub fn first(&self) -> Option<&T> {
        self.data.first()
    }

    /// Perform an operation on every element of
    /// a NdArray and return a new array correspondingly
    pub fn mapv<B, F>(&self, mut f: F) -> NdArray<B, N>
    where
        T: Clone,
        F: FnMut(T) -> B,
        B: Clone,
    {
        NdArray {
            data: self.data.iter().map(move |x| f(x.clone())).collect(),
            shape: self.shape,
        }
    }

    /// Performs [`NdArray::mapv`] and mutates the NdArray
    /// directly instead of returning a new NdArray
    pub fn mapv_inplace<F>(&mut self, mut f: F) 
    where
        T: Clone,
        F: FnMut(T) -> T,
    {
        let data = self.data.iter().map(move |x| f(x.clone())).collect();
        self.data = data;
    }

    /// Fill an NdArray with a value
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

    /// Find the largest element in an NdArray
    pub fn max(&self) -> Option<&T>
    where
        T: Ord,
    {
        self.data.iter().max()
    }

    /// Find the smallest element in an NdArray
    pub fn min(&self) -> Option<&T>
    where
        T: Ord,
    {
        self.data.iter().min()
    }

    /// Sum an NdArray
    pub fn sum(&self) -> T
    where
        T: Clone + Sum,
    {
        self.data.iter().cloned().sum()
    }

    /// Take the product of an NdArray
    pub fn product(&self) -> T
    where
        T: Clone + Product,
    {
        self.data.iter().cloned().product()
    }

    /// Find the mean of an NdArray
    pub fn mean(&self) -> T
    where
        T: Clone + Sum + Div<usize, Output = T>,
    {
        self.sum() / self.len()
    }

    /// Perform the operation `self += alpha * rhs`
    /// on an NdArray
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

fn _display_inner<T: Clone + Debug, const N: usize>(f: &mut fmt::Formatter<'_>, array: &NdArray<T, N>, axis: usize, offset: usize) -> std::fmt::Result {
    // Source: https://stackoverflow.com/questions/76735487/implementation-of-debug-for-ndarray-causes-subtraction-underflow-error
    let axisindent = min(2, max(0, array.shape.len().saturating_sub(axis + 1)));
    if axis < array.shape.len() {
        f.write_str("[")?;
        for (k_index, k) in (0..array.shape[axis]).into_iter().enumerate() {
            if k_index > 0 {
                for _ in 0..axisindent {
                    f.write_str("\n         ")?;
                    for _ in 0..axis {
                        f.write_str(" ")?;
                    }
                }
            }
            let offset_ = offset + k;
            _display_inner(f, array, axis + 1, offset_)?;
            if k_index < &array.shape[axis] - 1 {
                f.write_str(", ")?;
            }
        }
        f.write_str("]")?;
    } else {
        f.write_str(format!("{:?}", array.data[offset]).as_str())?;
    }
    Ok(())
}

impl<T: Clone, const N: usize> Debug for NdArray<T, N> 
where T: Debug
{ 
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.shape.len() {
            0 => write!(f, "NdArray([])"),
            1 => write!(f, "NdArray({:?}, shape={:?})", self.data, self.shape),
            _ => {
                f.write_str("NdArray(")?;
                _display_inner(f, &self, 0, 0)?;
                f.write_str(format!(", shape={:?})", self.shape).as_str())?;
                Ok(())
            }
        }
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
    /// Create an NdArray from a 1D vector; this is
    /// usually not used directly, but instead is
    /// called from [`arr!`]
    /// 
    /// [`arr!`]: #macro.arr
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

    /// Transposes a matrix
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
    [$trait:ident, $op_name:ident, $op:tt, $doc_type:expr] => {
        /// Performs elementwise
        #[doc=$doc_type]
        /// between
        /// `&NdArray` and `&NdArray`
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

        /// Performs elementwise
        #[doc=$doc_type]
        /// between
        /// `NdArray` and `&NdArray`
        impl<T: Clone + $trait<Output = T>, const N: usize> $trait<&NdArray<T, N>> for NdArray<T, N>
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

        /// Performs elementwise
        #[doc=$doc_type]
        /// between
        /// `NdArray` and `NdArray`
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

impl_binary_ops![Add, add, +, "addition"];
impl_binary_ops![Sub, sub, -, "subtraction"];
impl_binary_ops![Mul, mul, *, "multiplication"];
impl_binary_ops![Div, div, /, "division"];

// Scalar addassign
impl<T: Clone + Add<Output = T>, const N: usize> AddAssign<T> for NdArray<T, N> {
    fn add_assign(&mut self, rhs: T) {
        self.mapv_inplace(|a| a + rhs.clone())
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