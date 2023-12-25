use crate::num::randf;
use elara_log::prelude::*;
use num_traits::{Float, NumAssignOps};
use std::fmt::Display;
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

    /// Creates an allocated new NdArray
    /// without filling it with values
    pub fn empty(shape: [usize; N]) -> Self {
    	NdArray {
    		shape,
    		data: Vec::with_capacity(shape.iter().product())
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

    /// Returns the last element of an NdArray
    pub fn last(&self) -> Option<&T> {
        self.data.last()
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

    /// Returns the shape of an NdArray
    pub fn shape(&self) -> [usize; N] {
        self.shape
    }

    // Referenced https://codereview.stackexchange.com/questions/256345/n-dimensional-array-in-rust
    fn get_index(&self, idx: &[usize]) -> usize {
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

    /// Find number of dimensions of an NdArray
    pub fn ndims(&self) -> usize {
        self.shape.len()
    }

    /// Performs the operation `self[i]`
    pub fn index(&self, idx: usize) -> &[T] {
        let mut start_index_slice = vec![0; self.ndims()];
        let mut end_index_slice = vec![0; self.ndims()];
        start_index_slice[0] = idx;
        for i in 1..self.ndims() {
            end_index_slice[i] = self.shape[i] - 1;
        }
        let start_index = self.get_index(start_index_slice.as_slice());
        let end_index = self.get_index(end_index_slice.as_slice());
        &self.data[start_index..end_index + 1]
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

// Note! This is unreliable at the moment, use Display ("{}") for more consistent
// print formatting
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

fn _print_arrays<T: Clone + Debug, const N: usize>(f: &mut std::fmt::Formatter<'_>, array: &NdArray<T, N>) -> std::fmt::Result {
    match array.shape.len() {
        1 => {
            // f.write_str(format!("{:?}", array.data).as_str())?;
            f.write_str("[")?;
            let row_elements = 4;
            let debug_rows = array.len() / row_elements;
            for i in 0..debug_rows {
                let axisindent = if i == 0 { 0 } else { 9 };
                let num_commas = if i == debug_rows - 1 { 0 } else { 1 };
                let num_newlines = if i == debug_rows - 1 { 0 } else { 1 };
                let slice = &array.data[(i * row_elements)..(i * row_elements + row_elements)];
                let slice_str = slice.into_iter().map(|el| format!("{:?}", el)).collect::<Vec<String>>().join(", ");
                f.write_str(" ".repeat(axisindent).as_str())?;
                f.write_str(slice_str.as_str())?;
                f.write_str(",".repeat(num_commas).as_str())?;
                f.write_str("\n".repeat(num_newlines).as_str())?;
            }
            let last_slice = &array.data[(debug_rows * row_elements)..(debug_rows * row_elements + array.len() % row_elements)];
            let last_slice_str = last_slice.into_iter().map(|el| format!("{:?}", el)).collect::<Vec<String>>().join(", ");
            f.write_str(last_slice_str.as_str())?;
            f.write_str("]")?;
        },
        2 => {
            f.write_str("[")?;
            for row in 0..array.shape[0] {
                let axisindent = if row == 0 { 0 } else { 9 };
                let newlines = if row == array.shape[0] - 1 {0 } else { 1 };
                f.write_str(" ".repeat(axisindent).as_str())?;
                let start_index = array.get_index(&[row, 0]); // [0][0]
                let end_index = array.get_index(&[row, array.shape[1] - 1]); // 
                f.write_str(format!("{:?}", &array.data[start_index..end_index + 1]).as_str())?;
                f.write_str(",\n".repeat(newlines).as_str())?;
            }
            f.write_str("]")?;
        },
        3 => {
            f.write_str("[")?;
            for col in 0..array.shape[1] {
                let outer_newlines = if col == array.shape[1] - 1 { 0 } else { 2 };
                let outer_axisindent = if col == 0 { 0 } else { 9 };
                f.write_str(" ".repeat(outer_axisindent).as_str())?;
                f.write_str("[")?;
                let start_index = array.get_index(&[col, 0, 0]);
                let end_index = array.get_index(&[col, array.shape[1] - 1, array.shape[2] - 1]);
                let slice = &array.data[start_index..end_index + 1];
                for row in 0..array.shape[1] {
                    let axisindent = if row == 0 { 0 } else { 10 };
                    let newlines = if row == array.shape[1] - 1 {0 } else { 1 };
                    f.write_str(" ".repeat(axisindent).as_str())?;
                    let start_index_2d = row * array.shape[2];
                    let end_index_2d = row * array.shape[2] + array.shape[2];
                    f.write_str(format!("{:?}", &slice[start_index_2d..end_index_2d]).as_str())?;
                    f.write_str(",\n".repeat(newlines).as_str())?;
                }
                f.write_str("]")?;
                f.write_str("\n".repeat(outer_newlines).as_str())?;
            }
            f.write_str("]")?;
        }
        _ => {
            let shape_str: String = array.shape.into_iter().map(|el| el.to_string()).collect::<Vec<String>>().join(" x ");
            f.write_str(format!("[Array {}]", shape_str).as_str())?;
        }
    }
    Ok(())
}

// fn display_inner_new<T: Clone + Debug, const N: usize>(f: &mut fmt::Formatter<'_>, array: &NdArray<T, N>) -> std::fmt::Result {
//     f.write_str(data)
// }

impl<T: Clone, const N: usize> Debug for NdArray<T, N> 
where T: Debug
{ 
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.shape.len() {
            0 => write!(f, "NdArray([])"),
            _ => {
                f.write_str("NdArray(")?;
                _print_arrays(f, &self)?;
                f.write_str(format!(", shape={:?})\n", self.shape).as_str())?;
                Ok(())
            }
        }
    }
}

impl<T: Clone, const N: usize> Display for NdArray<T, N>
where T: Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NdArray({:?}, shape={:?})", self.data, self.shape)
    }
}

impl NdArray<f64, 1> {
    /// Creates a equally linearly-spaced vector
    pub fn linspace(x_start: f64, x_end: f64, n_samples: usize) -> NdArray<f64, 1> {
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

impl<T: Clone> NdArray<T, 2> {
    /// Indexes the rows of a 2D matrix NdArray and returns `&[T]`
    // pub fn index(&self, idx: usize) -> &[T] {
    //     let shape = self.shape;
    //     if idx >= self.shape[0] {
    //         error!("[elara-array] Attempting to index 2D matrix with index {} (equivalent to row {}) when the matrix only has {:?} rows", idx, idx + 1, shape[1]);
    //     }
    //     &self.data[(shape[1] * idx)..(shape[1] * idx + shape[1])]
    // }

    /// Indexes the rows of a 2D matrix NdArray and returns `Vec<T>`
    pub fn index_vec(&self, idx: usize) -> Vec<T> {
        let shape = self.shape;
        if idx >= self.shape[0] {
            error!("[elara-array] Attempting to index 2D matrix with index {} (equivalent to row {}) when the matrix only has {:?} rows", idx, idx + 1, shape[1]);
        }
        self.data[(shape[1] * idx)..(shape[1] * idx + shape[1])].iter().map(|el| el.clone()).collect()
    }
    
    /// Indexes the rows of a 2D matrix NdArray and returns a
    /// flat NdArray
    pub fn index_view(&self, idx: usize) -> NdArray<T, 1> {
        NdArray::from_vec1(self.index_vec(idx))
    }

    /// Assigns a provided slice of values to a row of a 
    /// 2D matrix NdArray
    pub fn assign(&mut self, idx: usize, contents: &[T]) {
        let shape = self.shape;
        if idx >= self.shape[0] {
            error!("[elara-array] Attempting to index 2D matrix with index {} (equivalent to row {}) when the matrix only has {:?} rows", idx, idx + 1, shape[1]);
        }
        if contents.len() > self.shape[1] {
            error!("[elara-array] Tried to assign a slice with {} elements (equivalent to {} columns) when the matrix only has {} columns", contents.len(), contents.len() + 1, self.shape[1]);
        }
        for i in 0..contents.len() {
            self[&[idx, i]] = contents[i].clone()
        }
    }

    /// Assigns a provided 1D NdArray of values to a row of a 
    /// 2D matrix NdArray
    pub fn assign_view(&mut self, idx: usize, contents: NdArray<T, 1>) {
        let shape = self.shape;
        if idx >= self.shape[0] {
            error!("[elara-array] Attempting to index 2D matrix with index {} (equivalent to row {}) when the matrix only has {:?} rows", idx, idx + 1, shape[1]);
        }
        if contents.len() > self.shape[1] {
            error!("[elara-array] Tried to assign a slice with {} elements (equivalent to {} columns) when the matrix only has {} columns", contents.len(), contents.len() + 1, self.shape[1]);
        }

        for i in 0..contents.len() {
            self[&[idx, i]] = contents[i].clone()
        }
    }

	/// Gets the ith column of a 2D NdArray
    pub fn index_column(&self, idx: usize) -> NdArray<T, 1> {
    	let column_len = self.shape[0];
    	let mut out = NdArray::empty([column_len]);
    	for j in 0..column_len {
    		out[j] = self[&[idx, j]].clone();
    	}
    	out
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

/// Convenience method for indexing 1D NdArrays
impl<T: Clone> Index<usize> for NdArray<T, 1> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self[&[index]]
    }
}

/// Convenience method for mutably indexing 1D NdArrays
impl<T: Clone> IndexMut<usize> for NdArray<T, 1> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self[&[index]]
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
        /// `&NdArray` and `NdArray`
        impl<T: Clone + $trait<Output = T>, const N: usize> $trait<NdArray<T, N>> for &NdArray<T, N>
        {
            type Output = NdArray<T, N>;

            fn $op_name(self, rhs: NdArray<T, N>) -> Self::Output {
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
