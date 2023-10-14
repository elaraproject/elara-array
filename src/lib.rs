#![warn(missing_docs)]
//! `elara-array` is a numerical and scientific
//! computation library that aims to offer
//! a fast, reliable, easy-to-use array type
//! for the [Elara Project's](https://github.com/elaraproject) 
//! computational needs
//! 
//! ## Hello World
//! 
//! ```rust 
//! use elara_array::prelude::*;
//! 
//! fn main() {
//!     let array = arr![[1.0, 2.0], [3.0, 4.0]];
//!     println!("{:?}", array);
//! }
//! ```
//! 
//! ## Important notes on usage
//! 
//! `elara-array` heavily integrates with the
//! `elara-log` library. While it can be used
//! without `elara-log`, this means all of its
//! errors will be silenced. To avoid this
//! from happening, in any project that uses
//! `elara-array`, do not forget to add the
//! following:
//! 
//! ```rust
//! use elara_log::prelude::*;
//! 
//! fn main() {
//!     Logger::new().init().unwrap();
//! }
//! ```

/// An n-dimensional data container, similar
/// to a NumPy array
pub mod array;
/// Various useful constants and numerical
/// functions
pub mod num;
/// Prelude for the library
pub mod prelude;

pub use array::NdArray;

/// A basic Runge-Kutta 4th order solver
pub fn rk4(f: &dyn Fn(NdArray<f64, 1>, f64) -> NdArray<f64, 1>, u0: &[f64], t0: f64, tf: f64, samples: usize) -> (NdArray<f64, 2>, NdArray<f64, 1>) {
	let mut u: NdArray<f64, 2> = NdArray::zeros([samples, u0.len()]);
    let t = NdArray::linspace(t0, tf, samples);
    for i in 0..u0.len() {
        u[&[0, i]] = u0[i];
    }
    let h = (tf - t0) / samples as f64;
    for i in 0..(samples - 1) {
        let k1 = f(u.index_view(i), t[i]) * h;
        let k2 = f(u.index_view(i) + &k1 * 0.5, t[i] + 0.5 * h) * h;
        let k3 = f(u.index_view(i) + &k2 * 0.5, t[i] + 0.5 * h) * h;
        let k4 = f(u.index_view(i) + &k3, t[i] + h) * h;
        for j in 0..u0.len() {
            u[&[i + 1, j]] = u[&[i, j]] + (k1[j] + 2.0 * (k2[j] + k3[j]) + k4[j]) / 6.0;
        }
    }
    (u, t)
}
