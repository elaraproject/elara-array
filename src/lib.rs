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