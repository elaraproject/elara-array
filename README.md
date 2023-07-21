# elara-array

This crate provides a fast, n-dimensional array for Project Elara with minimal dependencies. Note: this crate was once part of [`elara-math`](https://github.com/elaraproject/elara-math), but has been extracted into a standalone crate for separate development.

## Basic usage

```rust
use elara_array::prelude::*;

fn main() {
	let a = arr![[1.0, 2.0], [3.0, 4.0]];
	println!("{:?}", a);
}
