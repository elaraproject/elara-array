use elara_array::prelude::*;
use std::time::Instant;

fn main() {
    let a: NdArray<f64, 2> = NdArray::ones([1000, 1000]);
    let b: NdArray<f64, 2> = NdArray::ones([1000, 1000]);
    let now = Instant::now();
    let _c = &a + &b;
    println!("{:?}", now.elapsed());
}