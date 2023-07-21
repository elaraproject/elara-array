use elara_array::prelude::*;

fn main() {
    let arr: NdArray<f64, 2> = NdArray::ones([10, 10]);
    println!("{:?}", arr);
}