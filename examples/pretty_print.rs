use elara_array::prelude::*;

fn main() {
    let arr: NdArray<f64, 2> = NdArray::ones([10, 10]);
    println!("{:?}", arr);

    let arr2: NdArray<f64, 3> = NdArray::ones([2, 2, 3]);
    println!("{:?}", arr2);
}