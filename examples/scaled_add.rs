use elara_array::{arr, NdArray};

fn main() {
    let mut a = arr![1.0, 2.0, 3.0, 4.0, 5.0];
    let b: NdArray<f64, 1> = NdArray::ones([5]);
    a.scaled_add(1.0, &b);
    println!("{:?}", a);
}