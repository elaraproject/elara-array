use elara_log::prelude::*;
use elara_array::prelude::*;

fn main() {
    Logger::new().init().unwrap();
    let a = arr![1, 2, 3, 4, 5];
    println!("{}", a[4]);
    let mut b = NdArray::from_vec2(vec![[1.0, 2.0], [3.0, 4.0]]);
    println!("{:?}", b.index(0));
    println!("{}", b);
    b.assign(1, &[5.0, 6.0]);
    println!("{}", b);
}