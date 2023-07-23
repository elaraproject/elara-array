// Solves a series of coupled
// differential equations using a RK4 solver
use elara_array::prelude::*;
use elara_log::prelude::*;

fn dydt(u: NdArray<f64, 1>, _t: f64) -> NdArray<f64, 1> {
    // df/dt = g
    // dg/dt = -f
    let f = u[0];
    let g = u[1];
    let df_dt = g;
    let dg_dt = -f;
    arr![df_dt, dg_dt]
}

fn main() {
    Logger::new().init().unwrap();
    let x_start = 0.0;
    let x_end = 2.0 * PI;
    let samples = 100;
    // Set initial conditions
    let initial_conditions = &[1.0, 0.0];
    let mut u: NdArray<f64, 2> = NdArray::zeros([samples, initial_conditions.len()]);
    let t = NdArray::linspace(x_start, x_end, samples);
    for i in 0..initial_conditions.len() {
        u[&[0, i]] = initial_conditions[i];
    }
    let h = (x_end - x_start) / samples as f64;
    for i in 0..(samples - 1) {
        let k1 = dydt(u.index_view(i), t[i]) * h;
        let k2 = dydt(u.index_view(i) + &k1 * 0.5, t[i] + 0.5 * h) * h;
        let k3 = dydt(u.index_view(i) + &k2 * 0.5, t[i] + 0.5 * h) * h;
        let k4 = dydt(u.index_view(i) + &k3, t[i] + h) * h;
        for j in 0..initial_conditions.len() {
            u[&[i + 1, j]] = u[&[i, j]] + (k1[j] + 2.0 * (k2[j] + k3[j]) + k4[j]) / 6.0;
        }
    }
    println!("{:?}", t);

    for i in 0..u.shape()[0] {
        println!("[{}, {}],", u[&[i, 0]], u[&[i, 1]]);
    }

}