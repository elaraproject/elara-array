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
    let (u, t) = elara_array::rk4(&dydt, initial_conditions, x_start, x_end, samples);
    println!("{:?}", t);
    println!("{:?}", u);

}
