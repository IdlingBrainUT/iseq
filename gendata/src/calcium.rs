use num_traits::Float;

pub fn calcium<T: Float>(t: usize, tau: usize, c1: T, c2: T) -> T {
    let dt = T::from(t - tau).unwrap();
    (-c1 * dt).exp() - (-c2 * dt).exp()
}