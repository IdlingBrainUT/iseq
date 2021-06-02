//! Implementation of correlations.

use ndarray::*;

use crate::traits::FloatISeq;

/// Kendall correlation for binalized data.
pub fn kendall_01<A, S1, S2>(x: &ArrayBase<S1, Ix1>, y: &ArrayBase<S2, Ix1>) -> Option<A>
where
    A: FloatISeq,
    S1: Data<Elem = i32>,
    S2: Data<Elem = i32>,
{
    let n = x.len();
    if n == 1 || n != y.len() {
        return None;
    }

    let xpy = x + y;
    let a = xpy.iter().fold(0i32, |m, &e| if e == 2 { m + 1 } else { m });
    let s = xpy.iter().fold(0i32, |m, &e| if e == 0 { m + 1 } else { m });
    let kp = a * s;

    let xmy = x - y;
    let dx = xmy.iter().fold(0i32, |m, &e| if e == 1 { m + 1 } else { m });
    let dy = xmy.iter().fold(0i32, |m, &e| if e == -1 { m + 1 } else { m });
    let km = dx * dy;

    let n0 = (n * (n - 1) / 2) as i32;

    let x_sum = x.sum_axis(Axis(0)).into_scalar();
    let n_x_sum = n as i32 - x_sum;
    let n1 = (x_sum * (x_sum - 1) + n_x_sum * (n_x_sum - 1)) / 2;

    let y_sum = y.sum_axis(Axis(0)).into_scalar();
    let n_y_sum = n as i32 - y_sum;
    let n2 = (y_sum * (y_sum - 1) + n_y_sum * (n_y_sum - 1)) / 2;

    Some(A::from_i32(kp - km).unwrap() / (A::from_i32(n0 - n1).unwrap().sqrt() * A::from_i32(n0 - n2).unwrap().sqrt()))
}

/// The distribution of Kendall correration for binarized data.
pub fn kendall_01_dist<A, S1, S2>(x: &ArrayBase<S1, Ix1>, y: &ArrayBase<S2, Ix1>) -> Option<A>
where
    A: FloatISeq,
    S1: Data<Elem = i32>,
    S2: Data<Elem = i32>,
{
    let n = x.len() as i32;
    if n <= 2 || n != y.len() as i32 {
        return None;
    }

    let xpy = x + y;
    let a = xpy.iter().fold(0i32, |m, &e| if e == 2 { m + 1 } else { m });
    let s = xpy.iter().fold(0i32, |m, &e| if e == 0 { m + 1 } else { m });
    let kp = a * s;

    let xmy = x - y;
    let dx = xmy.iter().fold(0i32, |m, &e| if e == 1 { m + 1 } else { m });
    let dy = xmy.iter().fold(0i32, |m, &e| if e == -1 { m + 1 } else { m });
    let km = dx * dy;

    let n_2 = n * (n - 1);
    let v0 = n_2 * (2 * n + 5);

    let x_sum = x.sum_axis(Axis(0)).into_scalar();
    let n_x_sum = n - x_sum;
    let x_sum_2 = x_sum * (x_sum - 1);
    let n_x_sum_2 = n_x_sum * (n_x_sum - 1);
    let vx = x_sum_2 * (2 * x_sum + 5) + n_x_sum_2 * (2 * n_x_sum + 5);

    let y_sum = y.sum_axis(Axis(0)).into_scalar();
    let n_y_sum = n - y_sum;
    let y_sum_2 = y_sum * (y_sum - 1);
    let n_y_sum_2 = n_y_sum * (n_y_sum - 1);
    let vy = y_sum_2 * (2 * y_sum + 5) + n_y_sum_2 * (2 * n_y_sum + 5);

    let v1 = A::from_i32((x_sum_2 + n_x_sum_2) * (y_sum_2 + n_y_sum_2)).unwrap() / A::from_i32(2 * n_2).unwrap();
    let v2 = A::from_i32((x_sum_2 * (x_sum - 2) + n_x_sum_2 * (n_x_sum - 2)) * (y_sum_2 * (y_sum - 2) + n_y_sum_2 * (n_y_sum - 2))).unwrap() / A::from(9 * n_2 * (n - 2)).unwrap();

    let v = A::from_i32(v0 - vx - vy).unwrap() / A::from_usize(18).unwrap() + v1 + v2;
    Some(A::from_i32(kp - km).unwrap() / v.sqrt())
}