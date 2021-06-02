//! Functions for statistics.

use ndarray::*;
use num_traits::{Float, FromPrimitive, PrimInt};

/// Trait to calculate Max/Min of a vector of float values.
pub trait OrdVec<T: Float> {
    /// Maximum value.
    fn max(&self) -> T;
    /// Minimum value.
    fn min(&self) -> T;
    /// Index of Maximum value.
    fn argmax(&self) -> usize;
    /// Index of Minimum value.
    fn argmin(&self) -> usize;
}

impl<T: Float> OrdVec<T> for Vec<T> {
    fn max(&self) -> T {
        self.iter().fold(T::min_value(), |m, &x| if m > x { m } else { x })
    }
    fn min(&self) -> T {
        self.iter().fold(T::max_value(), |m, &x| if m < x { m } else { x })
    }
    fn argmax(&self) -> usize {
        self.iter()
            .enumerate()
            .fold((usize::MAX, T::min_value()), |(i, m), (j, &x)| if m > x { (i, m) } else { (j, x) })
            .0
    }
    fn argmin(&self) -> usize {
        self.iter()
            .enumerate()
            .fold((usize::MAX, T::max_value()), |(i, m), (j, &x)| if m < x { (i, m) } else { (j, x) })
            .0
    }
}

/// Trait to calculate Max/Min of a vector of integer values.
pub trait OrdVecInt<T: PrimInt> {
    /// Maximum value.
    fn max(&self) -> T;
    /// Minimum value.
    fn min(&self) -> T;
    /// Index of Maximum value.
    fn argmax(&self) -> usize;
    /// Index of Minimum value.
    fn argmin(&self) -> usize;
}

impl OrdVecInt<usize> for Vec<usize> {
    fn max(&self) -> usize {
        self.iter().fold(usize::MIN, |m, &x| if m > x { m } else { x })
    }
    fn min(&self) -> usize {
        self.iter().fold(usize::MAX, |m, &x| if m < x { m } else { x })
    }
    fn argmax(&self) -> usize {
        self.iter()
            .enumerate()
            .fold((usize::MAX, usize::MIN), |(i, m), (j, &x)| if m > x { (i, m) } else { (j, x) })
            .0
    }
    fn argmin(&self) -> usize {
        self.iter()
            .enumerate()
            .fold((usize::MAX, usize::MAX), |(i, m), (j, &x)| if m < x { (i, m) } else { (j, x) })
            .0
    }
}

/// Trait to calculate Max/Min of a N-dimensional array.
pub trait OrdArray<T: Clone, D: RemoveAxis> {
    /// Maximum value in `axis` direction.
    fn max_axis(&self, axis: Axis) -> Array<T, D::Smaller>;
    /// Minimum value in `axis` direction.
    fn min_axis(&self, axis: Axis) -> Array<T, D::Smaller>;
}

impl<A, S, D> OrdArray<A, D> for ArrayBase<S, D> 
where
    A: Float,
    S: Data<Elem = A>,
    D: RemoveAxis,
{
    fn max_axis(&self, axis: Axis) -> Array<A, D::Smaller> {
        self.fold_axis(axis, A::neg_infinity(), |&m, &x| if m < x { x } else { m })
    }

    fn min_axis(&self, axis: Axis) -> Array<A, D::Smaller> {
        self.fold_axis(axis, A::infinity(), |&m, &x| if m > x { x } else { m })
    }
}

/// Trait to calculate skewness of a N-dimensional array.
pub trait SkewArray<T: Clone, D: RemoveAxis> {
    /// Skewness in `axis` direction.
    fn skew_axis(&self, axis: Axis) -> Array<T, D::Smaller>;
}

impl<T: Float + FromPrimitive + ScalarOperand, S: Data<Elem = T>, D: RemoveAxis> SkewArray<T, D> for ArrayBase<S, D> {
    fn skew_axis(&self, axis: Axis) -> Array<T, D::Smaller> {
        let ax = axis.index();

        let n = self.shape()[ax];

        if n < 2 {
            panic!();
        }

        let pre = T::from(n).unwrap() / T::from((n - 1) * (n - 2)).unwrap();
        let mu = self.mean_axis(axis).unwrap();
        let sigma = self.std_axis(axis, T::one());
        let mut skew = Array::zeros(self.dim().clone()).remove_axis(axis);
        for subview in self.axis_iter(axis) {
            Zip::from(&mut skew)
                .and(subview)
                .and(&mu)
                .and(&sigma)
                .apply(|a, &b, &c, &d| {
                    let e = (b - c) / d;
                    *a = *a + e * e * e;
                });
        }

        skew.mapv_into(|s| s * pre)
    }
}

/// Trait to calculate convolution of a N-dimensional array.
pub trait ConvArray<T: Clone, D: RemoveAxis> {
    /// Convolution in `axis` direction.
    fn conv_sum_axis(&self, axis: Axis, size: usize) -> Result<Array<T, D>, ()>;
}

impl<T: Float, S: Data<Elem=T>, D: RemoveAxis> ConvArray<T, D> for ArrayBase<S, D> {
    fn conv_sum_axis(&self, axis: Axis, size: usize) -> Result<Array<T, D>, ()> {
        let index = axis.index();
        let len_axis = self.shape()[index];
        if size == 0 || size > len_axis {
            return Err(());
        }

        let mut sum_arr = self.slice_axis(axis, Slice::from(..size-1)).sum_axis(axis);

        let len_axis_new = len_axis - size + 1;
        
        let mut ret = self.slice_axis(axis, Slice::from(..len_axis_new)).to_owned();
        for (ret_sub, (add_sub, sub_sub)) in ret.axis_iter_mut(axis)
                            .zip(self.axis_iter(axis).skip(size - 1)
                                .zip(self.axis_iter(axis).take(len_axis_new)))
        {
            Zip::from(ret_sub)
                .and(&mut sum_arr)
                .and(add_sub)
                .and(sub_sub)
                .apply(|a, b, &c, &d| {
                    *b = *b + c;
                    *a = *b;
                    *b = *b - d;
                });
        }

        Ok(ret)
    }
}

/// Calculate cosine similarity between 2 arrays.
pub fn cos<A, S1, S2>(a: &ArrayBase<S1, Ix1>, b: &ArrayBase<S2, Ix1>) -> A
where
    A: Float,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let t = a.shape()[0];
    if t != b.shape()[0] {
        panic!()
    }

    let zero = A::zero();
    let a_norm = a.iter().fold(zero, |m, &x| m + x * x).sqrt();
    let b_norm = b.iter().fold(zero, |m, &x| m + x * x).sqrt();

    if a_norm == zero || b_norm == zero {
        zero
    } else {
        (a * b).sum() / (a_norm * b_norm)
    }
}

/// Calculate cos(a, b) while shifting them and return the maximum value.
pub fn shift_cos_max<A, S1, S2>(a: &ArrayBase<S1, Ix1>, b: &ArrayBase<S2, Ix1>, l: usize) -> A
where
    A: Float,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let t = a.shape()[0];
    if l == 0 || t != b.shape()[0] || t + 1 < 2 * l {
        panic!()
    }

    let mut m = A::zero();
    let l1 = l - 1;
    let dt = t - l1 * 2;
    let a_p = a.slice(s![l1..l1+dt]).to_owned();
    for li in 0..2*l-1 {
        let c = cos(&a_p, &b.slice(s![li..li+dt]));
        if m < c {
            m = c;
        }
    }

    m
}

#[test]
fn skew_test() {
    let a = arr2(&[
        [1.0, 2.0, 3.0],
        [4.0, 8.0, 6.0],
        [7.0, 8.0, 1.0],
    ]);

    let s0 = a.skew_axis(Axis(0));
    let s1 = a.skew_axis(Axis(1));
    assert!(s0[0] == 0.0);
    assert!((s0[1] + 1.73205).abs() < 0.01);
    assert!((s0[2] - 0.585583).abs() < 0.01);
    assert!(s1[0] == 0.0);
    assert!(s1[1] == 0.0);
    assert!((s1[2] + 1.5971).abs() < 0.01);
}

#[test]
fn conv_sum_test() {
    let a = arr2(&[
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0, 6.0],
    ]);

    let s_ax0 = a.conv_sum_axis(Axis(0), 2).unwrap();
    let s = s_ax0.shape();
    assert_eq!(s[0], 3);
    assert_eq!(s[1], 4);
    assert_eq!(s_ax0[[0, 0]], 6.0);
    assert_eq!(s_ax0[[0, 1]], 8.0);
    assert_eq!(s_ax0[[0, 2]], 10.0);
    assert_eq!(s_ax0[[0, 3]], 12.0);
    assert_eq!(s_ax0[[1, 0]], 14.0);
    assert_eq!(s_ax0[[1, 1]], 6.0);
    assert_eq!(s_ax0[[1, 2]], 8.0);
    assert_eq!(s_ax0[[1, 3]], 10.0);
    assert_eq!(s_ax0[[2, 0]], 12.0);
    assert_eq!(s_ax0[[2, 1]], 4.0);
    assert_eq!(s_ax0[[2, 2]], 6.0);
    assert_eq!(s_ax0[[2, 3]], 8.0);

    let s_ax1 = a.conv_sum_axis(Axis(1), 3).unwrap();
    let s = s_ax1.shape();
    assert_eq!(s[0], 4);
    assert_eq!(s[1], 2);
    assert_eq!(s_ax1[[0, 0]], 6.0);
    assert_eq!(s_ax1[[0, 1]], 9.0);
    assert_eq!(s_ax1[[1, 0]], 18.0);
    assert_eq!(s_ax1[[1, 1]], 21.0);
    assert_eq!(s_ax1[[2, 0]], 10.0);
    assert_eq!(s_ax1[[2, 1]], 3.0);
    assert_eq!(s_ax1[[3, 0]], 12.0);
    assert_eq!(s_ax1[[3, 1]], 15.0);
}