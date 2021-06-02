//! Functions for lambda value selection.

use ndarray::*;

use crate::error::ISeqError;
use crate::traits::FloatISeq;
use crate::utils::*;

/// Modes to search lambda value.
#[derive(Clone, Copy)]
pub enum NarrowMode<T: FloatISeq> {
    Narrow,
    Final(usize),
    Once(T)
}

/// Return all lambda values to be calculated from the lambda_range.
/// 
/// - Narrow: Enumerate in 10-fold increments.
/// - Final: Divide the range equally by the number of calculations.
/// - Once: Just returns `vec![lambda_once]`.
pub fn lambda_vec<T: FloatISeq>(lambda_range: (T, T), mode: NarrowMode<T>)
-> Result<Vec<T>, ISeqError>
{
    match mode {
        NarrowMode::Once(lambda_once) => {
            return Ok(vec![lambda_once]);
        }
        _ => {}
    }
    let one = T::one();
    let ten = T::from(10).unwrap();
    let (lambda_start, lambda_end) = lambda_range;
    if lambda_start >= lambda_end {
        return Err(
            ISeqError::InvalidRange {
                range: (lambda_start.to_f64().unwrap(), lambda_end.to_f64().unwrap())
            }
        );
    }

    let lambda_vec = match mode {
        NarrowMode::Final(procs) => {
            let mut v = Vec::with_capacity(procs);
            let mut lamb = lambda_start;
            v.push(lamb);
            if lambda_end / lambda_start > (ten + one) {
                if procs < 3 {
                    return Err(
                        ISeqError::InsufficientProcs {
                            procs: procs, needed: 3,
                        }
                    );
                }
                let half1 = (procs - 1) / 2;
                let half2 = procs - 1 - half1;
                let halves = vec![half1, half2];
                for &half in halves.iter() {
                    let dl = lamb * (ten - one) / T::from_usize(half).unwrap();
                    for _ in 0..half {
                        lamb += dl;
                        v.push(lamb);
                    }
                }
            } else {
                if procs < 2 {
                    return Err(
                        ISeqError::InsufficientProcs {
                            procs: procs, needed: 3,
                        }
                    );
                }
                let half = procs - 1;
                let dl = lamb * (ten - one) / T::from_usize(half).unwrap();
                for _ in 0..half {
                    lamb += dl;
                    v.push(lamb);
                }
            }
            v
        }
        NarrowMode::Narrow => {
            axis_log10(lambda_range)
        }
        _ => {
            panic!()
        }
    };

    Ok(lambda_vec)
}

/// Increase the number of each element by the factor of `iter_per_lambda`.
pub fn lambda_vec_all<T: FloatISeq>(lambda_vec: &Vec<T>, iter_per_lambda: usize) -> Vec<T> {
    let n_lambda = lambda_vec.len();
    let mut lambda_vec_all = Vec::with_capacity(n_lambda * iter_per_lambda);
    for &lambda in lambda_vec.iter() {
        for _ in 0..iter_per_lambda {
            lambda_vec_all.push(lambda);
        }
    }
    lambda_vec_all
}

/// Extract corresponding elements of `my_id`.
pub fn lambda_vec_run<T: FloatISeq>(lambda_vec_all: &Vec<T>, my_id: &Vec<usize>) -> Vec<T> {
    my_id.iter()
         .map(|&i| lambda_vec_all[i])
         .collect()
}

/// Store calculation results in an array.
/// 
/// If the result of the calculation under a different lambda is more optimal than the result under a certain lambda, 
/// overwrite the result with the more optimal one,
/// and record the value of lambda at the time separately.
pub fn rearrange_costs<A, S>(
    ret_arr: &mut ArrayBase<S, Ix2>,
    lambda_vec: &Vec<A>,
    lambda_vec_all: &Vec<A>,
    recon_vec: &Vec<A>,
    xortho_vec: &Vec<A>,
    seed_vec: &Vec<u32>,
    k_vec: &Vec<usize>,
    valid_vec: &Vec<usize>,
)
where
    A: FloatISeq,
    S: Data<Elem = A> + DataMut,
{
    let mut lambda_copy = lambda_vec_all.clone();
    let mut recon_copy = recon_vec.clone();
    let mut xortho_copy = xortho_vec.clone();
    let mut seed_copy = seed_vec.clone();
    let mut k_copy = k_vec.clone();
    let mut valid_copy = valid_vec.clone();

    let index = argsort(&recon_vec);
    rearrange(&mut lambda_copy, &index);
    rearrange(&mut recon_copy, &index);
    rearrange(&mut xortho_copy, &index);
    rearrange(&mut seed_copy, &index);
    rearrange(&mut k_copy, &index);
    rearrange(&mut valid_copy, &index);
    let (mut start, end) = (0, recon_vec.len());
    let mut pick = 0;

    for (i, &lambda) in lambda_vec.iter().enumerate() {
        let mut cost = A::infinity();
        for j in start..end {
            let cost_new = recon_copy[j] + lambda * xortho_copy[j];
            if cost >= cost_new {
                cost = cost_new;
                pick = j;
            }
        }
        start = pick;
        let v = vec![lambda, lambda_copy[pick], A::from_usize(index[pick]).unwrap(), recon_copy[pick], xortho_copy[pick], A::from(seed_copy[pick]).unwrap(), A::from(k_copy[pick]).unwrap(), A::from(valid_copy[pick]).unwrap()];
        Zip::from(ret_arr.slice_mut(s![i, ..])).and(&v).apply(|a, &b| *a = b);
    }
}