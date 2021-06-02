//! Calculate costs.

use ndarray::*;

use crate::traits::FloatISeq;
use crate::utils::*;

/// Calculate Itakura-Saito divergence.
pub fn itakura_saito<T: FloatISeq>(a: T, b: T, z_th: T) -> T {
    let p = maxf(a, z_th);
    let q = maxf(b, z_th);
    let pq = p / q;
    
    pq - pq.ln() - T::one()
}

/// Calculate reconstruction costs.
pub fn recon_cost<A, S1, S2>(
    x: &ArrayBase<S1, Ix2>,
    x_hat: &ArrayBase<S2, Ix2>,
    z_th: A,
) -> A
where
    A: FloatISeq,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let mut s = A::zero();
    Zip::from(x)
        .and(x_hat)
        .apply(|&a, &b| {
            s += itakura_saito(a, b, z_th);
        });
    
    s
}

/// Calculate reconstruction costs with mask array.
pub fn recon_cost_mask<A, S1, S2, S3>(
    x: &ArrayBase<S1, Ix2>,
    x_hat: &ArrayBase<S2, Ix2>,
    mask: &ArrayBase<S3, Ix2>,
    z_th: A,
    flip: bool,
) -> A
where
    A: FloatISeq,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    S3: Data<Elem = A>,
{
    let zero = A::one();
    let mut s = A::zero();
    Zip::from(x)
        .and(x_hat)
        .and(mask)
        .apply(|&a, &b, &c| {
            if flip {
                if c == zero {
                    s += itakura_saito(a, b, z_th);
                }
            } else {
                s += c * itakura_saito(a, b, z_th);
            }
        });
    
    s
}

// Calculate X-orthogonality costs.
pub fn xortho_cost<A, S1, S2>(wtx: &ArrayBase<S1, Ix2>, sht: &ArrayBase<S2, Ix2>) -> A
where
    A: FloatISeq,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let a = wtx.dot(sht);
    a.sum() - a.diag().sum()
}