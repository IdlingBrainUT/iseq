/// Update arrays used in iSeq.

use ndarray::*;

use crate::traits::FloatISeq;
use crate::utils::*;

/// Update X_HAT.
pub fn update_x_hat<A, S1, S2>(
    x_hat: &mut Array2<A>,
    w: &ArrayBase<S1, Ix3>,
    h: &ArrayBase<S2, Ix2>,
)
where
    A: FloatISeq,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let l = w.shape()[2];
    let (n, t) = (x_hat.shape()[0], x_hat.shape()[1]);
    let mut arr = Array2::zeros((n, t));
    for i in 0..l {
        let pos = l - 1 - i;
        let brr = w.slice(s![.., .., i])
                   .dot(&h.slice(s![.., pos..pos+t]));
        Zip::from(&mut arr).and(&brr).apply(|a, &b| {
            *a += b;
        });
    }
    *x_hat = arr;
}

/// Update WTX.
pub fn update_wtx<A, S1, S2>(
    wtx: &mut Array2<A>,
    w: &ArrayBase<S1, Ix3>,
    x: &ArrayBase<S2, Ix2>
)
where
    A: FloatISeq,
    S1: DataMut<Elem = A>,
    S2: Data<Elem = A>,
{
    let (k, t) = (wtx.shape()[0], wtx.shape()[1]);
    let l = w.shape()[2];
    let mut arr = Array2::zeros((k, t));
    for i in 0..l {
        let brr = w.slice(s![.., .., i])
                   .t()
                   .dot(&x.slice(s![.., i..i+t]));
        Zip::from(&mut arr).and(&brr).apply(|a, &b| {
            *a += b;
        })
    }
    *wtx = arr;
}

/// Update SH^T.
pub fn update_sht<A, S1, S2>(
    sht: &mut ArrayBase<S1, Ix2>,
    h: &ArrayBase<S2, Ix2>,
    l: usize
)
where
    A: FloatISeq,
    S1: DataMut<Elem = A>,
    S2: Data<Elem = A>,
{
    let (t, k) = (sht.shape()[0], sht.shape()[1]);
    let l22 = 2 * l - 2;
    for ki in 0..k {
        let mut hi = h.slice(s![ki, ..l22]).sum();
        for ti in 0..t {
            hi += h[[ki, ti+l22]];
            sht[[ti, ki]] = hi;
            hi -= h[[ki, ti]];
        }
    }
}

/// Update W and H.
pub fn update_wh<A, S1, S2, S3, S4, S5, S6, S7>(
    w: &mut ArrayBase<S2, Ix3>,
    h: &mut ArrayBase<S1, Ix2>,
    x: &ArrayBase<S3, Ix2>,
    x_hat: &ArrayBase<S4, Ix2>,
    op_mask: &Option<ArrayBase<S5, Ix2>>,

    wtx: &ArrayBase<S6, Ix2>,
    sht: &ArrayBase<S7, Ix2>,
    lambda: A,
    z_th: A,

    update_h: bool,
)
where
    A: FloatISeq,
    S1: DataMut<Elem = A> + RawDataClone,
    S2: DataMut<Elem = A> + RawDataClone,
    S3: Data<Elem = A>,
    S4: Data<Elem = A>,
    S5: Data<Elem = A>,
    S6: Data<Elem = A>,
    S7: Data<Elem = A>,
{
    let w_shape = w.shape();
    let (n, k, l) = (w_shape[0], w_shape[1], w_shape[2]);
    let tx = x.shape()[1];
    let th = h.shape()[1];
    let tsht = sht.shape()[0];

    let x_dim = x.dim();
    let mut vu2 = Array2::zeros(x_dim.clone());
    let mut u1 = Array2::zeros(x_dim.clone());
    let zero = A::zero();
    let epsilon = A::epsilon();

    // Calc for W and H.
    match op_mask {
        Some(mask) => {
            Zip::from(&mut vu2)
                .and(&mut u1)
                .and(x)
                .and(x_hat)
                .and(mask)
                .apply(|a, b, &c, &d, &e| {
                    if c >= z_th || d >= z_th {
                        let c_e = maxf(c, z_th);
                        let d_e = maxf(d, z_th);
                    
                        *a = e * c_e / (d_e * d_e);
                        *b = e / d_e;
                    }
                });
        }
        None => {
            let one = A::one();
            Zip::from(&mut vu2)
                .and(&mut u1)
                .and(x)
                .and(x_hat)
                .apply(|a, b, &c, &d| {
                    if c >= z_th || d >= z_th {
                        let c_e = maxf(c, z_th);
                        let d_e = maxf(d, z_th);
                    
                        *a = c_e / (d_e * d_e);
                        *b = one / d_e;
                    }
                });
        }
    }
    

    // Calc for H.
    let h_dim = h.dim();
    let mut nume_h: Array2<A> = Array2::zeros(h_dim.clone());
    let mut deno_h: Array2<A> = Array2::zeros(h_dim.clone());
    let mut xo_h: Array2<A> = Array2::zeros(h_dim.clone());
    if update_h {
        for li in 0..l {
            let ws = w.slice(s![.., .., li]);
            let start = l - 1 - li;
            let end = th - li;
            Zip::from(nume_h.slice_mut(s![.., start..end]))
                .and(&ws.t().dot(&vu2))
                .apply(|a, &b| {
                    *a += b;
                });
            Zip::from(deno_h.slice_mut(s![.., start..end]))
                .and(&ws.t().dot(&u1))
                .apply(|a, &b| {
                    *a += b;
                });
        }
        let mut wtxs_h = Array2::zeros(h_dim.clone());
        let shift = 2 * l - 1;
        for ki in 0..k {
            let line = wtx.slice(s![ki, ..]);
            let mut wtxss = wtxs_h.slice_mut(s![ki, ..]);
            let mut tmp = zero;
            for ti in 0..th {
                if ti < tsht {
                    tmp += line[ti];
                }
                if ti >= shift {
                    tmp -= line[ti - shift];
                }
                wtxss[ti] = tmp;
            }
        }
        xo_h = &wtxs_h.sum_axis(Axis(0)).broadcast((k, th)).unwrap() - &wtxs_h;
    }
    
    // Update W.
    for li in 0..l {
        let pos = l - 1 - li;
        let hs = h.slice(s![.., pos..pos+tx]);
        let nume_w = vu2.dot(&hs.t());
        let deno_w = u1.dot(&hs.t());
        let xsht_w = x.slice(s![.., li..tsht+li]).dot(&sht.view());
        let xo_w = &xsht_w.sum_axis(Axis(1)).broadcast((k, n)).unwrap().t() - &xsht_w;
        Zip::from(w.slice_mut(s![.., .., li]))
            .and(&nume_w)
            .and(&deno_w)
            .and(&xo_w)
            .apply(|a, &b, &c, &d| {
                *a *= b / (c + lambda * d + epsilon);
            });
    }
    
    if update_h {
        // Update H.
        Zip::from(h)
            .and(&nume_h)
            .and(&deno_h)
            .and(&xo_h)
            .apply(|a, &b, &c, &d| {
                *a *= b / (c + lambda * d + epsilon);
            });
    }
    
}