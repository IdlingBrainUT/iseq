//! The implementation of iSeqNMF.

use ndarray::*;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cmp::Reverse;
use std::collections::VecDeque;

use iseq_linalg::cost::*;
use iseq_linalg::heap::*;
use iseq_linalg::random::*;
use iseq_linalg::stats::*;
use iseq_linalg::traits::FloatISeq;
use iseq_linalg::update::*;

use crate::args::*;
use crate::io::*;

/// iSeq
pub struct ISeq<T: FloatISeq> {
    /// Data matrix X.
    /// 
    /// [SHAPE] (N, T)
    pub v: Array2<T>,
    ///
    pub v_max: T,

    /// Pattern matrix W.
    /// 
    /// [SHAPE] (N, K, L)
    pub w: Array3<T>,

    /// Intensity matrix H.
    /// 
    /// [SHAPE] (K, T+L-1)
    pub h: Array2<T>,

    /// 
    pub w_lim: T,
    /// 
    pub h_lim: T,

    /// Reconstructed data matrix, calculated from W and H.
    /// 
    /// [SHAPE] (N, T)
    pub u: Array2<T>,

    /// The number of neurons (N).
    pub n: usize,
    /// The number of patterns (K).
    pub k: usize,
    /// The length of time window of patterns (L).
    pub l: usize,
    /// The length of data (T).
    pub t: usize,

    /// 
    /// 
    /// (default) 100
    pub max_iter: usize,

    /// To what extent do you ignore the difference between the numbers below.
    pub tolerance: T,
    /// Zero threshold.
    /// 
    /// Values less than z_th are considered as zero.
    pub z_th: T,

    /// Seed values for the random number generator.
    pub random_seed: Option<u32>,

    /// Reconstruction + X-ortho cost at each iteration.
    pub cost: Array1<T>,

    /// Output the calculation process.
    pub verbose: bool,
}

impl<T> ISeq<T>
where
    T: FloatISeq
{
    /// Create a iSeqNMF struct.
    /// 
    /// You have to do init() before calculation.
    pub fn new(
        v: &Array2<T>, k: usize, l: usize,
        max_iter: usize, z_th: T, tolerance: T, 
        h_lim: T, w_lim: T, random_seed: Option<u32>
        verbose: bool,
    ) -> Self {
        let mut v_max = 0.0;
        Zip::from(v)
            .for_each(|&vi| if vi > v_max { v_max = vi; });
        let (n, t) = (x.shape()[0], x.shape()[1]);

        let w: Array3<T> = Array3::zeros((n, k, l));
        let h: Array2<T> = Array2::zeros((k, t+l-1));
        let u: Array2<T> = Array2::zeros((n, t));
        
        let cost = Array1::zeros(max_iter + 1);

        Self { 
            v, v_max, w, h, w_lim, h_lim, u,
            n, k, l, t, max_iter, tolerance, z_th,
            random_seed, cost, verbose
        }
    }

    /// Create a iSeqNMF struct from args.
    pub fn from_args(v: &Array<T>, a: &Args<T>) -> Self {
        Self::new(
            v, a.k, a.l, a.z_th, 
            a.max_iter, a.tolerance, a.n_null, 
            a.sd_significant, a.n_significant, a.p_integrate, 
            a.verbose,
        )
    }

    /// Reset parameters of iSeqNMF.
    pub fn init(
        &mut self, 
        k: usize, lambda: T, random_seed: u32,
    ) {
        if k < self.k {
            self.k = k;
            self.w.slice_collapse(s![.., ..k, ..]);
            self.h.slice_collapse(s![..k, ..]);
            self.wtx.slice_collapse(s![..k, ..]);
            self.sht.slice_collapse(s![.., ..k]);
        } else if k > self.k {
            self.k = k;
            self.w = Array3::zeros((self.w.shape()[0], k, self.w.shape()[2]));
            self.h = Array2::zeros((k, self.h.shape()[1]));
            self.wtx = Array2::zeros((k, self.wtx.shape()[1]));
            self.sht = Array2::zeros((self.sht.shape()[0], k));
        }
        self.lambda = lambda;
        self.random_seed = random_seed;
        let h_scale = self.x.sum() * T::from_usize(4).unwrap() / T::from_usize(self.n * self.k * self.l * self.t).unwrap();

        self.w.set_random(random_seed);
        self.h.set_random_scale(flip_u32(random_seed), h_scale);

        self.update_wtx();
        self.update_x_hat();
        self.update_sht();

        let zero = T::zero();
        let er = self.recon_cost_mask(false);
        let ex = self.xortho_cost();
        let e = er + self.lambda * ex;
        Zip::indexed(&mut self.cost)
            .and(&mut self.recon)
            .and(&mut self.xortho)
            .apply(|i, a, b, c| {
                if i == 0 {
                    *a = e;
                    *b = er;
                    *c = ex;
                } else {
                    *a = zero;
                    *b = zero;
                    *c = zero;
                }
            });
    }

    // Load the csv file on M or initialize it.
    pub fn reset_mask(&mut self, mask_path: Option<&String>) {
        match mask_path {
            Some(path) => {
                match self.mask.as_mut() {
                    Some(mask) => {
                        mask.read_csv_inplace(path, false);
                        return;
                    }
                    None => {}
                }
                let mut mask = Array2::zeros(self.x.dim());
                mask.read_csv_inplace(&path, false);
                self.mask = Some(mask);
            }
            None => {
                self.mask = None;
            }
        }
    }

    /// Run 1 iteration.
    pub fn run(&mut self, iter: usize, verbose: bool, drop: bool) {
        let cost_prev = self.cost[iter];
        let zpf = T::from(0.5).unwrap();

        // update
        let mut w_old = self.w.clone();
        let mut h_old = self.h.clone();
        self.update_wh();

        let mut drop_flg = false;
        for i in 0..10 {
            self.update_wtx();
            self.update_x_hat();
            self.update_sht();
            let e = self.recon_cost_mask(false) + self.lambda * self.xortho_cost();
            if verbose {
                println!("iter={:>4}, time={:>2} | now={:6.3e}, prev={:6.3e} | k={:>3}", iter, i, e, cost_prev, self.k);
            }
            if e < cost_prev {
                break;
            }
            if i == 9 {
                if drop {
                    drop_flg = true;
                }
                break;
            }
            if i == 0 {
                Zip::from(&mut w_old).apply(|a| *a *= zpf);
                Zip::from(&mut h_old).apply(|a| *a *= zpf);
            }
            Zip::from(&mut self.w).and(&w_old).apply(|a, &b| {
                *a = *a * zpf + b;
            });
            Zip::from(&mut self.h).and(&h_old).apply(|a, &b| {
                *a = *a * zpf + b;
            });
        }

        // remove weak sequences.
        let mut rem_k = Vec::new();
        if self.k > 1 {
            let w_flat_n = self.w.sum_axis(Axis(0));
            let mut kl: Array2<T> = Array2::zeros((self.k, self.l));
            let mut hk = self.h.slice(s![.., ..self.t-1]).sum_axis(Axis(1));
            for li in 0..self.l {
                hk = hk + self.h.slice(s![.., self.t+li-1]);
                Zip::from(&mut kl.slice_mut(s![.., self.l-li-1])).and(&hk).apply(|a, &b| {
                    *a = b;
                });
                hk = hk - self.h.slice(s![.., li]);
            }
            let wh_sum = (w_flat_n * kl).sum_axis(Axis(1));
            let wh_sum_max = wh_sum.max_axis(Axis(0)).into_scalar();
            let eps0 = self.epsilon * T::from(self.n * self.t).unwrap();

            if wh_sum_max < eps0 {
                let (mut i, mut s) = (0, wh_sum[0]);
                for ki in 1..self.k {
                    if wh_sum[ki] > s {
                        s = wh_sum[ki];
                        rem_k.push(i);
                        i = ki;
                    } else  {
                        rem_k.push(ki)
                    }
                }
            } else {
                let eps1 = wh_sum_max * T::from(0.01).unwrap();
                for (i, &ek) in wh_sum.iter().enumerate() {
                    if ek < eps0 || ek < eps1 {
                        rem_k.push(i);
                    }
                }
            }
            
            if drop_flg && rem_k.len() == 0 {
                let skew = self.h.skew_axis(Axis(1));
                let (mut i, mut s) = (0, skew[0].abs());
                for ki in 1..self.k {
                    let s_cand = skew[ki].abs();
                    if s_cand < s {
                        s = s_cand;
                        i = ki;
                    }
                }
                rem_k.push(i);
            }
        }

        // self.shift_factors();
        if rem_k.len() > 0 {
            self.remove_k(&rem_k);
            self.update_wtx();
            self.update_x_hat();
            self.update_sht();
        }

        let er = self.recon_cost_mask(false);
        let ex = self.xortho_cost();
        self.cost[iter + 1] = er + self.lambda * ex;
        self.recon[iter + 1] = er;
        self.xortho[iter + 1] = ex;
    }

    /// Run the all iterations up to max_iter.
    pub fn solve(&mut self) {
        let mut final_flg = false;
        for i in 0..self.max_iter {
            if i == self.max_iter - 1 ||
            (
                i >= 5 &&
                self.cost[i] + self.tolerance > self.cost.slice(s![i-5..i]).mean_axis(Axis(0)).unwrap().into_scalar()
            )
            {
                // The final iteration is convNMF without x-ortho penalty.
                // self.lambda = T::zero();
                final_flg = true;
            }
            self.run(i, self.verbose, true);
            if final_flg { break; }
        }
    }

    /// Remove patterns of ID `k` in `k_vec`.
    pub fn remove_k(&mut self, k_vec: &Vec<usize>) {
        let k_new = self.k - k_vec.len();
        if k_new == self.k { return; }

        let mut k_sort = k_vec.clone();
        k_sort.sort();
        let mut rem_k = VecDeque::from(k_sort);

        let h_old = self.h.clone();
        let w_old = self.w.clone();
        self.w.slice_collapse(s![.., ..k_new, ..]);
        self.h.slice_collapse(s![..k_new, ..]);
        self.wtx.slice_collapse(s![..k_new, ..]);
        self.sht.slice_collapse(s![.., ..k_new]);
        let mut k_index = 0;
        for i in 0..self.k {
            match rem_k.front() {
                Some(&e) => {
                    if i == e {
                        rem_k.pop_front();
                        continue;
                    }
                }
                None => {}
            };
            Zip::from(&mut self.h.slice_mut(s![k_index, ..])).and(&h_old.slice(s![i, ..])).apply(|a, &b| {
                *a = b;
            });
            Zip::from(&mut self.w.slice_mut(s![.., k_index, ..])).and(&w_old.slice(s![.., i, ..])).apply(|a, &b| {
                *a = b;
            });
            k_index += 1;
        }

        self.k = k_new;
    }

    /// Shift W and H so that the center of mass of the W is centered.
    pub fn shift_factors(&mut self) {
        shift_wh(&mut self.w, &mut self.h, self.random_seed);
    }

    /// Update W and H simultaneously.
    pub fn update_wh(&mut self) {
        update_wh(&mut self.w, &mut self.h, &self.x, &self.x_hat, &self.mask, &self.wtx, &self.sht, self.lambda, self.z_th, true);
    }
    /// Update transpose convolution calculated from W and X.
    pub fn update_wtx(&mut self) {
        update_wtx(&mut self.wtx, &self.w, &self.x);
    }
    /// Update reconstructed data matrix, calculated from W and H.
    pub fn update_x_hat(&mut self) {
        update_x_hat(&mut self.x_hat, &self.w, &self.h);
    }
    /// Update smoothed H^T.
    pub fn update_sht(&mut self) {
        update_sht(&mut self.sht, &self.h, self.l);
    }

    /// Calculate reconstruction cost.
    pub fn recon_cost_all(&self) -> T {
        recon_cost(&self.x, &self.x_hat, self.z_th)
    }
    /// Calculate reconstruction cost of masked data.
    pub fn recon_cost_mask(&self, flip: bool) -> T {
        match self.mask.as_ref() {
            Some(mask) => {
                recon_cost_mask(&self.x, &self.x_hat, mask, self.z_th, flip)
            }
            None => {
                if flip {
                    T::zero()
                } else {
                    recon_cost(&self.x, &self.x_hat, self.z_th)
                }
            }
        }
        
    }
    /// Calculate x-ortho penalty cost.
    pub fn xortho_cost(&self) -> T {
        xortho_cost(&self.wtx, &self.sht)
    }

    /// Make a matrix with each element is the divergence between x & x_hat.
    pub fn divergence_arr(&self) -> Array2<T> {
        let z_th = self.z_th;
        let mut div = Array2::zeros(self.x.dim().clone());
        Zip::from(&mut div)
            .and(&self.x)
            .and(&self.x_hat)
            .apply(|a, &b, &c| {
                *a = itakura_saito(b, c, z_th);
            });
        div
    }

    /// Returns the list of ID of significant sequences.
    pub fn significant(&self) -> Vec<usize> {
        let mut v = vec![1usize; self.k];

        // Sort sequences based on the number of significant neurons.
        let w_max_nk = self.w.max_axis(Axis(2));
        let n_cell = self.n - self.n_null;
        let mu_null = w_max_nk.slice(s![n_cell.., ..]).mean_axis(Axis(0)).unwrap();
        let sigma_null = w_max_nk.slice(s![n_cell.., ..]).std_axis(Axis(0), T::one());
        let sd = self.sd_significant;
        let mut th_null = Array::zeros(mu_null.dim().clone());
        Zip::from(&mut th_null)
            .and(&mu_null)
            .and(&sigma_null)
            .apply(|a, &b, &c| *a = b + sd * c);

        for (ki, (w_max_n, &th)) in w_max_nk.axis_iter(Axis(1))
                                            .zip(th_null.iter())
                                            .enumerate()
        {
            let mut count = 0;
            for &w_max in w_max_n.iter() {
                if w_max > th  {
                    count += 1;
                }
            }
            if count < self.n_significant {
                v[ki] = 0;
            }
        }

        let mut index = Vec::new();
        for ki in 0..self.k {
            if v[ki] == 1 {
                index.push(ki);
            }
        }
        let index_len = index.len();
        if index_len == 0 {
            return Vec::new();
        }

        // Integrate similar sequences.
        let mut seed = self.random_seed;
        let mut r0: Array1<usize> = Array1::zeros(1000);
        let mut r1: Array1<usize> = Array1::zeros(1000);
        r0.set_random(seed, index_len);
        r1.set_random(flip_u32(seed), index_len);
        seed += 1;

        let n_th = (T::from_usize(1000).unwrap() * self.p_integrate).to_usize().unwrap(); 
        let mut heap: HeapCap<Reverse<OrdFloat<T>>> = HeapCap::new(n_th).unwrap();
        for (&ri, &rj) in r0.iter().zip(r1.iter()) {
            let c = shift_cos_max(
                &self.wtx.slice(s![index[ri], ..]).to_owned().shuffle_axis(Axis(0), seed), 
                &self.wtx.slice(s![index[rj], ..]).to_owned().shuffle_axis(Axis(0), flip_u32(seed)), 
                self.l);
            heap.push(Reverse(OrdFloat::new(c)));
            seed += 1
        }
        let th = match heap.peek() {
            Some(Reverse(e)) => e.value,
            _ => panic!(),
        };

        let mut edge: Array2<usize> = Array2::zeros((index_len, index_len));
        for i in 0..index_len {
            let wtxi = self.wtx.slice(s![index[i], ..]).to_owned();
            for j in i..index_len {
                if i == j {
                    continue;
                }
                let c = shift_cos_max(&wtxi, &self.wtx.slice(s![index[j], ..]), self.l);
                if c > th {
                    edge[[i, j]] = 1;
                    edge[[j, i]] = 1;
                }
            }
        }

        loop {
            let edge_sum = edge.sum_axis(Axis(1));
            if edge_sum.sum() == 0 {
                break;
            }
            let i = edge_sum.to_vec().argmax();
            v[index[i]] = 0;
            for j in 0..index_len {
                edge[[i, j]] = 0;
                edge[[j, i]] = 0;
            }
        }

        let mut ret = Vec::new();
        for ki in 0..self.k {
            if v[ki] == 1 {
                ret.push(ki);
            }
        }

        ret
    }

    /// Save W & H to csv file.
    #[allow(bare_trait_objects)]
    pub fn save_wh(&self, f: &str) -> Result<(), Box::<std::error::Error>> {
        for i in 0..self.k {
            self.w.slice(s![.., i, ..]).save_to_csv(&format!("{}_W_{}.csv", f, i))?;
        }
        self.h.slice(s![.., self.l-1..]).t().save_to_csv(&format!("{}_H.csv", f))?;
        Ok(())
    }

    /// Save divergense_arr() & x_hat.
    #[allow(bare_trait_objects)]
    pub fn save_error(&self, f: &str) -> Result<(), Box::<std::error::Error>> {
        self.divergence_arr().t().save_to_csv(&format!("{}_error.csv", f))?;
        self.x_hat.t().save_to_csv(&format!("{}_x_hat.csv", f))?;
        Ok(())
    }
}

/// Rotate sequences so that their center of gravities are in the center of the time window.
pub fn shift_wh<T: FloatISeq>(w: &mut Array3<T>, h: &mut Array2<T>, random_seed: u32) {
    let (n, k, l) = (w.shape()[0], w.shape()[1], w.shape()[2]);
    let center = l / 2;
    let c_t = T::from_usize(center).unwrap();
    let zero = T::zero();
    let ud = Uniform::new(zero, T::one());
    let mut rng = StdRng::from_seed(seed_from_u32(random_seed));

    for ki in 0..k {
        let wk = w.slice(s![.., k, ..]).to_owned();
        let wk_sum = wk.sum_axis(Axis(0));
        let wk_sum_sum = wk_sum.sum_axis(Axis(0)).into_scalar();
        let wk_mean2 = wk_sum_sum / T::from(n * l).unwrap() * T::from(2).unwrap();

        let mut cmass = std::usize::MAX;
        let mut dist_min = T::infinity();
        for i in 0..l {
            let dist = (
                wk_sum.iter()
                      .enumerate()
                      .map(|(j, &e)| T::from_usize((i + j) % l).unwrap() * e)
                      .sum::<T>()
                / wk_sum_sum
                - c_t
            ).abs();
            if dist < dist_min {
                cmass = (l + center - i) % i;
                dist_min = dist;
            }
        }
        if cmass > center {
            let diff = cmass - center;
            Zip::from(&mut w.slice_mut(s![.., ki, ..l-diff]))
                .and(&wk.slice(s![.., diff..]))
                .apply(|a, &b| {
                    *a = b;
                }
            );
            Zip::from(&mut w.slice_mut(s![.., ki, l-diff..]))
                .and(&wk.slice(s![.., ..diff]))
                .apply(|a, &b| {
                    if b > zero {
                        *a = b;
                    } else {
                        *a = ud.sample(&mut rng) * wk_mean2;
                    }
                }
            );
            let hk = h.slice(s![ki, ..]).to_owned();
            let lh = hk.shape()[0];
            let hk_mean2 = hk.sum_axis(Axis(0)).into_scalar() / T::from(lh).unwrap() * T::from(2).unwrap();
            Zip::from(&mut h.slice_mut(s![ki, diff..]))
                .and(&hk.slice(s![..lh-diff]))
                .apply(|a, &b| {
                    *a = b;
                }
            );
            Zip::from(&mut h.slice_mut(s![ki, ..diff]))
                .apply(|a| {
                    *a = ud.sample(&mut rng) * hk_mean2;
                }
            );
        } else if cmass < center {
            let diff = center - cmass;
            Zip::from(&mut w.slice_mut(s![.., ki, diff..]))
                .and(&wk.slice(s![.., ..l-diff]))
                .apply(|a, &b| {
                    *a = b;
                }
            );
            Zip::from(&mut w.slice_mut(s![.., ki, ..diff]))
                .and(&wk.slice(s![.., l-diff..]))
                .apply(|a, &b| {
                    if b > zero {
                        *a = b;
                    } else {
                        *a = ud.sample(&mut rng) * wk_mean2;
                    }
                }
            );
            let hk = h.slice(s![ki, ..]).to_owned();
            let lh = hk.shape()[0];
            let hk_mean2 = hk.sum_axis(Axis(0)).into_scalar() / T::from(lh).unwrap() * T::from(2).unwrap();
            Zip::from(&mut h.slice_mut(s![ki, ..lh-diff]))
                .and(&hk.slice(s![diff..]))
                .apply(|a, &b| {
                    *a = b;
                }
            );
            Zip::from(&mut h.slice_mut(s![ki, lh-diff..]))
                .apply(|a| {
                    *a = ud.sample(&mut rng) * hk_mean2;
                }
            );
        }
    }
    
}