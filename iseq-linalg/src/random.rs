use ndarray::*;
use num_traits::{Float, PrimInt};
use rand::distributions::{uniform::SampleUniform, Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::utils::*;

pub fn flip_u32(n: u32) -> u32 {
    std::u32::MAX ^ n
}

pub fn incr_seed(n: u32) -> u32 {
    if n == std::u32::MAX {
        0
    } else {
        n + 1
    }
}

pub fn seed_from_u32(seed: u32) -> [u8; 32] {
    let mut v = [0; 32];
    for i in 0..32 {
        v[i] = (seed >> i & 1) as u8;
    }
    v
}

pub fn uniform_seed<T: Float>(seed: u32) -> T {
    let u = Uniform::new(0.0, 1.0);
    let mut rng = StdRng::from_seed(seed_from_u32(seed));
    let r: f64 = u.sample(&mut rng);
    T::from(r).unwrap()
}

pub trait RandomArray<T: Float + SampleUniform> {
    fn set_random(&mut self, random_seed: u32);
    fn set_random_scale(&mut self, random_seed: u32, scale: T);
    fn set_random_ord_range(&mut self, random_seed: u32, range: (T, T));
    fn set_random_ord_range_balance(&mut self, random_seed: u32, range: (T, T));
}

impl<A, S, D> RandomArray<A> for ArrayBase<S, D>
where
    A: Float + SampleUniform,
    S: DataMut<Elem = A> + RawDataClone,
    D: Dimension,
{

    fn set_random(&mut self, random_seed: u32) {
        self.set_random_scale(random_seed, A::one())
    }

    fn set_random_scale(&mut self, random_seed: u32, scale: A) {
        let mut rng = StdRng::from_seed(seed_from_u32(random_seed));
        let ud = Uniform::new::<A, A>(A::zero(), A::one());
        Zip::from(self).apply(|a| {
            *a = ud.sample(&mut rng) * scale;
        });
    }
    
    fn set_random_ord_range(&mut self, random_seed: u32, range: (A, A)) {
        let v = axis_log10(range);
        let n = v.len() - 1;

        let one = A::one();
        let dr = one / A::from(n).unwrap();
        let mut rng = StdRng::from_seed(seed_from_u32(random_seed));
        let ud = Uniform::new::<A, A>(A::zero(), one);
        Zip::from(self).apply(|a| {
            let r_ord = ud.sample(&mut rng);
            let i = (r_ord / dr).to_usize().unwrap();
            let core = v[i];
            *a = (ud.sample(&mut rng) * A::from(9).unwrap() + one) * core;
        });
    }

    fn set_random_ord_range_balance(&mut self, random_seed: u32, range: (A, A)) {
        let mut v = axis_log10(range);
        let _ = v.remove(v.len() - 1);
        let n_v = v.len();
        let n_self: usize = self.shape().iter().sum();
        let n_content = (n_self + n_v - 1) / n_v;
        let mut q = RandomQueue::new(&v, &vec![n_content; n_v], random_seed).unwrap();
    
        let one = A::one();
        let mut rng = StdRng::from_seed(seed_from_u32(flip_u32(random_seed)));
        let ud = Uniform::new::<A, A>(A::zero(), one);
        Zip::from(self).apply(|a| {
            let core = q.pop().unwrap();
            *a = (ud.sample(&mut rng) * A::from(9).unwrap() + one) * core;
        });
    }
}

pub trait RandomArrayInt<T: PrimInt> {
    fn set_random(&mut self, random_seed: u32, n: T) {
        self.set_random_range(random_seed, (T::zero(), n));
    }
    fn set_random_range(&mut self, random_seed: u32, range: (T, T));
}

impl<A, S, D> RandomArrayInt<A> for ArrayBase<S, D> 
where
    A: PrimInt,
    S: DataMut<Elem = A> + RawDataClone,
    D: Dimension,
{
    fn set_random_range(&mut self, random_seed: u32, range: (A, A)) {
        let (start, end) = range;
        if start >= end {
            panic!()
        }
        let (start_f32, end_f32) = (start.to_f32().unwrap(), end.to_f32().unwrap());
        
        let dr = 1.0 / (end_f32 - start_f32);
        let mut rng = StdRng::from_seed(seed_from_u32(random_seed));
        let ud = Uniform::new::<f32, f32>(0.0, 1.0);
        Zip::from(self).apply(|a| {
            *a = A::from(ud.sample(&mut rng) / dr).unwrap() + start;
        });
    }
}

pub trait ShuffleArray: Clone {
    fn shuffle_axis(&self, axis: Axis, random_seed: u32) -> Self {
        let mut a = self.clone();
        Self::shuffle_axis_core(&mut a, &self, axis, random_seed);
        a
    }
    fn shuffle_axis_inplace(&mut self, axis: Axis, random_seed: u32) {
        let a = self.clone();
        Self::shuffle_axis_core(self, &a, axis, random_seed);
    }
    fn shuffle_axis_core(a_save: &mut Self, a_tmp: &Self, axis: Axis, random_seed: u32);
}

impl<A, S, D> ShuffleArray for ArrayBase<S, D>
where
    A: Copy,
    S: DataMut<Elem = A> + RawDataClone,
    D: RemoveAxis,
{
    fn shuffle_axis_core(a_save: &mut Self, a_tmp: &Self, axis: Axis, random_seed: u32) {
        let n_axis = a_save.shape()[axis.index()];
        let mut index = vec![0usize; n_axis];
        shuffle_index(&mut index, random_seed);
        for (a_save_sub, &i) in a_save.axis_iter_mut(axis)
                                        .zip(index.iter())
        {
            Zip::from(a_save_sub)
                .and(&a_tmp.slice_axis(axis, Slice::from(i..i+1)).remove_axis(axis))
                .apply(|a, &b| {
                    *a = b;
                });
        }
    }
}

fn shuffle_index(v: &mut Vec<usize>, random_seed: u32) 
{
    let n = v.len();
    let ud = Uniform::new::<f64, f64>(0.0, 1.0);
    let mut rng = StdRng::from_seed(seed_from_u32(random_seed));
    let mut r = (0..n).map(|i| (ud.sample(&mut rng), i)).collect::<Vec<(f64, usize)>>();
    r.sort_by(|&(a0, _), &(b0, _)| a0.partial_cmp(&b0).unwrap());
    for (vi, &(_, i)) in v.iter_mut().zip(r.iter()) {
        *vi = i;
    }
}

pub struct RandomQueue<T> {
    pub dn: f64,
    pub contents: Vec<T>,
    pub counts: Vec<usize>,
    pub rng: StdRng,
    pub ud: Uniform<f64>
}

impl<T: Copy> RandomQueue<T> {
    pub fn new(contents: &Vec<T>, counts: &Vec<usize>, random_seed: u32) -> Result<Self, ()> {
        let contents = contents.clone();
        let counts = counts.clone();
        let n = contents.len();
        if n != counts.len() {
            return Err(());
        }

        let rng = StdRng::from_seed(seed_from_u32(random_seed));
        let ud = Uniform::new(0.0, 1.0);
        
        let dn = 1.0 / n as f64;

        Ok(
            Self { dn, contents, counts, rng, ud }
        )
    }

    pub fn pop(&mut self) -> Option<T> {
        loop {
            if self.dn == 0.0 {
                return None;
            }
            let r = self.ud.sample(&mut self.rng);
            let index = (r / self.dn) as usize;
            if self.counts[index] == 0 {
                self.drop(index);
            } else {
                self.counts[index] -= 1;
                return Some(self.contents[index]);
            }
        }
    }

    pub fn drop(&mut self, index: usize) {
        let _ = self.contents.remove(index);
        let _ = self.counts.remove(index);
        let n = self.contents.len();
        self.dn = if n > 0 {
            1.0 / n as f64
        } else {
            0.0
        };
    }
}