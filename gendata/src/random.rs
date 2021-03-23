use num_traits::{PrimInt, Float, float::FloatConst};
use rand::distributions::{uniform::SampleUniform, Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub fn flip_u32(n: u32) -> u32 {
    std::u32::MAX ^ n
}

pub fn seed_from_u32(seed: u32) -> [u8; 32] {
    let mut v = [0; 32];
    for i in 0..32 {
        v[i] = (seed >> i & 1) as u8;
    }
    v
}

pub struct DistUtil<F> 
where
    F: Float + FloatConst + SampleUniform ,
{
    pub ud: Uniform<F>,
    pub rng: StdRng,
}

impl<F> DistUtil<F>
where
    F: Float + FloatConst + SampleUniform,
{
    pub fn new() -> Self {
        let ud = Uniform::new::<F, F>(F::zero(), F::one());
        let seed = (ud.sample(&mut rand::thread_rng()) * F::from(u32::MAX).unwrap()).to_u32().unwrap();
        Self::from_seed(seed)
    }

    pub fn from_seed(seed: u32) -> Self {
        let ud = Uniform::new::<F, F>(F::zero(), F::one());
        let rng = StdRng::from_seed(seed_from_u32(seed));
        Self { ud, rng }
    }

    pub fn unin(&mut self) -> F {
        self.ud.sample(&mut self.rng)
    }

    pub fn unif(&mut self, range: (F, F)) -> F {
        range.0 + self.unin() * (range.1 - range.0)
    }

    pub fn unii<I: PrimInt>(&mut self, range: (I, I)) -> I {
        let s = F::from(range.0).unwrap();
        let e = F::from(range.1).unwrap();
        range.0 + I::from((e - s) * self.unin()).unwrap()
    }

    pub fn standard(&mut self) -> F {
        let two = F::from(2).unwrap();
        (-two * self.unin().ln()).sqrt() * (two * F::PI() * self.unin()).cos()
    }

    pub fn normal(&mut self, mu: F, sigma: F) -> F {
        mu + self.standard() * sigma
    }

    pub fn expon(&mut self, lambda: F) -> F {
        -(self.unin().ln()) / lambda
    }
}